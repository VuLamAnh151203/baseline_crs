#!/usr/bin/env python3
"""Run one or both dataset branches of the DPDP fitting pipeline.

Dataset-filtered invocations are used as disjoint one-GPU Slurm steps by the two-GPU
deadline controller.  The controller deliberately stops after self-play; development,
locking and test remain globally coordinated after both branches complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

DATASETS = {
    "TopDial": ("topdial.json", "TopDial"),
    "DuRecDial2.0": ("durecdial.json", "DuRecDial2.0"),
}
HEARTBEAT_SECONDS = 30


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def jsonl_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Pipeline:
    def __init__(
        self,
        root: Path,
        run_root: Path,
        python: Path,
        datasets: tuple[str, ...],
        configs: dict[str, Path],
        runtime_config: Path,
        worker_name: str,
    ):
        self.root = root.resolve()
        self.run_root = run_root.resolve()
        self.python = python.resolve()
        self.logs = self.run_root / "logs"
        safe_worker = worker_name.replace("/", "-")
        self.events = self.run_root / f"events-{safe_worker}.jsonl"
        self.status_path = self.run_root / f"status-{safe_worker}.json"
        self.stop_path = self.run_root / "stop.requested"
        self.active_output: Path | None = None
        self.stop_requested = False
        self.started_at = now()
        self.datasets = datasets
        self.configs = {key: value.resolve() for key, value in configs.items()}
        self.runtime_config = runtime_config.resolve()
        self.worker_name = safe_worker
        self.stage_number = 0
        self.stage_total = 6 * len(self.datasets)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, **fields) -> None:
        record = {
            "timestamp": now(),
            "event": event,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "worker": self.worker_name,
            **fields,
        }
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    def status(self, state: str, **fields) -> None:
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "pipeline": "dpdp_dataset_training",
                "worker": self.worker_name,
                "datasets": list(self.datasets),
                "state": state,
                "started_at": self.started_at,
                "updated_at": now(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "run_root": str(self.run_root),
                "stage_number": self.stage_number,
                "stage_total": self.stage_total,
                **fields,
            },
        )

    def request_stop(self, signum, _frame) -> None:
        self.stop_requested = True
        self.stop_path.touch(exist_ok=True)
        if self.active_output is not None:
            self.active_output.mkdir(parents=True, exist_ok=True)
            (self.active_output / "stop.requested").touch(exist_ok=True)
        self.event("stop_requested", signal=signum)

    def _progress(self, kind: str, output: Path, config_path: Path | None) -> dict:
        if kind == "annotate":
            prepared = output.with_name(output.name.replace(".annotated", ""))
            manifest = read_json(prepared.with_suffix(prepared.suffix + ".manifest.json")) or {}
            total = int(manifest.get("transition_count", 0)) * 10
            journal = output.parent / f"{output.name}.critic-journal"
            completed = max(
                0, len(list(journal.glob("*.json"))) - int((journal / "contract.json").exists())
            )
            return {
                "unit": "critic_requests",
                "completed": completed,
                "total": total,
                "percent": round(100 * completed / total, 2) if total else None,
            }
        if kind == "offline":
            metrics = output / "metrics.jsonl"
            rows = metrics.read_text(encoding="utf-8").splitlines() if metrics.exists() else []
            last = json.loads(rows[-1]) if rows else {}
            config = read_json(config_path) or {}
            data_manifest = (
                read_json(
                    self.run_root
                    / "data"
                    / config.get("dataset", "")
                    / "train.annotated.jsonl.manifest.json"
                )
                or {}
            )
            batch = max(1, int(config.get("offline_batch_size", 1)))
            transitions = int(data_manifest.get("transition_count", 0))
            batches = (transitions + batch - 1) // batch
            total = batches * int(config.get("offline_epochs", 0))
            completed = int(last.get("step", 0))
            return {
                "unit": "optimizer_steps",
                "completed": completed,
                "total": total,
                "epoch": last.get("epoch"),
                "batch": last.get("batch"),
                "percent": round(100 * completed / total, 2) if total else None,
            }
        if kind == "selfplay":
            config = read_json(config_path) or {}
            total = int(config.get("selfplay_epochs", 0)) * int(
                config.get("selfplay_cases_per_epoch", 0)
            )
            metrics = output / "metrics.jsonl"
            rows = metrics.read_text(encoding="utf-8").splitlines() if metrics.exists() else []
            last = json.loads(rows[-1]) if rows else {}
            completed = int(last.get("step", 0))
            return {
                "unit": "selfplay_cases",
                "completed": completed,
                "total": total,
                "epoch": last.get("epoch"),
                "case": last.get("case"),
                "scenario_id": last.get("scenario_id"),
                "percent": round(100 * completed / total, 2) if total else None,
            }
        return {"unit": "command", "completed": 0, "total": 1, "percent": 0.0}

    @staticmethod
    def _validate_complete(kind: str, output: Path, completion: Path, record: dict) -> None:
        if kind in {"prepare", "annotate"}:
            if not output.is_file():
                raise ValueError(f"complete stage lost output: {output}")
            expected = record.get("output_sha256")
            if not isinstance(expected, str) or file_hash(output) != expected:
                raise ValueError(f"complete stage output hash mismatch: {output}")
        elif kind == "offline":
            if not (output / "checkpoint-final.pt").is_file():
                raise ValueError("complete offline stage lost checkpoint-final.pt")
        elif kind == "selfplay":
            if not (output / "checkpoint-final.pt").is_file():
                raise ValueError("complete self-play stage lost checkpoint-final.pt")
        if not completion.is_file():
            raise ValueError(f"complete stage lost completion artifact: {completion}")

    def run_stage(
        self,
        name: str,
        command: list[str],
        *,
        kind: str,
        output: Path,
        completion: Path,
        config_path: Path | None = None,
    ) -> None:
        self.stage_number += 1
        complete = read_json(completion)
        if complete and complete.get("status") == "complete":
            self._validate_complete(kind, output, completion, complete)
            self.event("stage_skipped_complete", stage=name, artifact=str(completion))
            self.status("running", stage=name, stage_state="skipped_complete")
            return
        if self.stop_requested or self.stop_path.exists():
            raise InterruptedError("stop requested before next stage")
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{self.worker_name}-{self.stage_number:02d}-{name}.log"
        started_at = now()
        self.active_output = output if kind in {"offline", "selfplay"} else None
        self.event(
            "stage_started",
            stage=name,
            stage_number=self.stage_number,
            stage_total=self.stage_total,
            command=command,
            log=str(log_path),
        )
        self.status(
            "running",
            stage=name,
            stage_state="running",
            stage_started_at=started_at,
            stage_log=str(log_path),
            progress=self._progress(kind, output, config_path),
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n=== slurm_job_id={os.environ.get('SLURM_JOB_ID')} "
                f"stage={name} started_at={started_at} ===\n"
            )
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            def copy_output() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{name}] {line}", end="", flush=True)

            reader = threading.Thread(target=copy_output, daemon=True)
            reader.start()
            while process.poll() is None:
                progress = self._progress(kind, output, config_path)
                self.status(
                    "running",
                    stage=name,
                    stage_state="running",
                    stage_started_at=started_at,
                    stage_log=str(log_path),
                    progress=progress,
                )
                self.event("heartbeat", stage=name, progress=progress)
                if self.stop_path.exists() and kind in {"offline", "selfplay"}:
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "stop.requested").touch(exist_ok=True)
                try:
                    process.wait(timeout=HEARTBEAT_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            reader.join()
            returncode = process.wait()
        self.active_output = None
        if returncode != 0:
            self.event("stage_failed", stage=name, returncode=returncode, log=str(log_path))
            raise subprocess.CalledProcessError(returncode, command)
        complete = read_json(completion)
        if not complete or complete.get("status") != "complete":
            if self.stop_requested or self.stop_path.exists():
                raise InterruptedError(f"{name} stopped at a durable boundary")
            raise RuntimeError(f"{name} exited without complete artifact: {completion}")
        self.event("stage_completed", stage=name, artifact=str(completion))
        self.status("running", stage=name, stage_state="complete", progress={"percent": 100.0})

    def command(self, *values: object) -> list[str]:
        return [str(self.python), "-m", "tdrl.baselines.dpdp.cli", *(str(v) for v in values)]

    def preflight(self) -> Path:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()

        def label(path: Path) -> str:
            path = path.resolve()
            try:
                return str(path.relative_to(self.root))
            except ValueError:
                return str(path)

        metadata = {
            "schema_version": 1,
            "status": "ready" if not dirty else "blocked_dirty_tree",
            "created_at": now(),
            "git_commit": commit,
            "git_dirty": bool(dirty),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "python": str(self.python),
            "configs": {
                label(path): file_hash(path)
                for path in (
                    self.root / "configs/comparison.dpdp.json",
                    self.root / "configs/runtime.json",
                    self.root / "configs/dpdp.models.lock.json",
                    *(self.configs[dataset] for dataset in self.datasets),
                    self.runtime_config,
                )
            },
        }
        metadata_path = self.run_root / f"run-metadata-{self.worker_name}.json"
        atomic_json(metadata_path, metadata)
        if dirty:
            raise ValueError(
                "scientific DPDP training requires a clean committed worktree; "
                "commit or stash reviewed changes before submission"
            )
        return metadata_path

    def run(self) -> None:
        revision = (read_json(self.root / "configs/dpdp.models.lock.json") or {}).get("revision")
        if not revision:
            raise ValueError("DPDP RoBERTa revision is not pinned")
        model = self.root / "models/roberta-large" / revision
        if not model.is_dir():
            raise FileNotFoundError(model)
        common_model = (
            "--encoder",
            model,
            "--tokenizer",
            model,
            "--model-lock",
            self.root / "configs/dpdp.models.lock.json",
        )
        comparison = self.root / "configs/comparison.dpdp.json"
        runtime = self.runtime_config
        self.event(
            "pipeline_started",
            run_root=str(self.run_root),
            worker=self.worker_name,
            datasets=list(self.datasets),
        )
        self.status("running", stage="preflight", stage_state="running")
        metadata_path = self.preflight()
        self.event("preflight_completed", metadata=str(metadata_path))

        for dataset in self.datasets:
            _config_name, data_name = DATASETS[dataset]
            config = self.configs[dataset]
            for split in ("train", "dev"):
                output = self.run_root / "data" / dataset / f"{split}.jsonl"
                self.run_stage(
                    f"prepare-{dataset}-{split}".lower(),
                    self.command(
                        "prepare",
                        "--input",
                        self.root / "data/processed" / data_name / f"{split}.jsonl",
                        "--output",
                        output,
                        "--dataset",
                        dataset,
                        "--split",
                        split,
                    ),
                    kind="prepare",
                    output=output,
                    completion=output.with_suffix(output.suffix + ".manifest.json"),
                )
            for split in ("train", "dev"):
                source = self.run_root / "data" / dataset / f"{split}.jsonl"
                output = self.run_root / "data" / dataset / f"{split}.annotated.jsonl"
                self.run_stage(
                    f"annotate-{dataset}-{split}".lower(),
                    self.command(
                        "annotate-rewards",
                        "--root",
                        self.root,
                        "--runtime-config",
                        runtime,
                        "--config",
                        config,
                        "--input",
                        source,
                        "--output",
                        output,
                    ),
                    kind="annotate",
                    output=output,
                    completion=output.with_suffix(output.suffix + ".manifest.json"),
                    config_path=config,
                )

        for dataset in self.datasets:
            config = self.configs[dataset]
            output = self.run_root / "offline" / dataset
            resume = output / "checkpoint-last.pt"
            command = self.command(
                "train",
                "--config",
                config,
                "--train",
                self.run_root / "data" / dataset / "train.annotated.jsonl",
                "--dev",
                self.run_root / "data" / dataset / "dev.annotated.jsonl",
                *common_model,
                "--output",
                output,
            )
            if resume.exists():
                command.extend(("--resume", str(resume)))
            self.run_stage(
                f"offline-{dataset}".lower(),
                command,
                kind="offline",
                output=output,
                completion=output / "training-summary.json",
                config_path=config,
            )

        for dataset in self.datasets:
            _config_name, data_name = DATASETS[dataset]
            config = self.configs[dataset]
            output = self.run_root / "selfplay" / dataset
            resume = output / "checkpoint-last.pt"
            command = self.command(
                "selfplay",
                "--root",
                self.root,
                "--runtime-config",
                runtime,
                "--config",
                config,
                "--scenarios",
                self.root / "data/processed" / data_name / "train.jsonl",
                "--offline-checkpoint",
                self.run_root / "offline" / dataset / "checkpoint-final.pt",
                *common_model,
                "--output",
                output,
            )
            if resume.exists():
                command.extend(("--resume", str(resume)))
            self.run_stage(
                f"selfplay-{dataset}".lower(),
                command,
                kind="selfplay",
                output=output,
                completion=output / "selfplay-summary.json",
                config_path=config,
            )

        result = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": "dpdp_dataset_training",
            "worker": self.worker_name,
            "datasets": list(self.datasets),
            "started_at": self.started_at,
            "ended_at": now(),
            "comparison_config": str(comparison),
            "next_gate": "development_gate_and_budget_selection",
            "test_bodies_opened": False,
        }
        atomic_json(self.run_root / f"training-pipeline-summary-{self.worker_name}.json", result)
        self.event("pipeline_completed", next_gate=result["next_gate"])
        self.status(
            "complete",
            stage="training-complete",
            stage_state="complete",
            next_gate=result["next_gate"],
            test_bodies_opened=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPDP training stages in one allocation")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    parser.add_argument("--config-topdial", type=Path)
    parser.add_argument("--config-durecdial", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--worker-name")
    args = parser.parse_args()
    datasets = tuple(args.dataset or DATASETS)
    configs = {
        "TopDial": args.config_topdial or args.root / "configs/dpdp/topdial.json",
        "DuRecDial2.0": args.config_durecdial or args.root / "configs/dpdp/durecdial.json",
    }
    runtime_config = args.runtime_config or args.root / "configs/runtime.json"
    worker_name = args.worker_name or (datasets[0] if len(datasets) == 1 else "all")
    pipeline = Pipeline(
        args.root,
        args.run_root,
        args.python,
        datasets,
        configs,
        runtime_config,
        worker_name,
    )
    signal.signal(signal.SIGUSR1, pipeline.request_stop)
    signal.signal(signal.SIGTERM, pipeline.request_stop)
    try:
        pipeline.run()
    except InterruptedError as error:
        pipeline.event("pipeline_stopped", reason=str(error))
        pipeline.status("partial", stage_state="stopped", reason=str(error))
        raise SystemExit(75) from error
    except Exception as error:
        pipeline.event("pipeline_failed", error=type(error).__name__, message=str(error))
        pipeline.status(
            "failed", stage_state="failed", error=type(error).__name__, message=str(error)
        )
        raise


if __name__ == "__main__":
    main()
