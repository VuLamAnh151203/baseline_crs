#!/usr/bin/env python3
"""Run source-corrected EPL train/dev/selection/fixed-prefix evaluation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tdrl.baseline_test_setting import load_fixed_test_setting
from tdrl.baselines.epl.config import EPLConfig
from tdrl.baselines.epl.selection import verify_selection
from tdrl.provenance import atomic_json, file_hash

DATASETS = ("TopDial", "DuRecDial2.0")
CONFIGS = {
    "TopDial": Path("configs/epl/topdial.json"),
    "DuRecDial2.0": Path("configs/epl/durecdial.json"),
}
FIXED_SETTING = Path("configs/baseline-fixed1000-test-v1.json")
PORT_START, PORT_STOP, PORTS_PER_JOB = 30_000, 40_000, 4
PORT_OFFSETS = {"TopDial": (0, 1), "DuRecDial2.0": (2, 3)}


def now():
    return datetime.now(timezone.utc).isoformat()


def _ports_available(ports):
    probes = []
    try:
        for port in ports:
            probe = socket.socket()
            probes.append(probe)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
            probe.listen(1)
        return True
    except OSError:
        return False
    finally:
        for probe in probes:
            probe.close()


def select_ports(job_id):
    blocks = (PORT_STOP - PORT_START) // PORTS_PER_JOB
    initial = int(job_id) % blocks
    for offset in range(blocks):
        base = PORT_START + ((initial + offset) % blocks) * PORTS_PER_JOB
        ports = tuple(base + index for index in range(PORTS_PER_JOB))
        if _ports_available(ports):
            return {
                dataset: tuple(base + item for item in offsets)
                for dataset, offsets in PORT_OFFSETS.items()
            }
    raise OSError("no free EPL runtime port block")


class EPLE2E:
    def __init__(self, root, run_root, python, mode):
        self.root, self.run_root = Path(root).resolve(), Path(run_root).resolve()
        self.python, self.mode = Path(python).resolve(), mode
        self.fixed_count = {"mini": None, "initial400": 400, "extend1000": 1000}[mode]
        self.config_paths = CONFIGS
        self.comparison_path = self.root / (
            "configs/comparison.epl-mini.json"
            if mode == "mini"
            else "configs/comparison.epl.json"
        )
        self.logs = self.run_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_root / "events.jsonl"
        self.status_path = self.run_root / "status.json"
        self.stop_path = self.run_root / "stop.requested"
        self.children = []
        self.started_at = now()

    def event(self, event, **fields):
        record = {
            "timestamp": now(),
            "event": event,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "mode": self.mode,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(record, sort_keys=True), flush=True)

    def status(self, state, stage, **fields):
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "pipeline": "epl_source_corrected_v2_two_gpu",
                "state": state,
                "stage": stage,
                "mode": self.mode,
                "fixed_test_count": self.fixed_count,
                "started_at": self.started_at,
                "updated_at": now(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "run_root": str(self.run_root),
                **fields,
            },
        )

    def stop(self, signum, _frame):
        self.stop_path.touch(exist_ok=True)
        self.event("stop_requested", signal=signum)
        for child in self.children:
            if child.poll() is None:
                child.terminate()

    def preflight(self):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("scientific EPL E2E must run inside Slurm")
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()
        if dirty:
            raise ValueError("scientific EPL E2E requires a clean committed worktree")
        self.commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        required = (
            self.root / "runtime/uv.lock",
            self.root / "toolchains/cuda-12.9.1/bin/nvcc",
            self.root / "data/processed/TopDial/train.jsonl",
            self.root / "data/processed/DuRecDial2.0/train.jsonl",
            self.root / "configs/epl-reference.lock.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing EPL runtime assets: " + ", ".join(missing))
        source_lock = json.loads(
            (self.root / "configs/epl-reference.lock.json").read_text(encoding="utf-8")
        )
        if (
            source_lock.get("status") != "source_pinned_and_audited"
            or source_lock.get("revision")
            != "364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda"
        ):
            raise ValueError("unexpected EPL source revision lock")
        asset_root = Path(
            os.environ.get("EPL_MODEL_ROOT", "/mnt/data/u26anhvl/crs/epl/models")
        ).resolve()
        external_lock = Path(
            os.environ.get("EPL_MODEL_LOCK", asset_root / "epl.models.lock.json")
        ).resolve()
        self.model_lock = self.root / "configs/epl.models.lock.json"
        if json.loads(external_lock.read_text(encoding="utf-8")) != json.loads(
            self.model_lock.read_text(encoding="utf-8")
        ):
            raise ValueError("external EPL model lock differs from the committed lock")
        lock = json.loads(self.model_lock.read_text(encoding="utf-8"))
        self.encoder = asset_root / "bert-base-cased" / lock["policy_encoder"]["revision"]
        self.retrieval_encoder = (
            asset_root
            / "all-MiniLM-L6-v2"
            / lock["retrieval_encoder"]["revision"]
        )
        missing_assets = [
            str(path)
            for path in (self.encoder, self.retrieval_encoder)
            if not path.is_dir()
        ]
        if missing_assets:
            raise FileNotFoundError("missing EPL native model assets: " + ", ".join(missing_assets))
        self.configs = {
            dataset: self.root / path for dataset, path in self.config_paths.items()
        }
        for dataset, path in self.configs.items():
            config = EPLConfig.from_path(path)
            expected = (5, 100.0, 5, 376)
            if config.dataset != dataset or (
                config.puct_simulations,
                config.puct_cpuct,
                config.puct_max_depth,
                config.memory_cases,
            ) != expected:
                raise ValueError(f"unexpected frozen EPL config: {path}")
        self.setting_path = self.root / FIXED_SETTING
        self.setting = (
            None
            if self.mode == "mini"
            else load_fixed_test_setting(self.root, self.setting_path, self.fixed_count)
        )
        metadata_path = self.run_root / "run-metadata.json"
        if self.mode in {"mini", "initial400"}:
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("git_commit") != self.commit:
                    raise ValueError("EPL initial run root belongs to another source commit")
            else:
                atomic_json(
                    metadata_path,
                    {
                        "schema_version": 1,
                        "pipeline": "EPL-source-corrected-v2",
                        "git_commit": self.commit,
                        "created_at": now(),
                        "initial_job_id": os.environ["SLURM_JOB_ID"],
                        "fixed_setting_sha256": (
                            self.setting["sha256"] if self.setting is not None else None
                        ),
                        "initial_prefix": self.fixed_count,
                        "extension_prefix": 1000 if self.mode == "initial400" else None,
                        "diagnostic_only": self.mode == "mini",
                    },
                )
        else:
            if not metadata_path.is_file():
                raise FileNotFoundError("EPL extension requires the original run-metadata.json")
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("git_commit") != self.commit:
                raise ValueError("EPL extension must use the exact original source commit")
            summary400 = self.run_root / "e2e-summary-fixed1000-prefix-400.json"
            if not summary400.is_file() or json.loads(summary400.read_text()).get("status") != "complete":
                raise ValueError("EPL extension requires a completed immutable prefix 400")
            verify_selection(self.run_root / "evaluation/selection.lock.json")

    def materialize_runtime(self):
        base = json.loads((self.root / "configs/runtime.json").read_text())
        ports = select_ports(os.environ["SLURM_JOB_ID"])
        self.runtime = {}
        for dataset, assigned in ports.items():
            config = json.loads(json.dumps(base))
            config["generation"]["port"] = assigned[0]
            config["embedding"]["port"] = assigned[1]
            path = self.run_root / "runtime" / os.environ["SLURM_JOB_ID"] / f"{dataset}.json"
            atomic_json(path, config)
            self.runtime[dataset] = path
        atomic_json(
            self.run_root / "runtime" / os.environ["SLURM_JOB_ID"] / "ports.json",
            {"schema_version": 1, "job_id": os.environ["SLURM_JOB_ID"], "ports": ports},
        )

    def srun(self, command):
        return [
            "srun",
            "--exclusive",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=8",
            "--gpus-per-task=1",
            "--gpu-bind=single:1",
            "--mem=64G",
            *command,
        ]

    def parallel(self, stage, commands):
        if self.stop_path.exists():
            raise InterruptedError(f"stop requested before {stage}")
        self.status("running", stage, workers=list(commands))
        self.event("stage_started", stage=stage, workers=list(commands))
        handles = []
        self.children = []
        try:
            for name, command in commands.items():
                log = self.logs / f"{stage}-{name}-{os.environ['SLURM_JOB_ID']}.log"
                handle = log.open("a", encoding="utf-8")
                handles.append(handle)
                self.children.append(
                    subprocess.Popen(
                        self.srun(command),
                        cwd=self.root,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                )
            while True:
                codes = [child.poll() for child in self.children]
                failure = next((code for code in codes if code not in (None, 0)), None)
                if failure is not None:
                    for child in self.children:
                        if child.poll() is None:
                            child.terminate()
                    raise subprocess.CalledProcessError(failure, stage)
                if all(code == 0 for code in codes):
                    break
                if self.stop_path.exists():
                    raise InterruptedError(f"stop requested during {stage}")
                time.sleep(2)
        finally:
            for child in self.children:
                if child.poll() is None:
                    child.terminate()
            for handle in handles:
                handle.close()
            self.children = []
        self.event("stage_completed", stage=stage)

    def artifacts(self):
        return {
            "checkpoints": {
                dataset: self.run_root / "training" / dataset / "checkpoint-best.pt"
                for dataset in DATASETS
            },
            "memories": {
                dataset: self.run_root / "memory" / dataset / "experiences.jsonl"
                for dataset in DATASETS
            },
            "vectors": {
                dataset: self.run_root / "memory" / dataset / "vectors.pt"
                for dataset in DATASETS
            },
        }

    def common_args(self):
        items = self.artifacts()
        return [
            "--root", str(self.root),
            "--comparison-config", str(self.comparison_path),
            "--checkpoint-topdial", str(items["checkpoints"]["TopDial"]),
            "--checkpoint-durecdial", str(items["checkpoints"]["DuRecDial2.0"]),
            "--epl-config-topdial", str(self.configs["TopDial"]),
            "--epl-config-durecdial", str(self.configs["DuRecDial2.0"]),
            "--memory-topdial", str(items["memories"]["TopDial"]),
            "--memory-durecdial", str(items["memories"]["DuRecDial2.0"]),
            "--vectors-topdial", str(items["vectors"]["TopDial"]),
            "--vectors-durecdial", str(items["vectors"]["DuRecDial2.0"]),
            "--encoder", str(self.encoder),
            "--tokenizer", str(self.encoder),
            "--retrieval-encoder", str(self.retrieval_encoder),
            "--model-lock", str(self.model_lock),
            "--source-lock", str(self.root / "configs/epl-reference.lock.json"),
        ]

    def evaluate(self, split, selection=None):
        commands = {}
        for dataset in DATASETS:
            output = self.run_root / "evaluation" / f"{split}-workers" / dataset
            command = [
                str(self.python),
                "-m", "tdrl.baselines.epl.evaluate_cli", "evaluate",
                *self.common_args(),
                "--output", str(output),
                "--split", split,
                "--dataset", dataset,
                "--runtime-config", str(self.runtime[dataset]),
            ]
            if selection is not None:
                command.extend(("--selection-lock", str(selection)))
            if split == "test":
                command.extend(
                    (
                        "--fixed-test-setting", str(self.setting_path),
                        "--fixed-test-count", str(self.fixed_count),
                    )
                )
            commands[dataset] = command
        prefix = self.fixed_count if self.fixed_count is not None else "mini"
        self.parallel(f"evaluate-{split}-prefix-{prefix}", commands)
        leaf = "test-fixed1000" if split == "test" else "dev"
        filename = "epl-report.json" if split == "dev" else f"epl-report-fixed1000-prefix-{self.fixed_count}.json"
        inputs = [
            self.run_root / "evaluation" / f"{split}-workers" / dataset / leaf / filename
            for dataset in DATASETS
        ]
        output = self.run_root / "evaluation" / leaf / filename
        subprocess.run(
            [
                str(self.python), "scripts/epl/merge_evaluation.py",
                "--input", str(inputs[0]), "--input", str(inputs[1]), "--output", str(output),
            ],
            cwd=self.root,
            check=True,
        )
        return output

    def lock_selection(self, development):
        path = self.run_root / "evaluation/selection.lock.json"
        if path.exists():
            verify_selection(path)
            return path
        subprocess.run(
            [
                str(self.python), "-m", "tdrl.baselines.epl.evaluate_cli", "lock-selection",
                *self.common_args(),
                "--development-evidence", str(development),
                "--output", str(path),
            ],
            cwd=self.root,
            check=True,
        )
        return path

    def smoke(self, selection):
        artifacts = self.artifacts()
        commands = {
            dataset: [
                str(self.python),
                "scripts/epl/smoke.py",
                "--root", str(self.root),
                "--output", str(self.run_root / "smoke" / dataset),
                "--dataset", dataset,
                "--config", str(self.configs[dataset]),
                "--checkpoint", str(artifacts["checkpoints"][dataset]),
                "--memory", str(artifacts["memories"][dataset]),
                "--vectors", str(artifacts["vectors"][dataset]),
                "--encoder", str(self.encoder),
                "--retrieval-encoder", str(self.retrieval_encoder),
                "--model-lock", str(self.model_lock),
                "--selection-lock", str(selection),
                "--runtime-config", str(self.runtime[dataset]),
                "--comparison-config", str(self.comparison_path),
            ]
            for dataset in DATASETS
        }
        self.parallel("post-selection-smoke", commands)

    def feasibility(self, development):
        report = json.loads(Path(development).read_text())
        worker_wall = report["parallel_worker_wall_s"]
        dev_dialogues_per_dataset = self.config_development_dialogues()
        test_dialogues_per_dataset = (self.fixed_count // 2) * 2
        projected = {
            dataset: worker_wall[dataset] / dev_dialogues_per_dataset * test_dialogues_per_dataset
            for dataset in DATASETS
        }
        required = (max(projected.values()) + 4 * 3600) * 1.25
        remaining = self.remaining_wall_seconds()
        payload = {
            "schema_version": 1,
            "status": "passed",
            "prefix_count": self.fixed_count,
            "development_dialogues_per_dataset": dev_dialogues_per_dataset,
            "test_dialogues_per_dataset": test_dialogues_per_dataset,
            "projected_worker_seconds": projected,
            "required_wall_seconds_with_overhead_and_safety": required,
            "remaining_allocation_seconds": remaining,
            "fits": required <= remaining,
            "note": "measured development extrapolation with four-hour overhead and 1.25 safety",
        }
        payload["status"] = "passed" if payload["fits"] else "failed"
        atomic_json(self.run_root / f"feasibility-prefix-{self.fixed_count}.json", payload)
        if not payload["fits"]:
            raise RuntimeError("EPL fixed-prefix evaluation does not fit the remaining allocation")
        return payload

    @staticmethod
    def remaining_wall_seconds():
        value = os.environ.get("SLURM_JOB_END_TIME")
        if value and value.isdigit():
            return max(0.0, float(value) - time.time())
        raw = subprocess.check_output(
            ["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%L"],
            text=True,
        ).strip()
        if not raw or raw.upper() in {"INVALID", "UNLIMITED", "NOT_SET"}:
            raise RuntimeError("cannot determine remaining Slurm allocation time")
        days = 0
        if "-" in raw:
            day_text, raw = raw.split("-", 1)
            days = int(day_text)
        parts = [int(item) for item in raw.split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, (minutes, seconds) = 0, parts
        else:
            raise RuntimeError(f"unexpected Slurm remaining-time format: {raw}")
        return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)

    def config_development_dialogues(self):
        config = json.loads(self.comparison_path.read_text())
        return config["development_scenarios"] * len(config["protocols"])

    def run(self):
        self.status("running", "preflight")
        self.preflight()
        self.materialize_runtime()
        selection = self.run_root / "evaluation/selection.lock.json"
        development = self.run_root / "evaluation/dev/epl-report.json"
        if self.mode in {"mini", "initial400"} and not selection.exists():
            training = {
                dataset: [
                    str(self.python), "scripts/epl/train.py",
                    "--root", str(self.root),
                    "--run-root", str(self.run_root),
                    "--dataset", dataset,
                    "--config", str(self.configs[dataset]),
                    "--encoder", str(self.encoder),
                    "--retrieval-encoder", str(self.retrieval_encoder),
                    "--model-lock", str(self.model_lock),
                ]
                for dataset in DATASETS
            }
            if self.mode == "mini":
                for command in training.values():
                    command.extend(
                        ("--max-train-examples", "4", "--max-dev-examples", "2")
                    )
            self.parallel("training", training)
            self_simulation = {
                dataset: [
                    str(self.python), "scripts/epl/self_simulate.py",
                    "--root", str(self.root),
                    "--run-root", str(self.run_root),
                    "--dataset", dataset,
                    "--config", str(self.configs[dataset]),
                    "--checkpoint", str(
                        self.run_root / "training" / dataset / "checkpoint-best.pt"
                    ),
                    "--encoder", str(self.encoder),
                    "--retrieval-encoder", str(self.retrieval_encoder),
                    "--model-lock", str(self.model_lock),
                    "--runtime-config", str(self.runtime[dataset]),
                    "--comparison-config", str(self.comparison_path),
                ]
                for dataset in DATASETS
            }
            if self.mode == "mini":
                for command in self_simulation.values():
                    command.extend(("--max-cases", "1"))
            self.parallel("train-self-simulation", self_simulation)
            indexing = {
                dataset: [
                    str(self.python), "scripts/epl/build_memory_index.py",
                    "--root", str(self.root),
                    "--run-root", str(self.run_root),
                    "--dataset", dataset,
                    "--config", str(self.configs[dataset]),
                    "--encoder", str(self.encoder),
                    "--retrieval-encoder", str(self.retrieval_encoder),
                    "--model-lock", str(self.model_lock),
                ]
                for dataset in DATASETS
            }
            self.parallel("memory-index", indexing)
            development = self.evaluate("dev")
            selection = self.lock_selection(development)
        else:
            envelope = verify_selection(selection)
            if not development.is_file() or file_hash(development) != envelope["payload"]["development_evidence_sha256"]:
                raise ValueError("EPL selection lost its immutable development evidence")
        self.smoke(selection)
        if self.mode == "mini":
            summary = {
                "schema_version": 1,
                "status": "complete",
                "pipeline": "EPL-source-corrected-v2-mini-e2e",
                "mode": self.mode,
                "started_at": self.started_at,
                "ended_at": now(),
                "source_commit": self.commit,
                "selection_lock": str(selection),
                "selection_lock_sha256": file_hash(selection),
                "development_report": str(development),
                "official_test_opened": False,
                "claim_limit": "diagnostic train/dev/replay smoke only",
            }
            summary_path = self.run_root / "e2e-summary-mini.json"
            atomic_json(summary_path, summary)
            self.status("complete", "complete", summary=str(summary_path))
            self.event("pipeline_completed", summary=str(summary_path))
            return
        self.feasibility(development)
        test_report = self.evaluate("test", selection)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": "EPL-source-corrected-v2",
            "mode": self.mode,
            "started_at": self.started_at,
            "ended_at": now(),
            "source_commit": self.commit,
            "selection_lock": str(selection),
            "selection_lock_sha256": file_hash(selection),
            "development_report": str(development),
            "test_report": str(test_report),
            "test_report_sha256": file_hash(test_report),
            "official_test_scenarios": self.fixed_count,
            "official_test_scenarios_per_dataset": self.fixed_count // 2,
            "protocols": ["profile", "resistant"],
            "budget": "natural",
            "claim_limit": "single-seed sampled official-test prefix; EPL only",
        }
        summary_path = self.run_root / f"e2e-summary-fixed1000-prefix-{self.fixed_count}.json"
        atomic_json(summary_path, summary)
        self.status("complete", "complete", summary=str(summary_path))
        self.event("pipeline_completed", summary=str(summary_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--mode", choices=("mini", "initial400", "extend1000"), required=True
    )
    args = parser.parse_args()
    pipeline = EPLE2E(args.root, args.run_root, args.python, args.mode)
    signal.signal(signal.SIGUSR1, pipeline.stop)
    signal.signal(signal.SIGTERM, pipeline.stop)
    try:
        pipeline.run()
    except InterruptedError as error:
        pipeline.event("pipeline_stopped", reason=str(error))
        pipeline.status("partial", "stopped", reason=str(error))
        raise SystemExit(75) from error
    except Exception as error:
        pipeline.event("pipeline_failed", error=type(error).__name__, message=str(error))
        pipeline.status("failed", "failed", error=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
