#!/usr/bin/env python3
"""Run the two-GPU DPDP-250 train/dev/lock/fixed-prefix test workflow."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tdrl.baseline_test_setting import load_fixed_test_setting
from tdrl.baselines.dpdp.config import DPDPConfig
from tdrl.provenance import atomic_json, file_hash

DATASETS = ("TopDial", "DuRecDial2.0")
CONFIG_NAMES = {
    "TopDial": "topdial-deadline250.json",
    "DuRecDial2.0": "durecdial-deadline250.json",
}
PORT_BLOCK_START = 20_000
PORT_BLOCK_STOP = 30_000
PORTS_PER_JOB = 4
PORT_OFFSETS = {
    "TopDial": (0, 1),
    "DuRecDial2.0": (2, 3),
}
FIXED_TEST_SETTING = Path("configs/baseline-fixed1000-test-v1.json")
WORKER_MEMORY_GIB = 72
PARALLEL_SLOT_COUNT = 2
STOP_GRACE_SECONDS = 600


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _port_block_available(ports: tuple[int, ...]) -> bool:
    """Probe a complete per-job block while holding every successful bind."""
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


def select_runtime_ports(job_id: str, block_available=_port_block_available) -> dict:
    """Select a free, job-specific block instead of shared fixed localhost ports."""
    numeric_job_id = int(job_id)
    block_count = (PORT_BLOCK_STOP - PORT_BLOCK_START) // PORTS_PER_JOB
    initial_block = numeric_job_id % block_count
    for offset in range(block_count):
        block_index = (initial_block + offset) % block_count
        base = PORT_BLOCK_START + block_index * PORTS_PER_JOB
        block = tuple(base + index for index in range(PORTS_PER_JOB))
        if block_available(block):
            return {
                dataset: tuple(base + port_offset for port_offset in offsets)
                for dataset, offsets in PORT_OFFSETS.items()
            }
    raise OSError(f"no free {PORTS_PER_JOB}-port block in [{PORT_BLOCK_START}, {PORT_BLOCK_STOP})")


class DeadlineE2E:
    def __init__(
        self,
        root: Path,
        run_root: Path,
        python: Path,
        fixed_test_count: int,
        resume: bool = False,
    ):
        self.root = root.resolve()
        self.run_root = run_root.resolve()
        self.python = python.resolve()
        if fixed_test_count not in (400, 1000):
            raise ValueError("DPDP fixed-test count must be 400 or 1000")
        self.fixed_test_count = fixed_test_count
        self.logs = self.run_root / "logs"
        self.status_path = self.run_root / "status.json"
        self.events_path = self.run_root / "events.jsonl"
        self.stop_path = self.run_root / "stop.requested"
        self.children: list[subprocess.Popen] = []
        self.started_at = now()
        self.resume = resume
        self.job_id = os.environ.get("SLURM_JOB_ID", "local")
        self.restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
        self.segment_id = (
            self.job_id if self.restart_count == 0 else f"{self.job_id}-r{self.restart_count}"
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, **fields) -> None:
        record = {
            "timestamp": now(),
            "event": event,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    def status(self, state: str, stage: str, **fields) -> None:
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "pipeline": "dpdp_deadline250_two_gpu_e2e",
                "state": state,
                "stage": stage,
                "started_at": self.started_at,
                "updated_at": now(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "run_root": str(self.run_root),
                **fields,
            },
        )

    def stop(self, signum, _frame) -> None:
        self.stop_path.touch(exist_ok=True)
        self.event("stop_requested", signal=signum)

    def prepare_run_segment(self) -> None:
        prior_evidence = (
            self.status_path.exists()
            or self.events_path.exists()
            or any(self.run_root.glob("run-metadata-*.json"))
            or (self.run_root / "offline").exists()
            or (self.run_root / "selfplay").exists()
            or (self.run_root / "evaluation").exists()
        )
        if self.resume and not prior_evidence:
            raise ValueError("resume requested but run root has no prior DPDP evidence")
        if not self.resume and prior_evidence:
            raise ValueError(
                "fresh DPDP run refuses an existing run root; use the explicit resume wrapper"
            )
        markers = [
            self.stop_path,
            *(
                self.run_root / phase / dataset / "stop.requested"
                for phase in ("offline", "selfplay")
                for dataset in DATASETS
            ),
        ]
        present = [path for path in markers if path.exists()]
        if present and not self.resume:
            raise ValueError("fresh DPDP run found stale stop markers")
        archived = []
        if present:
            archive = self.run_root / "allocation" / "resume-markers" / self.segment_id
            archive.mkdir(parents=True, exist_ok=False)
            for path in present:
                destination = archive / str(path.relative_to(self.run_root)).replace("/", "__")
                path.replace(destination)
                archived.append({"source": str(path), "archive": str(destination)})
        segment = {
            "schema_version": 1,
            "status": "running",
            "started_at": self.started_at,
            "slurm_job_id": self.job_id,
            "slurm_restart_count": self.restart_count,
            "segment_id": self.segment_id,
            "resume": self.resume,
            "run_root": str(self.run_root),
            "archived_stop_markers": archived,
        }
        atomic_json(self.run_root / "allocation" / f"segment-{self.segment_id}.json", segment)
        self.event(
            "run_segment_started",
            resume=self.resume,
            archived_stop_markers=archived,
        )

    def preflight(self) -> None:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("DPDP deadline E2E must run inside Slurm")
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()
        if dirty:
            raise ValueError(
                "scientific DPDP E2E requires a clean committed worktree; "
                f"dirty entries:\n{dirty}"
            )
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        lock = json.loads((self.root / "configs/dpdp.models.lock.json").read_text(encoding="utf-8"))
        revision = lock.get("revision")
        self.model = self.root / "models/roberta-large" / str(revision)
        if not self.model.is_dir():
            raise FileNotFoundError(self.model)
        self.configs = {
            dataset: self.root / "configs/dpdp" / CONFIG_NAMES[dataset] for dataset in DATASETS
        }
        self.fixed_test_setting_path = self.root / FIXED_TEST_SETTING
        self.fixed_test_setting = load_fixed_test_setting(
            self.root, self.fixed_test_setting_path, self.fixed_test_count
        )
        for dataset, path in self.configs.items():
            config = DPDPConfig.from_path(path)
            if (
                config.dataset != dataset
                or config.selfplay_epochs != 5
                or config.selfplay_cases_per_epoch != 50
                or config.mcts_simulations != 10
                or config.mcts_max_depth != 5
                or config.critic_samples != 10
                or config.offline_dropout != 0.1
                or config.selfplay_dropout != 0.25
                or config.offline_epochs != 10
                or config.offline_batch_size != 8
                or config.offline_lr != 6e-6
                or config.offline_weight_decay != 0.001
                or config.offline_warmup_steps != 400
                or config.offline_seed != 42
                or config.selfplay_seed != 1
            ):
                raise ValueError(f"unexpected DPDP-250 config: {path}")
        metadata = {
            "schema_version": 1,
            "status": "ready",
            "pipeline": "DPDP-250",
            "git_commit": commit,
            "git_dirty": False,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "requested_gpus": 2,
            "walltime": "6-00:00:00",
            "configs": {
                str(path.relative_to(self.root)): file_hash(path) for path in self.configs.values()
            },
            "fixed_test_setting": str(FIXED_TEST_SETTING),
            "fixed_test_setting_sha256": self.fixed_test_setting["sha256"],
            "fixed_test_count": self.fixed_test_count,
            "fixed_test_per_dataset": self.fixed_test_count // 2,
            "official_test_budgets": ["natural"],
            "matched_budget_status": "minimum_grid_proxy_without_measured_tdrl_reference",
            "matched_budget": {"simulations": 2, "depth": 1},
            "claim_limit": "matched output is not an official compute-matched claim",
        }
        if self.resume:
            prior = [
                read
                for path in sorted(self.run_root.glob("run-metadata-*.json"))
                if path.name != f"run-metadata-{self.segment_id}.json"
                for read in [json.loads(path.read_text(encoding="utf-8"))]
            ]
            if prior:
                previous = prior[-1]
                for field in ("configs", "fixed_test_setting_sha256"):
                    if previous.get(field) != metadata[field]:
                        raise ValueError(
                            f"DPDP resume provenance mismatch in {field}; start a new run"
                        )
                metadata["resumed_from_job_id"] = previous.get("slurm_job_id")
                metadata["resumed_from_git_commit"] = previous.get("git_commit")
        atomic_json(self.run_root / f"run-metadata-{self.segment_id}.json", metadata)

    def materialize(self) -> None:
        base_runtime = json.loads((self.root / "configs/runtime.json").read_text(encoding="utf-8"))
        job_id = os.environ["SLURM_JOB_ID"]
        ports = select_runtime_ports(job_id)
        self.train_runtime, self.eval_runtime, self.budget_configs = {}, {}, {}
        for dataset in DATASETS:
            generation_port, embedding_port = ports[dataset]
            train_runtime = json.loads(json.dumps(base_runtime))
            train_runtime["generation"].update(
                {
                    "port": generation_port,
                    "gpu_memory_utilization": 0.78,
                    "max_num_seqs": 10,
                    "request_concurrency": 10,
                }
            )
            train_runtime["embedding"]["port"] = embedding_port
            eval_runtime = json.loads(json.dumps(base_runtime))
            eval_runtime["generation"]["port"] = generation_port
            eval_runtime["embedding"]["port"] = embedding_port
            self.train_runtime[dataset] = self.run_root / "runtime" / dataset / "training.json"
            self.eval_runtime[dataset] = self.run_root / "runtime" / dataset / "evaluation.json"
            atomic_json(self.train_runtime[dataset], train_runtime)
            atomic_json(self.eval_runtime[dataset], eval_runtime)

            natural = DPDPConfig.from_path(self.configs[dataset])
            matched = natural.with_budget(2, 1)
            if natural.training_sha256 != matched.training_sha256:
                raise ValueError("matched proxy changed fitted DPDP weights")
            self.budget_configs[dataset] = {}
            for budget, config in (("natural", natural), ("matched", matched)):
                path = self.run_root / "configs" / dataset / f"{budget}.json"
                raw = asdict(config)
                raw["actions"] = list(raw["actions"])
                atomic_json(path, raw)
                self.budget_configs[dataset][budget] = path
        port_manifest = {
            "schema_version": 1,
            "status": "selected_before_training",
            "strategy": "slurm_job_id_seeded_free_block_v1",
            "slurm_job_id": job_id,
            "range": [PORT_BLOCK_START, PORT_BLOCK_STOP],
            "ports": {
                dataset: {
                    "generation": assigned[0],
                    "embedding": assigned[1],
                }
                for dataset, assigned in ports.items()
            },
        }
        atomic_json(
            self.run_root / "runtime" / f"port-manifest-{job_id}.json",
            port_manifest,
        )
        atomic_json(self.run_root / "runtime/port-manifest.json", port_manifest)
        atomic_json(
            self.run_root / "configs/budget-manifest.json",
            {
                "schema_version": 1,
                "status": "frozen_before_development",
                "natural": {"simulations": 10, "depth": 5},
                "matched": {"simulations": 2, "depth": 1},
                "matched_status": "minimum_grid_proxy_without_measured_tdrl_reference",
                "gate_threshold": 0.2,
                "official_test_setting": str(FIXED_TEST_SETTING),
                "official_test_supported_prefix_counts": [400, 1000],
                "official_test_budgets": ["natural"],
                "test_opened": False,
            },
        )
        atomic_json(
            self.run_root / "configs" / f"fixed1000-prefix-{self.fixed_test_count}-manifest.json",
            {
                "schema_version": 1,
                "setting": str(FIXED_TEST_SETTING),
                "setting_sha256": self.fixed_test_setting["sha256"],
                "prefix_count": self.fixed_test_count,
                "per_dataset": self.fixed_test_count // 2,
                "seed": 17,
                "protocols": ["profile", "resistant"],
                "budget": "natural",
            },
        )

    def srun(self, command: list[str]) -> list[str]:
        return [
            "srun",
            "--exclusive",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=8",
            "--gpus-per-task=1",
            "--gpu-bind=single:1",
            f"--mem={WORKER_MEMORY_GIB}G",
            *command,
        ]

    def run_parallel(self, stage: str, commands: dict[str, list[str]]) -> None:
        if self.stop_path.exists():
            raise InterruptedError("stop requested before stage")
        self.status("running", stage, workers=list(commands))
        self.event("stage_started", stage=stage, workers=list(commands))
        opened = []
        self.children = []
        stop_seen_at = None
        try:
            for name, command in commands.items():
                job_id = os.environ.get("SLURM_JOB_ID", "local")
                log_path = self.logs / f"{stage}-{name}-{job_id}.log"
                handle = log_path.open("a", encoding="utf-8")
                opened.append(handle)
                self.event("worker_started", stage=stage, worker=name, log=str(log_path))
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
                failed = next((code for code in codes if code not in (None, 0)), None)
                stopping = self.stop_path.exists()
                if failed is not None and not stopping:
                    for child in self.children:
                        if child.poll() is None:
                            child.terminate()
                    raise subprocess.CalledProcessError(failed, stage)
                if all(code is not None for code in codes):
                    if stopping:
                        raise InterruptedError(f"stop requested during {stage}")
                    if any(code != 0 for code in codes):
                        raise subprocess.CalledProcessError(failed or 1, stage)
                    break
                if stopping and stop_seen_at is None:
                    stop_seen_at = time.monotonic()
                    self.event(
                        "graceful_stop_wait_started",
                        stage=stage,
                        grace_seconds=STOP_GRACE_SECONDS,
                    )
                if (
                    stop_seen_at is not None
                    and time.monotonic() - stop_seen_at >= STOP_GRACE_SECONDS
                ):
                    self.event("graceful_stop_timeout", stage=stage)
                    for child in self.children:
                        if child.poll() is None:
                            child.terminate()
                time.sleep(2)
        finally:
            for child in self.children:
                if child.poll() is None:
                    child.terminate()
            for handle in opened:
                handle.close()
            self.children = []
        self.event("stage_completed", stage=stage)

    def verify_parallel_slots(self) -> None:
        probe_root = (
            self.run_root
            / "allocation"
            / "two-gpu-probe"
            / f"{os.environ['SLURM_JOB_ID']}-{time.time_ns()}"
        )
        commands = {
            f"slot-{slot}": [
                str(self.python),
                "scripts/dpdp/probe_parallel_slot.py",
                "--output",
                str(probe_root),
                "--slot",
                str(slot),
                "--expected-slots",
                str(PARALLEL_SLOT_COUNT),
                "--timeout-seconds",
                "90",
            ]
            for slot in range(PARALLEL_SLOT_COUNT)
        }
        self.run_parallel("two-gpu-slot-preflight", commands)
        records = [
            json.loads((probe_root / f"passed-{slot}.json").read_text())
            for slot in range(PARALLEL_SLOT_COUNT)
        ]
        physical = [record["physical_gpu"] for record in records]
        if len(set(physical)) != PARALLEL_SLOT_COUNT:
            raise RuntimeError(f"DPDP slot preflight did not receive distinct GPUs: {physical}")
        atomic_json(
            probe_root / "summary.json",
            {
                "schema_version": 1,
                "status": "passed",
                "worker_memory_gib": WORKER_MEMORY_GIB,
                "parallel_slots": PARALLEL_SLOT_COUNT,
                "physical_gpus": physical,
                "records": records,
            },
        )

    def common_evaluation_args(self) -> list[str]:
        return [
            "--root",
            str(self.root),
            "--config",
            str(self.root / "configs/comparison.dpdp.json"),
            "--checkpoint-topdial",
            str(self.run_root / "selfplay/TopDial/checkpoint-final.pt"),
            "--checkpoint-durecdial",
            str(self.run_root / "selfplay/DuRecDial2.0/checkpoint-final.pt"),
            "--natural-config-topdial",
            str(self.budget_configs["TopDial"]["natural"]),
            "--matched-config-topdial",
            str(self.budget_configs["TopDial"]["matched"]),
            "--natural-config-durecdial",
            str(self.budget_configs["DuRecDial2.0"]["natural"]),
            "--matched-config-durecdial",
            str(self.budget_configs["DuRecDial2.0"]["matched"]),
            "--encoder",
            str(self.model),
            "--tokenizer",
            str(self.model),
            "--model-lock",
            str(self.root / "configs/dpdp.models.lock.json"),
        ]

    def evaluate(self, split: str, selection_lock: Path | None = None) -> Path:
        commands = {}
        for dataset in DATASETS:
            output = self.run_root / "evaluation" / f"{split}-workers" / dataset
            command = [
                str(self.python),
                "-m",
                "tdrl.baselines.dpdp.evaluate_cli",
                "evaluate",
                *self.common_evaluation_args(),
                "--output",
                str(output),
                "--split",
                split,
                "--dataset",
                dataset,
                "--runtime-config",
                str(self.eval_runtime[dataset]),
            ]
            if selection_lock is not None:
                command.extend(("--selection-lock", str(selection_lock)))
            if split == "test":
                command.extend(
                    (
                        "--fixed-test-setting",
                        str(self.fixed_test_setting_path),
                        "--fixed-test-count",
                        str(self.fixed_test_count),
                    )
                )
            commands[dataset] = command
        self.run_parallel(f"evaluate-{split}", commands)
        leaf = "test-fixed1000" if split == "test" else "dev"
        report_name = (
            f"dpdp-report-fixed1000-prefix-{self.fixed_test_count}.json"
            if split == "test"
            else "dpdp-report.json"
        )
        inputs = [
            self.run_root / "evaluation" / f"{split}-workers" / dataset / leaf / report_name
            for dataset in DATASETS
        ]
        output = self.run_root / "evaluation" / leaf / report_name
        subprocess.run(
            [
                str(self.python),
                "scripts/dpdp/merge_evaluation.py",
                "--input",
                str(inputs[0]),
                "--input",
                str(inputs[1]),
                "--output",
                str(output),
            ],
            cwd=self.root,
            check=True,
        )
        return output

    def lock_selection(self, development: Path) -> Path:
        lock = self.run_root / "evaluation/selection.lock.json"
        if lock.exists():
            from tdrl.baselines.dpdp.selection import verify_selection

            verify_selection(lock)
            return lock
        command = [
            str(self.python),
            "-m",
            "tdrl.baselines.dpdp.evaluate_cli",
            "lock-selection",
            *self.common_evaluation_args(),
            "--development-evidence",
            str(development),
            "--output",
            str(lock),
        ]
        subprocess.run(command, cwd=self.root, check=True)
        return lock

    def run(self) -> None:
        self.prepare_run_segment()
        self.status("running", "preflight")
        self.preflight()
        self.materialize()
        self.verify_parallel_slots()
        selection_lock = self.run_root / "evaluation/selection.lock.json"
        development = self.run_root / "evaluation/dev/dpdp-report.json"
        if selection_lock.exists():
            from tdrl.baselines.dpdp.selection import verify_selection

            envelope = verify_selection(selection_lock)
            if (
                not development.is_file()
                or file_hash(development) != envelope["payload"]["development_evidence_sha256"]
            ):
                raise ValueError(
                    "existing DPDP selection lock lost its immutable development evidence"
                )
            self.event(
                "selection_reused",
                selection_lock=str(selection_lock),
                development_report=str(development),
            )
        else:
            training = {}
            for dataset in DATASETS:
                training[dataset] = [
                    str(self.python),
                    "scripts/dpdp/run_training_pipeline.py",
                    "--root",
                    str(self.root),
                    "--run-root",
                    str(self.run_root),
                    "--dataset",
                    dataset,
                    "--worker-name",
                    dataset,
                    "--config-topdial",
                    str(self.configs["TopDial"]),
                    "--config-durecdial",
                    str(self.configs["DuRecDial2.0"]),
                    "--runtime-config",
                    str(self.train_runtime[dataset]),
                ]
            self.run_parallel("training", training)
            development = self.evaluate("dev")
            self.status("running", "selection-lock")
            selection_lock = self.lock_selection(development)
        test_report = self.evaluate("test", selection_lock)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": "DPDP-250",
            "started_at": self.started_at,
            "ended_at": now(),
            "selection_lock": str(selection_lock),
            "selection_lock_sha256": file_hash(selection_lock),
            "development_report": str(development),
            "test_report": str(test_report),
            "test_report_sha256": file_hash(test_report),
            "official_test_setting": str(FIXED_TEST_SETTING),
            "official_test_scenarios": self.fixed_test_count,
            "official_test_scenarios_per_dataset": self.fixed_test_count // 2,
            "protocols": ["profile", "resistant"],
            "official_test_budgets": ["natural"],
            "development_budgets": ["natural", "matched"],
            "matched_budget_status": "minimum_grid_proxy_without_measured_tdrl_reference",
            "claim_limit": "single-seed sampled official-test prefix; not a full split or training-seed uncertainty",
        }
        summary_path = self.run_root / (
            f"e2e-summary-fixed1000-prefix-{self.fixed_test_count}.json"
        )
        atomic_json(summary_path, summary)
        self.status("complete", "complete", summary=str(summary_path))
        self.event("pipeline_completed", summary=str(summary_path))
        segment_path = self.run_root / "allocation" / f"segment-{self.segment_id}.json"
        segment = json.loads(segment_path.read_text(encoding="utf-8"))
        segment.update(status="complete", ended_at=now(), summary=str(summary_path))
        atomic_json(segment_path, segment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--fixed-test-count", type=int, choices=(400, 1000), default=400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pipeline = DeadlineE2E(
        args.root, args.run_root, args.python, args.fixed_test_count, args.resume
    )
    signal.signal(signal.SIGUSR1, pipeline.stop)
    signal.signal(signal.SIGTERM, pipeline.stop)
    try:
        pipeline.run()
    except InterruptedError as error:
        pipeline.event("pipeline_stopped", reason=str(error))
        pipeline.status("partial", "stopped", reason=str(error))
        segment_path = pipeline.run_root / "allocation" / f"segment-{pipeline.segment_id}.json"
        if segment_path.exists():
            segment = json.loads(segment_path.read_text(encoding="utf-8"))
            segment.update(status="partial", ended_at=now(), reason=str(error))
            atomic_json(segment_path, segment)
        raise SystemExit(75) from error
    except Exception as error:
        pipeline.event("pipeline_failed", error=type(error).__name__, message=str(error))
        pipeline.status("failed", "failed", error=type(error).__name__, message=str(error))
        atomic_json(
            pipeline.run_root / "allocation" / f"failure-{pipeline.segment_id}.json",
            {
                "schema_version": 1,
                "status": "failed",
                "timestamp": now(),
                "slurm_job_id": pipeline.job_id,
                "error": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        segment_path = pipeline.run_root / "allocation" / f"segment-{pipeline.segment_id}.json"
        if segment_path.exists():
            segment = json.loads(segment_path.read_text(encoding="utf-8"))
            segment.update(
                status="failed",
                ended_at=now(),
                error=type(error).__name__,
                message=str(error),
            )
            atomic_json(segment_path, segment)
        raise


if __name__ == "__main__":
    main()
