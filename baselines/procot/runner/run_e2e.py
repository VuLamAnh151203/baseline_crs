#!/usr/bin/env python3
"""Run ProCoT dev/lock/smoke and optionally the immutable fixed-400 test."""

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
from tdrl.baselines.procot.selection import verify_reference_lock, verify_selection
from tdrl.provenance import atomic_json, file_hash

DATASETS = ("TopDial", "DuRecDial2.0")
FIXED_SETTING = Path("configs/baseline-fixed1000-test-v1.json")
SOURCE_LOCK = Path("configs/procot-reference.lock.json")
PORT_START, PORT_STOP, PORTS_PER_JOB = 50_000, 60_000, 4
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
    raise OSError("no free ProCoT runtime port block")


class ProCoTE2E:
    def __init__(self, root, run_root, python, mode):
        self.root, self.run_root = Path(root).resolve(), Path(run_root).resolve()
        self.python, self.mode = Path(python).resolve(), mode
        self.fixed_count = {"mini": None, "initial400": 400}[mode]
        self.comparison_path = self.root / (
            "configs/comparison.procot-mini.json"
            if mode == "mini"
            else "configs/comparison.procot.json"
        )
        self.source_lock_path = self.root / SOURCE_LOCK
        self.setting_path = self.root / FIXED_SETTING
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
                "pipeline": "procot_fixed400_two_gpu",
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
            raise RuntimeError("ProCoT scientific E2E must run inside Slurm")
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()
        if dirty:
            raise ValueError("ProCoT scientific E2E requires a clean committed worktree")
        self.commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        required = (
            self.root / "runtime/uv.lock",
            self.root / "toolchains/cuda-12.9.1/bin/nvcc",
            self.root / "configs/models.lock.json",
            self.root / "configs/runtime.json",
            self.source_lock_path,
            self.root / "data/processed/TopDial/dev.jsonl",
            self.root / "data/processed/DuRecDial2.0/dev.jsonl",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing ProCoT runtime assets: " + ", ".join(missing))
        verify_reference_lock(self.source_lock_path)
        self.setting = (
            None
            if self.mode == "mini"
            else load_fixed_test_setting(self.root, self.setting_path, self.fixed_count)
        )
        metadata_path = self.run_root / "run-metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("git_commit") != self.commit or metadata.get("mode") != self.mode:
                raise ValueError("ProCoT run root belongs to another source or mode")
            if metadata.get("source_lock_sha256") != file_hash(self.source_lock_path):
                raise ValueError("ProCoT run root belongs to another upstream source lock")
        else:
            atomic_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "pipeline": "ProCoT-adapted-post-hoc-baseline",
                    "git_commit": self.commit,
                    "created_at": now(),
                    "initial_job_id": os.environ["SLURM_JOB_ID"],
                    "mode": self.mode,
                    "source_lock_sha256": file_hash(self.source_lock_path),
                    "fixed_setting_sha256": (
                        self.setting["sha256"] if self.setting is not None else None
                    ),
                    "fixed_prefix": self.fixed_count,
                    "diagnostic_only": self.mode == "mini",
                    "training": "none",
                },
            )

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

    @staticmethod
    def srun(command):
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

    def common_args(self):
        return [
            "--root",
            str(self.root),
            "--comparison-config",
            str(self.comparison_path),
            "--source-lock",
            str(self.source_lock_path),
        ]

    def evaluate(self, split, selection=None):
        commands = {}
        for dataset in DATASETS:
            output = self.run_root / "evaluation" / f"{split}-workers" / dataset
            command = [
                str(self.python),
                "-m",
                "tdrl.baselines.procot.evaluate_cli",
                "evaluate",
                *self.common_args(),
                "--output",
                str(output),
                "--split",
                split,
                "--dataset",
                dataset,
                "--runtime-config",
                str(self.runtime[dataset]),
            ]
            if selection is not None:
                command.extend(("--selection-lock", str(selection)))
            if split == "test":
                command.extend(
                    (
                        "--fixed-test-setting",
                        str(self.setting_path),
                        "--fixed-test-count",
                        str(self.fixed_count),
                    )
                )
            commands[dataset] = command
        prefix = self.fixed_count if self.fixed_count is not None else "mini"
        self.parallel(f"evaluate-{split}-prefix-{prefix}", commands)
        leaf = "test-fixed1000" if split == "test" else "dev"
        filename = (
            "procot-report.json"
            if split == "dev"
            else f"procot-report-fixed1000-prefix-{self.fixed_count}.json"
        )
        inputs = [
            self.run_root / "evaluation" / f"{split}-workers" / dataset / leaf / filename
            for dataset in DATASETS
        ]
        output = self.run_root / "evaluation" / leaf / filename
        subprocess.run(
            [
                str(self.python),
                "scripts/procot/merge_evaluation.py",
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

    def lock_method(self, development):
        path = self.run_root / "evaluation/method.lock.json"
        if path.exists():
            verify_selection(path)
            return path
        subprocess.run(
            [
                str(self.python),
                "-m",
                "tdrl.baselines.procot.evaluate_cli",
                "lock-selection",
                *self.common_args(),
                "--development-evidence",
                str(development),
                "--output",
                str(path),
            ],
            cwd=self.root,
            check=True,
        )
        return path

    def smoke(self, selection):
        commands = {
            dataset: [
                str(self.python),
                "scripts/procot/smoke.py",
                "--root",
                str(self.root),
                "--output",
                str(self.run_root / "smoke" / dataset),
                "--dataset",
                dataset,
                "--selection-lock",
                str(selection),
                "--source-lock",
                str(self.source_lock_path),
                "--runtime-config",
                str(self.runtime[dataset]),
                "--comparison-config",
                str(self.comparison_path),
            ]
            for dataset in DATASETS
        }
        self.parallel("post-lock-smoke", commands)

    def feasibility(self, development):
        report = json.loads(Path(development).read_text(encoding="utf-8"))
        worker_wall = report["parallel_worker_wall_s"]
        development_per_dataset = self.config_development_dialogues()
        test_per_dataset = (self.fixed_count // 2) * 2
        projected = {
            dataset: worker_wall[dataset]
            / development_per_dataset
            * test_per_dataset
            for dataset in DATASETS
        }
        required = (max(projected.values()) + 2 * 3600) * 1.25
        remaining = self.remaining_wall_seconds()
        payload = {
            "schema_version": 1,
            "status": "passed" if required <= remaining else "failed",
            "prefix_count": self.fixed_count,
            "development_dialogues_per_dataset": development_per_dataset,
            "test_dialogues_per_dataset": test_per_dataset,
            "projected_worker_seconds": projected,
            "required_wall_seconds_with_overhead_and_safety": required,
            "remaining_allocation_seconds": remaining,
            "fits": required <= remaining,
            "note": "measured development extrapolation plus two-hour overhead and 1.25 safety",
        }
        atomic_json(self.run_root / "feasibility-prefix-400.json", payload)
        if not payload["fits"]:
            raise RuntimeError("ProCoT fixed-400 test does not fit allocation")
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
            raise RuntimeError("cannot determine remaining Slurm time")
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
        config = json.loads(self.comparison_path.read_text(encoding="utf-8"))
        return config["development_scenarios"] * len(config["protocols"])

    def run(self):
        self.status("running", "preflight")
        self.preflight()
        self.materialize_runtime()
        development = self.run_root / "evaluation/dev/procot-report.json"
        selection = self.run_root / "evaluation/method.lock.json"
        if not selection.exists():
            development = self.evaluate("dev")
            selection = self.lock_method(development)
        else:
            envelope = verify_selection(selection)
            if (
                not development.is_file()
                or file_hash(development)
                != envelope["payload"]["development_evidence_sha256"]
            ):
                raise ValueError("ProCoT method lock lost its development evidence")
        self.smoke(selection)
        if self.mode == "mini":
            summary = {
                "schema_version": 1,
                "status": "complete",
                "pipeline": "ProCoT-mini-e2e",
                "mode": self.mode,
                "started_at": self.started_at,
                "ended_at": now(),
                "source_commit": self.commit,
                "source_lock": str(self.source_lock_path),
                "source_lock_sha256": file_hash(self.source_lock_path),
                "method_lock": str(selection),
                "method_lock_sha256": file_hash(selection),
                "development_report": str(development),
                "official_test_opened": False,
                "claim_limit": "diagnostic development and replay smoke only",
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
            "pipeline": "ProCoT-adapted-post-hoc-baseline",
            "mode": self.mode,
            "started_at": self.started_at,
            "ended_at": now(),
            "source_commit": self.commit,
            "source_lock": str(self.source_lock_path),
            "source_lock_sha256": file_hash(self.source_lock_path),
            "method_lock": str(selection),
            "method_lock_sha256": file_hash(selection),
            "development_report": str(development),
            "test_report": str(test_report),
            "test_report_sha256": file_hash(test_report),
            "official_test_scenarios": self.fixed_count,
            "official_test_scenarios_per_dataset": self.fixed_count // 2,
            "official_test_dialogues": self.fixed_count * 2,
            "protocols": ["profile", "resistant"],
            "budget": "natural",
            "training": "none",
            "claim_limit": "adapted post-hoc single-seed sampled official-test prefix",
        }
        summary_path = self.run_root / "e2e-summary-fixed1000-prefix-400.json"
        atomic_json(summary_path, summary)
        self.status("complete", "complete", summary=str(summary_path))
        self.event("pipeline_completed", summary=str(summary_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mode", choices=("mini", "initial400"), required=True)
    args = parser.parse_args()
    pipeline = ProCoTE2E(args.root, args.run_root, args.python, args.mode)
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
