#!/usr/bin/env python3
"""Run SCOPE-CRS-lite gates, reward fitting, lock and optional fixed-400 test."""

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
from tdrl.baselines.scope.assets import (
    verify_llama_guard_snapshot,
    verify_transition_manifest,
)
from tdrl.baselines.scope.selection import verify_reference_lock, verify_selection
from tdrl.provenance import atomic_json, file_hash

DATASETS = ("TopDial", "DuRecDial2.0")
FIXED_SETTING = Path("configs/baseline-fixed1000-test-v1.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def ports_available(ports):
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
    start, stop = 50_000, 60_000
    initial = int(job_id) % (stop - start)
    for offset in range(stop - start):
        first = start + (initial + offset) % (stop - start - 1)
        ports = (first, first + 1)
        if ports_available(ports):
            return {"TopDial": ports[0], "DuRecDial2.0": ports[1]}
    raise OSError("no free SCOPE runtime ports")


class ScopeE2E:
    def __init__(self, root, run_root, python, mode, storage_root):
        self.root, self.run_root = Path(root).resolve(), Path(run_root).resolve()
        self.python, self.mode = Path(python).resolve(), mode
        self.storage_root = Path(storage_root).resolve()
        self.fixed_count = None if mode == "mini" else 400
        self.comparison = self.root / (
            "configs/comparison.scope-lite-mini.json"
            if mode == "mini"
            else "configs/comparison.scope-lite.json"
        )
        suffix = "-mini" if mode == "mini" else ""
        self.scope_configs = {
            "TopDial": self.root / f"configs/scope/topdial{suffix}.json",
            "DuRecDial2.0": self.root / f"configs/scope/durecdial{suffix}.json",
        }
        self.source_lock = self.root / "configs/scope-reference.lock.json"
        self.fixed_setting = self.root / FIXED_SETTING
        self.transitions = self.storage_root / "assets/transition_models/deterministic"
        self.llama_guard = self.storage_root / (
            "assets/Meta-Llama-Guard-2-8B/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef"
        )
        self.llama_manifest = self.llama_guard / "scope-asset-manifest.json"
        self.logs = self.run_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_root / "status.json"
        self.events_path = self.run_root / "events.jsonl"
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
                "pipeline": "scope-crs-lite-fixed400-two-gpu",
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
            raise RuntimeError("SCOPE scientific E2E must run inside Slurm")
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()
        if dirty:
            raise ValueError("SCOPE scientific E2E requires a clean committed worktree")
        self.commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        required = [
            self.root / "runtime/uv.lock",
            self.root / "toolchains/cuda-12.9.1/bin/nvcc",
            self.root / "configs/models.lock.json",
            self.source_lock,
            *self.scope_configs.values(),
        ]
        for dataset in DATASETS:
            required.extend(
                self.root / "data/processed" / dataset / f"{split}.jsonl"
                for split in ("train", "dev")
            )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing SCOPE runtime assets: " + ", ".join(missing))
        verify_reference_lock(self.source_lock)
        verify_transition_manifest(self.transitions)
        verify_llama_guard_snapshot(self.llama_guard, self.llama_manifest)
        self.setting = (
            None
            if self.mode == "mini"
            else load_fixed_test_setting(self.root, self.fixed_setting, self.fixed_count)
        )
        metadata = {
            "schema_version": 1,
            "pipeline": "SCOPE-CRS-lite",
            "git_commit": self.commit,
            "created_at": now(),
            "initial_job_id": os.environ["SLURM_JOB_ID"],
            "mode": self.mode,
            "source_lock_sha256": file_hash(self.source_lock),
            "transition_manifest_sha256": file_hash(self.transitions / "manifest.json"),
            "llama_guard_manifest_sha256": file_hash(self.llama_manifest),
            "fixed_setting_sha256": self.setting["sha256"] if self.setting else None,
            "official_test_opened": False,
        }
        path = self.run_root / "run-metadata.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            for key in (
                "git_commit",
                "mode",
                "source_lock_sha256",
                "transition_manifest_sha256",
                "llama_guard_manifest_sha256",
                "fixed_setting_sha256",
            ):
                if existing.get(key) != metadata[key]:
                    raise ValueError(f"SCOPE run root changed {key}")
        else:
            atomic_json(path, metadata)

    def materialize_runtime(self):
        base = json.loads((self.root / "configs/runtime.json").read_text(encoding="utf-8"))
        ports = select_ports(os.environ["SLURM_JOB_ID"])
        self.runtime = {}
        for dataset, port in ports.items():
            config = json.loads(json.dumps(base))
            config["generation"]["port"] = port
            config["generation"]["gpu_memory_utilization"] = 0.48
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
            "--mem=72G",
            *command,
        ]

    def parallel(self, stage, commands):
        if self.stop_path.exists():
            raise InterruptedError(f"stop requested before {stage}")
        self.status("running", stage, workers=list(commands))
        self.event("stage_started", stage=stage, workers=list(commands))
        handles, self.children = [], []
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

    def shared_asset_args(self, dataset):
        return [
            "--root",
            str(self.root),
            "--dataset",
            dataset,
            "--config",
            str(self.scope_configs[dataset]),
            "--llama-guard",
            str(self.llama_guard),
            "--llama-guard-manifest",
            str(self.llama_manifest),
        ]

    def smoke_asset_args(self, dataset):
        return [
            *self.shared_asset_args(dataset),
            "--transitions",
            str(self.transitions),
        ]

    def reward_preparation_args(self, dataset):
        return [
            *self.shared_asset_args(dataset),
            "--run-root",
            str(self.run_root),
            "--comparison-config",
            str(self.comparison),
        ]

    def smoke(self, name, fitted):
        commands = {}
        for dataset in DATASETS:
            command = [
                str(self.python),
                "scripts/scope/smoke.py",
                *self.smoke_asset_args(dataset),
                "--output",
                str(self.run_root / "smoke" / name / dataset),
                "--comparison-config",
                str(self.comparison),
                "--runtime-config",
                str(self.runtime[dataset]),
            ]
            if fitted:
                command.extend(
                    (
                        "--reward-checkpoint",
                        str(
                            self.run_root
                            / "reward-training"
                            / dataset
                            / "checkpoint-best.pt"
                        ),
                    )
                )
            commands[dataset] = command
        self.parallel(name, commands)

    def prepare_and_train(self):
        self.parallel(
            "prepare-reward-data",
            {
                dataset: [
                    str(self.python),
                    "scripts/scope/prepare_reward.py",
                    *self.reward_preparation_args(dataset),
                    "--runtime-config",
                    str(self.runtime[dataset]),
                ]
                for dataset in DATASETS
            },
        )
        self.parallel(
            "train-reward",
            {
                dataset: [
                    str(self.python),
                    "scripts/scope/train_reward.py",
                    "--run-root",
                    str(self.run_root),
                    "--dataset",
                    dataset,
                    "--config",
                    str(self.scope_configs[dataset]),
                    "--transitions",
                    str(self.transitions),
                    "--llama-guard",
                    str(self.llama_guard),
                    "--llama-guard-manifest",
                    str(self.llama_manifest),
                ]
                for dataset in DATASETS
            },
        )

    def common_evaluation_args(self):
        return [
            "--root",
            str(self.root),
            "--comparison-config",
            str(self.comparison),
            "--source-lock",
            str(self.source_lock),
            "--run-root",
            str(self.run_root),
            "--topdial-config",
            str(self.scope_configs["TopDial"]),
            "--durecdial-config",
            str(self.scope_configs["DuRecDial2.0"]),
            "--transitions",
            str(self.transitions),
            "--llama-guard",
            str(self.llama_guard),
            "--llama-guard-manifest",
            str(self.llama_manifest),
        ]

    def evaluate(self, split, selection=None):
        if split == "test":
            metadata_path = self.run_root / "run-metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["official_test_opened"] = True
            metadata["official_test_opened_at"] = now()
            metadata["method_lock_sha256"] = file_hash(selection)
            atomic_json(metadata_path, metadata)
        commands = {}
        for dataset in DATASETS:
            output = self.run_root / "evaluation" / f"{split}-workers" / dataset
            command = [
                str(self.python),
                "-m",
                "tdrl.baselines.scope.evaluate_cli",
                "evaluate",
                *self.common_evaluation_args(),
                "--output",
                str(output),
                "--split",
                split,
                "--dataset",
                dataset,
                "--runtime-config",
                str(self.runtime[dataset]),
            ]
            if selection:
                command.extend(("--selection-lock", str(selection)))
            if split == "test":
                command.extend(
                    (
                        "--fixed-test-setting",
                        str(self.fixed_setting),
                        "--fixed-test-count",
                        str(self.fixed_count),
                    )
                )
            commands[dataset] = command
        self.parallel(f"evaluate-{split}", commands)
        leaf = "test-fixed1000" if split == "test" else "dev"
        filename = (
            f"scope-report-fixed1000-prefix-{self.fixed_count}.json"
            if split == "test"
            else "scope-report.json"
        )
        inputs = [
            self.run_root / "evaluation" / f"{split}-workers" / dataset / leaf / filename
            for dataset in DATASETS
        ]
        output = self.run_root / "evaluation" / leaf / filename
        subprocess.run(
            [
                str(self.python),
                "scripts/scope/merge_evaluation.py",
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
                "tdrl.baselines.scope.evaluate_cli",
                "lock-selection",
                *self.common_evaluation_args(),
                "--development-evidence",
                str(development),
                "--output",
                str(path),
            ],
            cwd=self.root,
            check=True,
        )
        return path

    def feasibility(self, development):
        report = json.loads(Path(development).read_text(encoding="utf-8"))
        worker = report["parallel_worker_wall_s"]
        development_dialogues = self.config_development_dialogues()
        test_dialogues = (self.fixed_count // 2) * 2
        projected = {
            dataset: worker[dataset] / development_dialogues * test_dialogues
            for dataset in DATASETS
        }
        required = (max(projected.values()) + 3 * 3600) * 1.25
        remaining = self.remaining_wall_seconds()
        payload = {
            "schema_version": 1,
            "status": "passed" if required <= remaining else "failed",
            "projected_worker_seconds": projected,
            "required_wall_seconds_with_overhead_and_safety": required,
            "remaining_allocation_seconds": remaining,
            "fits": required <= remaining,
        }
        atomic_json(self.run_root / "feasibility-prefix-400.json", payload)
        if not payload["fits"]:
            raise RuntimeError("SCOPE fixed-400 does not fit measured allocation")

    def config_development_dialogues(self):
        config = json.loads(self.comparison.read_text(encoding="utf-8"))
        return config["development_scenarios"] * len(config["protocols"])

    @staticmethod
    def remaining_wall_seconds():
        value = os.environ.get("SLURM_JOB_END_TIME")
        if value and value.isdigit():
            return max(0.0, float(value) - time.time())
        raw = subprocess.check_output(
            ["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%L"], text=True
        ).strip()
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
            raise RuntimeError("cannot parse remaining Slurm time")
        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    def run(self):
        self.status("running", "preflight")
        self.preflight()
        self.materialize_runtime()
        self.smoke("pretrain-runtime-smoke", fitted=False)
        self.prepare_and_train()
        development = self.evaluate("dev")
        selection = self.lock_method(development)
        self.smoke("postfit-replay-smoke", fitted=True)
        test_report = None
        if self.mode != "mini":
            self.feasibility(development)
            test_report = self.evaluate("test", selection)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": "SCOPE-CRS-lite",
            "mode": self.mode,
            "started_at": self.started_at,
            "ended_at": now(),
            "source_commit": self.commit,
            "development_report": str(development),
            "method_lock": str(selection),
            "method_lock_sha256": file_hash(selection),
            "test_report": str(test_report) if test_report else None,
            "official_test_opened": test_report is not None,
            "official_test_scenarios": self.fixed_count,
            "protocols": ["profile", "resistant"],
            "budget": "natural",
            "claim_limit": (
                "task-adapted single-seed sampled official-test prefix"
                if test_report
                else "diagnostic train/dev and H100 smoke only"
            ),
        }
        path = self.run_root / (
            "e2e-summary-fixed1000-prefix-400.json"
            if test_report
            else "e2e-summary-mini.json"
        )
        atomic_json(path, summary)
        self.status("complete", "complete", summary=str(path))
        self.event("pipeline_completed", summary=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mode", choices=("mini", "initial400"), required=True)
    args = parser.parse_args()
    pipeline = ScopeE2E(
        args.root, args.run_root, args.python, args.mode, args.storage_root
    )
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
