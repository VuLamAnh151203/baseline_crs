#!/usr/bin/env python3
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

from tdrl.provenance import atomic_json, digest, file_hash

DATASETS = ("TopDial", "DuRecDial2.0")
MINI_JOB = "60458"


def now():
    return datetime.now(timezone.utc).isoformat()


def ports(job_id):
    for offset in range(1000):
        base = 42_000 + ((int(job_id) + offset) % 1000) * 2
        probes = []
        try:
            for port in (base, base + 1):
                probe = socket.socket()
                probes.append(probe)
                probe.bind(("127.0.0.1", port))
            return {"TopDial": base, "DuRecDial2.0": base + 1}
        except OSError:
            pass
        finally:
            for probe in probes:
                probe.close()
    raise OSError("no free EPL-native generation ports")


class Pipeline:
    def __init__(self, root, run_root, mini_root):
        self.root = Path(root).resolve()
        self.run_root = Path(run_root).resolve()
        self.mini_root = Path(mini_root).resolve()
        self.logs = self.run_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events = self.run_root / "events.jsonl"
        self.status_path = self.run_root / "status.json"
        self.children = []

    def event(self, event, **fields):
        row = {"timestamp": now(), "event": event, **fields}
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    def status(self, state, stage, **fields):
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "pipeline": "epl-native-fixed400-two-gpu",
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "state": state,
                "stage": stage,
                "run_root": str(self.run_root),
                "updated_at": now(),
                **fields,
            },
        )

    def stop(self, *_):
        self.event("stop_requested")
        for child in self.children:
            if child.poll() is None:
                child.terminate()

    def preflight(self):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("EPL-native full must run inside Slurm")
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()
        if dirty:
            raise ValueError("EPL-native full requires a clean committed checkout")
        self.commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        summary_path = self.mini_root / "e2e-summary-mini.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "complete"
            or summary.get("official_test_opened") is not False
            or len(summary.get("two_distinct_gpu_uuids", [])) != 2
        ):
            raise ValueError("source EPL-native readiness mini did not pass")
        for dataset in DATASETS:
            row = summary["datasets"][dataset]
            if not row["policy_diagnostics"]["noncollapsed"]:
                raise ValueError(f"collapsed EPL-native checkpoint: {dataset}")
        asset_root = Path(
            os.environ.get("EPL_MODEL_ROOT", "/mnt/data/u26anhvl/crs/epl/models")
        )
        lock = json.loads((self.root / "configs/epl.models.lock.json").read_text())
        self.encoder = asset_root / "bert-base-cased" / lock["policy_encoder"]["revision"]
        self.retrieval = (
            asset_root
            / "all-MiniLM-L6-v2"
            / lock["retrieval_encoder"]["revision"]
        )
        self.configs = {
            "TopDial": self.root / "configs/epl_native/topdial.json",
            "DuRecDial2.0": self.root / "configs/epl_native/durecdial.json",
        }
        self.checkpoints = {
            dataset: self.mini_root
            / "workers"
            / dataset
            / "training/checkpoint-best.pt"
            for dataset in DATASETS
        }
        self.vocabularies = {
            dataset: self.mini_root / "workers" / dataset / "action-vocabulary.json"
            for dataset in DATASETS
        }
        required = [self.encoder, self.retrieval]
        required += list(self.checkpoints.values()) + list(self.vocabularies.values())
        if any(not path.exists() for path in required):
            raise FileNotFoundError("missing EPL-native full source artifact")
        for dataset in DATASETS:
            source = summary["datasets"][dataset]
            if file_hash(self.checkpoints[dataset]) != source["checkpoint_sha256"]:
                raise ValueError(f"EPL-native checkpoint hash changed: {dataset}")
            vocabulary_sha256 = digest(
                json.loads(self.vocabularies[dataset].read_text(encoding="utf-8"))
            )
            if vocabulary_sha256 != source["action_vocabulary_sha256"]:
                raise ValueError(f"EPL-native vocabulary hash changed: {dataset}")
        expected = {
            "source_mini_job": MINI_JOB,
            "source_mini_summary_sha256": file_hash(summary_path),
            "source_commit": self.commit,
        }
        metadata = self.run_root / "run-metadata.json"
        if metadata.exists():
            if json.loads(metadata.read_text(encoding="utf-8")) != expected:
                raise ValueError("EPL-native full run root provenance changed")
        else:
            atomic_json(metadata, expected)

    def runtime_configs(self):
        base = json.loads((self.root / "configs/runtime.json").read_text())
        assigned = ports(os.environ["SLURM_JOB_ID"])
        result = {}
        for dataset, port in assigned.items():
            payload = json.loads(json.dumps(base))
            payload["generation"]["port"] = port
            path = self.run_root / "runtime" / f"{os.environ['SLURM_JOB_ID']}-{dataset}.json"
            atomic_json(path, payload)
            result[dataset] = path
        return result

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
        self.status("running", stage, workers=list(commands))
        self.event("stage_started", stage=stage)
        handles = []
        self.children = []
        try:
            for dataset, command in commands.items():
                handle = (self.logs / f"{stage}-{dataset}-{os.environ['SLURM_JOB_ID']}.log").open(
                    "a", encoding="utf-8"
                )
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
                time.sleep(2)
        finally:
            for handle in handles:
                handle.close()
            self.children = []
        self.event("stage_completed", stage=stage)

    def common(self, dataset, runtime):
        memory = self.run_root / "memory" / dataset / "experiences.jsonl"
        return [
            "--root",
            str(self.root),
            "--run-root",
            str(self.run_root),
            "--dataset",
            dataset,
            "--config",
            str(self.configs[dataset]),
            "--vocabulary",
            str(self.vocabularies[dataset]),
            "--checkpoint",
            str(self.checkpoints[dataset]),
            "--encoder",
            str(self.encoder),
            "--retrieval-encoder",
            str(self.retrieval),
            "--model-lock",
            str(self.root / "configs/epl.models.lock.json"),
            "--runtime-config",
            str(runtime[dataset]),
            "--comparison-config",
            str(self.root / "configs/comparison.epl.json"),
            "--memory",
            str(memory),
            "--vectors",
            str(memory.parent / "vectors.pt"),
        ]

    def worker_commands(self, command, runtime, extra=()):
        return {
            dataset: [
                sys.executable,
                "scripts/epl_native/full_worker.py",
                command,
                *self.common(dataset, runtime),
                *extra,
            ]
            for dataset in DATASETS
        }

    def merge(self, split):
        leaf = "test-fixed1000" if split == "test" else "dev"
        reports = {
            dataset: json.loads(
                (
                    self.run_root
                    / "evaluation"
                    / f"{split}-workers"
                    / dataset
                    / f"{leaf}-report.json"
                ).read_text(encoding="utf-8")
            )
            for dataset in DATASETS
        }
        if any(row["status"] != "complete" for row in reports.values()):
            raise ValueError("incomplete EPL-native evaluation shard")
        output = self.run_root / "evaluation" / leaf / "epl-native-report.json"
        payload = {
            "schema_version": 1,
            "status": "complete",
            "split": split,
            "method": "EPL-native",
            "dialogues": sum(row["dialogues"] for row in reports.values()),
            "parallel_worker_wall_s": {
                dataset: row["wall_s"] for dataset, row in reports.items()
            },
            "summary": [item for row in reports.values() for item in row["summary"]],
            "shards": {
                dataset: digest(row) for dataset, row in reports.items()
            },
        }
        atomic_json(output, payload)
        return output, reports

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

    def feasibility(self, reports, selection_sha256):
        comparison = json.loads(
            (self.root / "configs/comparison.epl.json").read_text(encoding="utf-8")
        )
        dev_dialogues = comparison["development_scenarios"] * len(
            comparison["protocols"]
        )
        test_dialogues = 200 * len(comparison["protocols"])
        projected = {
            dataset: reports[dataset]["wall_s"] / dev_dialogues * test_dialogues
            for dataset in DATASETS
        }
        required = (max(projected.values()) + 4 * 3600) * 1.25
        remaining = self.remaining_wall_seconds()
        payload = {
            "schema_version": 1,
            "status": "passed" if required <= remaining else "failed",
            "prefix_count": 400,
            "development_dialogues_per_dataset": dev_dialogues,
            "test_dialogues_per_dataset": test_dialogues,
            "projected_worker_seconds": projected,
            "required_wall_seconds_with_overhead_and_safety": required,
            "remaining_allocation_seconds": remaining,
            "fits": required <= remaining,
            "selection_sha256": selection_sha256,
            "note": "measured development extrapolation plus four-hour overhead and 1.25 safety",
        }
        atomic_json(self.run_root / "feasibility-prefix-400.json", payload)
        if not payload["fits"]:
            raise RuntimeError("EPL-native fixed-400 test does not fit allocation")

    def run(self):
        self.status("running", "preflight")
        self.preflight()
        runtime = self.runtime_configs()
        self.parallel("prepare-full-memory", self.worker_commands("prepare-memory", runtime))
        self.parallel(
            "evaluate-development",
            self.worker_commands("evaluate", runtime, ("--split", "dev")),
        )
        development, reports = self.merge("dev")
        self.parallel(
            "replay-development",
            self.worker_commands(
                "evaluate", runtime, ("--split", "dev", "--expect-replay")
            ),
        )
        lock = self.run_root / "evaluation/selection.lock.json"
        payload = {
            "schema_version": 1,
            "method": "EPL-native",
            "selection_rule": "fixed native configuration; train/dev only",
            "source_mini_job": MINI_JOB,
            "development_report_sha256": file_hash(development),
            "fixed_setting_sha256": file_hash(
                self.root / "configs/baseline-fixed1000-test-v1.json"
            ),
            "datasets": {
                dataset: reports[dataset]["provenance"] for dataset in DATASETS
            },
            "official_test_opened": False,
        }
        envelope = {"payload": payload, "sha256": digest(payload)}
        if lock.exists() and json.loads(lock.read_text()) != envelope:
            raise ValueError("EPL-native selection lock changed")
        atomic_json(lock, envelope)
        self.feasibility(reports, envelope["sha256"])
        self.parallel(
            "evaluate-fixed400",
            self.worker_commands(
                "evaluate",
                runtime,
                (
                    "--split",
                    "test",
                    "--selection-lock",
                    str(lock),
                    "--fixed-setting",
                    str(self.root / "configs/baseline-fixed1000-test-v1.json"),
                ),
            ),
        )
        report, _ = self.merge("test")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": "EPL-native-core-Qwen-fixed400",
            "source_commit": self.commit,
            "source_mini_job": MINI_JOB,
            "selection_lock": str(lock),
            "selection_sha256": envelope["sha256"],
            "test_report": str(report),
            "official_test_scenarios": 400,
            "official_test_dialogues": 800,
        }
        atomic_json(self.run_root / "e2e-summary-fixed400.json", summary)
        self.status("complete", "complete", summary=str(self.run_root / "e2e-summary-fixed400.json"))
        self.event("pipeline_completed")


def main():
    parser = argparse.ArgumentParser(description="Run EPL-native fixed-400 E2E")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mini-root", type=Path, required=True)
    args = parser.parse_args()
    pipeline = Pipeline(args.root, args.run_root, args.mini_root)
    signal.signal(signal.SIGUSR1, pipeline.stop)
    signal.signal(signal.SIGTERM, pipeline.stop)
    try:
        pipeline.run()
    except Exception as error:
        pipeline.event("pipeline_failed", error=type(error).__name__, message=str(error))
        pipeline.status("failed", "failed", error=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
