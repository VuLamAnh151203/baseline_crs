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


def now():
    return datetime.now(timezone.utc).isoformat()


def ports(job_id):
    start, blocks = 40_000, 1000
    for offset in range(blocks):
        base = start + ((int(job_id) + offset) % blocks) * 2
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
    raise OSError("no free EPL native generation ports")


class Pipeline:
    def __init__(self, root, run_root):
        self.root, self.run_root = Path(root).resolve(), Path(run_root).resolve()
        self.children = []
        self.logs = self.run_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events = self.run_root / "events.jsonl"
        self.status_path = self.run_root / "status.json"

    def event(self, event, **values):
        row = {"timestamp": now(), "event": event, **values}
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    def status(self, state, stage, **values):
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "pipeline": "epl-native-core-qwen-mini-two-gpu",
                "state": state,
                "stage": stage,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "run_root": str(self.run_root),
                "updated_at": now(),
                **values,
            },
        )

    def stop(self, *_):
        self.event("stop_requested")
        for child in self.children:
            if child.poll() is None:
                child.terminate()

    def preflight(self):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("EPL native mini must run inside Slurm")
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()
        if dirty:
            raise ValueError("EPL native mini requires a clean committed checkout")
        self.commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        required = (
            self.root / "data/raw/DuRecDial2.0/en_train.txt",
            self.root / "data/raw/DuRecDial2.0/en_dev.txt",
            self.root / "data/processed/TopDial/train.jsonl",
            self.root / "data/processed/TopDial/dev.jsonl",
            self.root / "configs/epl-reference.lock.json",
            self.root / "configs/epl.models.lock.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing EPL native mini inputs: " + ", ".join(missing))
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
        if not self.encoder.is_dir() or not self.retrieval.is_dir():
            raise FileNotFoundError("missing locked EPL encoder assets")

    def runtime_configs(self):
        base = json.loads((self.root / "configs/runtime.json").read_text())
        assigned = ports(os.environ["SLURM_JOB_ID"])
        result = {}
        for dataset, port in assigned.items():
            payload = json.loads(json.dumps(base))
            payload["generation"]["port"] = port
            path = self.run_root / "runtime" / f"{dataset}.json"
            atomic_json(path, payload)
            result[dataset] = path
        atomic_json(self.run_root / "runtime/ports.json", assigned)
        return result

    def run(self):
        self.status("running", "preflight")
        self.preflight()
        runtime = self.runtime_configs()
        commands = {}
        for dataset, config in (
            ("TopDial", "configs/epl_native/topdial.json"),
            ("DuRecDial2.0", "configs/epl_native/durecdial.json"),
        ):
            commands[dataset] = [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=8",
                "--gpus-per-task=1",
                "--gpu-bind=single:1",
                "--mem=64G",
                sys.executable,
                "scripts/epl_native/worker.py",
                "--root",
                str(self.root),
                "--run-root",
                str(self.run_root),
                "--dataset",
                dataset,
                "--config",
                str(self.root / config),
                "--runtime-config",
                str(runtime[dataset]),
                "--comparison-config",
                str(self.root / "configs/comparison.epl-mini.json"),
                "--encoder",
                str(self.encoder),
                "--retrieval-encoder",
                str(self.retrieval),
                "--model-lock",
                str(self.root / "configs/epl.models.lock.json"),
            ]
        self.status("running", "parallel-dataset-mini", workers=list(commands))
        self.event("stage_started", stage="parallel-dataset-mini")
        handles = []
        try:
            for dataset, command in commands.items():
                handle = (self.logs / f"{dataset}.log").open("a", encoding="utf-8")
                handles.append(handle)
                self.children.append(
                    subprocess.Popen(
                        command,
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
                    raise subprocess.CalledProcessError(failure, "parallel-dataset-mini")
                if all(code == 0 for code in codes):
                    break
                time.sleep(2)
        finally:
            for handle in handles:
                handle.close()
        summaries = {
            dataset: json.loads(
                (self.run_root / "workers" / dataset / "worker-summary.json").read_text()
            )
            for dataset in DATASETS
        }
        uuids = {row["gpu"]["uuid"] for row in summaries.values()}
        if len(uuids) != 2:
            raise ValueError("EPL native mini did not use two distinct GPUs")
        intervals = {
            dataset: (
                datetime.fromisoformat(row["gpu_work_started_at"]),
                datetime.fromisoformat(row["gpu_work_ended_at"]),
            )
            for dataset, row in summaries.items()
        }
        overlap_start = max(value[0] for value in intervals.values())
        overlap_end = min(value[1] for value in intervals.values())
        overlap_s = (overlap_end - overlap_start).total_seconds()
        if overlap_s <= 0:
            raise ValueError("EPL native dataset workers did not overlap GPU work")
        if any(row["official_test_opened"] for row in summaries.values()):
            raise ValueError("EPL native mini unexpectedly opened official test")
        collapsed = [
            dataset
            for dataset, row in summaries.items()
            if not row["policy_diagnostics"]["noncollapsed"]
        ]
        if collapsed:
            raise ValueError(
                "EPL native development policy collapsed: " + ", ".join(collapsed)
            )
        lock = {
            "schema_version": 1,
            "status": "diagnostic_complete",
            "method": "EPL-native-core-Qwen-fixed400",
            "source_commit": self.commit,
            "official_test_opened": False,
            "datasets": {
                dataset: {
                    "summary_sha256": file_hash(
                        self.run_root / "workers" / dataset / "worker-summary.json"
                    ),
                    "config_sha256": file_hash(
                        self.root
                        / (
                            "configs/epl_native/topdial.json"
                            if dataset == "TopDial"
                            else "configs/epl_native/durecdial.json"
                        )
                    ),
                    "vocabulary_sha256": row["action_vocabulary_sha256"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "memory_sha256": row["memory_sha256"],
                    "vectors_sha256": row["vectors_sha256"],
                    "gpu_uuid": row["gpu"]["uuid"],
                }
                for dataset, row in summaries.items()
            },
            "claim_limit": "train/dev two-GPU mini and exact replay only; not test evidence",
        }
        lock["sha256"] = digest(lock)
        atomic_json(self.run_root / "diagnostic.lock.json", lock)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": "EPL-native-core-Qwen-fixed400-mini-e2e",
            "source_commit": self.commit,
            "datasets": summaries,
            "two_distinct_gpu_uuids": sorted(uuids),
            "gpu_work_overlap_s": overlap_s,
            "diagnostic_lock_sha256": file_hash(self.run_root / "diagnostic.lock.json"),
            "official_test_opened": False,
            "claim_limit": lock["claim_limit"],
        }
        atomic_json(self.run_root / "e2e-summary-mini.json", summary)
        self.status("complete", "complete", summary=str(self.run_root / "e2e-summary-mini.json"))
        self.event("pipeline_completed")


def main():
    parser = argparse.ArgumentParser(description="Run two-GPU EPL native-core mini E2E")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    pipeline = Pipeline(args.root, args.run_root)
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
