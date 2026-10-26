#!/usr/bin/env python3
"""Run one bounded DPDP mini-E2E worker per H100 and verify real overlap."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from scripts.dpdp.run_deadline_e2e import (
    DATASETS,
    WORKER_MEMORY_GIB,
    select_runtime_ports,
)
from tdrl.provenance import atomic_json


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("two-GPU mini timestamps must be timezone-aware")
    return parsed


def validate_worker_summaries(summaries: dict[str, dict]) -> dict:
    if set(summaries) != set(DATASETS):
        raise ValueError("two-GPU mini requires exactly one worker per dataset")
    uuids = {}
    intervals = {}
    for dataset, summary in summaries.items():
        if summary.get("status") != "mini_non_scientific":
            raise ValueError(f"incomplete two-GPU mini worker: {dataset}")
        scope = summary.get("scope", {})
        if scope.get("datasets") != [dataset] or scope.get("official_test_opened") is not False:
            raise ValueError(f"invalid isolated mini scope: {dataset}")
        gpu = summary.get("gpu", {})
        if gpu.get("count") != 1 or not gpu.get("uuid"):
            raise ValueError(f"worker did not see exactly one GPU: {dataset}")
        uuids[dataset] = gpu["uuid"]
        intervals[dataset] = (
            _timestamp(summary["gpu_work_started_at"]),
            _timestamp(summary["gpu_work_ended_at"]),
        )
        if intervals[dataset][1] <= intervals[dataset][0]:
            raise ValueError(f"invalid GPU-work interval: {dataset}")
    if len(set(uuids.values())) != len(DATASETS):
        raise ValueError("two-GPU mini workers were not assigned distinct GPU UUIDs")
    overlap_start = max(value[0] for value in intervals.values())
    overlap_end = min(value[1] for value in intervals.values())
    overlap_s = (overlap_end - overlap_start).total_seconds()
    if overlap_s <= 0:
        raise ValueError("two-GPU mini workers did not overlap their GPU-work intervals")
    return {
        "distinct_gpu_uuids": uuids,
        "gpu_work_intervals": {
            dataset: [start.isoformat(), end.isoformat()]
            for dataset, (start, end) in intervals.items()
        },
        "gpu_work_overlap_s": overlap_s,
    }


def _runtime_configs(root: Path, output: Path, job_id: str) -> dict[str, Path]:
    base = json.loads((root / "configs/runtime.json").read_text(encoding="utf-8"))
    ports = select_runtime_ports(job_id)
    paths = {}
    for dataset, (generation_port, embedding_port) in ports.items():
        runtime = json.loads(json.dumps(base))
        runtime["generation"]["port"] = generation_port
        runtime["embedding"]["port"] = embedding_port
        path = output / "runtime" / dataset / "runtime.json"
        atomic_json(path, runtime)
        paths[dataset] = path
    atomic_json(
        output / "runtime/port-manifest.json",
        {
            "schema_version": 1,
            "slurm_job_id": job_id,
            "ports": {
                dataset: {"generation": values[0], "embedding": values[1]}
                for dataset, values in ports.items()
            },
        },
    )
    return paths


def run(args) -> dict:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("two-GPU mini must run inside Slurm")
    root = args.root.resolve()
    output = args.output.resolve()
    if (output / "two-gpu-mini-summary.json").exists() or (output / "workers").exists():
        raise FileExistsError("two-GPU mini output already contains a run")
    output.mkdir(parents=True, exist_ok=True)
    job_id = os.environ["SLURM_JOB_ID"]
    lock = json.loads((root / "configs/dpdp.models.lock.json").read_text(encoding="utf-8"))
    revision = str(lock["revision"])
    runtime_configs = _runtime_configs(root, output, job_id)
    processes = {}
    logs = {}
    handles = []
    started_at = datetime.now().astimezone().isoformat()
    try:
        for dataset in DATASETS:
            worker_output = output / "workers" / dataset
            log_path = output / "logs" / f"{dataset}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            logs[dataset] = str(log_path)
            command = [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=8",
                "--gpus-per-task=1",
                "--gpu-bind=single:1",
                f"--mem={WORKER_MEMORY_GIB}G",
                str(args.python.resolve()),
                "scripts/dpdp/mini_e2e.py",
                "--root",
                str(root),
                "--output",
                str(worker_output),
                "--roberta-revision",
                revision,
                "--dataset",
                dataset,
                "--runtime-config",
                str(runtime_configs[dataset]),
            ]
            processes[dataset] = subprocess.Popen(
                command,
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        while True:
            codes = {dataset: process.poll() for dataset, process in processes.items()}
            failed = {dataset: code for dataset, code in codes.items() if code not in (None, 0)}
            if failed:
                raise subprocess.CalledProcessError(next(iter(failed.values())), failed)
            if all(code == 0 for code in codes.values()):
                break
            time.sleep(2)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in handles:
            handle.close()

    summaries = {
        dataset: json.loads(
            (output / "workers" / dataset / "mini-e2e-summary.json").read_text(
                encoding="utf-8"
            )
        )
        for dataset in DATASETS
    }
    evidence = validate_worker_summaries(summaries)
    result = {
        "schema_version": 1,
        "status": "mini_non_scientific",
        "slurm_job_id": job_id,
        "requested_gpus": 2,
        "datasets_parallel": True,
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "worker_logs": logs,
        "worker_summaries": {
            dataset: str(output / "workers" / dataset / "mini-e2e-summary.json")
            for dataset in DATASETS
        },
        "evaluation_dialogues": sum(
            summary["evaluation_dialogues"] for summary in summaries.values()
        ),
        "official_test_opened": False,
        "claim_limit": "two-GPU integration evidence only; not a scientific result",
        **evidence,
    }
    atomic_json(output / "two-gpu-mini-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / "two-gpu-mini-failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "official_test_opened": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
