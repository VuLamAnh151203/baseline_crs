"""Manifest-backed stages and allocated GPU telemetry, with strict reuse contracts."""

import importlib.metadata
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .backend import allocated_gpu_uuid
from .provenance import (
    COMPUTE_FIELDS,
    PROVENANCE_FIELDS,
    atomic_json,
    digest,
    file_hash,
    validate_manifest,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def code_provenance(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    return {"commit": commit, "dirty": bool(dirty)}


def package_versions():
    result = {}
    for name in ("torch", "vllm", "transformers", "numpy", "httpx"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


class Stage:
    def __init__(self, root: Path, directory: Path, name: str, config_path: Path, extra: dict):
        self.root, self.directory, self.name = root, directory, name
        self.config_path = config_path
        self.code = code_provenance(root)
        if self.code["dirty"]:
            raise ValueError("scientific stages require the committed clean source tree")
        self.contract = {
            "code": self.code,
            "config_sha256": file_hash(config_path),
            "package_lock_sha256": file_hash(root / "runtime/uv.lock"),
            **extra,
        }
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "contract.json"
        if path.exists() and json.loads(path.read_text()) != self.contract:
            from .critic_execution import execution_evidence

            original = json.loads(path.read_text())
            evidence = execution_evidence(root, original["code"]["commit"])
            if {**original, "code": self.contract["code"]} != self.contract:
                raise ValueError("stage resume provenance mismatch")
            # RNG/checkpoint/journal provenance keeps its exact original contract.
            self.contract = original
            self.code = {**self.code, "critic_execution": evidence}
        atomic_json(path, self.contract)
        self.started = utc_now()
        self.start = time.monotonic()
        self.manifest_path = directory / "manifest.json"
        self.reused = False
        if self.manifest_path.exists():
            m = json.loads(self.manifest_path.read_text())
            validate_manifest(m)
            self.started = m["started_at"]
            if m["status"] == "complete":
                self.verify_outputs(m)
                self.reused = True

    def verify_outputs(self, manifest):
        for path, sha in manifest["outputs"].items():
            p = self.directory / path
            if not p.is_file() or file_hash(p) != sha:
                raise ValueError("completed stage artifact mismatch")

    def write(self, status, provenance_updates=None, compute_updates=None, error=None):
        provenance = {k: {"status": "not_applicable_to_this_stage"} for k in PROVENANCE_FIELDS}
        provenance.update(
            code=self.code,
            config={"path": str(self.config_path), "sha256": self.contract["config_sha256"]},
            packages={
                "lock_sha256": self.contract["package_lock_sha256"],
                "versions": package_versions(),
            },
            environment={
                "python": os.sys.version,
                "cuda": torch.version.cuda,
                "driver": self._driver_version(),
            },
            models=json.loads((self.root / "configs/models.lock.json").read_text()),
            slurm={
                "job_id": os.environ.get("SLURM_JOB_ID", "bootstrap"),
                "logs": str(
                    next(
                        (
                            p / "slurm-job.txt"
                            for p in self.directory.parents
                            if (p / "slurm-job.txt").exists()
                        ),
                        self.directory.parent / "slurm-job.txt",
                    )
                ),
            },
            preprocessing={
                "serialization_sha256": self.contract.get("serialization_sha256", "not_used")
            },
        )
        provenance.update(provenance_updates or {})
        compute = dict.fromkeys(COMPUTE_FIELDS)
        compute.update(
            generation_requests=0,
            embedding_requests=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            role_tokens={},
            embedding_texts=0,
            embedding_tokens=None,
            turn_latency_s=[],
            dialogue_latency_s=[],
            h100_seconds=0.0,
            cold_load_s=0.0,
            peak_allocated_bytes=0,
            peak_reserved_bytes=0,
            errors=int(error is not None),
            retries=0,
        )
        compute.update(compute_updates or {})
        outputs = {
            str(p.relative_to(self.directory)): file_hash(p)
            for p in sorted(self.directory.rglob("*"))
            if p.is_file() and p != self.manifest_path and p.suffix != ".part"
        }
        manifest = {
            "schema_version": 1,
            "run_id": f"{os.environ.get('SLURM_JOB_ID', 'bootstrap')}-{self.name.replace('/', '_')}-{digest(self.contract)[:10]}",
            "stage": self.name,
            "status": status,
            "started_at": self.started,
            "ended_at": utc_now() if status in {"complete", "failed"} else None,
            "provenance": provenance,
            "compute": compute,
            "outputs": outputs,
        }
        if error:
            manifest["error"] = str(error)
        validate_manifest(manifest)
        atomic_json(self.manifest_path, manifest)
        return manifest

    @staticmethod
    def _driver_version():
        if not os.environ.get("SLURM_JOB_ID"):
            return "not_applicable_bootstrap"
        try:
            return subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).splitlines()[0]
        except (subprocess.CalledProcessError, IndexError):
            return "unavailable"


class GpuMonitor:
    """Sample only the CUDA-visible allocated GPU UUID; never attribute other GPUs."""

    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.rows = []

    def __enter__(self):
        self.uuid = allocated_gpu_uuid()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as output:
            while not self.stop.is_set():
                try:
                    values = (
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "-i",
                                self.uuid,
                                "--query-gpu=memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        )
                        .strip()
                        .split(",")
                    )
                    used, total, utilization = [int(v.strip()) for v in values]
                    row = {
                        "timestamp": utc_now(),
                        "gpu_uuid": self.uuid,
                        "used_bytes": used * 1024**2,
                        "total_bytes": total * 1024**2,
                        "utilization_percent": utilization,
                    }
                    self.rows.append(row)
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                except (ValueError, subprocess.CalledProcessError) as e:
                    output.write(
                        json.dumps({"error": type(e).__name__, "timestamp": utc_now()}) + "\n"
                    )
                    output.flush()
                self.stop.wait(0.5)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)

    def summary(self):
        if not self.rows:
            raise ValueError("GPU telemetry absent")
        return {
            "gpu_uuid": self.uuid,
            "samples": len(self.rows),
            "idle_used_bytes": self.rows[0]["used_bytes"],
            "peak_used_bytes": max(r["used_bytes"] for r in self.rows),
            "steady_used_bytes": self.rows[-1]["used_bytes"],
        }


def counters(generation, embedding):
    """Actual completed HTTP executions across resumes; cache replays add no calls."""

    def records(backend, kind):
        ledger = getattr(backend, "ledger", None)
        if ledger is not None:
            if list(ledger.glob("pending-*.json")):
                raise ValueError(
                    "unresolved in-flight inference requests prohibit complete compute evidence"
                )
            return [json.loads(p.read_text()) for p in sorted(ledger.glob("completed-*.json"))]
        if kind == "generation":
            from dataclasses import asdict

            return [asdict(r) for r in backend.requests]
        return [
            {"batch_size": r.batch_size, "input_tokens": r.input_tokens} for r in backend.requests
        ]

    gs, es = records(generation, "generation"), records(embedding, "embedding")
    roles = {}
    for row in gs:
        record = roles.setdefault(
            row["role"], {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )
        record["requests"] += 1
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            record[name] += row[name]
    return {
        "generation_requests": len(gs),
        "embedding_requests": len(es),
        "input_tokens": sum(r["input_tokens"] for r in gs),
        "output_tokens": sum(r["output_tokens"] for r in gs),
        "total_tokens": sum(r["total_tokens"] for r in gs),
        "role_tokens": roles,
        "embedding_texts": sum(r["batch_size"] for r in es),
        "embedding_tokens": sum(r["input_tokens"] for r in es)
        if all(r["input_tokens"] is not None for r in es)
        else None,
    }


def combine_compute(records):
    result = {}
    for name in (
        "generation_requests",
        "embedding_requests",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "embedding_texts",
        "cold_load_s",
    ):
        result[name] = sum(r.get(name, 0) for r in records)
    result["embedding_tokens"] = (
        sum(r.get("embedding_tokens", 0) for r in records)
        if all(r.get("embedding_tokens", 0) is not None for r in records)
        else None
    )
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        result[name] = max((r.get(name, 0) for r in records), default=0)
    roles = {}
    for r in records:
        for role, counts in r.get("role_tokens", {}).items():
            into = roles.setdefault(role, dict.fromkeys(counts, 0))
            for key, value in counts.items():
                into[key] += value
    result["role_tokens"] = roles
    return result
