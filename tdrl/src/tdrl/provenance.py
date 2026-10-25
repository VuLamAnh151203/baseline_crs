import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(value) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


PROVENANCE_FIELDS = {
    "code",
    "config",
    "packages",
    "environment",
    "dataset",
    "scenarios",
    "seeds",
    "models",
    "prompts",
    "preprocessing",
    "checkpoint",
    "slurm",
    "gpu",
}
COMPUTE_FIELDS = {
    "generation_requests",
    "embedding_requests",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "role_tokens",
    "embedding_texts",
    "embedding_tokens",
    "turn_latency_s",
    "dialogue_latency_s",
    "h100_seconds",
    "cold_load_s",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "errors",
    "retries",
}


def validate_manifest(m: dict):
    if m.get("schema_version") != 1:
        raise ValueError("unknown run manifest version")
    if m.get("status") not in {"partial", "complete", "failed"}:
        raise ValueError("explicit run status required")
    for key in ("run_id", "stage", "started_at", "ended_at", "provenance", "compute", "outputs"):
        if key not in m:
            raise ValueError(f"missing manifest field: {key}")
    if set(m["provenance"]) != PROVENANCE_FIELDS:
        raise ValueError("incomplete provenance fields")
    if set(m["compute"]) != COMPUTE_FIELDS:
        raise ValueError("incomplete compute fields")
    canonical(m)  # Reject non-finite numeric values even in nested metadata.
    if m["status"] == "complete":
        if not m["ended_at"] or any(v is None for v in m["provenance"].values()):
            raise ValueError("complete run cannot contain unresolved provenance")
        if m["compute"]["errors"] != 0:
            raise ValueError("complete run has unresolved infrastructure errors")
        if not m["outputs"]:
            raise ValueError("complete run requires hashed outputs")
        for h in m["outputs"].values():
            if not re.fullmatch(r"[0-9a-f]{64}", h):
                raise ValueError("invalid output checksum")
    for key in (
        "generation_requests",
        "embedding_requests",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "h100_seconds",
        "errors",
        "retries",
    ):
        v = m["compute"][key]
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
            raise ValueError("invalid compute counter")


class Journal:
    """Single-writer atomic records; resume only for the exact provenance contract."""

    def __init__(self, directory: Path, contract: dict):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.contract = digest(contract)
        p = directory / "contract.json"
        if p.exists():
            if json.loads(p.read_text()) != {"sha256": self.contract, "contract": contract}:
                raise ValueError("resume/cache provenance mismatch")
        else:
            atomic_json(p, {"sha256": self.contract, "contract": contract})

    def read(self, request_id: str, request: dict):
        p = self.directory / (digest(request_id) + ".json")
        if not p.exists():
            return None
        record = json.loads(p.read_text())
        if record["request_sha256"] != digest(request) or record["contract"] != self.contract:
            raise ValueError("request ID reused with different inputs")
        return record["result"]

    def write(self, request_id: str, request: dict, result: dict):
        p = self.directory / (digest(request_id) + ".json")
        if p.exists():
            old = self.read(request_id, request)
            if old != result:
                raise ValueError("attempt to overwrite completed request")
            return
        atomic_json(
            p, {"contract": self.contract, "request_sha256": digest(request), "result": result}
        )


def stop_if_requested(directory):
    for parent in (directory, *directory.parents):
        if (parent / "stop.requested").exists():
            raise InterruptedError("allocation stop requested at a durable checkpoint")
        if (parent / "allocation.json").exists():
            break
