import json
import math
import os
from pathlib import Path

from .provenance import canonical, digest


def choose_head(candidates: dict, difference_ci: tuple[float, float]):
    valid = {
        k: v
        for k, v in candidates.items()
        if v["invariants_passed"] and v["stability_passed"] and v["provenance_passed"]
    }
    if set(candidates) != {"MRNFixed", "IQE"} or not valid:
        raise ValueError("both candidate records required; at least one valid head")
    for v in valid.values():
        if set(v["development_sr"]) != {"TopDial", "DuRecDial2.0"}:
            raise ValueError("both development datasets required")
        if any(not 0 <= s <= 1 for s in v["development_sr"].values()):
            raise ValueError("invalid SR")
        cost = v["h100_seconds_per_success"]
        successes = v.get("development_successes")
        if successes is not None:
            if set(successes) != {"TopDial", "DuRecDial2.0"} or any(
                type(n) is not int or n < 0 for n in successes.values()
            ):
                raise ValueError("invalid development success counts")
            if any((successes[d] == 0) != (v["development_sr"][d] == 0) for d in successes):
                raise ValueError("SR and success counts disagree")
        if successes is not None and sum(successes.values()) == 0 and cost is not None:
            raise ValueError("zero-success compute must remain undefined")
        if cost is None:
            if successes != {"TopDial": 0, "DuRecDial2.0": 0}:
                raise ValueError("undefined compute requires verified zero successes")
        elif not math.isfinite(cost) or cost <= 0:
            raise ValueError("finite positive compute per success required")
    if all(v["h100_seconds_per_success"] is None for v in valid.values()):
        raise ValueError(
            "all valid heads have zero successes; development evidence is insufficient"
        )
    if len(valid) == 1:
        return next(iter(valid))
    lo, hi = difference_ci
    if not math.isfinite(lo + hi) or lo > hi:
        raise ValueError("invalid paired CI")
    if not lo <= 0 <= hi:
        return max(valid, key=lambda k: sum(valid[k]["development_sr"].values()) / 2)
    a, b = valid["MRNFixed"], valid["IQE"]
    ca, cb = a["h100_seconds_per_success"], b["h100_seconds_per_success"]
    if ca is None or cb is None:
        return "IQE" if ca is None else "MRNFixed"
    if abs(ca - cb) / min(ca, cb) >= 0.01:
        return "MRNFixed" if ca < cb else "IQE"
    if a["parameters"] != b["parameters"]:
        return "MRNFixed" if a["parameters"] < b["parameters"] else "IQE"
    return "MRNFixed"


LOCK_FIELDS = {
    "schema_version",
    "selected_head",
    "candidates",
    "difference_ci",
    "thresholds",
    "prompts",
    "models",
    "budgets",
    "baseline_adaptations",
    "scenario_manifests",
    "config_sha256",
    "package_lock_sha256",
    "code",
    "gate_evidence",
}


def validate_selection(payload):
    version = payload.get("schema_version")
    fields = LOCK_FIELDS if version == 1 else LOCK_FIELDS | {"scope"}
    if version == 3:
        fields |= {"recovery_amendment"}
    if version not in {1, 2, 3} or set(payload) != fields:
        raise ValueError("unknown/incomplete selection schema")
    if version in {2, 3} and payload["scope"] != {
        "methods": ["PPDPP", "TDRL-R", "TDRL"],
        "claim": "three_method_automated_comparison",
    }:
        raise ValueError("invalid scoped comparison")
    if version == 3:
        if any("development_successes" not in c for c in payload["candidates"].values()):
            raise ValueError("recovered selection requires explicit success counts")
        amendment = payload["recovery_amendment"]
        if not all(amendment.get(k) for k in ("path", "sha256", "source_job_id")):
            raise ValueError("recovery amendment provenance is incomplete")
    canonical(payload)
    if payload["selected_head"] != choose_head(payload["candidates"], payload["difference_ci"]):
        raise ValueError("selection violates frozen decision rule")
    if set(payload["thresholds"]) != {"TopDial", "DuRecDial2.0"}:
        raise ValueError("dataset-specific thresholds required")
    for v in payload["thresholds"].values():
        if set(v) != {"H_edge", "H_cover"} or any(x < 0 for x in v.values()):
            raise ValueError("separate nonnegative thresholds required")
    for name in ("PPDPP",) if version in {2, 3} else ("PPDPP", "DPDP", "TRIP", "DiffTOD"):
        adaptation = payload["baseline_adaptations"].get(name, {})
        if adaptation.get("status") != "validated" or not adaptation.get("artifact_sha256"):
            raise ValueError("faithful baseline adaptation evidence required")
    if set(payload["gate_evidence"]) != {"0", "1", "2", "3"}:
        raise ValueError("gates 0-3 required before selection")
    if any(not v for v in payload.values()):
        raise ValueError("unresolved selection field")


def write_selection(path: Path, payload: dict):
    validate_selection(payload)
    envelope = {"payload": payload, "sha256": digest(payload)}
    # Exclusive creation: selection is never silently overwritten.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as f:
        f.write(canonical(envelope) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return envelope["sha256"]


def verify_selection(path: Path, expected_hash: str):
    envelope = json.loads(path.read_text())
    if envelope["sha256"] != expected_hash or digest(envelope["payload"]) != expected_hash:
        raise ValueError("selection lock mismatch")
    validate_selection(envelope["payload"])
    return envelope["payload"]
