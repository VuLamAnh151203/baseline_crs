"""Immutable candidate contract for disjoint development confirmation, never official tests."""

import json
import math
import os
from pathlib import Path

from .provenance import canonical, digest, file_hash

FIELDS = {
    "schema_version",
    "scope",
    "root_seed",
    "methods",
    "discovery_macro_differences",
    "thresholds",
    "head_sha256",
    "head_type",
    "code_commit",
    "package_lock_sha256",
    "difftod_policy_sha256",
    "files",
}


def finite_number(value):
    return type(value) in {int, float} and math.isfinite(value)


def validate_candidate(payload):
    if set(payload) != FIELDS or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete confirmation candidate schema")
    if (
        payload["scope"] != "TopDial disjoint development confirmation"
        or payload["root_seed"] != 17
    ):
        raise ValueError("confirmation scope/seed mismatch")
    if payload["methods"] != ["TDRL-R", "TDRL", "DiffTOD"]:
        raise ValueError("both MRN methods and fixed DiffTOD required")
    if payload["head_type"] != "MRNFixed":
        raise ValueError("current confirmation scope is MRNFixed only")
    if (
        not isinstance(payload["code_commit"], str)
        or len(payload["code_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in payload["code_commit"])
    ):
        raise ValueError("immutable code commit required")
    if set(payload["discovery_macro_differences"]) != {"TDRL-R", "TDRL"} or any(
        not finite_number(v) or not 0 < v <= 1
        for v in payload["discovery_macro_differences"].values()
    ):
        raise ValueError(
            "both candidates need positive discovery point estimates before confirmation"
        )
    if set(payload["thresholds"]) != {"H_edge", "H_cover"} or any(
        not finite_number(v) or v < 0 for v in payload["thresholds"].values()
    ):
        raise ValueError("frozen separate directed thresholds required")
    for name in ["head_sha256", "difftod_policy_sha256", "package_lock_sha256"]:
        if (
            not isinstance(payload[name], str)
            or len(payload[name]) != 64
            or any(c not in "0123456789abcdef" for c in payload[name])
        ):
            raise ValueError("immutable checkpoint hashes required")
    if set(payload["files"]) != {
        "candidate_config",
        "comparison_config",
        "confirmation_membership",
        "strategy_overlay_manifest",
        "discovery_report",
    }:
        raise ValueError("complete frozen configuration/evidence references required")
    for ref in payload["files"].values():
        if set(ref) != {"path", "sha256"} or not ref["path"]:
            raise ValueError("incomplete frozen artifact reference")
        if (
            not isinstance(ref["sha256"], str)
            or len(ref["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in ref["sha256"])
        ):
            raise ValueError("invalid artifact reference checksum")
    canonical(payload)


def freeze_candidate(path: Path, payload):
    validate_candidate(payload)
    envelope = {"payload": payload, "sha256": digest(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(canonical(envelope) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return envelope["sha256"]


def verify_candidate(path: Path, root: Path):
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "sha256"} or digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("confirmation lock checksum mismatch")
    payload = envelope["payload"]
    validate_candidate(payload)
    for ref in payload["files"].values():
        relative = Path(ref["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("confirmation references must be repository-relative")
        if file_hash(root / relative) != ref["sha256"]:
            raise ValueError("frozen confirmation artifact changed")
    return payload
