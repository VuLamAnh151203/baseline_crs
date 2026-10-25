"""Execute only a locked candidate on exact, disjoint development membership."""

import json
from pathlib import Path

from .confirmation_lock import verify_candidate
from .provenance import digest, file_hash, validate_manifest


def confirmation_cases(population, membership, processed_sha256):
    if (
        membership.get("schema_version") != 1
        or membership.get("dataset") != "TopDial"
        or membership.get("split") != "dev"
        or membership.get("root_seed") != 17
        or membership.get("protocols") != ["profile", "resistant"]
        or membership.get("population_sha256") != processed_sha256
        or membership.get("population_count") != len(population)
    ):
        raise ValueError("confirmation population/protocol contract changed")
    ordered = sorted(population, key=lambda row: digest([17, row["id"]]))
    if len({row["id"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate development population IDs")
    selected = ordered[80:144]
    ids = [row["id"] for row in selected]
    if len(ids) != 64 or membership.get("scenario_ids") != ids:
        raise ValueError("confirmation membership/order changed; never resample")
    expected = {row["id"]: digest(row) for row in selected}
    if membership.get("scenario_content_sha256") != expected:
        raise ValueError("confirmation scenario content changed")
    return selected


def confirmation_candidate(root: Path, lock: Path, commit: str, comparison_config: Path):
    payload = verify_candidate(lock, root)
    if payload["code_commit"] != commit:
        raise ValueError("code changed after candidate freeze")
    if payload["package_lock_sha256"] != file_hash(root / "runtime/uv.lock"):
        raise ValueError("packages changed after candidate freeze")
    if payload["files"]["comparison_config"]["sha256"] != file_hash(comparison_config):
        raise ValueError("shared comparison contract changed after freeze")
    membership = json.loads(
        (root / payload["files"]["confirmation_membership"]["path"]).read_text()
    )
    return payload, membership


def verify_planner_gates(paths, contract):
    ignored = {"slot", "scenario_ids", "calibration_ids", "confirmation_lock_sha256"}
    expected = {k: v for k, v in contract.items() if k not in ignored}
    checks = []
    for path in paths:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        validate_manifest(manifest)
        if manifest["status"] != "complete" or manifest["stage"] != "TopDial-planner-smoke":
            raise ValueError("complete planner smoke required")
        for name in ["checks.json", "contract.json"]:
            if file_hash(path.parent / name) != manifest["outputs"].get(name):
                raise ValueError("planner gate checksum mismatch")
        source = json.loads((path.parent / "contract.json").read_text())
        if {k: v for k, v in source.items() if k not in ignored} != expected:
            raise ValueError("planner gate source/config/provenance mismatch")
        check = json.loads(path.read_text())
        if (
            check["status"] != "passed"
            or not check["exact_journal_replay"]
            or check["optimizer_updates"] != 0
        ):
            raise ValueError("planner replay gate failed")
        checks.append(check)
    if len(checks) != 2 or len({c["gpu"]["uuid"] for c in checks}) != 2:
        raise ValueError("both independent planner GPU gates required")
    return checks
