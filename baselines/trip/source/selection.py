import json
import os
from pathlib import Path

from ...planner import ASSISTANT_POLICY
from ...provenance import canonical, digest, file_hash
from .planner import TRIP_REALIZER_POLICY

SCOPE = {
    "methods": ["PPDPP", "DiffTOD", "TRIP", "TDRL-R", "TDRL"],
    "claim": "five_method_automated_comparison",
}


def selection_payload(
    config_path,
    checkpoints,
    trip_config_sha256,
    asset_lock,
    development_evidence,
    base_selection=None,
):
    evidence = json.loads(Path(development_evidence).read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or evidence.get("split") != "dev":
        raise ValueError("complete TRIP development evidence is required")
    base_sha256 = None
    if base_selection is not None:
        envelope = json.loads(Path(base_selection).read_text(encoding="utf-8"))
        valid = set(envelope) == {"payload", "sha256"} and (
            digest(envelope["payload"]) == envelope["sha256"]
        )
        if not valid:
            raise ValueError("base comparison selection lock is invalid")
        base_sha256 = envelope["sha256"]
    checkpoint_sha256 = {dataset: file_hash(Path(path)) for dataset, path in checkpoints.items()}
    expected = {
        "method": "TRIP",
        "config_sha256": file_hash(Path(config_path)),
        "checkpoint_sha256": checkpoint_sha256,
        "trip_config_sha256": trip_config_sha256,
        "asset_lock_sha256": file_hash(Path(asset_lock)),
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("TRIP development evidence does not match selected artifacts")
    expected_groups = {
        (dataset, protocol, budget)
        for dataset in ("TopDial", "DuRecDial2.0")
        for protocol in ("profile", "resistant")
        for budget in ("natural", "matched")
    }
    found = {
        (row.get("dataset"), row.get("protocol"), row.get("budget"))
        for row in evidence.get("summary", [])
        if row.get("method") == "TRIP"
    }
    if found != expected_groups:
        raise ValueError("TRIP development evidence matrix is incomplete")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "config_sha256": expected["config_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "trip_config_sha256": trip_config_sha256,
        "asset_lock_sha256": expected["asset_lock_sha256"],
        "development_evidence_sha256": file_hash(Path(development_evidence)),
        "base_selection_sha256": base_sha256,
        "prompts": {
            "assistant": digest(ASSISTANT_POLICY),
            "trip_realizer": digest(TRIP_REALIZER_POLICY),
        },
        "test_bodies": "sealed",
    }


def validate_selection(payload):
    required = {
        "schema_version",
        "scope",
        "config_sha256",
        "checkpoint_sha256",
        "trip_config_sha256",
        "asset_lock_sha256",
        "development_evidence_sha256",
        "base_selection_sha256",
        "prompts",
        "test_bodies",
    }
    if set(payload) != required or payload["schema_version"] != 1 or payload["scope"] != SCOPE:
        raise ValueError("unknown/incomplete TRIP selection lock")
    hashes = ("config_sha256", "asset_lock_sha256", "development_evidence_sha256")
    if any(not isinstance(payload[key], str) or len(payload[key]) != 64 for key in hashes):
        raise ValueError("TRIP selection provenance is incomplete")
    datasets = {"TopDial", "DuRecDial2.0"}
    if set(payload["checkpoint_sha256"]) != datasets or any(
        len(value) != 64 for value in payload["checkpoint_sha256"].values()
    ):
        raise ValueError("TRIP selection checkpoint provenance is incomplete")
    configs = payload["trip_config_sha256"]
    if set(configs) != datasets or any(
        set(value) != {"natural", "matched"} or any(len(item) != 64 for item in value.values())
        for value in configs.values()
    ):
        raise ValueError("TRIP selection configuration provenance is incomplete")
    if payload["base_selection_sha256"] is not None and len(payload["base_selection_sha256"]) != 64:
        raise ValueError("invalid TRIP base selection hash")
    expected_prompts = {
        "assistant": digest(ASSISTANT_POLICY),
        "trip_realizer": digest(TRIP_REALIZER_POLICY),
    }
    if payload["prompts"] != expected_prompts or payload["test_bodies"] != "sealed":
        raise ValueError("TRIP prompt/test contract changed")
    canonical(payload)


def write_selection(path, payload):
    validate_selection(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "sha256": digest(payload)}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical(envelope) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return envelope["sha256"]


def verify_selection(path, expected=None):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "sha256"} or digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("TRIP selection lock hash mismatch")
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected TRIP selection lock")
    validate_selection(envelope["payload"])
    return envelope
