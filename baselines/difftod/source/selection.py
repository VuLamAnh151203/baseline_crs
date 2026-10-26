import json
import os
from pathlib import Path

from ...planner import ASSISTANT_POLICY
from ...provenance import canonical, digest, file_hash
from .guidance import GUIDANCE_POLICY
from .planner import DIFFTOD_REALIZER_POLICY

SCOPE = {
    "methods": ["PPDPP", "DiffTOD", "TDRL-R", "TDRL"],
    "claim": "four_method_automated_comparison",
}


def selection_payload(
    config_path,
    checkpoints,
    plan_config_sha256,
    development_evidence,
    base_selection=None,
):
    evidence = json.loads(Path(development_evidence).read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or evidence.get("split") != "dev":
        raise ValueError("complete DiffTOD development evidence is required before selection")
    base_sha256 = None
    if base_selection is not None:
        envelope = json.loads(Path(base_selection).read_text(encoding="utf-8"))
        valid = set(envelope) == {"payload", "sha256"} and digest(
            envelope["payload"]
        ) == envelope["sha256"]
        if not valid:
            raise ValueError("base comparison selection lock is invalid")
        base_sha256 = envelope["sha256"]
    if set(checkpoints) != {"TopDial", "DuRecDial2.0"}:
        raise ValueError("DiffTOD selection requires one checkpoint per dataset")
    if set(plan_config_sha256) != {"TopDial", "DuRecDial2.0"} or any(
        set(value) != {"natural", "matched"} for value in plan_config_sha256.values()
    ):
        raise ValueError("DiffTOD selection requires natural/matched configs per dataset")
    config_sha256 = file_hash(Path(config_path))
    checkpoint_sha256 = {
        dataset: file_hash(Path(path)) for dataset, path in checkpoints.items()
    }
    expected_evidence = {
        "method": "DiffTOD",
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "plan_config_sha256": plan_config_sha256,
    }
    if any(evidence.get(key) != value for key, value in expected_evidence.items()):
        raise ValueError("DiffTOD development evidence does not match the selected artifacts")
    expected_groups = {
        (dataset, protocol, budget)
        for dataset in ("TopDial", "DuRecDial2.0")
        for protocol in ("profile", "resistant")
        for budget in ("natural", "matched")
    }
    found_groups = {
        (row.get("dataset"), row.get("protocol"), row.get("budget"))
        for row in evidence.get("summary", [])
        if row.get("method") == "DiffTOD"
    }
    if found_groups != expected_groups:
        raise ValueError("DiffTOD development evidence has an incomplete evaluation matrix")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "plan_config_sha256": plan_config_sha256,
        "development_evidence_sha256": file_hash(Path(development_evidence)),
        "base_selection_sha256": base_sha256,
        "prompts": {
            "assistant": digest(ASSISTANT_POLICY),
            "difftod_realizer": digest(DIFFTOD_REALIZER_POLICY),
            "difftod_guidance": digest(GUIDANCE_POLICY),
        },
        "test_bodies": "sealed",
    }


def validate_selection(payload):
    if set(payload) != {
        "schema_version",
        "scope",
        "config_sha256",
        "checkpoint_sha256",
        "plan_config_sha256",
        "development_evidence_sha256",
        "base_selection_sha256",
        "prompts",
        "test_bodies",
    }:
        raise ValueError("unknown/incomplete DiffTOD selection lock")
    if payload["schema_version"] != 1 or payload["scope"] != SCOPE:
        raise ValueError("invalid DiffTOD comparison scope")
    for key in ("config_sha256", "development_evidence_sha256"):
        if not isinstance(payload[key], str) or len(payload[key]) != 64:
            raise ValueError("DiffTOD selection provenance is incomplete")
    checkpoints = payload["checkpoint_sha256"]
    if set(checkpoints) != {"TopDial", "DuRecDial2.0"} or any(
        not isinstance(value, str) or len(value) != 64 for value in checkpoints.values()
    ):
        raise ValueError("DiffTOD selection checkpoint provenance is incomplete")
    plan_configs = payload["plan_config_sha256"]
    valid_plan_configs = set(plan_configs) == {"TopDial", "DuRecDial2.0"} and all(
        set(value) == {"natural", "matched"}
        and all(isinstance(item, str) and len(item) == 64 for item in value.values())
        for value in plan_configs.values()
    )
    if not valid_plan_configs:
        raise ValueError("DiffTOD selection plan configuration is incomplete")
    if payload["base_selection_sha256"] is not None and len(payload["base_selection_sha256"]) != 64:
        raise ValueError("invalid base selection hash")
    expected_prompts = {
        "assistant": digest(ASSISTANT_POLICY),
        "difftod_realizer": digest(DIFFTOD_REALIZER_POLICY),
        "difftod_guidance": digest(GUIDANCE_POLICY),
    }
    if payload["prompts"] != expected_prompts or payload["test_bodies"] != "sealed":
        raise ValueError("DiffTOD selection prompt/test contract changed")
    canonical(payload)


def write_selection(path: Path, payload):
    validate_selection(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "sha256": digest(payload)}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical(envelope) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return envelope["sha256"]


def verify_selection(path: Path, expected=None):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "sha256"} or digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("DiffTOD selection lock hash mismatch")
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected DiffTOD selection lock")
    validate_selection(envelope["payload"])
    return envelope
