import json
import os
from pathlib import Path

from ...provenance import canonical, digest, file_hash
from ...stages import code_provenance
from .planner import EPL_METHOD_CONTRACT, EPL_METHOD_REVISION, EPL_POLICY_REALIZER
from .rollout import EPL_ROLLOUT_POLICY

SCOPE = {
    "method": "EPL",
    "claim": "single_method_shared_fixed1000_evaluation",
}


def selection_payload(
    root,
    comparison_config,
    configs,
    checkpoints,
    memories,
    vectors,
    asset_lock,
    source_lock,
    development_evidence,
):
    root = Path(root)
    evidence = json.loads(Path(development_evidence).read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "complete"
        or evidence.get("split") != "dev"
        or evidence.get("method") != "EPL"
        or evidence.get("method_revision") != EPL_METHOD_REVISION
        or evidence.get("method_contract") != EPL_METHOD_CONTRACT
    ):
        raise ValueError("complete EPL development evidence is required")
    datasets = {"TopDial", "DuRecDial2.0"}
    mappings = (configs, checkpoints, memories, vectors)
    if any(set(mapping) != datasets for mapping in mappings):
        raise ValueError("EPL selection requires both dataset artifacts")
    hashes = {
        "config_sha256": {key: file_hash(Path(value)) for key, value in configs.items()},
        "checkpoint_sha256": {
            key: file_hash(Path(value)) for key, value in checkpoints.items()
        },
        "memory_sha256": {key: file_hash(Path(value)) for key, value in memories.items()},
        "memory_vector_sha256": {
            key: file_hash(Path(value)) for key, value in vectors.items()
        },
    }
    for key, value in hashes.items():
        if evidence.get(key) != value:
            raise ValueError(f"EPL development evidence changed selected {key}")
    expected = {
        (dataset, protocol, "natural")
        for dataset in datasets
        for protocol in ("profile", "resistant")
    }
    observed = {
        (row.get("dataset"), row.get("protocol"), row.get("budget"))
        for row in evidence.get("summary", [])
    }
    if observed != expected or len(evidence.get("summary", [])) != len(expected):
        raise ValueError("EPL development evidence matrix is incomplete")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "method_revision": EPL_METHOD_REVISION,
        "method_contract": EPL_METHOD_CONTRACT,
        "comparison_config_sha256": file_hash(Path(comparison_config)),
        **hashes,
        "asset_lock_sha256": file_hash(Path(asset_lock)),
        "source_lock_sha256": file_hash(Path(source_lock)),
        "qwen_model_lock_sha256": file_hash(root / "configs/models.lock.json"),
        "fixed1000_resource_sha256": "c0bb70f77d305616a54cfc546bb75a15ad6651c42b7ef8fa7ae98508a103af97",
        "development_evidence_sha256": file_hash(Path(development_evidence)),
        "code": code_provenance(root),
        "prompts": {
            "rollout": digest(EPL_ROLLOUT_POLICY),
            "self_simulation_realizer": digest(EPL_POLICY_REALIZER),
        },
        "test_bodies": "sealed",
    }


def validate_selection(payload):
    required = {
        "schema_version",
        "scope",
        "method_revision",
        "method_contract",
        "comparison_config_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "memory_sha256",
        "memory_vector_sha256",
        "asset_lock_sha256",
        "source_lock_sha256",
        "qwen_model_lock_sha256",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
        "code",
        "prompts",
        "test_bodies",
    }
    if (
        set(payload) != required
        or payload["schema_version"] != 1
        or payload["scope"] != SCOPE
        or payload["method_revision"] != EPL_METHOD_REVISION
        or payload["method_contract"] != EPL_METHOD_CONTRACT
    ):
        raise ValueError("unknown/incomplete EPL selection lock")
    datasets = {"TopDial", "DuRecDial2.0"}
    for field in ("config_sha256", "checkpoint_sha256", "memory_sha256", "memory_vector_sha256"):
        if set(payload[field]) != datasets or any(len(value) != 64 for value in payload[field].values()):
            raise ValueError(f"incomplete EPL {field}")
    scalar_hashes = (
        "comparison_config_sha256",
        "asset_lock_sha256",
        "source_lock_sha256",
        "qwen_model_lock_sha256",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
    )
    if any(len(payload[field]) != 64 for field in scalar_hashes):
        raise ValueError("invalid EPL selection provenance hash")
    if payload["code"].get("dirty") or payload["test_bodies"] != "sealed":
        raise ValueError("EPL selection requires clean code and sealed tests")
    if payload["prompts"] != {
        "rollout": digest(EPL_ROLLOUT_POLICY),
        "self_simulation_realizer": digest(EPL_POLICY_REALIZER),
    }:
        raise ValueError("EPL prompt contract changed")
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
        raise ValueError("EPL selection lock hash mismatch")
    validate_selection(envelope["payload"])
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected EPL selection lock")
    return envelope
