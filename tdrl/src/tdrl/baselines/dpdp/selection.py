import json
import os
from pathlib import Path

from ...planner import ASSISTANT_POLICY
from ...provenance import canonical, digest, file_hash
from ...stages import code_provenance
from ...strategies import ACTION_INSTRUCTIONS
from .planner import DPDP_REALIZER_POLICY
from .rewards import CRITIC_POLICY
from .rollout import ROLLOUT_ASSISTANT_POLICY

SCOPE = {
    "methods": ["PPDPP", "DiffTOD", "TRIP", "DPDP", "TDRL-R", "TDRL"],
    "claim": "six_method_automated_comparison",
}


def _action_prompt_digest(policy):
    return digest({"policy": policy, "action_instructions": ACTION_INSTRUCTIONS})


def selection_payload(
    root,
    config_path,
    checkpoints,
    offline_checkpoint_sha256,
    dpdp_config_sha256,
    training_config_sha256,
    gate_thresholds,
    search_budgets,
    asset_lock,
    development_evidence,
    base_selection=None,
):
    evidence = json.loads(Path(development_evidence).read_text(encoding="utf-8"))
    if evidence.get("status") != "complete" or evidence.get("split") != "dev":
        raise ValueError("complete DPDP development evidence is required")
    base_sha256 = None
    if base_selection is not None:
        envelope = json.loads(Path(base_selection).read_text(encoding="utf-8"))
        if (
            set(envelope) != {"payload", "sha256"}
            or digest(envelope["payload"]) != envelope["sha256"]
        ):
            raise ValueError("base comparison selection lock is invalid")
        base_sha256 = envelope["sha256"]
    checkpoint_sha256 = {dataset: file_hash(Path(path)) for dataset, path in checkpoints.items()}
    code = code_provenance(Path(root))
    qwen_model_lock_sha256 = file_hash(Path(root) / "configs/models.lock.json")
    expected = {
        "method": "DPDP",
        "config_sha256": file_hash(Path(config_path)),
        "checkpoint_sha256": checkpoint_sha256,
        "offline_checkpoint_sha256": offline_checkpoint_sha256,
        "dpdp_config_sha256": dpdp_config_sha256,
        "training_config_sha256": training_config_sha256,
        "gate_thresholds": gate_thresholds,
        "search_budgets": search_budgets,
        "asset_lock_sha256": file_hash(Path(asset_lock)),
        "code": code,
        "qwen_model_lock_sha256": qwen_model_lock_sha256,
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("DPDP development evidence does not match selected artifacts")
    expected_groups = {
        (dataset, protocol, budget)
        for dataset in ("TopDial", "DuRecDial2.0")
        for protocol in ("profile", "resistant")
        for budget in ("natural", "matched")
    }
    found = {
        (row.get("dataset"), row.get("protocol"), row.get("budget"))
        for row in evidence.get("summary", [])
        if row.get("method") == "DPDP"
    }
    if found != expected_groups:
        raise ValueError("DPDP development evidence matrix is incomplete")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "config_sha256": expected["config_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "offline_checkpoint_sha256": offline_checkpoint_sha256,
        "dpdp_config_sha256": dpdp_config_sha256,
        "training_config_sha256": training_config_sha256,
        "gate_thresholds": gate_thresholds,
        "search_budgets": search_budgets,
        "asset_lock_sha256": expected["asset_lock_sha256"],
        "development_evidence_sha256": file_hash(Path(development_evidence)),
        "base_selection_sha256": base_sha256,
        "code": code,
        "qwen_model_lock_sha256": qwen_model_lock_sha256,
        "prompts": {
            "assistant": digest(ASSISTANT_POLICY),
            "realizer": _action_prompt_digest(DPDP_REALIZER_POLICY),
            "rollout_assistant": _action_prompt_digest(ROLLOUT_ASSISTANT_POLICY),
            "critic": digest(CRITIC_POLICY),
        },
        "source_revision": "4d1e22e35db452d51f0f39388456d310ec1c971b",
        "test_bodies": "sealed",
    }


def validate_selection(payload):
    required = {
        "schema_version",
        "scope",
        "config_sha256",
        "checkpoint_sha256",
        "offline_checkpoint_sha256",
        "dpdp_config_sha256",
        "training_config_sha256",
        "gate_thresholds",
        "search_budgets",
        "asset_lock_sha256",
        "development_evidence_sha256",
        "base_selection_sha256",
        "code",
        "qwen_model_lock_sha256",
        "prompts",
        "source_revision",
        "test_bodies",
    }
    if set(payload) != required or payload["schema_version"] != 1 or payload["scope"] != SCOPE:
        raise ValueError("unknown/incomplete DPDP selection lock")
    hashes = ("config_sha256", "asset_lock_sha256", "development_evidence_sha256")
    if any(not isinstance(payload[key], str) or len(payload[key]) != 64 for key in hashes):
        raise ValueError("DPDP selection provenance is incomplete")
    datasets = {"TopDial", "DuRecDial2.0"}
    if set(payload["checkpoint_sha256"]) != datasets or any(
        len(value) != 64 for value in payload["checkpoint_sha256"].values()
    ):
        raise ValueError("DPDP selection checkpoint provenance is incomplete")
    if set(payload["offline_checkpoint_sha256"]) != datasets or any(
        len(value) != 64 for value in payload["offline_checkpoint_sha256"].values()
    ):
        raise ValueError("DPDP offline-checkpoint provenance is incomplete")
    for field in ("dpdp_config_sha256", "training_config_sha256"):
        configs = payload[field]
        if set(configs) != datasets:
            raise ValueError("DPDP selection configuration provenance is incomplete")
        for value in configs.values():
            values = value.values() if isinstance(value, dict) else (value,)
            if any(not isinstance(item, str) or len(item) != 64 for item in values):
                raise ValueError("invalid DPDP configuration hash")
    if set(payload["gate_thresholds"]) != datasets or set(payload["search_budgets"]) != datasets:
        raise ValueError("DPDP gate/search selection is incomplete")
    for dataset in datasets:
        if set(payload["gate_thresholds"][dataset]) != {"natural", "matched"}:
            raise ValueError("DPDP selected gate thresholds are incomplete")
        if set(payload["search_budgets"][dataset]) != {"natural", "matched"}:
            raise ValueError("DPDP selected search budgets are incomplete")
        if any(
            not isinstance(value, (int, float)) or not 0 <= value <= 1
            for value in payload["gate_thresholds"][dataset].values()
        ):
            raise ValueError("invalid selected DPDP gate threshold")
        if len(set(payload["gate_thresholds"][dataset].values())) != 1:
            raise ValueError("DPDP natural/matched budgets changed the frozen gate")
        for budget in payload["search_budgets"][dataset].values():
            if (
                set(budget) != {"simulations", "depth"}
                or budget["simulations"] < 2
                or budget["depth"] < 1
            ):
                raise ValueError("invalid selected DPDP search budget")
    expected_prompts = {
        "assistant": digest(ASSISTANT_POLICY),
        "realizer": _action_prompt_digest(DPDP_REALIZER_POLICY),
        "rollout_assistant": _action_prompt_digest(ROLLOUT_ASSISTANT_POLICY),
        "critic": digest(CRITIC_POLICY),
    }
    if payload["prompts"] != expected_prompts or payload["test_bodies"] != "sealed":
        raise ValueError("DPDP prompt/test contract changed")
    if payload["source_revision"] != "4d1e22e35db452d51f0f39388456d310ec1c971b":
        raise ValueError("DPDP source revision changed")
    if payload["base_selection_sha256"] is not None and len(payload["base_selection_sha256"]) != 64:
        raise ValueError("invalid DPDP base selection hash")
    if set(payload["code"]) != {"commit", "dirty"} or payload["code"]["dirty"]:
        raise ValueError("DPDP scientific selection requires a clean committed tree")
    if len(payload["qwen_model_lock_sha256"]) != 64:
        raise ValueError("invalid Qwen model-lock hash")
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
        raise ValueError("DPDP selection lock hash mismatch")
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected DPDP selection lock")
    validate_selection(envelope["payload"])
    return envelope
