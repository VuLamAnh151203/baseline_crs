"""Immutable no-training method lock for adapted ProCoT evaluation."""

import json
import os
from pathlib import Path

from ...provenance import canonical, digest, file_hash
from ...stages import code_provenance
from ...test_prefix import RESOURCE_SHA256
from .prompts import (
    PROCOT_INSTRUCTION,
    PROCOT_ONE_SHOT_DEMONSTRATION,
    PROCOT_POLICY_VERSION,
    procot_system_prompt,
)

SCOPE = {
    "method": "ProCoT",
    "claim": "post_hoc_adapted_one_shot_shared_fixed1000_evaluation",
}
DATASETS = {"TopDial", "DuRecDial2.0"}
REFERENCE = {
    "schema_version": 1,
    "status": "source_pinned_and_audited",
    "method": "ProCoT",
    "upstream_repository": "https://github.com/dengyang17/LLM-Proactive.git",
    "revision": "77ff4ed7dfda11afd719db3bd40886a1a51edf4e",
    "files": {
        "topkg/prompt.txt": (
            "0926dfa356dc7ba77ec6c0003727722cdd73236b292fed407c90a33d7705bb37"
        ),
        "topkg/self_play.py": (
            "4a6f5e22cd5abebe55f404a8b630aa3ef9dcc4d90c3e0a2817da43cb190ab710"
        ),
        "topkg/chatgpt_self_play.py": (
            "bd2102bd7dfe9b0c3d8089d97dad3399d5b75df9058b6ac32857becb5317cc01"
        ),
    },
}


def verify_reference_lock(path):
    """Fail closed if the committed upstream source identity changes."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in REFERENCE.items():
        if payload.get(key) != value:
            raise ValueError(f"ProCoT reference lock changed {key}")
    allowed = set(REFERENCE) | {"paper", "adaptation"}
    if set(payload) != allowed:
        raise ValueError("unknown/incomplete ProCoT reference lock")
    if not isinstance(payload.get("paper"), str) or not isinstance(
        payload.get("adaptation"), dict
    ):
        raise ValueError("invalid ProCoT reference metadata")
    return payload


def prompt_contract():
    fixture = {
        "id": "prompt-contract",
        "target": ["Movie recommendation", "Example"],
        "profile": {"preference": "example"},
        "knowledge": [["Example", "type", "movie"]],
    }
    return {
        "policy_version": PROCOT_POLICY_VERSION,
        "system_prompt_sha256": digest(procot_system_prompt(fixture)),
        "instruction_sha256": digest(PROCOT_INSTRUCTION),
        "demonstration_sha256": digest(PROCOT_ONE_SHOT_DEMONSTRATION),
        "visible_fields": ["target", "profile", "knowledge"],
        "hidden_fields": ["personality", "future_messages", "outcome"],
        "assistant_calls_per_executed_turn": 1,
        "policy_embedding_calls_per_turn": 0,
        "response_parser": "final-marker-deterministic-observable-v3",
        "malformed_output": "execute_raw_and_record_violation_without_retry",
    }


def selection_payload(root, comparison_config, source_lock, development_evidence):
    root = Path(root)
    source_path = Path(source_lock)
    verify_reference_lock(source_path)
    evidence_path = Path(development_evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "complete"
        or evidence.get("split") != "dev"
        or evidence.get("method") != "ProCoT"
    ):
        raise ValueError("complete ProCoT development evidence is required")
    expected = {
        (dataset, protocol, "natural")
        for dataset in DATASETS
        for protocol in ("profile", "resistant")
    }
    observed = {
        (row.get("dataset"), row.get("protocol"), row.get("budget"))
        for row in evidence.get("summary", [])
    }
    if observed != expected or len(evidence.get("summary", [])) != len(expected):
        raise ValueError("ProCoT development evidence matrix is incomplete")
    hashes = {
        "comparison_config_sha256": file_hash(Path(comparison_config)),
        "source_lock_sha256": file_hash(source_path),
        "qwen_model_lock_sha256": file_hash(root / "configs/models.lock.json"),
    }
    for key, value in hashes.items():
        if evidence.get(key) != value:
            raise ValueError(f"ProCoT development evidence changed {key}")
    contract = prompt_contract()
    if evidence.get("prompt_sha256") != contract["system_prompt_sha256"]:
        raise ValueError("ProCoT development evidence used another prompt")
    code = code_provenance(root)
    if evidence.get("code") != code:
        raise ValueError("ProCoT development evidence used another source revision")
    evidence_policy = evidence.get("policy_contract", {})
    if (
        evidence_policy.get("version") != PROCOT_POLICY_VERSION
        or evidence_policy.get("training") != "none"
        or evidence_policy.get("planning") != "one_call_explicit_cot"
    ):
        raise ValueError("ProCoT development evidence used another policy")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        **hashes,
        "fixed1000_resource_sha256": RESOURCE_SHA256,
        "development_evidence_sha256": file_hash(evidence_path),
        "code": code,
        "prompt_contract": contract,
        "policy_contract": {
            "training": "none",
            "checkpoint": "none",
            "retrieval": "none",
            "planning": "one_call_explicit_cot",
            "selection": "frozen_single_prompt_no_candidate_selection",
        },
        "test_bodies": "sealed",
    }


def validate_selection(payload):
    required = {
        "schema_version",
        "scope",
        "comparison_config_sha256",
        "source_lock_sha256",
        "qwen_model_lock_sha256",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
        "code",
        "prompt_contract",
        "policy_contract",
        "test_bodies",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete ProCoT method lock")
    if payload["scope"] != SCOPE:
        raise ValueError("unexpected ProCoT method-lock scope")
    for field in (
        "comparison_config_sha256",
        "source_lock_sha256",
        "qwen_model_lock_sha256",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid ProCoT provenance hash: {field}")
    if payload["fixed1000_resource_sha256"] != RESOURCE_SHA256:
        raise ValueError("ProCoT fixed resource changed")
    if payload["code"].get("dirty") or payload["test_bodies"] != "sealed":
        raise ValueError("ProCoT official test requires clean code and sealed tests")
    if payload["prompt_contract"] != prompt_contract():
        raise ValueError("ProCoT prompt contract changed")
    if payload["policy_contract"] != {
        "training": "none",
        "checkpoint": "none",
        "retrieval": "none",
        "planning": "one_call_explicit_cot",
        "selection": "frozen_single_prompt_no_candidate_selection",
    }:
        raise ValueError("ProCoT policy contract changed")
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
    if set(envelope) != {"payload", "sha256"}:
        raise ValueError("invalid ProCoT method-lock envelope")
    if digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("ProCoT method-lock hash mismatch")
    validate_selection(envelope["payload"])
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected ProCoT method lock")
    return envelope
