"""Immutable no-training method lock for DirectLLM official-test evaluation."""

import json
import os
from pathlib import Path

from ...provenance import canonical, digest, file_hash
from ...stages import code_provenance
from ...test_prefix import RESOURCE_SHA256
from .planner import (
    DIRECT_LLM_METHODS,
    DIRECT_LLM_V2_METHOD,
    policy_version,
    prompt_hashes,
)

DATASETS = {"TopDial", "DuRecDial2.0"}


def _scope(method):
    if method not in DIRECT_LLM_METHODS:
        raise ValueError(f"unsupported DirectLLM method: {method}")
    return {
        "method": method,
        "claim": "post_hoc_single_method_shared_fixed1000_evaluation",
    }


def _prompt_contract(method):
    prompt_field = (
        "system_prompt_sha256_by_dataset"
        if method == DIRECT_LLM_V2_METHOD
        else "system_prompt_sha256"
    )
    contract = {
        "policy_version": policy_version(method),
        prompt_field: prompt_hashes(method),
        "visible_fields": ["target", "profile", "knowledge"],
        "hidden_fields": ["personality", "future_messages", "outcome"],
        "assistant_calls_per_executed_turn": 1,
        "policy_embedding_calls_per_turn": 0,
    }
    if method == DIRECT_LLM_V2_METHOD:
        contract.update(
            {
                "response_scaffold": "tdrl-final-response-exact",
                "planner_guidance": "empty",
                "target_fact_grounding_by_dataset": {
                    "TopDial": True,
                    "DuRecDial2.0": False,
                },
            }
        )
    return contract


def _method_from_config(config):
    scope = config.get("scope", {})
    methods = scope.get("methods") if isinstance(scope, dict) else None
    if (
        not isinstance(methods, list)
        or len(methods) != 1
        or methods[0] not in DIRECT_LLM_METHODS
    ):
        raise ValueError("unsupported DirectLLM comparison method")
    return methods[0]


def selection_payload(root, comparison_config, development_evidence):
    root = Path(root)
    comparison_config = Path(comparison_config)
    method = _method_from_config(
        json.loads(comparison_config.read_text(encoding="utf-8"))
    )
    evidence_path = Path(development_evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "complete"
        or evidence.get("split") != "dev"
        or evidence.get("method") != method
    ):
        raise ValueError("complete DirectLLM development evidence is required")
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
        raise ValueError("DirectLLM development evidence matrix is incomplete")
    hashes = {
        "comparison_config_sha256": file_hash(comparison_config),
        "qwen_model_lock_sha256": file_hash(root / "configs/models.lock.json"),
    }
    for key, value in hashes.items():
        if evidence.get(key) != value:
            raise ValueError(f"DirectLLM development evidence changed {key}")
    prompt_contract = _prompt_contract(method)
    expected_prompt = prompt_contract.get(
        "system_prompt_sha256", prompt_contract.get("system_prompt_sha256_by_dataset")
    )
    if evidence.get("prompt_sha256") != expected_prompt:
        raise ValueError("DirectLLM development evidence used another prompt")
    code = code_provenance(root)
    if evidence.get("code") != code:
        raise ValueError("DirectLLM development evidence used another source revision")
    evidence_policy = evidence.get("policy_contract", {})
    if (
        evidence_policy.get("version") != policy_version(method)
        or evidence_policy.get("training") != "none"
        or evidence_policy.get("planning") != "none"
    ):
        raise ValueError("DirectLLM development evidence used another policy")
    return {
        "schema_version": 1,
        "scope": _scope(method),
        **hashes,
        "fixed1000_resource_sha256": RESOURCE_SHA256,
        "development_evidence_sha256": file_hash(evidence_path),
        "code": code,
        "prompt_contract": prompt_contract,
        "policy_contract": {
            "training": "none",
            "checkpoint": "none",
            "retrieval": "none",
            "planning": "none",
            "selection": "frozen_single_prompt_no_candidate_selection",
        },
        "test_bodies": "sealed",
    }


def validate_selection(payload):
    required = {
        "schema_version",
        "scope",
        "comparison_config_sha256",
        "qwen_model_lock_sha256",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
        "code",
        "prompt_contract",
        "policy_contract",
        "test_bodies",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete DirectLLM method lock")
    scope = payload.get("scope", {})
    method = scope.get("method") if isinstance(scope, dict) else None
    if method not in DIRECT_LLM_METHODS or payload["scope"] != _scope(method):
        raise ValueError("unexpected DirectLLM method-lock scope")
    for field in (
        "comparison_config_sha256",
        "qwen_model_lock_sha256",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid DirectLLM provenance hash: {field}")
    if payload["fixed1000_resource_sha256"] != RESOURCE_SHA256:
        raise ValueError("DirectLLM fixed resource changed")
    if payload["code"].get("dirty") or payload["test_bodies"] != "sealed":
        raise ValueError("DirectLLM official test requires clean code and sealed tests")
    if payload["prompt_contract"] != _prompt_contract(method):
        raise ValueError("DirectLLM prompt contract changed")
    if payload["policy_contract"] != {
        "training": "none",
        "checkpoint": "none",
        "retrieval": "none",
        "planning": "none",
        "selection": "frozen_single_prompt_no_candidate_selection",
    }:
        raise ValueError("DirectLLM policy contract changed")
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
        raise ValueError("invalid DirectLLM method-lock envelope")
    if digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("DirectLLM method-lock hash mismatch")
    validate_selection(envelope["payload"])
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected DirectLLM method lock")
    return envelope
