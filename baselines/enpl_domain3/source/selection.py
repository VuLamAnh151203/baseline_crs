"""Immutable method lock for independent EnPL reconstruction evaluation."""

import json
import os
from pathlib import Path

from ...provenance import canonical, digest, file_hash
from ...stages import code_provenance
from ...test_prefix import RESOURCE_SHA256
from .bank import load_bank
from .planner import ENPL_POLICY_VERSION
from .prompts import template_hashes

SCOPE = {
    "method": "EnPL",
    "claim": "independent_reconstruction_adapted_shared_fixed1000_evaluation",
}
DATASETS = {"TopDial", "DuRecDial2.0"}
REFERENCE = {
    "schema_version": 1,
    "status": "source_pinned_and_audited",
    "method": "EnPL",
    "upstream_repository": "https://github.com/pandazzh2020/ConvPlan.git",
    "revision": "a76d5a7d6ea1d9c7a07a9574f4901ff79ed440bd",
    "files": {
        "README.md": "f817c6692ec74a35bc66db3346e58dbd9068e34746eafc1c1a8230510c8e238d",
        "ConvPlan.zip": "be487e50e95a202645b3dcc6440cb5d88cb113ae9fee2e73017502ff7bd002d3",
        "ConvPlan.json": "cbb8c656455ecfd3b45defa0d7d2c3b56f19637a9f5988166d9bc2eee5ea6b28",
    },
}
REFERENCE_PAPER = {
    "title": "Thoughts to Target: Enhance Planning for Target-driven Conversation",
    "venue": "EMNLP 2024",
    "url": "https://aclanthology.org/2024.emnlp-main.1175/",
    "attached_pdf_sha256": (
        "334e11ef9ef4d88da18f13230c19e6f3bf5a2907620434071ed11f1f835922b9"
    ),
}
REFERENCE_GUIDE_SHA256 = (
    "8add2abba7c06020192b68b74eb72136c9bd8627661ce6a894fa5ed4dad2201c"
)


def verify_reference_lock(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, expected in REFERENCE.items():
        if value.get(key) != expected:
            raise ValueError(f"EnPL reference lock changed {key}")
    if set(value) != set(REFERENCE) | {
        "paper",
        "implementation_guide_sha256",
        "limitations",
    }:
        raise ValueError("unknown/incomplete EnPL reference lock")
    if value["paper"] != REFERENCE_PAPER:
        raise ValueError("EnPL paper provenance changed")
    if value["implementation_guide_sha256"] != REFERENCE_GUIDE_SHA256:
        raise ValueError("EnPL implementation-guide provenance changed")
    if not value["limitations"] or not all(
        isinstance(item, str) and item for item in value["limitations"]
    ):
        raise ValueError("invalid EnPL reference metadata")
    return value


def prompt_contract():
    return {
        "policy_version": ENPL_POLICY_VERSION,
        "template_sha256": template_hashes(),
        "planning_mode": "history_conditioned",
        "retrieval": "same_domain_exact_entity_overlap_cardinality_k3",
        "plan_output": "numbered_or_bulleted_at_least_2_steps_observed_v4",
        "malformed_plan": "execute_nonempty_raw_plan_without_retry",
        "truncated_plan": "execute_and_record_without_retry",
        "visible_fields": ["target", "profile", "knowledge", "observed_history"],
        "hidden_fields": ["personality", "future_messages", "outcome"],
        "planner_calls_per_executed_turn": 1,
        "response_calls_per_executed_turn": 1,
        "policy_embedding_calls_per_turn": 0,
        "max_input_characters": 12000,
        "context_overflow": "drop_lowest_ranked_examples_then_fail_closed",
    }


def selection_payload(
    root,
    comparison_config,
    enpl_config,
    source_lock,
    bank,
    bank_manifest,
    development_evidence,
):
    root = Path(root)
    source_lock = Path(source_lock)
    verify_reference_lock(source_lock)
    _entries, manifest = load_bank(bank, bank_manifest, require_complete=True)
    evidence_path = Path(development_evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "complete"
        or evidence.get("split") != "dev"
        or evidence.get("method") != "EnPL"
    ):
        raise ValueError("complete EnPL development evidence is required")
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
        raise ValueError("EnPL development evidence matrix is incomplete")
    hashes = {
        "comparison_config_sha256": file_hash(Path(comparison_config)),
        "enpl_config_sha256": file_hash(Path(enpl_config)),
        "source_lock_sha256": file_hash(source_lock),
        "bank_sha256": file_hash(Path(bank)),
        "bank_manifest_sha256": file_hash(Path(bank_manifest)),
        "qwen_model_lock_sha256": file_hash(root / "configs/models.lock.json"),
    }
    if hashes["bank_sha256"] != manifest["bank_sha256"]:
        raise ValueError("EnPL bank manifest does not identify selected bank")
    for key, expected_hash in hashes.items():
        if evidence.get(key) != expected_hash:
            raise ValueError(f"EnPL development evidence changed {key}")
    contract = prompt_contract()
    if evidence.get("prompt_contract") != contract:
        raise ValueError("EnPL development evidence used another prompt contract")
    code = code_provenance(root)
    if evidence.get("code") != code:
        raise ValueError("EnPL development evidence used another source revision")
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
            "offline_preparation": "textrank_consistency_top2",
            "retrieval": "same_domain_exact_entity_overlap_cardinality_k3",
            "planning": "replan_each_turn_then_generate_response",
            "selection": "frozen_reconstruction_no_test_tuning",
        },
        "test_bodies": "sealed",
    }


def validate_selection(payload):
    required = {
        "schema_version",
        "scope",
        "comparison_config_sha256",
        "enpl_config_sha256",
        "source_lock_sha256",
        "bank_sha256",
        "bank_manifest_sha256",
        "qwen_model_lock_sha256",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
        "code",
        "prompt_contract",
        "policy_contract",
        "test_bodies",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete EnPL method lock")
    if payload["scope"] != SCOPE or payload["prompt_contract"] != prompt_contract():
        raise ValueError("EnPL method contract changed")
    for field in required - {
        "schema_version",
        "scope",
        "code",
        "prompt_contract",
        "policy_contract",
        "test_bodies",
    }:
        if not isinstance(payload[field], str) or len(payload[field]) != 64:
            raise ValueError(f"invalid EnPL provenance hash: {field}")
    if payload["fixed1000_resource_sha256"] != RESOURCE_SHA256:
        raise ValueError("EnPL fixed resource changed")
    if payload["code"].get("dirty") or payload["test_bodies"] != "sealed":
        raise ValueError("EnPL official test requires clean code and sealed tests")
    expected_policy = {
        "training": "none",
        "offline_preparation": "textrank_consistency_top2",
        "retrieval": "same_domain_exact_entity_overlap_cardinality_k3",
        "planning": "replan_each_turn_then_generate_response",
        "selection": "frozen_reconstruction_no_test_tuning",
    }
    if payload["policy_contract"] != expected_policy:
        raise ValueError("EnPL policy contract changed")
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
        raise ValueError("invalid EnPL method-lock envelope")
    if digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("EnPL method-lock hash mismatch")
    validate_selection(envelope["payload"])
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected EnPL method lock")
    return envelope
