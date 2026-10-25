"""Immutable selection lock for SCOPE-CRS-lite reward checkpoints."""

import json
import os
from pathlib import Path

from ...provenance import canonical, digest, file_hash
from ...stages import code_provenance
from ...test_prefix import RESOURCE_SHA256
from ..direct_llm.planner import direct_v2_system_prompt
from .assets import verify_llama_guard_snapshot, verify_transition_manifest
from .config import ScopeConfig
from .planner import POLICY_VERSION

DATASETS = ("TopDial", "DuRecDial2.0")
REFERENCE_REVISION = "b35dbc64808afce25e8f8e5d6838ecf271ec4d0b"


def verify_reference_lock(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "method",
        "upstream_repository",
        "revision",
        "paper",
        "transition_release_folder_id",
        "files",
        "adaptation",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete SCOPE reference lock")
    if (
        payload["status"] != "source_pinned_and_audited"
        or payload["method"] != "SCOPE-CRS-lite"
        or payload["revision"] != REFERENCE_REVISION
        or payload["upstream_repository"]
        != "https://github.com/chenzhiliang94/convo-plan-SCOPE.git"
        or len(payload["files"]) != 11
    ):
        raise ValueError("SCOPE upstream source identity changed")
    return payload


def prompt_contract():
    fixture = {
        "id": "prompt-contract",
        "target": ["Movie recommendation", "Example"],
        "profile": {"preference": "example"},
        "knowledge": [["Example", "type", "movie"]],
    }
    return {
        "policy_version": POLICY_VERSION,
        "system_prompt_sha256_by_dataset": {
            dataset: digest(direct_v2_system_prompt(fixture, dataset)) for dataset in DATASETS
        },
        "visible_fields": ["target", "profile", "knowledge", "observed_history"],
        "hidden_fields": ["personality", "future_messages", "test_outcome"],
        "candidate_calls_per_executed_turn": 5,
        "shared_qwen_embedding_calls_per_turn": 0,
        "intrinsic_encoder": "Meta-Llama-Guard-2-8B projected 1024-D",
    }


def selection_payload(
    root,
    comparison_config,
    source_lock,
    development_evidence,
    run_root,
    configs,
    transitions,
    llama_guard,
    llama_guard_manifest,
):
    root, run_root = Path(root), Path(run_root)
    verify_reference_lock(source_lock)
    evidence_path = Path(development_evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "complete"
        or evidence.get("split") != "dev"
        or evidence.get("method") != "SCOPE-CRS-lite"
    ):
        raise ValueError("complete SCOPE development evidence required")
    expected = {
        (dataset, protocol, "natural")
        for dataset in DATASETS
        for protocol in ("profile", "resistant")
    }
    observed = {
        (row.get("dataset"), row.get("protocol"), row.get("budget"))
        for row in evidence.get("summary", [])
    }
    if observed != expected or len(evidence["summary"]) != len(expected):
        raise ValueError("SCOPE development evidence matrix incomplete")
    transition = verify_transition_manifest(transitions)
    llama = verify_llama_guard_snapshot(llama_guard, llama_guard_manifest)
    reward_checkpoints = {}
    config_hashes = {}
    for dataset in DATASETS:
        config = ScopeConfig.from_path(configs[dataset])
        checkpoint = run_root / "reward-training" / dataset / "checkpoint-best.pt"
        summary = run_root / "reward-training" / dataset / "summary.json"
        if not checkpoint.is_file() or not summary.is_file():
            raise FileNotFoundError(f"missing SCOPE reward selection artifacts: {dataset}")
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("dataset") != dataset:
            raise ValueError("invalid SCOPE reward training summary")
        reward_checkpoints[dataset] = {
            "path": str(checkpoint),
            "sha256": file_hash(checkpoint),
            "summary_sha256": file_hash(summary),
            "selection_rule": "minimum development BCE; train/dev only",
        }
        config_hashes[dataset] = config.sha256
    hashes = {
        "comparison_config_sha256": file_hash(comparison_config),
        "source_lock_sha256": file_hash(source_lock),
        "qwen_model_lock_sha256": file_hash(root / "configs/models.lock.json"),
        "transition_manifest_sha256": transition["sha256"],
        "llama_guard_manifest_sha256": llama["manifest_sha256"],
    }
    for key, value in hashes.items():
        if evidence.get(key) != value:
            raise ValueError(f"SCOPE development evidence changed {key}")
    if evidence.get("reward_checkpoint_sha256_by_dataset") != {
        dataset: reward_checkpoints[dataset]["sha256"] for dataset in DATASETS
    }:
        raise ValueError("SCOPE development used other reward checkpoints")
    code = code_provenance(root)
    if evidence.get("code") != code:
        raise ValueError("SCOPE development used another source revision")
    return {
        "schema_version": 1,
        "scope": {
            "method": "SCOPE-CRS-lite",
            "claim": "task_adapted_scope_shared_fixed1000_evaluation",
        },
        **hashes,
        "scope_config_sha256_by_dataset": config_hashes,
        "reward_checkpoints": reward_checkpoints,
        "fixed1000_resource_sha256": RESOURCE_SHA256,
        "development_evidence_sha256": file_hash(evidence_path),
        "code": code,
        "prompt_contract": prompt_contract(),
        "policy_contract": {
            "planning": "semantic-space-online-mcts",
            "transition": "released LMSYS four-seed ensemble",
            "reward": "dataset-specific goal-conditioned train/dev head",
            "selection": "minimum reward development BCE before dialogue test",
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
        "transition_manifest_sha256",
        "llama_guard_manifest_sha256",
        "scope_config_sha256_by_dataset",
        "reward_checkpoints",
        "fixed1000_resource_sha256",
        "development_evidence_sha256",
        "code",
        "prompt_contract",
        "policy_contract",
        "test_bodies",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unknown/incomplete SCOPE method lock")
    if payload["scope"] != {
        "method": "SCOPE-CRS-lite",
        "claim": "task_adapted_scope_shared_fixed1000_evaluation",
    }:
        raise ValueError("unexpected SCOPE selection scope")
    if set(payload["reward_checkpoints"]) != set(DATASETS):
        raise ValueError("SCOPE selection requires both reward checkpoints")
    if payload["fixed1000_resource_sha256"] != RESOURCE_SHA256:
        raise ValueError("SCOPE fixed resource changed")
    if payload["code"].get("dirty") or payload["test_bodies"] != "sealed":
        raise ValueError("SCOPE test requires clean committed code")
    if payload["prompt_contract"] != prompt_contract():
        raise ValueError("SCOPE prompt contract changed")
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
        raise ValueError("invalid SCOPE method-lock envelope")
    validate_selection(envelope["payload"])
    if expected is not None and envelope["sha256"] != expected:
        raise ValueError("unexpected SCOPE method lock")
    return envelope
