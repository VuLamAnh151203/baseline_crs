"""Strict validation for the frozen EnPL reconstruction profile."""

import json
from pathlib import Path

from ...contracts import Decoding


def load_enpl_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "method",
        "profile",
        "data",
        "filtering",
        "retrieval",
        "planning",
        "context",
        "plan_decoding",
        "provenance_labels",
    }
    if set(value) != expected or value["schema_version"] != 1:
        raise ValueError("unknown/incomplete EnPL method configuration")
    if value["method"] != "EnPL" or value["profile"] != "tdrl_domain_adapted":
        raise ValueError("unexpected EnPL method profile")
    if value["data"].get("require_verified_training_bank") is not True:
        raise ValueError("EnPL scored evaluation requires a verified training bank")
    if value["filtering"] != {
        "expected_candidates_per_record": 10,
        "keep_per_conversation": 2,
        "extractor": "independent_token_textrank_v1",
        "keyword_budget": 20,
        "sequence_order": "first_occurrence",
        "empty_sequence_policy": "invalid",
        "tie_break": "candidate_index",
    }:
        raise ValueError("unexpected EnPL filtering contract")
    if value["retrieval"] != {
        "strategy": "domain_conditioned_entity_overlap_cardinality",
        "k": 3,
        "distinct_source_conversations": True,
        "same_domain_only": True,
        "zero_overlap_fallback": "same_domain_consistency_then_deterministic",
    }:
        raise ValueError("unexpected EnPL retrieval contract")
    if value["planning"] != {
        "replan_each_turn": True,
        "mode": "history_conditioned",
        "semantic_regeneration_attempts": 0,
    }:
        raise ValueError("unexpected EnPL planning contract")
    if value["context"] != {
        "max_input_characters": 12000,
        "overflow_policy": "drop_lowest_ranked_examples_then_fail_closed",
        "preserve": ["target", "profile", "knowledge", "observed_history"],
    }:
        raise ValueError("unexpected EnPL context budget contract")
    Decoding(**value["plan_decoding"])
    if not isinstance(value["provenance_labels"], dict):
        raise ValueError("missing EnPL settings provenance labels")
    return value
