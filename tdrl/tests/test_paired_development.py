from copy import deepcopy

import pytest

from tdrl.paired_development import compare_records
from tdrl.planner import derived_seed


def matrix(accepted):
    return [
        {
            "status": "complete",
            "infrastructure_errors": 0,
            "seed": 17,
            "scenario_id": str(i),
            "protocol": p,
            "accepted": accepted,
            "judge_protocol_sha256": "judge",
            "acceptance_rule_version": "semantic-v2",
            "turn_budget": 10,
            "turns": 1 if accepted else 10,
            "generation_requests": 3,
            "embedding_requests": 1,
            "input_tokens": 30,
            "output_tokens": 3,
            "total_tokens": 33,
            "h100_seconds": 1,
            "wall_s": 1,
            "selection_sha256": "development_or_smoke",
            "turns_trace": [
                {
                    "turn": 0,
                    "simulator_seed": derived_seed(17, f"{i}/{p}/0", "simulator"),
                    "generation": [
                        {
                            "role": role,
                            "model": {"revision": "pinned"},
                            "decoding": {"temperature": 0.7},
                        }
                        for role in ["assistant", "simulator", "judge"]
                    ],
                }
            ],
        }
        for p in ["profile", "resistant"]
        for i in range(2)
    ]


def test_paired_macro_clusters_both_protocols_and_is_development_only():
    result = compare_records(matrix(True), matrix(False), expected_count=2)
    assert result["paired_macro_protocol_sr"]["difference"] == 1
    assert result["paired_macro_protocol_sr"]["ci95"] == [1, 1]
    assert result["paired_macro_protocol_sr"]["resamples"] == 10000
    assert "disjoint confirmation" in result["scope_limits"]


def test_pairing_rejects_missing_duplicate_ids_or_outcome_contracts():
    baseline = matrix(False)
    with pytest.raises(ValueError, match="incomplete development"):
        compare_records(matrix(True)[:-1], baseline, expected_count=2)
    with pytest.raises(ValueError, match="duplicate"):
        compare_records(matrix(True) + matrix(True)[:1], baseline, expected_count=2)
    candidate = matrix(True)
    candidate[0]["judge_protocol_sha256"] = "different"
    with pytest.raises(ValueError, match="outcome/simulator"):
        compare_records(candidate, baseline, expected_count=2)
    candidate = matrix(True)
    candidate[0]["scenario_id"] = "other"
    with pytest.raises(ValueError, match="protocol populations"):
        compare_records(candidate, baseline, expected_count=2)


def test_pairing_rejects_model_decoding_and_nonfinite_compute():
    baseline = matrix(False)
    candidate = deepcopy(baseline)
    candidate[0]["turns_trace"][0]["generation"][0]["model"]["revision"] = "other"
    with pytest.raises(ValueError, match="model/decoding"):
        compare_records(candidate, baseline, expected_count=2)
    candidate = deepcopy(baseline)
    candidate[0]["turns_trace"][0]["generation"][0]["decoding"]["temperature"] = 1.1
    with pytest.raises(ValueError, match="model/decoding"):
        compare_records(candidate, baseline, expected_count=2)
    candidate = deepcopy(baseline)
    candidate[0]["turns_trace"][0]["simulator_seed"] += 1
    with pytest.raises(ValueError, match="simulator seed"):
        compare_records(candidate, baseline, expected_count=2)
    candidate = deepcopy(baseline)
    candidate[0]["h100_seconds"] = float("nan")
    with pytest.raises(ValueError, match="invalid result counter"):
        compare_records(candidate, baseline, expected_count=2)
