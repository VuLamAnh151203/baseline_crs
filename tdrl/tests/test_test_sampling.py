import json
from pathlib import Path

import pytest

from tdrl.comparison import validate_config
from tdrl.test_sampling import projected_sample_counts, sample_test_scenarios

ROOT = Path(__file__).resolve().parents[1]
SPEC = {
    "schema_version": 1,
    "method": "pooled_uniform_without_replacement",
    "total_scenarios": 2000,
    "seed": 42,
}


def population():
    return {
        d: [{"id": str(i), "outcome": i % 2} for i in range(n)]
        for d, n in (("TopDial", 3606), ("DuRecDial2.0", 1752))
    }


def test_sample_is_paired_stable_and_outcome_independent():
    populations = population()
    rows, manifest = sample_test_scenarios(populations, SPEC)
    assert sum(map(len, rows.values())) == 2000
    assert all(len({r["id"] for r in group}) == len(group) for group in rows.values())
    reordered = {
        d: [{**r, "outcome": "changed"} for r in reversed(group)]
        for d, group in reversed(list(populations.items()))
    }
    _, replay = sample_test_scenarios(reordered, SPEC)
    assert replay == manifest
    assert manifest["claim"] == "sampled_official_test_subset"
    assert all(manifest["selected_ids"].values())
    assert manifest["population_membership_sha256"] != manifest["selected_membership_sha256"]


def test_population_changes_are_detectable_and_invalid_inputs_fail():
    populations = population()
    _, original = sample_test_scenarios(populations, SPEC)
    populations["TopDial"].append({"id": "additional"})
    _, changed = sample_test_scenarios(populations, SPEC)
    assert changed["population_membership_sha256"] != original["population_membership_sha256"]
    populations["TopDial"].append({"id": "additional"})
    with pytest.raises(ValueError, match="duplicate"):
        sample_test_scenarios(populations, SPEC)
    with pytest.raises(ValueError, match="smaller"):
        sample_test_scenarios({"TopDial": [{"id": "one"}]}, SPEC)
    with pytest.raises(ValueError, match="unknown"):
        sample_test_scenarios(population(), {**SPEC, "seed": 43})


def test_only_evaluation_membership_changes_and_projection_matches_scope():
    full = json.loads((ROOT / "configs/comparison-judge-v2.json").read_text())
    sampled = json.loads((ROOT / "configs/comparison-judge-v2-sample2000.json").read_text())
    validate_config(sampled)
    assert sampled.pop("test_sampling") == SPEC
    assert sampled == full
    counts = projected_sample_counts({"TopDial": 3606, "DuRecDial2.0": 1752}, SPEC)
    assert sum(counts.values()) == pytest.approx(2000)
    assert counts["TopDial"] / counts["DuRecDial2.0"] == pytest.approx(3606 / 1752)
    assert projected_sample_counts({"TopDial": 3606}, None) == {"TopDial": 3606}


def test_single_seed_scope_preserves_training_and_sampling_contracts():
    old = json.loads((ROOT / "configs/comparison-judge-v2-sample2000.json").read_text())
    cfg = json.loads((ROOT / "configs/comparison-judge-v2-sample2000-seed17.json").read_text())
    validate_config(cfg)
    assert cfg["seeds"] == [17]
    assert cfg["test_sampling"] == SPEC
    assert {**cfg, "seeds": old["seeds"]} == old
    with pytest.raises(ValueError, match="authorized seeds"):
        validate_config({**cfg, "seeds": [42]})
    with pytest.raises(ValueError, match="single-seed scope"):
        validate_config({**cfg, "head": {**cfg["head"], "seed": 42}})
    with pytest.raises(ValueError, match="single-seed scope"):
        validate_config({k: v for k, v in cfg.items() if k != "test_sampling"})


def test_single_seed_pairing_and_scenario_confidence_intervals():
    from tdrl.metrics import paired_bootstrap, validate_records

    records = [{"scenario_id": str(i), "seed": 17, "selection_sha256": "lock"} for i in range(4)]
    # Use full records required by the validator, including finite compute metrics.
    for row in records:
        row.update(
            accepted=False,
            turns=10,
            turn_budget=10,
            infrastructure_errors=0,
            generation_requests=1,
            embedding_requests=1,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            wall_s=1.0,
            h100_seconds=1.0,
        )
    result = paired_bootstrap([[1], [0], [1], [0]], [[0], [0], [1], [0]])
    assert result["difference"] == 0.25
    assert result["unit"] == "scenario_cluster"
    assert result == paired_bootstrap([[1], [0], [1], [0]], [[0], [0], [1], [0]])
    validate_records(records, [str(i) for i in range(4)], [17], "lock")
    with pytest.raises(ValueError):
        validate_records(records, [str(i) for i in range(4)], [17, 29, 43], "lock")


def test_500_sample_is_shared_reproducible_and_preserves_training(monkeypatch):
    freeze_fixture(monkeypatch, population())
    spec = {**SPEC, "total_scenarios": 500}
    first, manifest = sample_test_scenarios(population(), spec)
    assert sum(map(len, first.values())) == 500
    _, replay = sample_test_scenarios(
        {d: list(reversed(rows)) for d, rows in population().items()}, spec
    )
    assert replay == manifest
    assert sum(
        projected_sample_counts({"TopDial": 3606, "DuRecDial2.0": 1752}, spec).values()
    ) == pytest.approx(500)
    full = json.loads((ROOT / "configs/comparison-judge-v2-sample2000-seed17.json").read_text())
    cfg = json.loads((ROOT / "configs/comparison-judge-v2-sample500-seed17.json").read_text())
    validate_config(cfg)
    assert cfg["test_sampling"] == spec
    assert {**cfg, "test_sampling": full["test_sampling"]} == full
    with pytest.raises(ValueError, match="unknown"):
        sample_test_scenarios(population(), {**spec, "total_scenarios": 499})


def freeze_fixture(monkeypatch, populations):
    import random

    import tdrl.test_sampling as sampling
    from tdrl.provenance import digest

    keys = sorted((d, row["id"]) for d, rows in populations.items() for row in rows)
    chosen = sorted(random.Random(42).sample(keys, 500))
    ids = {d: [i for ds, i in chosen if ds == d] for d in populations}
    ordered = {d: sorted(rows, key=lambda r: r["id"]) for d, rows in populations.items()}
    lookup = {d: {r["id"]: (i, r) for i, r in enumerate(rows)} for d, rows in ordered.items()}
    fixed = {
        "schema_version": 1,
        "protocol": {**SPEC, "total_scenarios": 500},
        "index_basis": "zero_based_per_dataset_lexicographic_id_order",
        "population_counts": {d: len(rows) for d, rows in populations.items()},
        "population_membership_sha256": digest(keys),
        "selected_membership_sha256": digest(chosen),
        "selected_ids": ids,
        "selected_indices": {d: [lookup[d][i][0] for i in group] for d, group in ids.items()},
        "selected_content_sha256": {
            d: {i: digest(lookup[d][i][1]) for i in group} for d, group in ids.items()
        },
    }
    monkeypatch.setattr(sampling, "_fixed_subset", lambda: (fixed, digest(fixed)))
    return fixed


def test_fixed_subset_never_calls_rng_and_rejects_dataset_drift(monkeypatch):
    import copy

    import tdrl.test_sampling as sampling

    rows = population()
    fixed = freeze_fixture(monkeypatch, rows)

    def forbid_sampling(*args, **kwargs):
        raise AssertionError("fixed evaluation must never resample")

    monkeypatch.setattr(sampling.random, "Random", forbid_sampling)
    spec = {**SPEC, "total_scenarios": 500}
    first, manifest = sample_test_scenarios(rows, spec)
    second, replay = sample_test_scenarios(
        {d: list(reversed(r)) for d, r in reversed(list(rows.items()))}, spec
    )
    assert first == second and manifest == replay
    assert list(first) == list(second) == ["TopDial", "DuRecDial2.0"]
    assert manifest["frozen_subset"]["selected_indices"] == fixed["selected_indices"]
    changed = copy.deepcopy(rows)
    selected = first["TopDial"][0]["id"]
    next(r for r in changed["TopDial"] if r["id"] == selected)["outcome"] = "drift"
    with pytest.raises(ValueError, match="content mismatch"):
        sample_test_scenarios(changed, spec)
    rows["TopDial"].append({"id": "extra"})
    with pytest.raises(ValueError, match="population membership"):
        sample_test_scenarios(rows, spec)


def test_fixed_subset_checksum_and_index_mismatch_fail_closed(monkeypatch):
    import tdrl.test_sampling as sampling

    pinned, checksum = sampling._fixed_subset()
    assert checksum == sampling.FIXED_500_SHA256
    assert {d: len(ids) for d, ids in pinned["selected_ids"].items()} == {
        "TopDial": 330,
        "DuRecDial2.0": 170,
    }
    monkeypatch.setattr(sampling, "FIXED_500_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        sampling._fixed_subset()
    rows = population()
    fixed = freeze_fixture(monkeypatch, rows)
    fixed["selected_indices"]["TopDial"][0] += 1
    with pytest.raises(ValueError, match="index/ID"):
        sample_test_scenarios(rows, {**SPEC, "total_scenarios": 500})
