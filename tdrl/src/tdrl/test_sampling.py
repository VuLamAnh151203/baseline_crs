"""Outcome-independent, post-lock sampling of official test scenarios."""

import hashlib
import json
import random
from importlib.resources import files

from .provenance import digest

FIXED_500_RESOURCE = "resources/test_scenarios_500_seed42.json"
FIXED_500_SHA256 = "a56f7a0c909bd1ac460f41904da12f99582e98d4ef0b602f2593f7ab21870cb0"


def _fixed_subset():
    # Git may check out JSON with CRLF on Windows; frozen hashes use LF bytes.
    raw = files("tdrl").joinpath(FIXED_500_RESOURCE).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != FIXED_500_SHA256:
        raise ValueError("fixed test subset checksum mismatch")
    return json.loads(raw), FIXED_500_SHA256


def _select_fixed(populations, keys, spec):
    fixed, checksum = _fixed_subset()
    if (
        fixed.get("schema_version") != 1
        or fixed.get("protocol") != spec
        or fixed.get("index_basis") != "zero_based_per_dataset_lexicographic_id_order"
    ):
        raise ValueError("unknown fixed test subset protocol")
    if fixed["population_counts"] != {d: len(rows) for d, rows in populations.items()} or fixed[
        "population_membership_sha256"
    ] != digest(keys):
        raise ValueError("fixed test population membership mismatch")
    if set(fixed["selected_ids"]) != set(populations):
        raise ValueError("fixed test subset dataset mismatch")
    selected, result = [], {}
    for dataset in ("TopDial", "DuRecDial2.0"):
        rows = populations[dataset]
        ordered = sorted(rows, key=lambda row: row["id"])
        ids, indices = fixed["selected_ids"][dataset], fixed["selected_indices"][dataset]
        if not ids or ids != sorted(set(ids)) or len(ids) != len(indices):
            raise ValueError("fixed test IDs/order mismatch")
        chosen = []
        for scenario_id, index in zip(ids, indices, strict=True):
            if (
                type(index) is not int
                or not 0 <= index < len(ordered)
                or ordered[index]["id"] != scenario_id
            ):
                raise ValueError("fixed test index/ID mismatch")
            row = ordered[index]
            if digest(row) != fixed["selected_content_sha256"][dataset][scenario_id]:
                raise ValueError(
                    "fixed test scenario content mismatch: " + dataset + "/" + scenario_id
                )
            chosen.append(row)
            selected.append((dataset, scenario_id))
        result[dataset] = chosen
    selected.sort()
    if len(selected) != 500 or digest(selected) != fixed["selected_membership_sha256"]:
        raise ValueError("fixed test subset membership mismatch")
    return (
        selected,
        result,
        {
            "resource": FIXED_500_RESOURCE,
            "sha256": checksum,
            "index_basis": fixed["index_basis"],
            "selected_indices": fixed["selected_indices"],
            "execution": "replay_frozen_seed42_membership_without_resampling",
        },
    )


def validate_sampling(spec):
    if spec is None:
        return
    if spec.get("total_scenarios") not in (500, 2000) or spec != {
        "schema_version": 1,
        "method": "pooled_uniform_without_replacement",
        "total_scenarios": spec.get("total_scenarios"),
        "seed": 42,
    }:
        raise ValueError("unknown test sampling protocol")


def sample_test_scenarios(populations, spec):
    validate_sampling(spec)
    keys = sorted((dataset, row["id"]) for dataset, rows in populations.items() for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate test population IDs")
    frozen = None
    if spec is not None and spec["total_scenarios"] == 500:
        selected, result, frozen = _select_fixed(populations, keys, spec)
    else:
        if spec is None:
            selected = keys
        else:
            if len(keys) < spec["total_scenarios"]:
                raise ValueError("test population smaller than requested sample")
            selected = sorted(random.Random(spec["seed"]).sample(keys, spec["total_scenarios"]))
        selected_set = set(selected)
        result = {
            dataset: sorted(
                (row for row in rows if (dataset, row["id"]) in selected_set),
                key=lambda row: row["id"],
            )
            for dataset, rows in populations.items()
        }
    if any(not rows for rows in result.values()):
        raise ValueError("test sample must represent both datasets")
    manifest = {
        "schema_version": 1,
        "protocol": spec,
        "population_counts": {d: len(rows) for d, rows in populations.items()},
        "population_membership_sha256": digest(keys),
        "selected_ids": {d: [row["id"] for row in rows] for d, rows in result.items()},
        "selected_membership_sha256": digest(selected),
        "claim": "sampled_official_test_subset" if spec else "full_official_test",
    }
    if frozen is not None:
        manifest["frozen_subset"] = frozen
    return result, manifest


def projected_sample_counts(counts, spec):
    """Expected pooled sample composition; official test bodies remain sealed."""
    validate_sampling(spec)
    if spec is None:
        return dict(counts)
    total = sum(counts.values())
    if total < spec["total_scenarios"]:
        raise ValueError("projected test population smaller than requested sample")
    return {d: spec["total_scenarios"] * n / total for d, n in counts.items()}
