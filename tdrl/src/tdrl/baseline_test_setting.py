"""Shared, immutable fixed-test contract for every baseline evaluator."""

import json
from pathlib import Path

from .official_data import SPLITS, read_release
from .provenance import digest
from .test_prefix import RESOURCE_SHA256, prefix_cases

METHODS = ["PPDPP", "DiffTOD", "TRIP", "DPDP", "TDRL-R", "TDRL"]
DATASETS = ["TopDial", "DuRecDial2.0"]
PROTOCOLS = ["profile", "resistant"]
EXECUTION_ORDER = (
    "first_200_TopDial_then_200_DuRecDial_then_300_TopDial_then_300_DuRecDial"
)


def _expected():
    return {
        "schema_version": 1,
        "scope": "all_baselines_fixed1000_official_test",
        "methods": METHODS,
        "datasets": DATASETS,
        "train_dev": "full_official_train_and_development_splits",
        "test": {
            "resource": "tdrl.resources.test_scenarios_1000_td500_dr500_v1.json",
            "resource_sha256": RESOURCE_SHA256,
            "scenario_count": 1000,
            "per_dataset": {"TopDial": 500, "DuRecDial2.0": 500},
            "execution_order": EXECUTION_ORDER,
        },
        "evaluation": {
            "seed": 17,
            "protocols": PROTOCOLS,
            "turn_budget": 10,
            "budget": "natural",
        },
        "claim": "single_seed_sampled_official_test_evaluation",
    }


def load_fixed_test_setting(root, path, count=None):
    """Load and strictly validate the common setting; None preserves legacy behavior."""
    if path is None:
        if count is not None:
            raise ValueError("a fixed-test count requires a fixed-test setting")
        return None
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected():
        raise ValueError("unknown or modified shared fixed-test setting")
    if count is None:
        count = payload["test"]["scenario_count"]
    if count not in (400, 1000):
        raise ValueError("fixed-test count must be 400 or 1000")
    resolved_root = Path(root).resolve()
    resolved_path = path.resolve()
    try:
        display_path = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        display_path = str(resolved_path)
    return {
        "path": display_path,
        "sha256": digest(payload),
        "payload": payload,
        "count": count,
    }


def select_fixed_test_populations(populations, setting):
    """Apply the fixed resource to already-opened official test populations."""
    if setting is None:
        raise ValueError("fixed-test setting is required")
    chosen, sample, membership = prefix_cases(populations, setting["count"])
    selected = {dataset: [] for dataset in DATASETS}
    for dataset, scenario in chosen:
        selected[dataset].append(scenario)
    counts = {dataset: len(rows) for dataset, rows in selected.items()}
    expected_per_dataset = {
        "TopDial": setting["count"] // 2,
        "DuRecDial2.0": setting["count"] // 2,
    }
    if counts != expected_per_dataset:
        raise ValueError("fixed-test sample does not have the expected per-dataset counts")
    manifest = {
        "schema_version": 1,
        "setting_path": setting["path"],
        "setting_sha256": setting["sha256"],
        "requested_prefix_count": setting["count"],
        "scenario_count": len(chosen),
        "selected_counts": counts,
        "sample": sample,
        "membership": membership,
        "claim": setting["payload"]["claim"],
    }
    return selected, manifest


def fixed_test_scenarios(root, setting):
    """Open the sealed official test splits only after caller verifies selection."""
    if setting is None:
        raise ValueError("fixed-test setting is required")
    populations = {
        dataset: [
            scenario
            for split in SPLITS[dataset]
            if split.startswith("test")
            for scenario in read_release(root, dataset, split, selection_verified=True)
        ]
        for dataset in DATASETS
    }
    return select_fixed_test_populations(populations, setting)


def evaluation_axes(setting, config, split):
    """Use the common seed/budget for fixed official test; retain dev selection axes."""
    if split != "test" or setting is None:
        return config["seeds"], config.get("budgets", ["natural", "matched"])
    evaluation = setting["payload"]["evaluation"]
    if (
        config["protocols"] != evaluation["protocols"]
        or config["turn_budget"] != evaluation["turn_budget"]
    ):
        raise ValueError("baseline config conflicts with the shared test protocol")
    return [evaluation["seed"]], [evaluation["budget"]]


def evaluation_output_directory(output, split, setting):
    """Keep fixed-test artifacts separate while allowing prefix extension."""
    folder = "test-fixed1000" if split == "test" and setting is not None else split
    return Path(output) / folder
