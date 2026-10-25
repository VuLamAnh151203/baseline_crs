from copy import deepcopy

import pytest

from tdrl.confirmation import confirmation_cases, verify_planner_gates
from tdrl.provenance import COMPUTE_FIELDS, PROVENANCE_FIELDS, atomic_json, digest, file_hash


def cohort():
    rows = [{"id": f"scenario-{i}", "profile": {"preference": i}} for i in range(160)]
    selected = sorted(rows, key=lambda r: digest([17, r["id"]]))[80:144]
    membership = {
        "schema_version": 1,
        "dataset": "TopDial",
        "split": "dev",
        "root_seed": 17,
        "protocols": ["profile", "resistant"],
        "population_sha256": "a" * 64,
        "population_count": len(rows),
        "scenario_ids": [r["id"] for r in selected],
        "scenario_content_sha256": {r["id"]: digest(r) for r in selected},
    }
    return rows, membership, selected


def test_confirmation_replays_exact_order_independent_of_population_order():
    rows, membership, selected = cohort()
    assert confirmation_cases(list(reversed(rows)), membership, "a" * 64) == selected
    discovery = sorted(rows, key=lambda r: digest([17, r["id"]]))[:80]
    assert not set(membership["scenario_ids"]) & {r["id"] for r in discovery}


def test_confirmation_rejects_changed_population_membership_and_content():
    rows, membership, selected = cohort()
    with pytest.raises(ValueError, match="population"):
        confirmation_cases(rows, membership, "b" * 64)
    changed = deepcopy(membership)
    changed["scenario_ids"].reverse()
    with pytest.raises(ValueError, match="membership/order"):
        confirmation_cases(rows, changed, "a" * 64)
    changed_rows = deepcopy(rows)
    next(r for r in changed_rows if r["id"] == selected[0]["id"])["profile"] = {}
    with pytest.raises(ValueError, match="content"):
        confirmation_cases(changed_rows, membership, "a" * 64)
    duplicated = [*rows[:-1], rows[0]]
    with pytest.raises(ValueError, match="duplicate"):
        confirmation_cases(duplicated, membership, "a" * 64)


def gates(root):
    expected = {"code": {"commit": "code"}, "config_sha256": "config", "slot": 1}
    paths = []
    for i in [0, 1]:
        folder = root / str(i)
        folder.mkdir()
        atomic_json(folder / "contract.json", {**expected, "slot": i})
        atomic_json(
            folder / "checks.json",
            {
                "status": "passed",
                "exact_journal_replay": True,
                "optimizer_updates": 0,
                "gpu": {"uuid": f"fake-unit-gpu-{i}"},
            },
        )
        atomic_json(
            folder / "manifest.json",
            {
                "schema_version": 1,
                "run_id": "cpu-unit-fixture",
                "stage": "TopDial-planner-smoke",
                "status": "complete",
                "started_at": "unit",
                "ended_at": "unit",
                "provenance": {k: {"status": "unit-fixture"} for k in PROVENANCE_FIELDS},
                "compute": {k: 0 for k in COMPUTE_FIELDS},
                "outputs": {
                    name: file_hash(folder / name) for name in ["checks.json", "contract.json"]
                },
            },
        )
        paths.append(folder / "checks.json")
    return paths, expected


def test_planner_gate_binds_config_and_checksum(tmp_path):
    paths, expected = gates(tmp_path)
    assert len(verify_planner_gates(paths, expected)) == 2
    with pytest.raises(ValueError, match="source/config"):
        verify_planner_gates(paths, {**expected, "config_sha256": "changed"})
    check = paths[0]
    check.write_text(check.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        verify_planner_gates(paths, expected)
