import importlib.util
import sys
from pathlib import Path

import pytest

from tdrl.two_gpu import merged_selection_inputs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_two_gpu.py"
spec = importlib.util.spec_from_file_location("two_gpu_queue", SCRIPT)
queue = importlib.util.module_from_spec(spec)

sys.path.insert(0, str(SCRIPT.parent))
spec.loader.exec_module(queue)


def test_queue_requires_both_bootstraps_and_lock_before_test():
    plan = queue.task_plan()
    assert {t[0] for t in queue.eligible_tasks(plan, set(), set())} == {
        "bootstrap-0",
        "bootstrap-1",
    }
    assert not queue.eligible_tasks(plan, {"bootstrap-0"}, {"bootstrap-1"})
    ready = {t[0] for t in queue.eligible_tasks(plan, {"bootstrap-0", "bootstrap-1"}, set())}
    assert {"policy-TopDial", "policy-DuRecDial2.0", "offline-DuRecDial2.0"} <= ready
    assert not {"lock", "test-policy", "test-full"} & ready
    assert {t[0] for t in queue.eligible_tasks(plan, {"lock"}, {"bootstrap-0", "bootstrap-1"})} == {
        "test-policy",
        "test-full",
    }


def shard(dataset):
    return {
        "schema_version": 1,
        "candidates": {
            h: {
                "development_sr": {dataset: 0.5},
                "parameters": 10,
                "invariants_passed": True,
                "stability_passed": True,
                "provenance_passed": True,
            }
            for h in ["MRNFixed", "IQE"]
        },
        "rows": {h: {dataset: [{"accepted": True}]} for h in ["MRNFixed", "IQE"]},
        "thresholds": {h: {dataset: {"H_edge": 1, "H_cover": 2}} for h in ["MRNFixed", "IQE"]},
        "scenario_ids": {dataset: ["fixed"]},
    }


def test_selection_shards_merge_before_global_head_decision():
    candidates, rows, thresholds, ids = merged_selection_inputs(
        [shard("TopDial"), shard("DuRecDial2.0")]
    )
    assert set(ids) == {"TopDial", "DuRecDial2.0"}
    assert candidates["IQE"]["development_sr"] == {"TopDial": 0.5, "DuRecDial2.0": 0.5}
    assert set(rows["MRNFixed"]) == set(thresholds["IQE"]) == set(ids)
    with pytest.raises(ValueError, match="both selection"):
        merged_selection_inputs([shard("TopDial")])
    with pytest.raises(ValueError, match="duplicate"):
        merged_selection_inputs([shard("TopDial"), shard("TopDial")])
    bad = shard("DuRecDial2.0")
    bad["candidates"]["IQE"]["parameters"] = 20
    with pytest.raises(ValueError, match="architecture"):
        merged_selection_inputs([shard("TopDial"), bad])


def test_parallel_branches_use_fresh_paired_selection_namespaces(monkeypatch):
    from tdrl.branch_execution import selection_task_name

    monkeypatch.setenv("TDRL_CANDIDATE_WORKERS", "3")
    plan = queue.task_plan()
    names = {t[0] for t in plan}
    assert {"selection-branches-TopDial", "selection-branches-DuRecDial2.0"} <= names
    assert not {"selection-TopDial", "selection-DuRecDial2.0"} & names
    lock = next(t for t in plan if t[0] == "lock")
    assert {selection_task_name(d) for d in ["TopDial", "DuRecDial2.0"]} <= lock[4]
    assert {"policy-TopDial", "policy-DuRecDial2.0"} <= names


def test_restart_assembles_fitting_that_finished_without_controller(tmp_path):
    for prefix in ["PPDPP-SFT", "PPDPP-RL"]:
        target = tmp_path / "tasks/policy-DuRecDial2.0" / prefix / "DuRecDial2.0"
        target.mkdir(parents=True)
        (target / "checkpoint.pt").write_text("completed while controller paused")
    queue.assemble_completed_fit(tmp_path, "policy-DuRecDial2.0")
    queue.assemble_completed_fit(tmp_path, "policy-DuRecDial2.0")
    assert (
        tmp_path / "PPDPP-RL/DuRecDial2.0/checkpoint.pt"
    ).read_text() == "completed while controller paused"
