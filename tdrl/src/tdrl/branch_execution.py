"""Audited lineage for concurrent Full candidate branches and retained fitting."""

import ast
import json
import os
import subprocess
from pathlib import Path

from .provenance import file_hash


def selection_task_name(dataset):
    return (
        "selection-branches-" if os.environ.get("TDRL_CANDIDATE_WORKERS") == "3" else "selection-"
    ) + dataset


def unchanged_planner_protocol(root, previous):
    path = "tdrl/src/tdrl/planner.py"
    before = ast.parse(subprocess.check_output(["git", "show", previous + ":" + path], cwd=root))
    after = ast.parse((root / "src/tdrl/planner.py").read_text())

    def protocol(tree):
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        cls = next(n for n in tree.body if getattr(n, "name", None) == "ConditionalPlanner")
        cls.body = [n for n in cls.body if getattr(n, "name", None) != "act"]
        return ast.dump(tree, include_attributes=False)

    if protocol(before) != protocol(after):
        raise ValueError("candidate scheduling changed planner prompts/config/seeds/interfaces")


def branch_evidence(root, previous_commit=None):
    record_path = Path(os.environ["TDRL_BRANCH_AMENDMENT"])
    record = json.loads(record_path.read_text())
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (
        record.get("schema_version") != 1
        or record.get("status") != "passed"
        or record.get("commit") != current
        or os.environ.get("TDRL_CANDIDATE_WORKERS") != "3"
        or os.environ.get("TDRL_CRITIC_WORKERS") != "10"
    ):
        raise ValueError("candidate execution evidence/code mismatch")
    inherited_path = Path(record["critic_amendment"]["path"])
    if file_hash(inherited_path) != record["critic_amendment"]["sha256"]:
        raise ValueError("critic execution lineage mismatch")
    inherited = json.loads(inherited_path.read_text())
    if inherited["status"] != "passed" or inherited["commit"] != record["source_commit"]:
        raise ValueError("candidate source has no passed critic amendment")
    if previous_commit and previous_commit not in {
        record["source_commit"],
        inherited["source_commit"],
    }:
        raise ValueError("candidate execution source commit mismatch")
    allowed = {
        "AGENTS.md",
        "tdrl/src/tdrl/planner.py",
        "tdrl/src/tdrl/branch_execution.py",
        "tdrl/src/tdrl/critic_execution.py",
        "tdrl/src/tdrl/two_gpu.py",
        "tdrl/scripts/run_two_gpu.py",
        "tdrl/scripts/smoke_branch_parallel.py",
        "tdrl/scripts/transition_branch_parallel.py",
        "tdrl/tests/test_concurrency.py",
        "tdrl/tests/test_two_gpu.py",
        "tdrl/docs/PARALLEL_BRANCHES.md",
    }
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", record["source_commit"], current], cwd=root, text=True
    ).splitlines()
    if set(changed) - allowed:
        raise ValueError("candidate amendment changed unrelated scientific code/config")
    unchanged_planner_protocol(root, record["source_commit"])
    checks = []
    for item in record["smokes"]:
        if file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError("candidate smoke checksum mismatch")
        checks.append(json.loads(Path(item["path"]).read_text()))
    if len(checks) != 2 or len({c["gpu"]["uuid"] for c in checks}) != 2:
        raise ValueError("candidate execution requires both physical GPUs")
    if any(
        c["status"] != "passed"
        or not c["exact_journal_replay"]
        or c["pending_requests"]
        or c["optimizer_updates"] != 0
        or set(c["heads"]) != {"MRNFixed", "IQE"}
        or not c["critic_exact_replay"]
        for c in checks
    ):
        raise ValueError("candidate smoke failed")
    return {
        "path": str(record_path.resolve()),
        "sha256": file_hash(record_path),
        "candidate_workers": 3,
        "critic_workers": 10,
        "source_commit": record["source_commit"],
    }
