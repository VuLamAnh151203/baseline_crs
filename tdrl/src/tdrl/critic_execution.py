"""Explicit, smoke-backed migration of critic scheduling; scientific inputs stay frozen."""

import ast
import json
import os
import subprocess
from pathlib import Path

from .provenance import file_hash


def unchanged_judge_protocol(root, previous_commit):
    path = "tdrl/src/tdrl/judge.py"
    before = ast.parse(
        subprocess.check_output(["git", "show", previous_commit + ":" + path], cwd=root)
    )
    after = ast.parse((root / "src/tdrl/judge.py").read_text())

    # Only feedback's scheduling and its imports may change; all prompts, validation,
    # protocol hashing, outcome judgment and message construction must be identical.
    def protocol(tree):
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        cls = next(n for n in tree.body if getattr(n, "name", None) == "LocalJudge")
        cls.body = [n for n in cls.body if getattr(n, "name", None) != "feedback"]
        return ast.dump(tree, include_attributes=False)

    if protocol(before) != protocol(after):
        raise ValueError("critic execution changed the scientific judge protocol")


def execution_evidence(root, previous_commit=None):
    if os.environ.get("TDRL_DIFFTOD_RESUME_AMENDMENT"):
        from .diffusion_resume_execution import resume_evidence

        return resume_evidence(root, previous_commit)
    if os.environ.get("TDRL_BRANCH_AMENDMENT"):
        from .branch_execution import branch_evidence

        return branch_evidence(root, previous_commit)
    path = os.environ.get("TDRL_CRITIC_AMENDMENT")
    if not path or os.environ.get("TDRL_CRITIC_WORKERS") != "10":
        raise ValueError("critic scheduling migration requires an explicit H100 smoke amendment")
    record = json.loads(Path(path).read_text())
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (
        record.get("schema_version") != 1
        or record.get("status") != "passed"
        or record.get("commit") != current
    ):
        raise ValueError("critic execution evidence/code mismatch")
    if previous_commit and record["source_commit"] != previous_commit:
        raise ValueError("critic execution source commit mismatch")
    allowed = {
        "AGENTS.md",
        "tdrl/src/tdrl/judge.py",
        "tdrl/src/tdrl/critic_execution.py",
        "tdrl/src/tdrl/stages.py",
        "tdrl/src/tdrl/parallel_comparison.py",
        "tdrl/scripts/smoke_critic_parallel.py",
        "tdrl/tests/test_ppdpp_fidelity.py",
        "tdrl/docs/PARALLEL_CRITIC.md",
    }
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", record["source_commit"], current], cwd=root, text=True
    ).splitlines()
    if set(changed) - allowed:
        raise ValueError("critic amendment changed unrelated scientific code/config")
    unchanged_judge_protocol(root, record["source_commit"])
    checks = []
    for item in record["smokes"]:
        if file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError("critic smoke checksum mismatch")
        checks.append(json.loads(Path(item["path"]).read_text()))
    if len(checks) != 2 or len({c["gpu"]["uuid"] for c in checks}) != 2:
        raise ValueError("critic execution needs independent smoke on both GPUs")
    if any(
        c["status"] != "passed"
        or c["requests"] != 10
        or not c["exact_journal_replay"]
        or c["pending_requests"]
        or c["optimizer_updates"] != 0
        for c in checks
    ):
        raise ValueError("critic execution smoke failed")
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_hash(Path(path)),
        "workers": 10,
        "source_commit": record["source_commit"],
    }
