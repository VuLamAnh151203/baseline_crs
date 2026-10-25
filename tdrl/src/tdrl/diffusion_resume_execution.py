"""Checksum-bound H100 evidence for an execution-only durable restore repair."""

import ast
import json
import os
import subprocess
from pathlib import Path

from .provenance import file_hash

ALLOWED = {
    "tdrl/scripts/train_difftod.py",
    "tdrl/scripts/audit_difftod_resume.py",
    "tdrl/src/tdrl/diffusion_resume.py",
    "tdrl/src/tdrl/diffusion_resume_execution.py",
    "tdrl/src/tdrl/critic_execution.py",
    "tdrl/tests/test_diffusion_resume.py",
    "tdrl/docs/difftod-resume-repair.md",
}


def execution_only(root, previous):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", previous, "HEAD"], cwd=root, text=True
    ).splitlines()
    if set(changed) - ALLOWED:
        raise ValueError("DiffTOD restore amendment changed unrelated source")
    path = "tdrl/scripts/train_difftod.py"
    before = ast.parse(subprocess.check_output(["git", "show", previous + ":" + path], cwd=root))
    after = ast.parse((root / "scripts/train_difftod.py").read_text())

    class Normalize(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "restore_optimizer":
                if len(node.args) != 2 or node.keywords or ast.unparse(node.args[0]) != "optimizer":
                    raise ValueError("Unexpected optimizer restore call")
                return ast.Call(
                    func=ast.Attribute(value=node.args[0], attr="load_state_dict", ctx=ast.Load()),
                    args=[node.args[1]],
                    keywords=[],
                )
            return node

    def normalize(tree):
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        return ast.dump(Normalize().visit(tree), include_attributes=False)

    if normalize(before) != normalize(after):
        raise ValueError("DiffTOD fitting math/control flow changed beyond restore isolation")
    path = "tdrl/src/tdrl/critic_execution.py"
    old_dispatch = ast.parse(
        subprocess.check_output(["git", "show", previous + ":" + path], cwd=root)
    )
    new_dispatch = ast.parse((root / "src/tdrl/critic_execution.py").read_text())
    function = next(
        n for n in new_dispatch.body if getattr(n, "name", None) == "execution_evidence"
    )
    expected_dispatch = ast.parse(
        "if os.environ.get('TDRL_DIFFTOD_RESUME_AMENDMENT'):\n"
        " from .diffusion_resume_execution import resume_evidence\n"
        " return resume_evidence(root, previous_commit)\n"
    ).body[0]
    if ast.dump(function.body[0], include_attributes=False) != ast.dump(
        expected_dispatch, include_attributes=False
    ):
        raise ValueError("Unexpected restore evidence dispatch")
    function.body.pop(0)
    if ast.dump(old_dispatch, include_attributes=False) != ast.dump(
        new_dispatch, include_attributes=False
    ):
        raise ValueError("Restore amendment changed another execution protocol")
    helper = ast.parse((root / "src/tdrl/diffusion_resume.py").read_text())
    helper.body = [
        n
        for n in helper.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    expected = ast.parse(
        "import copy\ndef restore_optimizer(optimizer, state):\n optimizer.load_state_dict(copy.deepcopy(state))\n"
    )
    if ast.dump(helper, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("Restore helper must only isolate checkpoint ownership")


def resume_evidence(root, previous_commit=None):
    path = Path(os.environ["TDRL_DIFFTOD_RESUME_AMENDMENT"])
    record = json.loads(path.read_text())
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (
        record.get("schema_version") != 1
        or record.get("status") != "passed"
        or record["commit"] != current
    ):
        raise ValueError("Unknown/unpassed DiffTOD restore amendment")
    if previous_commit and record["source_commit"] != previous_commit:
        raise ValueError("DiffTOD restore source lineage mismatch")
    execution_only(root, record["source_commit"])
    checks = []
    for ref in record["smokes"]:
        p = Path(ref["path"])
        if file_hash(p) != ref["sha256"]:
            raise ValueError("DiffTOD restore smoke checksum mismatch")
        checks.append(json.loads(p.read_text()))
    if len(checks) != 2 or len({c["gpu"]["uuid"] for c in checks}) != 2:
        raise ValueError("Restore audit requires both physical H100s")
    for check in checks:
        if (
            check["status"] != "passed"
            or check["commit"] != current
            or check["checkpoint_sha256"] != record["checkpoint_sha256"]
            or check["before_updates"] != 82360
            or check["committed_updates"] != 0
            or not check["checkpoint_unchanged"]
            or not check["source_state_immutable"]
            or not check["exact_next_update_replay"]
            or not check["unchanged_forward"]
            or not check["original_counter_alias_reproduced"]
        ):
            raise ValueError("DiffTOD restore audit incomplete or failed")
    if file_hash(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
        raise ValueError("Immutable migration source checkpoint changed")
    return {
        "kind": "DiffTOD immutable optimizer restore",
        "path": str(path),
        "sha256": file_hash(path),
        "source_commit": record["source_commit"],
    }
