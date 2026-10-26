#!/usr/bin/env python3
"""Fail-fast, non-inference validation for a fresh baseline_crs checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--require-upstreams", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "tdrl"
    required = [
        project / "pyproject.toml",
        project / "runtime" / "uv.lock",
        project / "configs" / "models.lock.json",
        project / "src" / "tdrl" / "resources" / "test_scenarios_1000_td500_dr500_v1.json",
        root / "configs" / "baseline_registry.json",
        root / ".gitmodules",
    ]
    method_sources = {
        "direct_llm_v2": (
            project / "src" / "tdrl" / "baselines" / "direct_llm" / "planner.py",
            project / "scripts" / "direct_llm" / "run_e2e.py",
        ),
        "tepl_v2": (
            project / "src" / "tdrl" / "baselines" / "epl" / "planner.py",
            project / "scripts" / "epl" / "run_e2e.py",
        ),
        "enpl_domain3": (
            project / "src" / "tdrl" / "baselines" / "enpl" / "planner.py",
            project / "scripts" / "enpl" / "run_e2e.py",
        ),
        "procot": (
            project / "src" / "tdrl" / "baselines" / "procot" / "planner.py",
            project / "scripts" / "procot" / "run_e2e.py",
        ),
        "dpdp": (
            project / "src" / "tdrl" / "baselines" / "dpdp" / "planner.py",
            project / "scripts" / "dpdp" / "run_training_pipeline.py",
        ),
        "scope_crs_lite": (
            project / "src" / "tdrl" / "baselines" / "scope" / "planner.py",
            project / "scripts" / "scope" / "run_e2e.py",
        ),
        "epl_native_qwen": (
            project / "src" / "tdrl" / "baselines" / "epl_native" / "planner.py",
            project / "scripts" / "epl_native" / "run_full_e2e.py",
        ),
        "difftod": (
            project / "src" / "tdrl" / "baselines" / "difftod" / "planner.py",
            project / "scripts" / "train_difftod.py",
        ),
        "trip": (
            project / "src" / "tdrl" / "baselines" / "trip" / "planner.py",
            project / "src" / "tdrl" / "baselines" / "trip" / "cli.py",
        ),
    }
    for source_paths in method_sources.values():
        required.extend(source_paths)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required repository files:\n" + "\n".join(missing))

    upstreams = {
        "baselines/dpdp/upstream": "4d1e22e35db452d51f0f39388456d310ec1c971b",
        "baselines/tepl_v2/upstream": "364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda",
        "baselines/epl_native_qwen/upstream": "364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda",
        "baselines/enpl_domain3/upstream": "a76d5a7d6ea1d9c7a07a9574f4901ff79ed440bd",
        "baselines/procot/upstream": "77ff4ed7dfda11afd719db3bd40886a1a51edf4e",
        "baselines/scope_crs_lite/upstream": "b35dbc64808afce25e8f8e5d6838ecf271ec4d0b",
        "baselines/difftod/upstream": "fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1",
        "baselines/trip/upstream": "ee845a5306902ae673e92083711b283b9f33bb9f",
    }
    for path, expected_revision in upstreams.items():
        entry = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "HEAD", "--", path], text=True
        ).strip()
        fields = entry.split(None, 3)
        if len(fields) != 4 or fields[:3] != ["160000", "commit", expected_revision]:
            raise SystemExit(
                f"invalid upstream gitlink for {path}: expected {expected_revision}, got {entry}"
            )
        if args.require_upstreams:
            checkout = root / path
            if not checkout.is_dir():
                raise SystemExit(f"upstream submodule is not initialized: {checkout}")
            actual_revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual_revision != expected_revision:
                raise SystemExit(
                    f"upstream checkout revision mismatch for {path}: "
                    f"{actual_revision} != {expected_revision}"
                )

    registry = json.loads((root / "configs" / "baseline_registry.json").read_text())
    if registry.get("schema_version") != 1 or not registry.get("methods"):
        raise SystemExit("invalid baseline registry")
    missing_registered = sorted(set(registry["methods"]) - set(method_sources))
    if missing_registered:
        raise SystemExit(
            "registered methods have no packaged source contract: "
            + ", ".join(missing_registered)
        )
    resource = project / "src" / "tdrl" / "resources" / "test_scenarios_1000_td500_dr500_v1.json"
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit("checkout is dirty; scientific jobs would fail closed:\n" + dirty)

    report: dict[str, object] = {
        "status": "repository_ok",
        "root": str(root),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "fixed_resource_sha256": digest(resource),
        "methods": sorted(registry["methods"]),
        "packaged_method_sources": sorted(method_sources),
        "pinned_upstream_gitlinks": upstreams,
    }
    if args.require_assets:
        for relative in ("data/raw", "data/processed", "models"):
            path = project / relative
            if not path.exists():
                raise SystemExit(f"required asset path is missing: {path}")
        python = Path(
            os.environ.get(
                "BASELINE_PYTHON", project / "runtime" / ".venv" / "bin" / "python"
            )
        )
        if not python.is_file() or not os.access(python, os.X_OK):
            raise SystemExit(f"runtime Python is missing: {python}")
        report["assets"] = "paths_present_not_scientifically_verified"
        report["python"] = str(python)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
