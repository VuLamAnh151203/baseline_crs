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
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "tdrl"
    required = [
        project / "pyproject.toml",
        project / "runtime" / "uv.lock",
        project / "configs" / "models.lock.json",
        project / "src" / "tdrl" / "resources" / "test_scenarios_1000_td500_dr500_v1.json",
        root / "configs" / "baseline_registry.json",
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
