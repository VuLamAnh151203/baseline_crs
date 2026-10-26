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


def tree_digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


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
    method_mirrors = {
        "direct_llm_v2": ("direct_llm", "direct_llm"),
        "tepl_v2": ("epl", "epl"),
        "enpl_domain3": ("enpl", "enpl"),
        "procot": ("procot", "procot"),
        "dpdp": ("dpdp", "dpdp"),
        "scope_crs_lite": ("scope", "scope"),
        "epl_native_qwen": ("epl_native", "epl_native"),
        "difftod": ("difftod", "difftod"),
        "trip": ("trip", "trip"),
    }
    required_directories: list[Path] = []
    for public_name, (source_name, runner_name) in method_mirrors.items():
        required_directories.extend(
            (
                root / "baselines" / public_name / "source",
                root / "baselines" / public_name / "runner",
                project / "src" / "tdrl" / "baselines" / source_name,
                project / "scripts" / runner_name,
            )
        )
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(str(path) for path in required_directories if not path.is_dir())
    if missing:
        raise SystemExit("missing required repository files:\n" + "\n".join(missing))

    for public_name, (source_name, runner_name) in method_mirrors.items():
        comparisons = (
            (
                project / "src" / "tdrl" / "baselines" / source_name,
                root / "baselines" / public_name / "source",
            ),
            (
                project / "scripts" / runner_name,
                root / "baselines" / public_name / "runner",
            ),
        )
        for canonical, mirror in comparisons:
            if tree_digests(canonical) != tree_digests(mirror):
                raise SystemExit(
                    f"baseline mirror differs from canonical source: {mirror} != {canonical}"
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
        "verified_source_mirrors": sorted(method_mirrors),
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
