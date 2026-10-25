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
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required repository files:\n" + "\n".join(missing))

    registry = json.loads((root / "configs" / "baseline_registry.json").read_text())
    if registry.get("schema_version") != 1 or not registry.get("methods"):
        raise SystemExit("invalid baseline registry")
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
