#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/ruff check src tests scripts instrumentation --config pyproject.toml
.venv/bin/ruff format --check src tests scripts instrumentation --config pyproject.toml
.venv/bin/python -m pytest -q
.venv/bin/tdrl check
.venv/bin/tdrl runtime-audit


for script in scripts/*.sh scripts/slurm/*.sbatch; do
  bash -n "$script"
done
