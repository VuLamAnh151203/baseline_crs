#!/usr/bin/env bash
#SBATCH --job-name=baseline-crs-cpu-gate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
repo_root="${BASELINE_REPO_ROOT:?missing BASELINE_REPO_ROOT}"
python_bin="${BASELINE_PYTHON:?missing BASELINE_PYTHON}"
cd "$repo_root"
export PYTHONPATH="$repo_root/tdrl/src:$repo_root/tdrl"
export PYTHONDONTWRITEBYTECODE=1

bash -n scripts/bootstrap_environment.sh
bash -n scripts/submit_baseline.sh
bash -n scripts/slurm/baseline_e2e.sbatch
"$python_bin" -m pytest -q \
  tdrl/tests/test_public_packaging.py \
  tdrl/tests/test_direct_llm.py \
  tdrl/tests/test_epl.py \
  tdrl/tests/test_enpl.py \
  tdrl/tests/test_procot.py \
  tdrl/tests/test_dpdp.py \
  tdrl/tests/test_scope.py
