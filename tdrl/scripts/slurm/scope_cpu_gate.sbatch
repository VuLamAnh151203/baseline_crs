#!/usr/bin/env bash
#SBATCH --job-name=scope-cpu-gate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/u26anhvl/crs/scope/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/scope/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export OMP_NUM_THREADS=2
run_root="/mnt/data/u26anhvl/crs/scope/scope-cpu-gate-${SLURM_JOB_ID}"
mkdir -p -- "$run_root"
python -m compileall -q \
  src/tdrl/baselines/scope \
  src/tdrl/comparison_scope.py \
  scripts/scope
ruff check \
  src/tdrl/baselines/scope \
  src/tdrl/comparison_scope.py \
  scripts/scope \
  tests/test_scope.py
bash -n \
  scripts/slurm/scope_llama_manifest.sbatch \
  scripts/slurm/scope_transition_import.sbatch \
  scripts/slurm/scope_cpu_gate.sbatch \
  scripts/slurm/scope_mini_e2e.sbatch \
  scripts/slurm/scope_e2e400.sbatch \
  scripts/download_scope_transition.sh \
  scripts/submit_scope_llama_manifest.sh \
  scripts/submit_scope_transition_import.sh \
  scripts/submit_scope_cpu_gate.sh \
  scripts/submit_scope_mini.sh \
  scripts/submit_scope400.sh
python scripts/scope/run_e2e.py --help > /dev/null
python -m tdrl.baselines.scope.evaluate_cli --help > /dev/null
python -m pytest -q \
  tests/test_scope.py \
  tests/test_baseline_test_setting.py \
  tests/test_test_prefix.py
printf 'SCOPE CPU gate completed: %s\n' "$(date --iso-8601=seconds)"
touch "$run_root/complete"
