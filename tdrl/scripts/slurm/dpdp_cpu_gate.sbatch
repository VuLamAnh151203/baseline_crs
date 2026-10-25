#!/usr/bin/env bash
#SBATCH --job-name=dpdp-cpu-gate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/u26anhvl/crs/dpdp/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/dpdp/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export OMP_NUM_THREADS=2
run_root="/mnt/data/u26anhvl/crs/dpdp/cpu-gate-${SLURM_JOB_ID}"
mkdir -p -- "$run_root"
printf 'DPDP CPU gate start: %s\nNode: %s\nPython: %s\nRun root: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$run_root"
bash -n \
  scripts/dpdp/prepare_clean_checkout.sh \
  scripts/submit_dpdp_cpu_gate.sh \
  scripts/submit_dpdp_deadline.sh \
  scripts/slurm/dpdp_cpu_gate.sbatch \
  scripts/slurm/dpdp_stage.sbatch
python -m compileall -q src/tdrl/baselines/dpdp scripts/dpdp
ruff check src/tdrl/baselines/dpdp scripts/dpdp tests/test_dpdp.py tests/test_dpdp_mini_e2e.py
python -m pytest -q \
  tests/test_dpdp.py \
  tests/test_dpdp_mini_e2e.py \
  tests/test_baseline_test_setting.py \
  tests/test_test_prefix.py
printf 'DPDP CPU gate completed: %s\n' "$(date --iso-8601=seconds)"
touch "$run_root/complete"
