#!/usr/bin/env bash
#SBATCH --job-name=epl-cpu-mini
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/u26anhvl/crs/epl/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/epl/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export OMP_NUM_THREADS=2
run_root="/mnt/data/u26anhvl/crs/epl/epl-cpu-mini-${SLURM_JOB_ID}"
mkdir -p -- "$run_root"
printf 'EPL CPU mini start: %s\nNode: %s\nPython: %s\nRun root: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$run_root"
python -m compileall -q \
  src/tdrl/baselines/epl \
  src/tdrl/comparison_epl.py \
  scripts/epl
ruff check \
  src/tdrl/baselines/epl \
  src/tdrl/comparison_epl.py \
  scripts/epl \
  tests/test_epl.py
python scripts/epl/run_e2e.py --help > /dev/null
python -m pytest -q \
  tests/test_epl.py \
  tests/test_baseline_test_setting.py \
  tests/test_test_prefix.py
printf 'EPL CPU mini completed: %s\n' "$(date --iso-8601=seconds)"
touch "$run_root/complete"
