#!/usr/bin/env bash
#SBATCH --job-name=procot-cpu-gate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/u26anhvl/crs/procot/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/procot/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export OMP_NUM_THREADS=2
run_root="/mnt/data/u26anhvl/crs/procot/procot-cpu-gate-${SLURM_JOB_ID}"
mkdir -p -- "$run_root"
printf 'ProCoT CPU gate start: %s\nNode: %s\nPython: %s\nRun root: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$run_root"
python -m compileall -q \
  src/tdrl/baselines/procot \
  src/tdrl/comparison_procot.py \
  scripts/procot
ruff check \
  src/tdrl/baselines/procot \
  src/tdrl/comparison_procot.py \
  scripts/procot \
  tests/test_procot.py
bash -n \
  scripts/slurm/procot_cpu_gate.sbatch \
  scripts/slurm/procot_mini_e2e.sbatch \
  scripts/slurm/procot_e2e400.sbatch \
  scripts/submit_procot_cpu_gate.sh \
  scripts/submit_procot_mini.sh \
  scripts/submit_procot400.sh
python scripts/procot/run_e2e.py --help > /dev/null
python -m tdrl.baselines.procot.evaluate_cli --help > /dev/null
python -m pytest -q \
  tests/test_procot.py \
  tests/test_baseline_test_setting.py \
  tests/test_test_prefix.py
printf 'ProCoT CPU gate completed: %s\n' "$(date --iso-8601=seconds)"
touch "$run_root/complete"
