#!/usr/bin/env bash
#SBATCH --job-name=enpl-cpu-gate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/u26anhvl/crs/enpl/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/enpl/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export OMP_NUM_THREADS=2
run_root="/mnt/data/u26anhvl/crs/enpl/enpl-cpu-gate-${SLURM_JOB_ID}"
mkdir -p -- "$run_root"
python -m compileall -q src/tdrl/baselines/enpl src/tdrl/comparison_enpl.py scripts/enpl
ruff check src/tdrl/baselines/enpl src/tdrl/comparison_enpl.py scripts/enpl tests/test_enpl.py
bash -n scripts/slurm/enpl_cpu_gate.sbatch scripts/slurm/enpl_bank_audit.sbatch \
  scripts/slurm/enpl_mini_e2e.sbatch \
  scripts/slurm/enpl_e2e400.sbatch scripts/submit_enpl_cpu_gate.sh \
  scripts/submit_enpl_bank_audit.sh scripts/submit_enpl_mini.sh \
  scripts/submit_enpl400.sh scripts/submit_enpl400_resume_60429.sh \
  scripts/enpl/prepare_clean_checkout.sh
python scripts/enpl/run_e2e.py --help > /dev/null
python -m tdrl.baselines.enpl.evaluate_cli --help > /dev/null
python -m pytest -q tests/test_enpl.py tests/test_baseline_test_setting.py tests/test_test_prefix.py
printf 'EnPL CPU gate completed: %s\n' "$(date --iso-8601=seconds)"
touch "$run_root/complete"
