#!/usr/bin/env bash
#SBATCH --job-name=enpl-bank-audit
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/u26anhvl/crs/enpl/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/enpl/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export OMP_NUM_THREADS=8
run_root="/mnt/data/u26anhvl/crs/enpl/enpl-bank-audit-${SLURM_JOB_ID}"
mkdir -p -- "$run_root"
python scripts/enpl/prepare_bank.py \
  --root "$SLURM_SUBMIT_DIR" \
  --convplan-zip "${ENPL_CONVPLAN_ZIP:-/home/u26anhvl/crs/ConvPlan/ConvPlan.zip}" \
  --output "$run_root"
printf 'EnPL full bank audit completed: %s\n' "$(date --iso-8601=seconds)"
