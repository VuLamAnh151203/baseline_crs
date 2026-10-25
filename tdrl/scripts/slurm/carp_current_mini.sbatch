#!/usr/bin/env bash
#SBATCH --job-name=carp2-mini
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from baseline_crs/tdrl}"
export CARP_BASELINE_ROOT="$SLURM_SUBMIT_DIR"
export CARP_CURRENT_COUNT=2
export CARP_CURRENT_WORKERS=2
export CARP_CURRENT_RUN_ROOT="/mnt/data/u26anhvl/crs/carp_eval/carp-current-e5bf703-mini-${SLURM_JOB_ID}"
bash scripts/carp_current/run_job.sh
