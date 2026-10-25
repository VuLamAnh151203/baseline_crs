#!/usr/bin/env bash
#SBATCH --job-name=carp2-full
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from baseline_crs/tdrl}"
export CARP_BASELINE_ROOT="$SLURM_SUBMIT_DIR"
export CARP_CURRENT_COUNT=200
export CARP_CURRENT_WORKERS=16
export CARP_CURRENT_RUN_ROOT="/mnt/data/u26anhvl/crs/carp_eval/carp-current-e5bf703-full-v1"
bash scripts/carp_current/run_job.sh
