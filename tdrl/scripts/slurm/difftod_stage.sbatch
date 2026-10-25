#!/usr/bin/env bash
#SBATCH --job-name=difftod-stage
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=20-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=8
test "$#" -gt 0
scontrol show job "$SLURM_JOB_ID"
exec srun --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:1 "$@"
