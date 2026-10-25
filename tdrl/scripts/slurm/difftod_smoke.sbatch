#!/usr/bin/env bash
#SBATCH --job-name=difftod-smoke
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
runtime_python="${SLURM_SUBMIT_DIR}/runtime/.venv/bin/python"
test -x "$runtime_python"
output="artifacts/difftod-smoke-${SLURM_JOB_ID}"
mkdir -p "$output"
scontrol show job "$SLURM_JOB_ID" > "$output/slurm-job.txt"
srun --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  "$runtime_python" -m tdrl.baselines.difftod.cli smoke \
  --require-h100 --output "$output"
