#!/usr/bin/env bash
#SBATCH --job-name=dpdp-smoke
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=8
runtime_python="$(command -v python)"
test -x "$runtime_python"
: "${DPDP_ROBERTA_REVISION:?export the pinned RoBERTa-large commit SHA before submission}"
storage_root="/mnt/data/u26anhvl/crs/dpdp"
output="${storage_root}/smoke/dpdp-smoke-${SLURM_JOB_ID}"
model="models/roberta-large/${DPDP_ROBERTA_REVISION}"
mkdir -p -- "$output"
scontrol show job "$SLURM_JOB_ID" > "$output/slurm-job.txt"
printf 'Start at: %s\nRunning on node: %s\nPython: %s\n' "$(date --iso-8601=seconds)" "$(hostname)" "$runtime_python"
srun --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  "$runtime_python" -m tdrl.baselines.dpdp.cli smoke-model \
  --config configs/dpdp/topdial.json --encoder "$model" --tokenizer "$model" \
  --model-lock configs/dpdp.models.lock.json --output "$output"
srun --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  "$runtime_python" -m tdrl.baselines.dpdp.cli smoke-shared \
  --root . --runtime-config configs/runtime.json \
  --comparison-config configs/comparison.dpdp.json \
  --config configs/dpdp/topdial.json --encoder "$model" --tokenizer "$model" \
  --model-lock configs/dpdp.models.lock.json --output "$output"
