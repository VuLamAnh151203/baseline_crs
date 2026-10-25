#!/usr/bin/env bash
#SBATCH --job-name=dpdp-mini-e2e
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
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
: "${DPDP_ROBERTA_REVISION:?export the pinned RoBERTa-large commit SHA before submission}"

runtime_python="$(command -v python)"
storage_root="/mnt/data/u26anhvl/crs"
baseline_name="dpdp"
output="${storage_root}/${baseline_name}/dpdp-mini-e2e-${SLURM_JOB_ID}"
# mkdir -p creates crs/, the baseline directory, and this run directory when absent.
mkdir -p -- "$output"
scontrol show job "$SLURM_JOB_ID" > "$output/slurm-job.txt"
printf 'Start at: %s\nRunning on node: %s\nPython: %s\nOutput: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$runtime_python" "$output"

srun --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  "$runtime_python" scripts/dpdp/mini_e2e.py \
  --root . --output "$output" --roberta-revision "$DPDP_ROBERTA_REVISION"

printf 'Completed mini E2E: %s\n' "$output/mini-e2e-summary.json"
