#!/usr/bin/env bash
#SBATCH --job-name=dpdp-2gpu-mini
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
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
export PYTHONPATH="src:."
export OMP_NUM_THREADS=4

runtime_python="$(command -v python)"
output="/mnt/data/u26anhvl/crs/dpdp/two-gpu-mini-${SLURM_JOB_ID}"
mkdir -p -- "$output"
scontrol show job "$SLURM_JOB_ID" > "$output/slurm-job.txt"
printf 'Two-GPU DPDP mini start: %s\nNode: %s\nPython: %s\nOutput: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$runtime_python" "$output"

"$runtime_python" scripts/dpdp/run_two_gpu_mini.py \
  --root "$SLURM_SUBMIT_DIR" \
  --output "$output" \
  --python "$runtime_python"

printf 'Completed two-GPU mini: %s\n' "$output/two-gpu-mini-summary.json"
