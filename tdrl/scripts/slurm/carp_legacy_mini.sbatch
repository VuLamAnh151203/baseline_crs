#!/usr/bin/env bash
#SBATCH --job-name=carp-mini
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
method="${CARP_METHOD:?set CARP_METHOD}"
count="${CARP_COUNT:-1}"
storage_root="/mnt/data/u26anhvl/crs/carp_eval"
run_root="${CARP_RUN_ROOT:-${storage_root}/${method}-mini-${SLURM_JOB_ID}}"
mkdir -p -- "$run_root/allocation" "$run_root/logs"
git rev-parse HEAD > "$run_root/allocation/source-commit.txt"
git status --porcelain --untracked-files=all > "$run_root/allocation/source-status.txt"
if [[ -s "$run_root/allocation/source-status.txt" ]]; then
  printf 'CARP legacy evaluation requires a clean committed checkout.\n' >&2
  cat "$run_root/allocation/source-status.txt" >&2
  exit 3
fi
scontrol show job "$SLURM_JOB_ID" > "$run_root/allocation/slurm-job.txt"
nvidia-smi -L > "$run_root/allocation/gpus.txt"
python -m compileall -q src/tdrl/carp_legacy.py scripts/carp_legacy/run.py
ruff check --select E4,E7,E9,F,B src/tdrl/backend.py src/tdrl/carp_legacy.py \
  scripts/carp_legacy/run.py tests/test_carp_legacy.py
python -m pytest -q tests/test_carp_legacy.py
python scripts/carp_legacy/run.py controller \
  --root "$SLURM_SUBMIT_DIR" \
  --run-root "$run_root" \
  --method "$method" \
  --count "$count"
