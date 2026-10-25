#!/usr/bin/env bash
#SBATCH --job-name=enpl-e2e400
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/mnt/data/u26anhvl/crs/enpl/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/enpl/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
storage_root="/mnt/data/u26anhvl/crs/enpl"
run_root="${ENPL_RUN_ROOT:-${storage_root}/enpl-fixed1000-${SLURM_JOB_ID}}"
mkdir -p -- "${run_root}/allocation"
scontrol show job "$SLURM_JOB_ID" > "${run_root}/allocation/slurm-job-400.txt"
git rev-parse HEAD > "${run_root}/allocation/source-commit-${SLURM_JOB_ID}.txt"
pwd > "${run_root}/allocation/source-checkout-${SLURM_JOB_ID}.txt"
git status --porcelain --untracked-files=all \
  > "${run_root}/allocation/source-status-${SLURM_JOB_ID}.txt"
if [[ -s "${run_root}/allocation/source-status-${SLURM_JOB_ID}.txt" ]]; then
  printf 'EnPL allocation checkout is dirty; refusing scientific execution.\n' >&2
  cat "${run_root}/allocation/source-status-${SLURM_JOB_ID}.txt" >&2
  exit 3
fi
python -m compileall -q src/tdrl/backend.py src/tdrl/baselines/enpl \
  src/tdrl/comparison_enpl.py scripts/enpl
ruff check --select E4,E7,E9,F,B \
  src/tdrl/backend.py src/tdrl/baselines/enpl src/tdrl/comparison_enpl.py \
  scripts/enpl tests/test_enpl.py
python -m pytest -q tests/test_enpl.py
forward_stop() {
  touch "${run_root}/stop.requested"
  if [[ -n "${pipeline_pid:-}" ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    kill -USR1 "$pipeline_pid"
  fi
}
trap forward_stop USR1 TERM
python scripts/enpl/run_e2e.py --root "$SLURM_SUBMIT_DIR" --run-root "$run_root" \
  --convplan-zip "${ENPL_CONVPLAN_ZIP:-/home/u26anhvl/crs/ConvPlan/ConvPlan.zip}" \
  --mode initial400 &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
status=$?
set -e
exit "$status"
