#!/usr/bin/env bash
#SBATCH --job-name=epl-extend1000
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/mnt/data/u26anhvl/crs/epl/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/epl/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
: "${EPL_RUN_ROOT:?set EPL_RUN_ROOT to the completed prefix400 run root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
mkdir -p -- "${EPL_RUN_ROOT}/allocation"
printf '%s\n' "$SLURM_JOB_ID" > "${EPL_RUN_ROOT}/allocation/job-id-1000.txt"
scontrol show job "$SLURM_JOB_ID" > "${EPL_RUN_ROOT}/allocation/slurm-job-1000.txt"
printf 'EPL extension start: %s\nNode: %s\nPython: %s\nRun root: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$EPL_RUN_ROOT"
forward_stop() {
  touch "${EPL_RUN_ROOT}/stop.requested"
  if [[ -n "${pipeline_pid:-}" ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    kill -USR1 "$pipeline_pid"
  fi
}
trap forward_stop USR1 TERM
python scripts/epl/run_e2e.py \
  --root "$SLURM_SUBMIT_DIR" \
  --run-root "$EPL_RUN_ROOT" \
  --mode extend1000 &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
status=$?
set -e
exit "$status"
