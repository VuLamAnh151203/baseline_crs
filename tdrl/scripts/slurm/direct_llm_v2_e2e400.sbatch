#!/usr/bin/env bash
#SBATCH --job-name=direct-llm-v2-e2e400
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/mnt/data/u26anhvl/crs/direct_llm/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/direct_llm/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
storage_root="/mnt/data/u26anhvl/crs/direct_llm"
run_root="${DIRECT_LLM_V2_RUN_ROOT:-${storage_root}/direct-llm-v2-fixed1000-${SLURM_JOB_ID}}"
mkdir -p -- "${run_root}/allocation"
printf '%s\n' "$SLURM_JOB_ID" > "${run_root}/allocation/job-id-400.txt"
printf '%s\n' "$run_root" > "${run_root}/allocation/run-root.txt"
scontrol show job "$SLURM_JOB_ID" > "${run_root}/allocation/slurm-job-400.txt"
printf 'DirectLLM-v2 fixed-400 start: %s\nNode: %s\nPython: %s\nRun root: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$run_root"
forward_stop() {
  touch "${run_root}/stop.requested"
  if [[ -n "${pipeline_pid:-}" ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    kill -USR1 "$pipeline_pid"
  fi
}
trap forward_stop USR1 TERM
python scripts/direct_llm/run_e2e.py \
  --root "$SLURM_SUBMIT_DIR" \
  --run-root "$run_root" \
  --mode initial400 \
  --variant shared_prompt_v2 &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
status=$?
set -e
exit "$status"
