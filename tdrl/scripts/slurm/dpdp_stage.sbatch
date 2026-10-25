#!/usr/bin/env bash
#SBATCH --job-name=dpdp-e2e250-2gpu
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=/mnt/data/u26anhvl/crs/dpdp/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/dpdp/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
storage_root="/mnt/data/u26anhvl/crs/dpdp"
run_root="${DPDP_RUN_ROOT:-${storage_root}/deadline250-${SLURM_JOB_ID}}"
fixed_test_count="${DPDP_FIXED_TEST_COUNT:-400}"
resume_mode="${DPDP_RESUME:-0}"
restart_count="${SLURM_RESTART_COUNT:-0}"
if [[ ! "$restart_count" =~ ^[0-9]+$ ]]; then
  printf 'SLURM_RESTART_COUNT must be a non-negative integer, got: %s\n' "$restart_count" >&2
  exit 2
fi
segment_tag="$SLURM_JOB_ID"
if [[ "$restart_count" != "0" ]]; then
  segment_tag="${segment_tag}-r${restart_count}"
fi
if [[ "$fixed_test_count" != "400" && "$fixed_test_count" != "1000" ]]; then
  printf 'DPDP_FIXED_TEST_COUNT must be 400 or 1000, got: %s\n' "$fixed_test_count" >&2
  exit 2
fi
if [[ "$resume_mode" != "0" && "$resume_mode" != "1" ]]; then
  printf 'DPDP_RESUME must be 0 or 1, got: %s\n' "$resume_mode" >&2
  exit 2
fi
if [[ "$restart_count" =~ ^[1-9][0-9]*$ ]]; then
  resume_mode=1
fi
if [[ "$resume_mode" == "1" && ( ! -d "$run_root" || -L "$run_root" ) ]]; then
  printf 'Resume root is missing or is a symlink: %s\n' "$run_root" >&2
  exit 2
fi
holder="${run_root}/allocation"
mkdir -p -- "$holder" "$run_root/logs"
source_commit="$(git rev-parse HEAD)"
source_status="$(git status --porcelain --untracked-files=all)"
printf '%s\n' "$SLURM_SUBMIT_DIR" > "$holder/source-checkout-${segment_tag}.txt"
printf '%s\n' "$source_commit" > "$holder/source-commit-${segment_tag}.txt"
printf '%s\n' "$source_status" > "$holder/source-status-${segment_tag}.txt"
if [[ -n "$source_status" ]]; then
  printf 'DPDP allocation received a dirty pinned checkout:\n%s\n' "$source_status" >&2
  exit 3
fi
printf '%s\n' "$SLURM_JOB_ID" > "$holder/job-id-${segment_tag}.txt"
printf '%s\n' "$run_root" > "$holder/run-root.txt"
scontrol show job "$SLURM_JOB_ID" > "$holder/slurm-job-${segment_tag}.txt"
{
  printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
  printf 'SLURM_JOB_NAME=%s\n' "${SLURM_JOB_NAME:-}"
  printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-}"
  printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-}"
  printf 'SLURM_RESTART_COUNT=%s\n' "$restart_count"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-}"
  printf 'PYTHON=%s\n' "$(command -v python)"
} > "$holder/runtime-environment-${segment_tag}.txt"
nvidia-smi -L > "$holder/gpus-${segment_tag}.txt"
nvidia-smi -q > "$holder/nvidia-smi-${segment_tag}.txt"
python --version > "$holder/python-version-${segment_tag}.txt" 2>&1
python -c 'import platform, torch, transformers, vllm; print({"platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "vllm": vllm.__version__})' \
  > "$holder/package-versions-${segment_tag}.txt" 2>&1
printf 'Start at: %s\nRunning on node: %s\nPython: %s\nRun root: %s\nFixed test count: %s\nResume: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$run_root" "$fixed_test_count" "$resume_mode"

nvidia-smi \
  --query-gpu=timestamp,uuid,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv -l 30 > "$holder/gpu-telemetry-${segment_tag}.csv" 2>&1 &
telemetry_pid=$!

cleanup() {
  if [[ -n "${telemetry_pid:-}" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  scontrol show job "$SLURM_JOB_ID" > "$holder/slurm-job-final-${segment_tag}.txt" 2>&1 || true
}
forward_stop() {
  touch "$run_root/stop.requested"
  if [[ -n "${pipeline_pid:-}" ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    kill -USR1 "$pipeline_pid"
  fi
}
trap forward_stop USR1 TERM
trap cleanup EXIT
pipeline_args=(
  --root "$SLURM_SUBMIT_DIR"
  --run-root "$run_root"
  --fixed-test-count "$fixed_test_count"
)
if [[ "$resume_mode" == "1" ]]; then
  pipeline_args+=(--resume)
fi
python scripts/dpdp/run_deadline_e2e.py "${pipeline_args[@]}" \
  > >(tee -a "$run_root/logs/controller-${segment_tag}.log") 2>&1 &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
pipeline_status=$?
if kill -0 "$pipeline_pid" 2>/dev/null; then
  wait "$pipeline_pid"
  pipeline_status=$?
fi
exit "$pipeline_status"
