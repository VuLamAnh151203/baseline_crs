#!/usr/bin/env bash
set -euo pipefail

repo_root="${BASELINE_REPO_ROOT:?missing BASELINE_REPO_ROOT}"
project_root="$repo_root/tdrl"
method="${BASELINE_METHOD:?missing BASELINE_METHOD}"
mode="${BASELINE_MODE:?missing BASELINE_MODE}"
storage_root="${BASELINE_STORAGE_ROOT:?missing BASELINE_STORAGE_ROOT}"
python_bin="${BASELINE_PYTHON:?missing BASELINE_PYTHON}"

if [[ ! -d "$repo_root/.git" || ! -f "$project_root/pyproject.toml" ]]; then
  printf 'Invalid baseline_crs checkout: %s\n' "$repo_root" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Runtime Python is not executable: %s\n' "$python_bin" >&2
  exit 2
fi

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export EPL_MODEL_ROOT="${EPL_MODEL_ROOT:-$project_root/models/epl}"

run_root="${BASELINE_RUN_ROOT:-$storage_root/runs/$method/${method}-${mode}-${SLURM_JOB_ID}}"
if [[ "$run_root" != /* || "$run_root" == *'/../'* || "$run_root" == */.. || -L "$run_root" ]]; then
  printf 'Unsafe run root: %s\n' "$run_root" >&2
  exit 2
fi
mkdir -p -- "$run_root/allocation" "$run_root/logs"

segment="${SLURM_JOB_ID:-manual}"
git -C "$repo_root" rev-parse HEAD > "$run_root/allocation/source-commit-$segment.txt"
git -C "$repo_root" status --porcelain --untracked-files=all \
  > "$run_root/allocation/source-status-$segment.txt"
if [[ -s "$run_root/allocation/source-status-$segment.txt" ]]; then
  printf 'Scientific execution requires a clean committed checkout.\n' >&2
  cat "$run_root/allocation/source-status-$segment.txt" >&2
  exit 3
fi
printf '%s\n' "$repo_root" > "$run_root/allocation/source-checkout-$segment.txt"
printf '%s\n' "$run_root" > "$run_root/allocation/run-root.txt"
scontrol show job "$SLURM_JOB_ID" > "$run_root/allocation/slurm-job-$segment.txt"
nvidia-smi -L > "$run_root/allocation/gpus-$segment.txt"
"$python_bin" --version > "$run_root/allocation/python-$segment.txt" 2>&1

telemetry_pid=''
nvidia-smi \
  --query-gpu=timestamp,uuid,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv -l 30 > "$run_root/allocation/gpu-telemetry-$segment.csv" 2>&1 &
telemetry_pid=$!

pipeline_pid=''
cleanup() {
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  scontrol show job "$SLURM_JOB_ID" > "$run_root/allocation/slurm-job-final-$segment.txt" 2>&1 || true
}
request_stop() {
  touch "$run_root/stop.requested"
  if [[ -n "$pipeline_pid" ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    kill -USR1 "$pipeline_pid" 2>/dev/null || true
  fi
}
trap request_stop USR1 TERM
trap cleanup EXIT

if [[ "$mode" == mini ]]; then
  pipeline_mode=mini
else
  pipeline_mode=initial400
fi
common=(--root "$project_root" --run-root "$run_root" --python "$python_bin")
case "$method" in
  direct_llm_v2)
    cmd=("$python_bin" scripts/direct_llm/run_e2e.py "${common[@]}" --mode "$pipeline_mode" --variant shared_prompt_v2)
    ;;
  tepl_v2)
    cmd=("$python_bin" scripts/epl/run_e2e.py "${common[@]}" --mode "$pipeline_mode")
    ;;
  enpl_domain3)
    cmd=("$python_bin" scripts/enpl/run_e2e.py "${common[@]}" --mode "$pipeline_mode" --convplan-zip "${ENPL_CONVPLAN_ZIP:?missing ENPL_CONVPLAN_ZIP}")
    ;;
  procot)
    cmd=("$python_bin" scripts/procot/run_e2e.py "${common[@]}" --mode "$pipeline_mode")
    ;;
  scope_crs_lite)
    scope_asset_root="${SCOPE_ASSET_ROOT:?missing SCOPE_ASSET_ROOT}"
    scope_storage_root="$(cd "$scope_asset_root/.." && pwd -P)"
    cmd=("$python_bin" scripts/scope/run_e2e.py "${common[@]}" --storage-root "$scope_storage_root" --mode "$pipeline_mode")
    ;;
  dpdp)
    if [[ "$mode" == mini ]]; then
      revision="$("$python_bin" -c 'import json; print(json.load(open("configs/dpdp.models.lock.json"))["revision"])')"
      cmd=("$python_bin" scripts/dpdp/mini_e2e.py --root "$project_root" --output "$run_root" --roberta-revision "$revision" --runtime-config configs/runtime.json)
    else
      cmd=("$python_bin" scripts/dpdp/run_deadline_e2e.py --root "$project_root" --run-root "$run_root" --python "$python_bin" --fixed-test-count 400)
      if [[ "${BASELINE_RESUME:-0}" == 1 ]]; then
        cmd+=(--resume)
      fi
    fi
    ;;
  epl_native_qwen)
    if [[ "$mode" == mini ]]; then
      cmd=("$python_bin" scripts/epl_native/run_mini_e2e.py --root "$project_root" --run-root "$run_root")
    else
      mini_root="${EPL_NATIVE_MINI_ROOT:?Set EPL_NATIVE_MINI_ROOT to a completed mini run}"
      cmd=("$python_bin" scripts/epl_native/run_full_e2e.py --root "$project_root" --run-root "$run_root" --mini-root "$mini_root")
    fi
    ;;
  *) printf 'Unsupported method: %s\n' "$method" >&2; exit 2 ;;
esac

launch_record="$run_root/allocation/launch-$segment.txt"
printf 'method=%s\nmode=%s\nrun_root=%s\nstart=%s\n' \
  "$method" "$mode" "$run_root" "$(date --iso-8601=seconds)" | tee "$launch_record"
printf 'command:' | tee -a "$launch_record"
printf ' %q' "${cmd[@]}" | tee -a "$launch_record"
printf '\n' | tee -a "$launch_record"

"${cmd[@]}" > >(tee -a "$run_root/logs/controller-$segment.log") 2>&1 &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
status=$?
set -e
printf 'exit_code=%s\nend=%s\n' "$status" "$(date --iso-8601=seconds)" | tee -a "$launch_record"
exit "$status"
