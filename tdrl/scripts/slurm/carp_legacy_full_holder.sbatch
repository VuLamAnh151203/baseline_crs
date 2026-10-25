#!/usr/bin/env bash
#SBATCH --job-name=carp-full-holder
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.err

# This allocation deliberately survives evaluator/preflight failures.  The batch
# shell remains alive and waits for control/resume.request, so corrected code can
# resume the immutable per-dialogue snapshots without another queue wait.
set -uo pipefail

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

storage_root="/mnt/data/u26anhvl/crs/carp_eval"
run_root="${CARP_FULL_RUN_ROOT:-${storage_root}/carp-legacy-full-v1}"
control_dir="${run_root}/control"
allocation_dir="${run_root}/allocation"
holder_log_dir="${run_root}/logs/holder"
mkdir -p -- "$control_dir" "$allocation_dir" "$holder_log_dir"

methods=(direct_llm_v2 tepl_v2 enpl_domain3 procot dpdp)
external_mini_jobs="${CARP_MINI_JOB_IDS:-}"
current_stage="allocation"

timestamp() {
  date --iso-8601=seconds
}

write_state() {
  local status="$1"
  local detail="${2:-}"
  local temporary="${control_dir}/state.tsv.tmp-${SLURM_JOB_ID}"
  {
    printf 'status\t%s\n' "$status"
    printf 'stage\t%s\n' "$current_stage"
    printf 'detail\t%s\n' "$detail"
    printf 'job_id\t%s\n' "$SLURM_JOB_ID"
    printf 'node\t%s\n' "${SLURM_JOB_NODELIST:-unknown}"
    printf 'run_root\t%s\n' "$run_root"
    printf 'updated_at\t%s\n' "$(timestamp)"
  } > "$temporary"
  mv -f -- "$temporary" "${control_dir}/state.tsv"
}

record_segment() {
  local tag="$1"
  local commit status_file
  commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  status_file="${allocation_dir}/source-status-${tag}.txt"
  printf '%s\n' "$commit" > "${allocation_dir}/source-commit-${tag}.txt"
  git status --porcelain --untracked-files=all > "$status_file"
  printf '%s\n' "$SLURM_SUBMIT_DIR" > "${allocation_dir}/source-checkout-${tag}.txt"
  printf '%s\n' "$(timestamp)" > "${allocation_dir}/segment-start-${tag}.txt"
}

wait_for_resume() {
  local detail="$1"
  write_state "HOLDING_AFTER_FAILURE" "$detail"
  printf '[%s] HOLDING allocation after failure: %s\n' "$(timestamp)" "$detail"
  printf '[%s] Fix and commit the code, then run: bash scripts/resume_carp_legacy_full_holder.sh %s\n' \
    "$(timestamp)" "$SLURM_JOB_ID"
  while true; do
    if [[ -f "${control_dir}/release.request" ]]; then
      write_state "RELEASED_BY_REQUEST" "$detail"
      return 1
    fi
    if [[ -f "${control_dir}/resume.request" ]]; then
      mv -f -- "${control_dir}/resume.request" \
        "${control_dir}/resume-consumed-$(date -u +%Y%m%dT%H%M%SZ).txt"
      printf '[%s] Resume request received for stage %s.\n' "$(timestamp)" "$current_stage"
      write_state "RESUMING" "$detail"
      return 0
    fi
    sleep 15
  done
}

source_is_clean() {
  [[ -z "$(git status --porcelain --untracked-files=all)" ]]
}

run_preflight() {
  local attempt="$1"
  local log="${holder_log_dir}/preflight-attempt-${attempt}.log"
  {
    printf '[%s] preflight attempt=%s\n' "$(timestamp)" "$attempt"
    python -m compileall -q src/tdrl/carp_legacy.py scripts/carp_legacy/run.py &&
    ruff check --select E4,E7,E9,F,B src/tdrl/backend.py src/tdrl/carp_legacy.py \
      scripts/carp_legacy/run.py tests/test_carp_legacy.py &&
    python -m pytest -q tests/test_carp_legacy.py
  } > >(tee -a "$log") 2>&1
}

report_is_complete() {
  local report="$1"
  local expected_count="$2"
  python -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    row = json.loads(p.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
expected = int(sys.argv[2])
raise SystemExit(0 if row.get("status") == "complete" and row.get("count_per_dataset") == expected and row.get("diagnostic_only") == (expected < 200) else 1)' \
    "$report" "$expected_count"
}

run_controller() {
  local method="$1"
  local count="$2"
  local method_root="$3"
  local log_tag="$4"
  local log="${holder_log_dir}/${log_tag}.log"
  mkdir -p -- "$method_root"
  printf '[%s] method=%s count=%s run_root=%s\n' \
    "$(timestamp)" "$method" "$count" "$method_root" | tee -a "$log"
  python scripts/carp_legacy/run.py controller \
    --root "$SLURM_SUBMIT_DIR" \
    --run-root "$method_root" \
    --method "$method" \
    --count "$count" \
    > >(tee -a "$log") 2>&1
}

on_walltime_warning() {
  printf '[%s] Slurm wall-time warning received during stage %s.\n' \
    "$(timestamp)" "$current_stage" | tee -a "${holder_log_dir}/signals.log"
  write_state "WALLTIME_WARNING" "Slurm sent USR1 with less than 15 minutes remaining"
}
trap on_walltime_warning USR1

printf '%s\n' "$SLURM_JOB_ID" > "${allocation_dir}/job-id.txt"
printf '%s\n' "$run_root" > "${allocation_dir}/run-root.txt"
scontrol show job "$SLURM_JOB_ID" > "${allocation_dir}/slurm-job-start.txt" 2>&1
nvidia-smi -L > "${allocation_dir}/gpus.txt" 2>&1
record_segment "${SLURM_JOB_ID}-initial"
write_state "STARTING" "holder allocation received"

# Never surrender the allocation merely because the checkout or a local gate is
# temporarily broken.  Wait for a clean committed correction and a resume signal.
preflight_attempt=0
while true; do
  current_stage="preflight"
  preflight_attempt=$((preflight_attempt + 1))
  record_segment "${SLURM_JOB_ID}-preflight-${preflight_attempt}"
  if ! source_is_clean; then
    if ! wait_for_resume "checkout is dirty; scientific execution refused"; then
      exit 0
    fi
    continue
  fi
  write_state "RUNNING" "preflight attempt ${preflight_attempt}"
  if run_preflight "$preflight_attempt"; then
    break
  fi
  if ! wait_for_resume "preflight failed on attempt ${preflight_attempt}"; then
    exit 0
  fi
done

# All submitted diagnostics are allowed to finish even if one fails.  If any
# external mini did fail, validate the corrected/current checkout inside this
# retained allocation before opening the full test run.
repair_smoke_required=0
if [[ -n "$external_mini_jobs" ]]; then
  : > "${allocation_dir}/external-mini-states.tsv"
  for mini_job in ${external_mini_jobs//:/ }; do
    mini_state="$(sacct -j "$mini_job" -X -n -P --format=State | sed -n '1{s/|.*//;p;}')"
    printf '%s\t%s\n' "$mini_job" "${mini_state:-UNKNOWN}" \
      >> "${allocation_dir}/external-mini-states.tsv"
    if [[ "$mini_state" != "COMPLETED" ]]; then
      repair_smoke_required=1
    fi
  done
fi

if [[ "$repair_smoke_required" -eq 1 ]]; then
  for method in "${methods[@]}"; do
    current_stage="repair_smoke_${method}"
    mini_root="${run_root}/repair-smoke/${method}"
    attempt=0
    while ! report_is_complete "${mini_root}/report.json" 1; do
      attempt=$((attempt + 1))
      record_segment "${SLURM_JOB_ID}-repair-smoke-${method}-${attempt}"
      if ! source_is_clean; then
        if ! wait_for_resume "checkout is dirty before repair smoke ${method}"; then
          exit 0
        fi
        continue
      fi
      write_state "RUNNING_REPAIR_SMOKE" "${method} attempt ${attempt}"
      if run_controller "$method" 1 "$mini_root" \
          "repair-smoke-${method}-attempt-${attempt}" && \
          report_is_complete "${mini_root}/report.json" 1; then
        break
      fi
      if [[ "$attempt" -eq 1 ]]; then
        write_state "AUTO_RETRY_WAIT" \
          "repair smoke ${method} failed once; retrying in 60 seconds"
        sleep 60
        continue
      fi
      if ! wait_for_resume "repair smoke ${method} failed on attempt ${attempt}"; then
        exit 0
      fi
    done
  done
fi

for method in "${methods[@]}"; do
  current_stage="$method"
  method_root="${run_root}/${method}"
  if report_is_complete "${method_root}/report.json" 200; then
    printf '[%s] method=%s already complete; skipping.\n' "$(timestamp)" "$method"
    continue
  fi

  attempt=0
  while ! report_is_complete "${method_root}/report.json" 200; do
    attempt=$((attempt + 1))
    record_segment "${SLURM_JOB_ID}-${method}-${attempt}"
    if ! source_is_clean; then
      if ! wait_for_resume "checkout became dirty before ${method}"; then
        exit 0
      fi
      continue
    fi
    write_state "RUNNING" "${method} attempt ${attempt}"
    if run_controller "$method" 200 "$method_root" \
        "${method}-attempt-${attempt}" && \
        report_is_complete "${method_root}/report.json" 200; then
      printf '[%s] method=%s completed.\n' "$(timestamp)" "$method"
      break
    fi

    # One unattended retry handles transient server/port failures.  Persistent
    # failures retain the allocation and require an explicit resume request.
    if [[ "$attempt" -eq 1 ]]; then
      write_state "AUTO_RETRY_WAIT" "${method} failed once; retrying in 60 seconds"
      sleep 60
      continue
    fi
    if ! wait_for_resume "${method} failed on attempt ${attempt}"; then
      exit 0
    fi
  done
done

current_stage="complete"
write_state "COMPLETE" "all five CARP-legacy full evaluations completed"
scontrol show job "$SLURM_JOB_ID" > "${allocation_dir}/slurm-job-final.txt" 2>&1 || true
printf '[%s] All methods completed; releasing allocation.\n' "$(timestamp)"

