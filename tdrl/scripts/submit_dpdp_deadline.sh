#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested sbatch from inside allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi

fixed_test_count="${DPDP_FIXED_TEST_COUNT:-400}"
if [[ "$fixed_test_count" != "400" && "$fixed_test_count" != "1000" ]]; then
  printf 'DPDP_FIXED_TEST_COUNT must be 400 or 1000, got: %s\n' "$fixed_test_count" >&2
  exit 2
fi
export DPDP_FIXED_TEST_COUNT="$fixed_test_count"

resume_run_root="${DPDP_RESUME_RUN_ROOT:-}"
if [[ -n "$resume_run_root" ]]; then
  if [[ "$resume_run_root" != /mnt/data/u26anhvl/crs/dpdp/deadline250-* ]]; then
    printf 'DPDP_RESUME_RUN_ROOT must name an existing deadline250 run under shared storage: %s\n' \
      "$resume_run_root" >&2
    exit 2
  fi
  if [[ ! -d "$resume_run_root" || -L "$resume_run_root" ]]; then
    printf 'DPDP resume root is missing or is a symlink: %s\n' "$resume_run_root" >&2
    exit 2
  fi
  resume_run_root="$(realpath -- "$resume_run_root")"
  export DPDP_RUN_ROOT="$resume_run_root"
  export DPDP_RESUME=1
elif [[ -n "${DPDP_RUN_ROOT:-}" ]]; then
  printf '%s\n' 'Set DPDP_RESUME_RUN_ROOT, not DPDP_RUN_ROOT, when resubmitting a run.' >&2
  exit 2
else
  export DPDP_RESUME=0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf '%s\n' 'Refusing scientific submission from a dirty worktree.' >&2
  printf '%s\n' 'Review and commit the DPDP implementation, then rerun this script.' >&2
  git status --short >&2
  exit 3
fi

python_path="${HOME}/miniconda3/envs/dpdp/bin/python"
if [[ ! -x "$python_path" ]]; then
  printf 'Missing DPDP conda Python: %s\n' "$python_path" >&2
  exit 4
fi

revision="722cf37b1afa9454edce342e7895e588b6ff1d59"
model_path="$project_root/models/roberta-large/$revision"
if [[ ! -f "$model_path/model.safetensors" ]]; then
  printf 'Missing pinned RoBERTa snapshot: %s\n' "$model_path" >&2
  exit 5
fi

storage_root="/mnt/data/u26anhvl/crs/dpdp"
source_log_root="$project_root/logs"
mkdir -p -- "$source_log_root" "$storage_root/slurm"
checkout_project="$(
  bash scripts/dpdp/prepare_clean_checkout.sh "$project_root" "$storage_root"
)"
cd "$checkout_project"
submission="$(sbatch --parsable scripts/slurm/dpdp_stage.sbatch)"
job_id="${submission%%;*}"
printf '%s\n' "$job_id" > "$source_log_root/active-dpdp-deadline-job-id.txt"
printf '%s\n' "$checkout_project" > "$source_log_root/active-dpdp-deadline-checkout.txt"
run_root="${resume_run_root:-/mnt/data/u26anhvl/crs/dpdp/deadline250-$job_id}"

if [[ -n "$resume_run_root" ]]; then
  printf '%s\n' "$job_id" > "$source_log_root/active-dpdp-resume-job-id.txt"
fi

printf 'Submitted DPDP-250 E2E job: %s\n' "$job_id"
printf 'Submission mode: %s\n' "$([[ -n "$resume_run_root" ]] && printf resume || printf fresh)"
printf 'Run root: %s\n' "$run_root"
printf 'Pinned checkout: %s\n' "$checkout_project"
printf 'Fixed test prefix: %s scenarios (%s per dataset)\n' \
  "$fixed_test_count" "$((fixed_test_count / 2))"
printf 'Slurm stdout: %s/slurm/dpdp-e2e250-2gpu-%s.out\n' "$storage_root" "$job_id"
printf 'Monitor: squeue -j %s -o %s\n' "$job_id" "'%.18i %.28j %.9T %.10M %.30R'"
printf 'Progress: python -m json.tool %s/status.json\n' "$run_root"
printf 'Events: tail -f %s/events.jsonl\n' "$run_root"
