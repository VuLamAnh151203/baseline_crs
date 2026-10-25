#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested EnPL resume submission.\n' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'Refusing EnPL resume from a dirty worktree.\n' >&2
  git status --short >&2
  exit 3
fi
run_root="/mnt/data/u26anhvl/crs/enpl/enpl-fixed1000-60429"
if [[ ! -f "$run_root/evaluation/method.lock.json" ]]; then
  printf 'Missing frozen EnPL selection in %s\n' "$run_root" >&2
  exit 4
fi
storage_root="/mnt/data/u26anhvl/crs/enpl"
checkout="$(bash "$project_root/scripts/enpl/prepare_clean_checkout.sh" \
  "$project_root" "$storage_root")"
cd "$checkout"
submission="$(sbatch --parsable \
  --export=ALL,ENPL_RUN_ROOT="$run_root",ENPL_AUDITED_RESUME=1 \
  scripts/slurm/enpl_e2e400.sbatch)"
job_id="${submission%%;*}"
mkdir -p -- "$project_root/logs"
printf '%s\n' "$job_id" > "$project_root/logs/active-enpl400-job-id.txt"
printf '%s\n' "$run_root" > "$project_root/logs/active-enpl-run-root.txt"
printf '%s\n' "$checkout" > "$project_root/logs/active-enpl-checkout.txt"
printf 'Submitted audited EnPL fixed-400 resume: %s\nRun root: %s\nCheckout: %s\n' \
  "$job_id" "$run_root" "$checkout"
