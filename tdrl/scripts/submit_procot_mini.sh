#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested submission from allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf '%s\n' 'Refusing ProCoT mini submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
storage_root="/mnt/data/u26anhvl/crs/procot"
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/procot_mini_e2e.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/procot-mini-e2e-${job_id}"
printf '%s\n' "$job_id" > logs/active-procot-mini-job-id.txt
printf '%s\n' "$run_root" > logs/active-procot-mini-run-root.txt
printf 'Submitted ProCoT mini E2E: %s\nRun root: %s\n' "$job_id" "$run_root"
