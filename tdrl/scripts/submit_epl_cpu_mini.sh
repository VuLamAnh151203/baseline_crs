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
  printf '%s\n' 'Refusing EPL CPU mini submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
storage_root="/mnt/data/u26anhvl/crs/epl"
mkdir -p logs "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/epl_cpu_mini.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/epl-cpu-mini-${job_id}"
printf '%s\n' "$job_id" > logs/active-epl-cpu-mini-job-id.txt
printf '%s\n' "$run_root" > logs/active-epl-cpu-mini-run-root.txt
printf 'Submitted EPL CPU mini job: %s\n' "$job_id"
printf 'Run root: %s\n' "$run_root"
printf 'Slurm stdout: %s/slurm/epl-cpu-mini-%s.out\n' "$storage_root" "$job_id"
