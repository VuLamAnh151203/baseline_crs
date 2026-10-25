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
  printf '%s\n' 'Refusing ProCoT fixed-400 submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
for path in \
  runtime/uv.lock \
  toolchains/cuda-12.9.1/bin/nvcc \
  configs/procot-reference.lock.json \
  data/processed/TopDial/dev.jsonl \
  data/processed/DuRecDial2.0/dev.jsonl; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required ProCoT runtime path: %s\n' "$path" >&2
    exit 4
  fi
done
storage_root="/mnt/data/u26anhvl/crs/procot"
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/procot_e2e400.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/procot-fixed1000-${job_id}"
printf '%s\n' "$job_id" > logs/active-procot400-job-id.txt
printf '%s\n' "$run_root" > logs/active-procot-run-root.txt
printf 'Submitted ProCoT fixed-400 E2E: %s\n' "$job_id"
printf 'Run root: %s\n' "$run_root"
printf 'Slurm stdout: %s/slurm/procot-e2e400-%s.out\n' "$storage_root" "$job_id"
