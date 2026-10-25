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
  printf '%s\n' 'Refusing EPL mini-E2E submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
for path in \
  runtime/uv.lock \
  toolchains/cuda-12.9.1/bin/nvcc \
  data/processed/TopDial/train.jsonl \
  data/processed/DuRecDial2.0/train.jsonl; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required EPL mini-E2E asset: %s\n' "$path" >&2
    exit 4
  fi
done
storage_root="/mnt/data/u26anhvl/crs/epl"
mkdir -p logs "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/epl_mini_e2e.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/epl-mini-e2e-${job_id}"
printf '%s\n' "$job_id" > logs/active-epl-mini-job-id.txt
printf '%s\n' "$run_root" > logs/active-epl-mini-run-root.txt
printf 'Submitted EPL mini-E2E job: %s\n' "$job_id"
printf 'Run root: %s\n' "$run_root"
printf 'Slurm stdout: %s/slurm/epl-mini-e2e-%s.out\n' "$storage_root" "$job_id"
printf 'Monitor: squeue -j %s -o %s\n' "$job_id" "'%.18i %.28j %.9T %.10M %.30R'"
