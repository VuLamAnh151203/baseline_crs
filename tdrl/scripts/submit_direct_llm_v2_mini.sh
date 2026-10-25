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
  printf '%s\n' 'Refusing DirectLLM-v2 mini submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
storage_root="/mnt/data/u26anhvl/crs/direct_llm"
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/direct_llm_v2_mini_e2e.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/direct-llm-v2-mini-e2e-${job_id}"
printf '%s\n' "$job_id" > logs/active-direct-llm-v2-mini-job-id.txt
printf '%s\n' "$run_root" > logs/active-direct-llm-v2-mini-run-root.txt
printf 'Submitted DirectLLM-v2 mini E2E: %s\nRun root: %s\n' "$job_id" "$run_root"
