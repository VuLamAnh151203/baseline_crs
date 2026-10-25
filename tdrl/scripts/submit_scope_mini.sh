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
  printf '%s\n' 'Refusing SCOPE mini submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
storage_root="/mnt/data/u26anhvl/crs/scope"
for path in \
  "${storage_root}/assets/transition_models/deterministic/manifest.json" \
  "${storage_root}/assets/Meta-Llama-Guard-2-8B/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef/scope-asset-manifest.json"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing required SCOPE asset: %s\n' "$path" >&2
    exit 4
  fi
done
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/scope_mini_e2e.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/scope-mini-e2e-${job_id}"
printf '%s\n' "$job_id" > logs/active-scope-mini-job-id.txt
printf '%s\n' "$run_root" > logs/active-scope-mini-run-root.txt
printf 'Submitted SCOPE mini E2E: %s\nRun root: %s\n' "$job_id" "$run_root"
