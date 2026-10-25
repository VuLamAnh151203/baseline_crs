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
  printf '%s\n' 'Refusing SCOPE fixed-400 submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
storage_root="/mnt/data/u26anhvl/crs/scope"
for path in \
  runtime/uv.lock \
  toolchains/cuda-12.9.1/bin/nvcc \
  configs/scope-reference.lock.json \
  "${storage_root}/assets/transition_models/deterministic/manifest.json" \
  "${storage_root}/assets/Meta-Llama-Guard-2-8B/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef/scope-asset-manifest.json"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required SCOPE runtime path: %s\n' "$path" >&2
    exit 4
  fi
done
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/scope_e2e400.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/scope-fixed1000-${job_id}"
printf '%s\n' "$job_id" > logs/active-scope400-job-id.txt
printf '%s\n' "$run_root" > logs/active-scope-run-root.txt
printf 'Submitted SCOPE fixed-400 E2E: %s\nRun root: %s\n' "$job_id" "$run_root"
