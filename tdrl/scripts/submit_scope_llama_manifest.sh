#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested submission from allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
storage_root="/mnt/data/u26anhvl/crs/scope"
snapshot="${storage_root}/assets/Meta-Llama-Guard-2-8B/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef"
if [[ ! -d "$snapshot" ]]; then
  printf 'Authorized Llama Guard snapshot is absent: %s\n' "$snapshot" >&2
  exit 4
fi
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/scope_llama_manifest.sbatch)"
job_id="${submission%%;*}"
printf 'Submitted SCOPE Llama Guard manifest job: %s\n' "$job_id"
printf 'Log: %s/slurm/scope-llama-hash-%s.out\n' "$storage_root" "$job_id"
