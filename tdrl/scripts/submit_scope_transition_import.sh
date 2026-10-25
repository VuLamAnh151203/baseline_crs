#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested submission from allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
assets="/mnt/data/u26anhvl/crs/scope/assets/transition_models"
for index in 001 002 003; do
  archive="${assets}/deterministic-20261016T193326Z-1-${index}.zip"
  if [[ ! -f "$archive" ]]; then
    printf 'Missing SCOPE transition archive: %s\n' "$archive" >&2
    exit 4
  fi
done
mkdir -p -- "/mnt/data/u26anhvl/crs/scope/slurm"
submission="$(sbatch --parsable scripts/slurm/scope_transition_import.sbatch)"
job_id="${submission%%;*}"
printf 'Submitted SCOPE transition import: %s\n' "$job_id"
printf 'Log: /mnt/data/u26anhvl/crs/scope/slurm/scope-transition-import-%s.out\n' "$job_id"
