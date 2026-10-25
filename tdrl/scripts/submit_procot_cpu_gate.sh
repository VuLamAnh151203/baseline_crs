#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested submission from allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
storage_root="/mnt/data/u26anhvl/crs/procot"
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/procot_cpu_gate.sbatch)"
job_id="${submission%%;*}"
printf 'Submitted ProCoT CPU gate: %s\n' "$job_id"
printf 'Log: %s/slurm/procot-cpu-gate-%s.out\n' "$storage_root" "$job_id"
