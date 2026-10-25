#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested submission from allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$script_dir/.." && pwd)"
storage_root="/mnt/data/u26anhvl/crs/enpl"
mkdir -p -- "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/enpl_bank_audit.sbatch)"
job_id="${submission%%;*}"
printf 'Submitted EnPL full bank audit: %s\n' "$job_id"
printf 'Output: %s/enpl-bank-audit-%s\n' "$storage_root" "$job_id"
