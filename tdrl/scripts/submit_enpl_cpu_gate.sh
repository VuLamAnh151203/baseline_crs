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
  printf 'Refusing EnPL CPU gate from a dirty worktree.\n' >&2
  git status --short >&2
  exit 3
fi
storage_root="/mnt/data/u26anhvl/crs/enpl"
mkdir -p -- "${storage_root}/slurm"
checkout="$(bash "$project_root/scripts/enpl/prepare_clean_checkout.sh" \
  "$project_root" "$storage_root")"
cd "$checkout"
submission="$(sbatch --parsable scripts/slurm/enpl_cpu_gate.sbatch)"
job_id="${submission%%;*}"
printf 'Submitted EnPL CPU gate: %s\nLog: %s/slurm/enpl-cpu-gate-%s.out\nCheckout: %s\n' \
  "$job_id" "$storage_root" "$job_id" "$checkout"
