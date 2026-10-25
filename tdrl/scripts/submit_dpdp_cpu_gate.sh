#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested sbatch from inside allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
storage_root="/mnt/data/u26anhvl/crs/dpdp"
cd "$project_root"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf '%s\n' 'Refusing DPDP CPU gate submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi

mkdir -p -- "$storage_root/slurm"
checkout_project="$(
  bash scripts/dpdp/prepare_clean_checkout.sh "$project_root" "$storage_root"
)"
cd "$checkout_project"
submission="$(sbatch --parsable scripts/slurm/dpdp_cpu_gate.sbatch)"
job_id="${submission%%;*}"

printf 'Submitted DPDP CPU gate: %s\n' "$job_id"
printf 'Pinned checkout: %s\n' "$checkout_project"
printf 'Log: %s/slurm/dpdp-cpu-gate-%s.out\n' "$storage_root" "$job_id"
