#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Refusing nested submission from allocation %s\n' "$SLURM_JOB_ID" >&2
  exit 2
fi
if [[ "$#" -ne 1 ]]; then
  printf 'Usage: %s /mnt/data/.../epl-fixed1000-JOB_ID\n' "$0" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf '%s\n' 'Refusing scientific EPL extension from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
run_root="$1"
if [[ "$run_root" != /mnt/data/u26anhvl/crs/epl/* ]]; then
  printf 'EPL run root must be under /mnt/data/u26anhvl/crs/epl: %s\n' "$run_root" >&2
  exit 4
fi
for path in \
  "$run_root/run-metadata.json" \
  "$run_root/evaluation/selection.lock.json" \
  "$run_root/e2e-summary-fixed1000-prefix-400.json"; do
  if [[ ! -f "$path" ]]; then
    printf 'Prefix400 evidence is incomplete: %s\n' "$path" >&2
    exit 5
  fi
done
storage_root="/mnt/data/u26anhvl/crs/epl"
mkdir -p logs "${storage_root}/slurm"
submission="$(sbatch --parsable --export=ALL,EPL_RUN_ROOT="$run_root" scripts/slurm/epl_extend1000.sbatch)"
job_id="${submission%%;*}"
printf '%s\n' "$job_id" > logs/active-epl1000-job-id.txt
printf 'Submitted EPL 400->1000 extension job: %s\n' "$job_id"
printf 'Reusing run root: %s\n' "$run_root"
printf 'Slurm stdout: %s/slurm/epl-extend1000-%s.out\n' "$storage_root" "$job_id"
