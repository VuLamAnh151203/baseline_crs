#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /mnt/data/u26anhvl/crs/carp_eval/slurm logs

if [[ -n "$(git -C .. status --porcelain --untracked-files=all)" ]]; then
  printf 'Refusing submission from a dirty baseline_crs checkout.\n' >&2
  exit 3
fi
if [[ -n "$(git -C /home/u26anhvl/crs/carp_www status --porcelain --untracked-files=all)" ]]; then
  printf 'Refusing submission from a dirty carp_www checkout.\n' >&2
  exit 3
fi

cpu_job="$(sbatch --parsable scripts/slurm/carp_current_cpu_gate.sbatch)"
mini_job="$(sbatch --parsable --dependency="afterok:${cpu_job}" scripts/slurm/carp_current_mini.sbatch)"
full_job="$(sbatch --parsable --dependency="afterok:${mini_job}" scripts/slurm/carp_current_full.sbatch)"
printf '%s\n' "$cpu_job" > logs/active-carp-current-cpu-job-id.txt
printf '%s\n' "$mini_job" > logs/active-carp-current-mini-job-id.txt
printf '%s\n' "$full_job" > logs/active-carp-current-full-job-id.txt
printf 'CPU gate: %s\nMini (afterok:%s): %s\nFull (afterok:%s): %s\n' \
  "$cpu_job" "$cpu_job" "$mini_job" "$mini_job" "$full_job"
printf 'Mini root: /mnt/data/u26anhvl/crs/carp_eval/carp-current-e5bf703-mini-%s\n' "$mini_job"
printf 'Full root: /mnt/data/u26anhvl/crs/carp_eval/carp-current-e5bf703-full-v1\n'
