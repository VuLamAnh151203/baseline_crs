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
  printf '%s\n' 'Refusing EPL native-core mini submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi
for path in \
  runtime/uv.lock \
  toolchains/cuda-12.9.1/bin/nvcc \
  data/raw/DuRecDial2.0/en_train.txt \
  data/raw/DuRecDial2.0/en_dev.txt \
  data/processed/TopDial/train.jsonl \
  data/processed/TopDial/dev.jsonl; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing EPL native-core mini asset: %s\n' "$path" >&2
    exit 4
  fi
done
storage_root="/mnt/data/u26anhvl/crs/epl_native"
mkdir -p logs "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/epl_native_core_mini_e2e.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/epl-native-core-mini-e2e-${job_id}"
printf '%s\n' "$job_id" > logs/active-epl-native-mini-job-id.txt
printf '%s\n' "$run_root" > logs/active-epl-native-mini-run-root.txt
printf 'Submitted EPL native-core two-GPU mini-E2E job: %s\n' "$job_id"
printf 'Run root: %s\n' "$run_root"
printf 'Slurm stdout: %s/slurm/epl-native-mini-%s.out\n' "$storage_root" "$job_id"
printf 'Monitor: squeue -j %s -o %s\n' "$job_id" "'%.18i %.28j %.9T %.10M %.30R'"
