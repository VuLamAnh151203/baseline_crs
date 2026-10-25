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
  printf '%s\n' 'Refusing EPL-native fixed-400 submission from a dirty worktree.' >&2
  git status --short >&2
  exit 3
fi

storage_root="/mnt/data/u26anhvl/crs/epl_native"
mini_root="${storage_root}/epl-native-core-mini-e2e-60458"
for path in \
  "${mini_root}/e2e-summary-mini.json" \
  "${mini_root}/workers/TopDial/training/checkpoint-best.pt" \
  "${mini_root}/workers/TopDial/action-vocabulary.json" \
  "${mini_root}/workers/DuRecDial2.0/training/checkpoint-best.pt" \
  "${mini_root}/workers/DuRecDial2.0/action-vocabulary.json" \
  runtime/uv.lock \
  toolchains/cuda-12.9.1/bin/nvcc \
  configs/baseline-fixed1000-test-v1.json \
  data/processed/TopDial/train.jsonl \
  data/processed/TopDial/dev.jsonl \
  data/processed/DuRecDial2.0/train.jsonl \
  data/processed/DuRecDial2.0/dev.jsonl; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing EPL-native fixed-400 asset: %s\n' "$path" >&2
    exit 4
  fi
done

mkdir -p logs "${storage_root}/slurm"
submission="$(sbatch --parsable scripts/slurm/epl_native_core_e2e400.sbatch)"
job_id="${submission%%;*}"
run_root="${storage_root}/epl-native-core-fixed400-${job_id}"
printf '%s\n' "$job_id" > logs/active-epl-native-e2e400-job-id.txt
printf '%s\n' "$run_root" > logs/active-epl-native-e2e400-run-root.txt
printf 'Submitted EPL-native two-GPU fixed-400 job: %s\n' "$job_id"
printf 'Run root: %s\n' "$run_root"
printf 'Slurm stdout: %s/slurm/epl-native-e2e400-%s.out\n' "$storage_root" "$job_id"
printf 'Monitor: squeue -j %s -o %s\n' "$job_id" "'%.18i %.28j %.9T %.10M %.30R'"
