#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
python_bin="${BASELINE_PYTHON:-$repo_root/tdrl/runtime/.venv/bin/python}"
storage_root="${BASELINE_STORAGE_ROOT:?Set BASELINE_STORAGE_ROOT}"
[[ -x "$python_bin" ]] || { printf 'Missing runtime Python: %s\n' "$python_bin" >&2; exit 2; }
mkdir -p -- "$storage_root/slurm"
export BASELINE_REPO_ROOT="$repo_root"
export BASELINE_PYTHON="$python_bin"
job_id="$(sbatch --parsable \
  --partition="${SLURM_PARTITION:-main}" \
  --account="${SLURM_ACCOUNT:-normal}" \
  --output="$storage_root/slurm/%x-%j.out" \
  --error="$storage_root/slurm/%x-%j.err" \
  --export=ALL \
  "$repo_root/scripts/slurm/repository_cpu_gate.sbatch")"
printf 'Submitted repository CPU gate: %s\n' "$job_id"
printf 'Log: %s/slurm/baseline-crs-cpu-gate-%s.out\n' "$storage_root" "$job_id"
