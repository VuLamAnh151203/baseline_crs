#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'SCOPE CARP evaluation must run inside an owned Slurm allocation.\n' >&2
  exit 2
fi

project_root="${SCOPE_CARP_PROJECT_ROOT:-/home/u26anhvl/crs/crs_www/tdrl}"
run_root="${SCOPE_CARP_RUN_ROOT:-/mnt/data/u26anhvl/crs/carp_eval/carp-legacy-full-v1}"
method_root="${run_root}/scope_crs_lite"
smoke_root="${run_root}/scope_crs_lite-smoke"
log_dir="${run_root}/logs/scope-crs-lite"
carp_data_root="${CARP_DATA_ROOT:-/home/u26anhvl/crs/carp_www/data/eval}"
mkdir -p -- "$method_root" "$smoke_root" "$log_dir"

cd "$project_root"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4

report_complete() {
  local report="$1"
  local expected="$2"
  python -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    row = json.loads(p.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
expected = int(sys.argv[2])
ok = row.get("status") == "complete"
ok = ok and row.get("count_per_dataset") == expected
ok = ok and row.get("diagnostic_only") == (expected < 200)
raise SystemExit(0 if ok else 1)' "$report" "$expected"
}

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'SCOPE CARP evaluation requires a clean committed checkout.\n' >&2
  git status --short >&2
  exit 3
fi

printf '%s\n' "$(git rev-parse HEAD)" > "${method_root}/source-commit.txt"
printf '%s\n' "$SLURM_JOB_ID" > "${method_root}/slurm-job-id.txt"

if ! report_complete "${smoke_root}/report.json" 1; then
  printf '[%s] Starting SCOPE CARP two-dataset smoke.\n' "$(date --iso-8601=seconds)"
  python scripts/carp_legacy/run.py controller \
    --root "$project_root" \
    --run-root "$smoke_root" \
    --method scope_crs_lite \
    --count 1 \
    --carp-data-root "$carp_data_root" \
    > >(tee -a "${log_dir}/smoke-${SLURM_JOB_ID}.log") 2>&1
fi

if ! report_complete "${method_root}/report.json" 200; then
  printf '[%s] Starting/resuming SCOPE CARP fixed-400 evaluation.\n' \
    "$(date --iso-8601=seconds)"
  python scripts/carp_legacy/run.py controller \
    --root "$project_root" \
    --run-root "$method_root" \
    --method scope_crs_lite \
    --count 200 \
    --carp-data-root "$carp_data_root" \
    > >(tee -a "${log_dir}/full-${SLURM_JOB_ID}.log") 2>&1
fi

report_complete "${method_root}/report.json" 200
printf '[%s] SCOPE CARP fixed-400 report is complete.\n' "$(date --iso-8601=seconds)"
