#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' 'Run this I/O-only download on the networked login node.' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
storage_root="/mnt/data/u26anhvl/crs/scope"
bootstrap_python="${storage_root}/bootstrap-python/gdown-5.2.0"
mkdir -p -- "$bootstrap_python"
if [[ ! -f "${bootstrap_python}/gdown/__main__.py" ]]; then
  python -m pip install --disable-pip-version-check --target "$bootstrap_python" \
    "gdown==5.2.0"
fi
export PYTHONPATH="${bootstrap_python}:src:."
python scripts/scope/bootstrap_transition.py \
  --output "${storage_root}/assets/transition_models/deterministic"
