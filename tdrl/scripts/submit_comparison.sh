#!/usr/bin/env bash
# Bootstrap only on login; all model execution is in the new gated Slurm allocation.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
checkout_root="$(dirname "$project_root")"
ssh_alias="${TDRL_SSH_ALIAS:-H100}"
remote_root="${TDRL_REMOTE_ROOT:-/home/u26anhvl/crs}"
remote_data_root="/mnt/data/duclm3/crs_refactor/tdrl/data"
remote_model_root="/mnt/data/duclm3/crs_refactor/tdrl/models"
if [[ ! "$remote_root" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
  printf '%s\n' 'Remote root must be an absolute path without shell metacharacters.' >&2
  exit 2
fi
if [[ -n "$(git -C "$checkout_root" status --porcelain)" ]]; then
  printf '%s\n' 'Commit the source/configuration before submission.' >&2
  exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=15 "$ssh_alias" true
bash "$project_root/scripts/check.sh"
ssh "$ssh_alias" "mkdir -p '$remote_root/tdrl'"
rsync -az --exclude='*.pdf' --exclude='.venv' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='*.egg-info' \
  --exclude='models' --exclude='artifacts' --exclude='logs' --exclude='toolchains' \
  --exclude='data/raw' --exclude='data/processed' \
  "$checkout_root/" "$ssh_alias:$remote_root/"
ssh "$ssh_alias" "cd '$remote_root/tdrl' && bash scripts/link_h100_storage.sh '$remote_data_root' '$remote_model_root'"
rsync -az "$project_root/data/raw/" "$ssh_alias:$remote_data_root/raw/"
ssh "$ssh_alias" "bash -s -- '$remote_root'" <<'REMOTE'
set -euo pipefail
cd "$1/tdrl"
~/.local/bin/uv sync --project runtime --locked --python 3.11 > ../tdrl-bootstrap-runtime.log 2>&1
runtime/.venv/bin/python scripts/bootstrap_toolchain.py > ../tdrl-bootstrap-toolchain.log 2>&1
runtime/.venv/bin/python scripts/verify_runtime_cli.py > ../tdrl-bootstrap-imports.log 2>&1
runtime/.venv/bin/python scripts/bootstrap_models.py > ../tdrl-bootstrap-models.log 2>&1
runtime/.venv/bin/python scripts/bootstrap_models.py --lock configs/policy-model.lock.json > ../tdrl-bootstrap-policy.log 2>&1
runtime/.venv/bin/python -m tdrl verify-models > ../tdrl-bootstrap-verified-models.log
runtime/.venv/bin/python -m tdrl runtime-audit
if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' 'Remote source is dirty; not submitting.' >&2
  exit 2
fi
mkdir -p logs
# Inspect current scheduler state; do not reuse, cancel or attach to existing jobs.
squeue -u "$(id -un)"
sinfo -p main -o '%N %T %G'
job_id="$(sbatch --parsable scripts/slurm/allocation.sbatch)"
printf '%s\n' "$job_id" > logs/submitted-job-id.txt
scontrol show job "${job_id%%;*}"
printf 'Submitted new allocation: %s\n' "$job_id"
REMOTE
