#!/usr/bin/env bash
# Transfer committed tracked source only; preserve unrelated local files and old evidence.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
checkout_root="$(dirname "$project_root")"
ssh_alias="${TDRL_SSH_ALIAS:-H100}"
remote_root="${TDRL_REMOTE_ROOT:-/home/u26anhvl/crs}"
[[ "$remote_root" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 2
if ! git -C "$checkout_root" diff --quiet HEAD; then
  printf '%s\n' 'Tracked source must be committed before recovery.' >&2
  exit 2
fi
# Untracked files are not synchronized or loaded by the scientific runtime.
bash "$project_root/scripts/check.sh"
source_commit="$(git -C "$checkout_root" rev-parse HEAD)"
bundle_path="$(mktemp /tmp/tdrl-recovery-XXXXXX.bundle)"
trap 'rm -f "$bundle_path"' EXIT
git -C "$checkout_root" bundle create "$bundle_path" HEAD
ssh -o ConnectTimeout=15 "$ssh_alias" "cd '$remote_root' && test -z \"\$(git status --porcelain)\" && test -z \"\$(squeue -h -u \$(id -un))\""
ssh "$ssh_alias" "mkdir -p '$remote_root/tdrl/artifacts/debug-57166-DuRecDial'"
scp "$bundle_path" "$ssh_alias:$remote_root/tdrl/artifacts/recovery-source.bundle"
scp "$project_root/artifacts/debug-57166-DuRecDial/inspection-replay-fix.json" "$ssh_alias:$remote_root/tdrl/artifacts/debug-57166-DuRecDial/inspection-replay-fix.json"
ssh "$ssh_alias" "bash -s -- '$remote_root' '$source_commit'" <<'REMOTE'
set -euo pipefail
cd "$1"
test -z "$(git status --porcelain)"
git fetch "$1/tdrl/artifacts/recovery-source.bundle" HEAD
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = "$2"
rm "$1/tdrl/artifacts/recovery-source.bundle"
cd tdrl
bash scripts/link_h100_storage.sh
runtime/.venv/bin/python -m tdrl check
runtime/.venv/bin/python -m tdrl runtime-audit
runtime/.venv/bin/python scripts/verify_runtime_cli.py
mkdir -p logs
squeue -u "$(id -un)"
sinfo -p main -o '%N %T %G'
test -z "$(squeue -h -u "$(id -un)")"
job_id="$(sbatch --parsable scripts/slurm/recovery.sbatch)"
printf '%s\n' "$job_id" > logs/submitted-recovery-job-id.txt
printf '%s\n' "${job_id%%;*}" > logs/active-allocation-job-id.txt
scontrol show job "${job_id%%;*}"
REMOTE
