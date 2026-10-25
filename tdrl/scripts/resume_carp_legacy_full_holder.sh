#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  printf 'Usage: %s HOLDER_JOB_ID\n' "$0" >&2
  exit 2
fi

job_id="$1"
state="$(squeue -h -j "$job_id" -o '%T')"
if [[ "$state" != "RUNNING" ]]; then
  printf 'Holder job %s is not RUNNING (state=%s).\n' "$job_id" "${state:-not-in-queue}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'Checkout is dirty. Commit the correction before resuming scientific execution.\n' >&2
  git status --short >&2
  exit 1
fi

control_dir="${CARP_FULL_RUN_ROOT:-/mnt/data/u26anhvl/crs/carp_eval/carp-legacy-full-v1}/control"
mkdir -p -- "$control_dir"
printf 'requested_at\t%s\nrequested_by\t%s\njob_id\t%s\n' \
  "$(date --iso-8601=seconds)" "$USER" "$job_id" > "${control_dir}/resume.request"
printf 'Resume requested for holder %s. Follow: %s/state.tsv\n' "$job_id" "$control_dir"
