#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s /mnt/data/u26anhvl/crs/dpdp/deadline250-ORIGINAL_JOB_ID\n' "$0" >&2
  exit 2
fi

export DPDP_RESUME_RUN_ROOT="$1"
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/submit_dpdp_deadline.sh"
