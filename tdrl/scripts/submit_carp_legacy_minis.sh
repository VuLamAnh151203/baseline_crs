#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /mnt/data/u26anhvl/crs/carp_eval/slurm

methods=(direct_llm_v2 tepl_v2 enpl_domain3 procot dpdp)
previous=""
for method in "${methods[@]}"; do
  arguments=(--parsable --job-name="carp-${method}-mini" --export="ALL,CARP_METHOD=${method},CARP_COUNT=1")
  if [[ -n "$previous" ]]; then
    # Every method is an independent compatibility gate.  Run the next mini
    # even if the preceding method fails so one shared defect cannot hide the
    # status of the remaining baselines.
    arguments+=(--dependency="afterany:${previous}")
  fi
  job_id="$(sbatch "${arguments[@]}" scripts/slurm/carp_legacy_mini.sbatch)"
  printf '%s\t%s\n' "$method" "$job_id"
  previous="$job_id"
done
