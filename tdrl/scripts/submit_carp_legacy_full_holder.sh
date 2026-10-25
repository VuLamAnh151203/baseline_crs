#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p /mnt/data/u26anhvl/crs/carp_eval/slurm

# The mini jobs are intentionally chained with afterany.  The holder verifies
# their terminal states and runs an in-allocation repair smoke before full
# evaluation if any external diagnostic failed.
dependency_job="${CARP_MINI_TAIL_JOB_ID:-61985}"
mini_jobs="${CARP_MINI_JOB_IDS:-61981:61982:61983:61984:61985}"
job_id="$(sbatch --parsable --dependency="afterany:${dependency_job}" \
  --export="ALL,CARP_MINI_JOB_IDS=${mini_jobs}" \
  scripts/slurm/carp_legacy_full_holder.sbatch)"

printf 'Submitted CARP full-evaluation holder: %s\n' "$job_id"
printf 'Dependency: afterany:%s\n' "$dependency_job"
printf 'Mini jobs checked by holder: %s\n' "$mini_jobs"
printf 'Run root: /mnt/data/u26anhvl/crs/carp_eval/carp-legacy-full-v1\n'
printf 'Slurm log: /mnt/data/u26anhvl/crs/carp_eval/slurm/carp-full-holder-%s.out\n' "$job_id"
printf 'State: /mnt/data/u26anhvl/crs/carp_eval/carp-legacy-full-v1/control/state.tsv\n'
printf 'Resume after a retained failure: bash scripts/resume_carp_legacy_full_holder.sh %s\n' "$job_id"

