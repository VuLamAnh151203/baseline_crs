#!/usr/bin/env bash
#SBATCH --job-name=tdrl-persistent-allocation
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from tdrl root}"
holder="artifacts/allocation-${SLURM_JOB_ID}"
mkdir -p "$holder"
scontrol show job "$SLURM_JOB_ID" > "$holder/slurm-job.txt"
python3 - "$holder" <<'PY'
import json, os, sys, time
from pathlib import Path
Path(sys.argv[1], "allocation.json").write_text(json.dumps({
    "schema_version": 1, "job_id": os.environ["SLURM_JOB_ID"],
    "started_unix": time.time(), "gpu_count": 2, "gpu": "H100 80GB",
    "cpus": 16, "ram": "128G", "walltime_seconds": 1728000,
    "replaces_job": "57711", "persistent": True,
}) + "\n")
PY
trap 'touch "$holder/stop.requested"' USR1
while [[ ! -e "$holder/stop.requested" ]]; do
  sleep 30 &
  wait "$!" || true
done
