#!/usr/bin/env bash
#SBATCH --job-name=tdrl-persistent-allocation
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=8
# Controller-only workspace; owned inference servers strip these two flags.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TDRL_POLICY_DETERMINISM=strict-backward-v1
runtime_python="${SLURM_SUBMIT_DIR}/runtime/.venv/bin/python"
test -x "$runtime_python"
run_dir="artifacts/recovery-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
printf '%s\n' "$SLURM_JOB_ID" > "$run_dir/job_id.txt"
scontrol show job "$SLURM_JOB_ID" > "$run_dir/slurm-job.txt"
trap 'touch "$run_dir/stop.requested"' USR1
"$runtime_python" -m tdrl check
"$runtime_python" -m tdrl runtime-audit
"$runtime_python" -m tdrl verify-models > "$run_dir/verified-models.json"
# The controller checks Gate 2 before entering any learning/selection/test stage.
srun --ntasks=1 --cpus-per-task=16 --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  "$runtime_python" -m tdrl recover-comparison --config configs/comparison.json \
    --source artifacts/allocation-57166 --audit artifacts/debug-57166-DuRecDial/inspection-replay-fix.json \
    --output "$run_dir" &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
pipeline_status=$?
# USR1 interrupts wait, allowing the trap to write the stop marker immediately.
if kill -0 "$pipeline_pid" 2>/dev/null; then
  wait "$pipeline_pid"
  pipeline_status=$?
fi
"$runtime_python" - "$run_dir" "$pipeline_status" <<'PYTHON'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1]) / "allocation-holder.json"
p.write_text(json.dumps({
    "schema_version": 1, "job_id": os.environ["SLURM_JOB_ID"],
    "pipeline_exit_code": int(sys.argv[2]),
    "state": "idle_after_pipeline; allocation retained for subsequent srun steps",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}) + "\n")
PYTHON
# Keep the owned allocation alive for later steps, even when a scientific gate stops.
# No background inference runs in this idle loop. Steps reserve the whole GPU/16 CPUs.
while [[ ! -e "$run_dir/stop.requested" ]]; do
  sleep 30 &
  wait "$!" || true
done
exit 0
