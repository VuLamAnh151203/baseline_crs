#!/usr/bin/env bash
#SBATCH --job-name=tdrl-ppdpp-comparison
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
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
runtime_python="${SLURM_SUBMIT_DIR}/runtime/.venv/bin/python"
test -x "$runtime_python"
run_dir="artifacts/allocation-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
printf '%s\n' "$SLURM_JOB_ID" > "$run_dir/job_id.txt"
scontrol show job "$SLURM_JOB_ID" > "$run_dir/slurm-job.txt"
trap 'touch "$run_dir/stop.requested"' USR1
"$runtime_python" -m tdrl check
"$runtime_python" -m tdrl runtime-audit
"$runtime_python" -m tdrl verify-models > "$run_dir/verified-models.json"
# The controller checks Gate 2 before entering any learning/selection/test stage.
srun --ntasks=1 --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  "$runtime_python" -m tdrl comparison --config configs/comparison.json --output "$run_dir" &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
pipeline_status=$?
# USR1 interrupts wait, allowing the trap to write the stop marker immediately.
if kill -0 "$pipeline_pid" 2>/dev/null; then
  wait "$pipeline_pid"
  pipeline_status=$?
fi
exit "$pipeline_status"
