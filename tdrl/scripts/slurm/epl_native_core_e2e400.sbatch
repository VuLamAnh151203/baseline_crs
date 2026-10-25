#!/usr/bin/env bash
#SBATCH --job-name=epl-native-e2e400
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/mnt/data/u26anhvl/crs/epl_native/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/epl_native/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4

storage_root="/mnt/data/u26anhvl/crs/epl_native"
run_root="${storage_root}/epl-native-core-fixed400-${SLURM_JOB_ID}"
mini_root="${storage_root}/epl-native-core-mini-e2e-60458"
mkdir -p -- "${run_root}/allocation"
printf '%s\n' "$SLURM_JOB_ID" > "${run_root}/allocation/job-id.txt"
printf '%s\n' "$run_root" > "${run_root}/allocation/run-root.txt"
scontrol show job "$SLURM_JOB_ID" > "${run_root}/allocation/slurm-job.txt"
printf 'EPL-native fixed-400 start: %s\nNode: %s\nPython: %s\nRun root: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$run_root"

# All executable checks run inside the allocation, never on the login node.
python -m compileall -q src/tdrl/baselines/epl_native scripts/epl_native
ruff check --select E4,E7,E9,F,B \
  src/tdrl/baselines/epl_native scripts/epl_native tests/test_epl_native.py
env PYTHONPATH=src:. python -m pytest -q tests/test_epl_native.py

forward_stop() {
  if [[ -n "${pipeline_pid:-}" ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    kill -USR1 "$pipeline_pid"
  fi
}
trap forward_stop USR1 TERM
env PYTHONPATH=src:. python scripts/epl_native/run_full_e2e.py \
  --root "$SLURM_SUBMIT_DIR" \
  --run-root "$run_root" \
  --mini-root "$mini_root" &
pipeline_pid=$!
set +e
wait "$pipeline_pid"
status=$?
set -e
exit "$status"
