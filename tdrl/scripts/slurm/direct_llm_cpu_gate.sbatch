#!/usr/bin/env bash
#SBATCH --job-name=direct-llm-cpu-gate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/u26anhvl/crs/direct_llm/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/direct_llm/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export OMP_NUM_THREADS=2
run_root="/mnt/data/u26anhvl/crs/direct_llm/direct-llm-cpu-gate-${SLURM_JOB_ID}"
mkdir -p -- "$run_root"
printf 'DirectLLM CPU gate start: %s\nNode: %s\nPython: %s\nRun root: %s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$(command -v python)" "$run_root"
python -m compileall -q \
  src/tdrl/baselines/direct_llm \
  src/tdrl/comparison_direct_llm.py \
  scripts/direct_llm
ruff check \
  src/tdrl/baselines/direct_llm \
  src/tdrl/comparison_direct_llm.py \
  scripts/direct_llm \
  tests/test_direct_llm.py
bash -n \
  scripts/slurm/direct_llm_cpu_gate.sbatch \
  scripts/slurm/direct_llm_mini_e2e.sbatch \
  scripts/slurm/direct_llm_e2e400.sbatch \
  scripts/submit_direct_llm_cpu_gate.sh \
  scripts/submit_direct_llm_mini.sh \
  scripts/submit_direct_llm400.sh
python scripts/direct_llm/run_e2e.py --help > /dev/null
python -m tdrl.baselines.direct_llm.evaluate_cli --help > /dev/null
python -m pytest -q \
  tests/test_direct_llm.py \
  tests/test_baseline_test_setting.py \
  tests/test_test_prefix.py
printf 'DirectLLM CPU gate completed: %s\n' "$(date --iso-8601=seconds)"
touch "$run_root/complete"
