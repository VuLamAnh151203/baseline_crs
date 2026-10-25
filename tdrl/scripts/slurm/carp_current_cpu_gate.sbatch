#!/usr/bin/env bash
#SBATCH --job-name=carp2-cpu-gate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/carp_eval/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from baseline_crs/tdrl}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:/home/u26anhvl/crs/carp_www/src"
python -m compileall -q src/tdrl/carp_current.py scripts/carp_current/run.py
ruff check --select E4,E7,E9,F,B src/tdrl/carp_current.py scripts/carp_current/run.py tests/test_carp_current.py
python -m pytest -q tests/test_carp_current.py
python scripts/carp_current/run.py --help >/dev/null
