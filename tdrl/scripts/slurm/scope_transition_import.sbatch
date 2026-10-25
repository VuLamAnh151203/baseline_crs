#!/usr/bin/env bash
#SBATCH --job-name=scope-transition-import
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/u26anhvl/crs/scope/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/scope/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
assets="/mnt/data/u26anhvl/crs/scope/assets/transition_models"
python scripts/scope/import_transition_archives.py \
  --archive "${assets}/deterministic-20261016T193326Z-1-001.zip" \
  --archive "${assets}/deterministic-20261016T193326Z-1-002.zip" \
  --archive "${assets}/deterministic-20261016T193326Z-1-003.zip" \
  --output "${assets}/deterministic"
printf 'SCOPE transition import completed: %s\n' "$(date --iso-8601=seconds)"
