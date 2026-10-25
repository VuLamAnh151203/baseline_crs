#!/usr/bin/env bash
#SBATCH --job-name=epl-assets
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/u26anhvl/crs/epl/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/epl/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
export HF_HUB_DISABLE_XET=1
asset_root="/mnt/data/u26anhvl/crs/epl/models"
mkdir -p -- "$asset_root" /mnt/data/u26anhvl/crs/epl/slurm
python scripts/epl/bootstrap_assets.py \
  --output "$asset_root" \
  --lock "$asset_root/epl.models.lock.json"
