#!/usr/bin/env bash
#SBATCH --job-name=scope-llama-hash
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/u26anhvl/crs/scope/slurm/%x-%j.out
#SBATCH --error=/mnt/data/u26anhvl/crs/scope/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the tdrl project root}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dpdp
export PYTHONPATH="src:."
snapshot="/mnt/data/u26anhvl/crs/scope/assets/Meta-Llama-Guard-2-8B/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef"
python scripts/scope/build_llama_guard_manifest.py \
  --snapshot "$snapshot" \
  --output "${snapshot}/scope-asset-manifest.json"
printf 'SCOPE Llama Guard manifest completed: %s\n' "$(date --iso-8601=seconds)"
