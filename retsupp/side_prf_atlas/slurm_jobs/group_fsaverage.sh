#!/bin/bash -l
#SBATCH --job-name=side_prf_atlas_group
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633,u24-chaiam0-613
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gdehol/logs/side_prf_atlas_group_%j.txt
# Group fsaverage maps + region stats for the side pRF atlas (after fit_signed_prf.sh).
set -euo pipefail
export TMPDIR=/scratch/$USER/side_prf_atlas_group_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONUNBUFFERED=1
cd "$HOME/retsupp_plan"
echo "host $(hostname) commit $(git rev-parse --short HEAD)"
exec "$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.side_prf_atlas.group_fsaverage \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
