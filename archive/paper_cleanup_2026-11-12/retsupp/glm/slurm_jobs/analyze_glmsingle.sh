#!/bin/bash
#SBATCH --job-name=glmsingle_analyze
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Run analyze_glmsingle.py on all 30 subjects.
# Loads ~10 GB single-trial pe arrays per subject; do this on a fat
# compute node, not the login node.

set -euo pipefail

LOGFILE="$HOME/logs/glmsingle_analyze_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

cd "$HOME/git/retsupp"
~/data/conda/envs/retsupp_cuda/bin/python -m retsupp.glm.analyze_glmsingle \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --out notes/figures/glmsingle.pdf \
    --tsv-out notes/data/glmsingle_summary.tsv

echo "Finished: $(date)"
