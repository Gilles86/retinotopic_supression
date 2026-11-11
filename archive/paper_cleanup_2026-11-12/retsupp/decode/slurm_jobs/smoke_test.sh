#!/bin/bash
# Smoke test: sub-02 V1 ses-1 run-1, low-res, fast.
#   sbatch retsupp/decode/slurm_jobs/smoke_test.sh
#SBATCH --job-name=decode_smoke
#SBATCH --output=/dev/null
#SBATCH --time=30:00
#SBATCH --account=zne.uzh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-decode_smoke}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2

export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

cd $HOME/git/retsupp

python -u -m retsupp.decode.smoke_test \
    --bids-folder "$bids_folder" \
    --subject 2 --roi V1 \
    --session 1 --run 1 \
    --resolution 30 --max-voxels 200 \
    --max-n-iterations 600 --resid-max-iter 300

echo "[$(date)] done"
