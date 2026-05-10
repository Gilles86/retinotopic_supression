#!/bin/bash
# Fast linear-filter decode for all 28 subjects, single CPU node.
#   sbatch retsupp/decode/slurm_jobs/decode_drive_fast.sh
#SBATCH --job-name=decode_drive_fast
#SBATCH --output=/dev/null
#SBATCH --time=120:00
#SBATCH --account=zne.uzh
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-decode_drive_fast}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2

export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
out="$bids_folder/derivatives/decode/decoded_drive.tsv"
mkdir -p "$(dirname "$out")"

cd "$HOME/git/retsupp"

python -u -m retsupp.decode.decode_drive_fast \
    --bids-folder "$bids_folder" \
    --out "$out" \
    --n-jobs 4

echo "[$(date)] decoding done; making figure"

python -u -m retsupp.decode.plot_decoded_drive \
    --tsv "$out" \
    --out "notes/figures/decoded_drive_HP_vs_LP.pdf" \
    --repo-root "$HOME/git/retsupp"

echo "[$(date)] done"
