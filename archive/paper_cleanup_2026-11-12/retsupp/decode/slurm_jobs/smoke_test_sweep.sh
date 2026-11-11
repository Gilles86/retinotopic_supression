#!/bin/bash
# Hyperparameter sweep for StimulusFitter on one (subject, ROI, run).
# Runs the full grid (l2 x lr) sequentially in a single job.
#
# Submit:
#   sbatch retsupp/decode/slurm_jobs/smoke_test_sweep.sh           # defaults: sub-02 V1 ses-1 run-1
#   sbatch retsupp/decode/slurm_jobs/smoke_test_sweep.sh 5 V1 1 1  # sub-05 V1 ses-1 run-1
#
# Outputs land in ~/git/retsupp/notes/ -- rsync back after the job:
#   rsync -av sciencecluster:git/retsupp/notes/data/decode_sweep/ \
#             /Users/gdehol/git/retsupp/notes/data/decode_sweep/
#   rsync -av sciencecluster:git/retsupp/notes/figures/decode_sweep/ \
#             /Users/gdehol/git/retsupp/notes/figures/decode_sweep/
#SBATCH --job-name=decode_sweep
#SBATCH --output=/dev/null
#SBATCH --time=120:00
#SBATCH --account=zne.uzh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

SUBJECT="${1:-2}"
ROI="${2:-V1}"
SESSION="${3:-1}"
RUN="${4:-1}"

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-decode_sweep}_sub-${SUBJECT}_${ROI}_ses-${SESSION}_run-${RUN}_${SLURM_JOB_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="decode_sweep_sub-$(printf %02d "$SUBJECT")_${ROI}_ses-${SESSION}_run-${RUN}" \
    2>/dev/null || true

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2

export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

echo "[$(date)] sub-${SUBJECT} ${ROI} ses-${SESSION} run-${RUN} on $(hostname)"
echo "BIDS: ${bids_folder}"

python -u -m retsupp.decode.smoke_test_sweep \
    --bids-folder "$bids_folder" \
    --subject "$SUBJECT" \
    --roi "$ROI" \
    --session "$SESSION" \
    --run "$RUN"

echo "[$(date)] done"
