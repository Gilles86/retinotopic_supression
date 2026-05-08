#!/bin/bash
#SBATCH --job-name=prf_merge
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Merge per-chunk NPZ files into final per-parameter NIfTIs.
# One task per subject. Submit AFTER the chunked fit array completes
# (use --dependency=afterok:JOB_ID).
#
# Usage:
#   sbatch --array=1-30 --export=ALL,MODEL=1 \
#          --dependency=afterok:CHUNKED_JOB_ID \
#          retsupp/modeling/slurm_jobs/merge_prf_chunks.sh

set -euo pipefail

LOGFILE="$HOME/logs/prf_merge_m${MODEL:-?}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${MODEL:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID and MODEL required." >&2; exit 2
fi
subject="${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) | sub-${subject} | model ${MODEL}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/merge_prf_chunks.py" \
    "$subject" --model "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp

echo "Finished: $(date)"
