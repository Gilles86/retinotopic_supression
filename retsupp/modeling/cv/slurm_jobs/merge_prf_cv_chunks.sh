#!/bin/bash
#SBATCH --job-name=prf_cv_merge
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Merge CV chunks → per-(sub, model, fold) r2_test NIfTI.
# Submitted by submit_cv_persub.sh with --array=$subject and
# --dependency=afterok on the chunk array.
#
# Required env: MODEL, FOLD

set -eo pipefail
sub_pad=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
subject="$SLURM_ARRAY_TASK_ID"

LOGFILE="$HOME/logs/prf_cv_merge_m${MODEL}_f${FOLD}_sub-${sub_pad}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_cv_merge_m${MODEL}_f${FOLD}_sub-${sub_pad}" 2>/dev/null || true

echo "Host: $(hostname) | sub-${subject} | model ${MODEL} | fold ${FOLD}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

$PYTHON -u -m retsupp.modeling.cv.merge_prf_cv_chunks \
    "$subject" --model "$MODEL" --fold "$FOLD" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp

echo "Finished: $(date)"
