#!/bin/bash
#SBATCH --job-name=learn_gains
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#
# Learning curves of the gain fields on top of the nested full-data fit
# (fit_learning_gains_nested). One array task per subject, all 11 ROIs.
#
# Usage (from the repo root on the cluster login node):
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_learning_gains_nested.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/learn_gains_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | sub-${subject} | $(date)"

export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2
export PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"

"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_learning_gains_nested \
    "$subject" --bids-folder /shares/zne.uzh/gdehol/ds-retsupp ${ROIS:+--rois $ROIS}

echo "Finished: $(date)"
