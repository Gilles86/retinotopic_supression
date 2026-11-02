#!/bin/bash
#SBATCH --job-name=fixed_mech
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#
# Sufficient stats for the leave-one-subject-out test of every ladder mechanism
# (fit_fixed_group_mechanisms). One array task per subject, all 11 ROIs.
#
# Usage (from the repo root on the cluster login node):
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_fixed_group_mechanisms.sh
#   OVERLAY_ROOT=<derivatives folder> OUTPUT_SUBDIR=<new folder> sbatch --array=1-30 ...

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/fixed_mech_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
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

"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_fixed_group_mechanisms \
    "$subject" --bids-folder /shares/zne.uzh/gdehol/ds-retsupp ${ROIS:+--rois $ROIS} \
    ${OVERLAY_ROOT:+--overlay-root "$OVERLAY_ROOT"} ${OUTPUT_SUBDIR:+--output-subdir "$OUTPUT_SUBDIR"}

echo "Finished: $(date)"
