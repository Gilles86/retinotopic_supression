#!/bin/bash
#SBATCH --job-name=fixed_mech
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=18G   # dS+dD on 9 ROIs: 4 min, 1 CPU, 14.7 GB peak (sub-03, 2026-11-07)
#
# Sufficient stats for the leave-one-subject-out test of every ladder mechanism
# (fit_fixed_group_mechanisms). One array task per subject, all 11 ROIs.
#
# Usage (from the repo root on the cluster login node):
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_fixed_group_mechanisms.sh
#   OVERLAY_ROOT=<derivatives folder> OUTPUT_SUBDIR=<new folder> sbatch --array=1-30 ...
#   MECHANISMS=dT runs only the listed mechanisms
#   HRF_TABLE=<tsv> one HRF per subject x ROI (must match the HRF of the base fits)
#   HP_ROTATION=K relabels HP K ring steps (null of the conditional dS, dD); SESSIONS="1" uses session 1 only; RUN_PARITY=odd|even odd or even runs

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
    ${OVERLAY_ROOT:+--overlay-root "$OVERLAY_ROOT"} ${OVERLAY_PER_MODEL:+--overlay-per-model} ${OUTPUT_SUBDIR:+--output-subdir "$OUTPUT_SUBDIR"} \
    ${BASE_ROOT:+--base-root "$BASE_ROOT"} ${FAMILY:+--family "$FAMILY"} ${STEP:+--step "$STEP"} ${EXACT:+--exact} \
    ${MECHANISMS:+--mechanisms $MECHANISMS} ${HP_ROTATION:+--hp-rotation $HP_ROTATION} ${SESSIONS:+--sessions $SESSIONS} ${RUN_PARITY:+--run-parity $RUN_PARITY} \
    ${HRF_TABLE:+--hrf-table $HRF_TABLE}

echo "Finished: $(date)"
