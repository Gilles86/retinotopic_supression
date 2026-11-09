#!/bin/bash
#SBATCH --job-name=dnbase_recovery
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=6
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#
# Model and parameter recovery for the DN baseline-field variants (recover_dn_baseline_field): one task per
# (subject, ROI). SUBJECTS and ROIS (space-separated) define the grid; task index -> (subject, ROI), ROI fastest.
#
#   SUBJECTS="3 8 13 23" ROIS="V1 V3 V3AB hV4" sbatch --array=1-16 retsupp/modeling/slurm_jobs/recover_dn_baseline_field.sh
set -eo pipefail
SUBJ=(${SUBJECTS:-3 8 13 23})
ROI_ARR=(${ROIS:-V1 V3 V3AB hV4})
idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
subject=${SUBJ[$(( idx0 / ${#ROI_ARR[@]} ))]}
roi=${ROI_ARR[$(( idx0 % ${#ROI_ARR[@]} ))]}
ROOT="${ROOT:-/shares/zne.uzh/gdehol/ds-retsupp/derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_dn}"
OUT_DIR="${OUT_DIR:-/shares/zne.uzh/gdehol/ds-retsupp/derivatives/dn_baseline_field_recovery_2026-11}"

LOGFILE="$HOME/logs/dnbase_recovery_sub-$(printf %02d "$subject")_${roi}_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} | sub-${subject} ${roi} | $(date)"
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TF_NUM_INTEROP_THREADS=2
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.recover_dn_baseline_field \
    "$subject" --roi "$roi" --root "$ROOT" --out-dir "$OUT_DIR" --n-noise "${N_NOISE:-2}" \
    ${SIZES:+--sizes $SIZES}
echo "Finished: $(date)"
