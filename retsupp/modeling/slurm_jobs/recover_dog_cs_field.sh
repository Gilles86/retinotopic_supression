#!/bin/bash
#SBATCH --job-name=dogcs_recovery
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=6
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#
# Recovery of the centre/surround tilt κ of the DoG sustained field (recover_dog_cs_field): one task per
# (subject, ROI). SUBJECTS and ROIS (space-separated) define the grid; task index -> (subject, ROI), ROI fastest.
#
#   SUBJECTS="3 8 13 23" ROIS="V1 V3 V3AB hV4" sbatch --array=1-16 retsupp/modeling/slurm_jobs/recover_dog_cs_field.sh
set -eo pipefail
SUBJ=(${SUBJECTS:-3 8 13 23})
ROI_ARR=(${ROIS:-V1 V3 V3AB hV4})
idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
subject=${SUBJ[$(( idx0 / ${#ROI_ARR[@]} ))]}
roi=${ROI_ARR[$(( idx0 % ${#ROI_ARR[@]} ))]}
D=/shares/zne.uzh/gdehol/ds-retsupp/derivatives
ROOT="${ROOT:-$D/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf}"
HRF_TABLE="${HRF_TABLE:-$D/paper_2026-11/roi_hrf_dog_final2.tsv}"
OUT_DIR="${OUT_DIR:-$D/dog_cs_field_recovery_2026-11}"

LOGFILE="$HOME/logs/dogcs_recovery_sub-$(printf %02d "$subject")_${roi}_${SLURM_JOB_ID}.txt"
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
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.recover_dog_cs_field \
    "$subject" --roi "$roi" --root "$ROOT" --hrf-table "$HRF_TABLE" --out-dir "$OUT_DIR" --n-noise "${N_NOISE:-2}" \
    ${SIZES:+--sizes $SIZES}
echo "Finished: $(date)"
