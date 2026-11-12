#!/bin/bash
#SBATCH --job-name=cvl_nested
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4   # tasks used ~45-60% of 8 CPUs (sacct TotalCPU, 2026-11-06)
#SBATCH --time=08:00:00
#SBATCH --mem=28G            # peak RSS 8-23 GB in recent tasks
#
# Nested CV ladder (fit_cv_ladder_nested): one array task per (subject, ROI)
# runs the whole m00 -> m0 -> gain-field chain. CPU only. Resumable: models
# whose pickle exists are loaded, not refit.
#
# Usage (from the repo root on the cluster login node):
#   sbatch --array=1-330 retsupp/modeling/slurm_jobs/fit_cv_ladder_nested.sh
#   CV_SCHEME=run3 sbatch --array=1-330 ...
#   POLISH=1 sbatch --array=1-330 ...    # + all-free polish pass
#   POLISH_MODELS="m4_TDS m7_full" restricts the polish pass to these models; POLISH_METHOD=alt alternates Adam (pRFs)
#   and L-BFGS (shared terms); POLISH_SUFFIX=_x writes the polish to <root>_x (tests)
#   CV_SCHEME=full POLISH=1 sbatch --array=1-330 ...  # main fits
#   BOUNDED=1 boxes the shared terms (root gets a _bounded suffix)
#   LOSS=nll fits the shared terms by likelihood (root gets a _nll suffix)
#   BASE_ROOT="<path> ..." reuses the m00 and m0 fits of these roots, first hit wins
#   MAX_BOLD_SD=20 drops voxels above that BOLD SD (root gets a _maxsd20 suffix)
#   FAMILY=dn uses the divisive-normalization pRF (root gets a _dn suffix)
#   FAMILY=gauss uses a Gaussian pRF (DoG with surround amplitude 0; m00 from the joint-HRF Gaussian fold fits; _gauss suffix)
#   HP_ROTATION=K relabels HP K ring steps away (root gets a _hprot<K> suffix), needs BASE_ROOT
#   MODELS="m5_TDS_dS m7_full" fits only these models and their ancestors
#   MODELS=m8_full_dT or m8_full_adjopp (target split; LP gain split next to HP vs opposite) are fitted on
#     top of a finished ladder: BASE_ROOT=<that root> plus the same CV_SCHEME/BOUNDED/MAX_BOLD_SD/RELIABLE_ONLY/HRF_TABLE
#   PER_SUBJECT=1: array index = subject, all ROIS in one task (data loaded once)
#   WARM_ROOT="<path> ..." starts m00/m0 from earlier fits; GD_ITER, GD_MIN_ITER, SIGMA_STARTS, LBFGS_ROUNDS tune the search
#   ROI_LIST="V1 V2" restricts the ROIs (with PER_SUBJECT=1); OUTPUT_ROOT=<path> writes elsewhere (tests)
#   HRF_TABLE=<tsv> one HRF per subject x ROI (build_roi_hrf_table.py; root gets a _roihrf suffix)
#   FROZEN_PRF=1 with CV_SCHEME=run3 and BASE_ROOT=<all-runs root>: pRFs frozen at the all-runs fit, m7 + MODELS refitted per fold (_frozenprf)
#   RELIABLE_ONLY=1 keeps voxels with cross-run reliability > 0 (root gets a _reliable suffix), needs BASE_ROOT
# Task index -> (subject, ROI): idx0 = task-1; subject = idx0/11 + 1; roi = ROIS[idx0%11].

set -eo pipefail
CV_SCHEME="${CV_SCHEME:-loco}"
POLISH="${POLISH:-}"
BOUNDED="${BOUNDED:-}"
MAX_BOLD_SD="${MAX_BOLD_SD:-}"
LOSS="${LOSS:-sse}"
BASE_ROOT="${BASE_ROOT:-}"
FAMILY="${FAMILY:-dog}"
HP_ROTATION="${HP_ROTATION:-0}"
MODELS="${MODELS:-}"
TAG="${CV_SCHEME}${FROZEN_PRF:+_frozenprf}${BOUNDED:+_bounded}${POLISH:+_polish}${MAX_BOLD_SD:+_maxsd$MAX_BOLD_SD}_${LOSS}_${FAMILY}_hprot${HP_ROTATION}${RELIABLE_ONLY:+_reliable}${HRF_TABLE:+_roihrf}"
ROIS=(${ROI_LIST:-V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF})
if [[ -n "${PER_SUBJECT:-}" ]]; then
    subject=$SLURM_ARRAY_TASK_ID
    roi="${ROIS[*]}"
else
    idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
    subject=$(( idx0 / ${#ROIS[@]} + 1 ))
    roi="${ROIS[$(( idx0 % ${#ROIS[@]} ))]}"
fi

LOGFILE="$HOME/logs/cvl_nested_${TAG}_sub-$(printf %02d "$subject")_${roi// /-}_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} | sub-${subject} ${roi} | cv=${CV_SCHEME} polish=${POLISH:-0} | $(date)"
scontrol update jobid="${SLURM_JOB_ID}" name="cvl_nested_${TAG}" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2
export PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"

"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_cv_ladder_nested \
    "$subject" --roi $roi \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --cv-scheme "$CV_SCHEME" ${POLISH:+--polish} ${BOUNDED:+--bounded} \
    ${MAX_BOLD_SD:+--max-bold-sd "$MAX_BOLD_SD"} --loss "$LOSS" \
    ${BASE_ROOT:+--base-root $BASE_ROOT} --family "$FAMILY" --hp-rotation "$HP_ROTATION" \
    ${MODELS:+--models $MODELS} ${RELIABLE_ONLY:+--reliable-only} ${FROZEN_PRF:+--frozen-prf} \
    ${WARM_ROOT:+--warm-root $WARM_ROOT} ${GD_ITER:+--gd-iterations $GD_ITER} \
    ${GD_MIN_ITER:+--gd-min-iterations $GD_MIN_ITER} ${SIGMA_STARTS:+--sigma-starts $SIGMA_STARTS} \
    ${LBFGS_ROUNDS:+--lbfgs-rounds $LBFGS_ROUNDS} ${OUTPUT_ROOT:+--output-root $OUTPUT_ROOT} \
    ${HRF_TABLE:+--hrf-table $HRF_TABLE} ${POLISH_MODELS:+--polish-models $POLISH_MODELS} ${POLISH_METHOD:+--polish-method $POLISH_METHOD} ${POLISH_SUFFIX:+--polish-suffix $POLISH_SUFFIX}

echo "Finished: $(date)"
