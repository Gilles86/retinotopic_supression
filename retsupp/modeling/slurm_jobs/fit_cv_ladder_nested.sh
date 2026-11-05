#!/bin/bash
#SBATCH --job-name=cvl_nested
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00  # m00+m0 long GD ~3.5h + gain models ~1h on 8 CPUs (resumable)
#SBATCH --mem=48G  # large-brain subjects (e.g. sub-08) OOM at 16G loading full BOLD
#
# Nested CV ladder (fit_cv_ladder_nested): one array task per (subject, ROI)
# runs the whole m00 -> m0 -> gain-field chain. CPU only. Resumable: models
# whose pickle exists are loaded, not refit.
#
# Usage (from the repo root on the cluster login node):
#   sbatch --array=1-330 retsupp/modeling/slurm_jobs/fit_cv_ladder_nested.sh
#   CV_SCHEME=run3 sbatch --array=1-330 ...
#   POLISH=1 sbatch --array=1-330 ...    # + all-free polish pass
#   CV_SCHEME=full POLISH=1 sbatch --array=1-330 ...  # main fits
#   BOUNDED=1 boxes the shared terms (root gets a _bounded suffix)
#   LOSS=nll fits the shared terms by likelihood (root gets a _nll suffix)
#   BASE_ROOT="<path> ..." reuses the m00 and m0 fits of these roots, first hit wins
#   MAX_BOLD_SD=20 drops voxels above that BOLD SD (root gets a _maxsd20 suffix)
#   FAMILY=dn uses the divisive-normalization pRF (root gets a _dn suffix)
#   HP_ROTATION=K relabels HP K ring steps away (root gets a _hprot<K> suffix), needs BASE_ROOT
#   MODELS="m5_TDS_dS m7_full" fits only these models and their ancestors
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
TAG="${CV_SCHEME}${BOUNDED:+_bounded}${POLISH:+_polish}${MAX_BOLD_SD:+_maxsd$MAX_BOLD_SD}_${LOSS}_${FAMILY}_hprot${HP_ROTATION}${RELIABLE_ONLY:+_reliable}"
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
subject=$(( idx0 / ${#ROIS[@]} + 1 ))
roi="${ROIS[$(( idx0 % ${#ROIS[@]} ))]}"

LOGFILE="$HOME/logs/cvl_nested_${TAG}_sub-$(printf %02d "$subject")_${roi}_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} | sub-${subject} ${roi} | cv=${CV_SCHEME} polish=${POLISH:-0} | $(date)"
scontrol update jobid="${SLURM_JOB_ID}" name="cvl_nested_${TAG}" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2
export PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"

"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_cv_ladder_nested \
    "$subject" --roi "$roi" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --cv-scheme "$CV_SCHEME" ${POLISH:+--polish} ${BOUNDED:+--bounded} \
    ${MAX_BOLD_SD:+--max-bold-sd "$MAX_BOLD_SD"} --loss "$LOSS" \
    ${BASE_ROOT:+--base-root $BASE_ROOT} --family "$FAMILY" --hp-rotation "$HP_ROTATION" \
    ${MODELS:+--models $MODELS} ${RELIABLE_ONLY:+--reliable-only}

echo "Finished: $(date)"
