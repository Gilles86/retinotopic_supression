#!/bin/bash
#SBATCH --job-name=cv_ladder
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=48G  # large-brain subjects (e.g. sub-08) OOM at 16G loading full BOLD

# One arm of the nested gain-field model comparison (leave-one-HP-condition-
# out CV, 4 folds inside one job). Submit via submit_cv_ladder.sh, which sets
# the env vars below. Identical voxels / folds / HRF across arms.
#
# Required env: MODEL SUS_HP SUS_LP DYN_HP DYN_LP TGT
# Optional env: ITEM_SHAPE (rectangle) ITEM_SCALE (free) PILOT (0) CV_SCHEME (loco|run3)
#
# (subject, ROI) decode from SLURM_ARRAY_TASK_ID:
#   idx0 = SLURM_ARRAY_TASK_ID - 1; sub = idx0 // N_ROIS; roi = idx0 % N_ROIS

set -eo pipefail

: "${MODEL:?MODEL not set}" "${SUS_HP:?}" "${SUS_LP:?}" "${DYN_HP:?}" "${DYN_LP:?}" "${TGT:?}"
ITEM_SHAPE="${ITEM_SHAPE:-rectangle}"
ITEM_SCALE="${ITEM_SCALE:-free}"
CV_SCHEME="${CV_SCHEME:-loco}"
if [[ "$CV_SCHEME" == "loco" ]]; then
    OUT_ROOT="af_prf_cv_ladder"
else
    OUT_ROOT="af_prf_cv_ladder_${CV_SCHEME}"
fi

LOGFILE="$HOME/logs/cv_ladder_${MODEL}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:           $(hostname)"
echo "Job:            ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:        $(date)"
echo "Model:          ${MODEL}: sus=(${SUS_HP},${SUS_LP}) dyn=(${DYN_HP},${DYN_LP}) tgt=${TGT} items=${ITEM_SHAPE}/${ITEM_SCALE} cv=${CV_SCHEME}"

PILOT="${PILOT:-0}"
if [[ "$PILOT" == "1" ]]; then
    SUB_IDS=(3 17 23)
else
    SUB_IDS=($(seq 1 30))
fi
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
    exit 2
fi
idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_idx=$(( idx0 / N_ROIS ))
roi_idx=$(( idx0 % N_ROIS ))
if [[ "$sub_idx" -ge "$N_SUBS" ]]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range; max is $(( N_SUBS * N_ROIS ))." >&2
    exit 2
fi
subject="${SUB_IDS[$sub_idx]}"
roi="${ROIS[$roi_idx]}"
echo "Subject:        ${subject}"
echo "ROI:            ${roi}"

scontrol update jobid="${SLURM_JOB_ID}" name="cvl_${CV_SCHEME}_${MODEL}" 2>/dev/null || true

# NFS profile-read dogpile defense.
sleep $(( RANDOM % 15 ))

source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

set +u
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
# Code root = the checkout/snapshot that was submitted from (slurm_jobs/../../..).
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}/../../.." && pwd)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code dir:       ${CODE_DIR}"

if [[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]]; then
# cuInit-race defense: per-node flock on /dev/shm.
LOCK="/dev/shm/cuinit_warm_$(hostname -s).flock"
(
    flock -w 60 -x 200 || { echo "WARN: couldn't acquire $LOCK in 60s"; exit 0; }
    "$PYTHON" -c "
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'cuInit warm-up: {len(gpus)} GPU(s) visible to TF', flush=True)
sys.exit(0 if gpus else 1)
"
) 200>"$LOCK"
if [[ $? -ne 0 ]]; then
    echo "FATAL: cuInit warm-up failed; exiting fast."
    exit 1
fi
else
    echo "CPU mode: ${SLURM_CPUS_PER_TASK:-?} cores"
    export CUDA_VISIBLE_DEVICES=""
    export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export TF_NUM_INTEROP_THREADS=2
fi

"$PYTHON" -u -m retsupp.modeling.fit_af_prf_cv_v2 \
    "$subject" \
    --roi "$roi" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --sus-hp-sign "$SUS_HP" \
    --sus-lp-sign "$SUS_LP" \
    --dyn-hp-sign "$DYN_HP" \
    --dyn-lp-sign "$DYN_LP" \
    --target-gain "$TGT" \
    --item-shape "$ITEM_SHAPE" \
    --item-scale "$ITEM_SCALE" \
    --resolution 50 \
    --max-voxels 500 \
    --p-signal-thr 0.5 \
    --max-n-iterations 1500 \
    --cv-scheme "$CV_SCHEME" \
    --output-subdir "${OUT_ROOT}/${MODEL}"

echo "Finished:       $(date)"
