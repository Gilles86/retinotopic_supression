#!/bin/bash
#SBATCH --job-name=klein_shift_cv
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --constraint="L4"
#SBATCH --mem=16G

# SHIFT arm of the cross-validated shift-vs-gain comparison.
# Runs retsupp.modeling.fit_klein_shift_cv (DoGKleinShift_v3_target_6sigma),
# leave-one-condition-out CV, all 4 folds inside one job.
#
# GPU twin of fit_klein_shift_pilot_gpu.sh (same L4/16G, cuInit flock
# warm-up, CUDA libs, retsupp_cuda env). The forward model is the fast
# separable-Gaussian klein forward, so 45 min walltime is generous.
#
# (subject, ROI) decode from SLURM_ARRAY_TASK_ID:
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   sub_idx = idx0 // N_ROIS ;  roi_idx = idx0 % N_ROIS
#   subject = SUB_IDS[sub_idx]
#
# PILOT (3 subjects x 11 ROIs = 33 tasks; the default SUB_IDS below):
#   sbatch --array=1-33 fit_klein_shift_cv.sh
#
# FULL (30 subjects x 11 ROIs = 330 tasks). Throttle to avoid the NFS
# env-read dogpile on large arrays:
#   PILOT=0 sbatch --array=1-330%150 fit_klein_shift_cv.sh
#
# Override GPU type / host RAM at submit time if a bigger GPU is needed:
#   sbatch --constraint=A100 --mem=32G --array=1-33 fit_klein_shift_cv.sh

set -eo pipefail

LOGFILE="$HOME/logs/klein_shift_cv_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:           $(hostname)"
echo "Job:            ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:        $(date)"

# PILOT=1 (default): the 3-subject pilot set. PILOT=0: all 30 subjects.
PILOT="${PILOT:-1}"
if [[ "$PILOT" == "1" ]]; then
    SUB_IDS=(3 17 23)
else
    SUB_IDS=($(seq 1 30))
fi
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

echo "Mode:           PILOT=${PILOT} (N_SUBS=${N_SUBS}, N_ROIS=${N_ROIS})"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-33 (pilot) or --array=1-330 (full)." >&2
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

# SLURM Name field is array-shared; encode only sub (roi is recoverable
# from the _N array suffix).
sub_pad=$(printf "%02d" "$subject")
scontrol update jobid="${SLURM_JOB_ID}" \
    name="klein_cv_sub-${sub_pad}" 2>/dev/null || true

# NFS profile-read dogpile defense.
sleep $(( RANDOM % 15 ))

# CUDA runtime (TF 2.14 / cuDNN 8.7 ABI; CUDA 11.8 spack install).
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

# cuInit-race defense: per-node flock on /dev/shm.
LOCK="/dev/shm/cuinit_warm_$(hostname -s).flock"
echo "cuInit warm-up under flock $LOCK ..."
(
    flock -w 60 -x 200 || {
        echo "WARN: couldn't acquire $LOCK in 60s; skipping warm-up";
        exit 0;
    }
    "$PYTHON" -c "
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'cuInit warm-up: {len(gpus)} GPU(s) visible to TF', flush=True)
sys.exit(0 if gpus else 1)
"
) 200>"$LOCK"
WARMUP_RC=$?
if [[ $WARMUP_RC -ne 0 ]]; then
    echo "FATAL: cuInit warm-up returned $WARMUP_RC (no GPU after lock-protected init)." \
         " Node driver may be in a bad state from a prior cuInit race; exiting fast."
    exit 1
fi

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

echo "Running fit_klein_shift_cv (SHIFT arm, all 4 folds) for sub-${subject}, roi=${roi}"

"$PYTHON" -u -m retsupp.modeling.fit_klein_shift_cv \
    "$subject" \
    --roi "$roi" \
    --bids-folder "$bids_folder" \
    --max-voxels 500 \
    --p-signal-thr 0.5 \
    --max-n-iterations 1500 \
    --output-subdir af_prf_cv_shiftvsgain/shift

echo "Finished:       $(date)"
