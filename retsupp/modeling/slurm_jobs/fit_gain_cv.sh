#!/bin/bash
#SBATCH --job-name=gain_cv
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --constraint="L4"
#SBATCH --mem=48G  # 48G: large-brain subjects (e.g. sub-08) OOM at 16G loading full BOLD

# GAIN arm of the cross-validated shift-vs-gain comparison.
# Runs retsupp.modeling.fit_af_prf_cv_v2 with ALL FIVE gains FREE
# (multiplicative-AF target-matched model), leave-one-condition-out CV,
# all 4 folds inside one job.
#
# GPU L4 twin of fit_klein_shift_cv.sh — IDENTICAL array geometry, ROI
# list, --max-voxels, and compute so the two arms are matched. CV-v2's
# own slurm (fit_af_prf_cv_v2.sh) runs CPU/standard and only 8 ROIs; this
# twin uses the same 11-ROI L4 setup as the shift arm for fairness.
#
# CV-v2 already uses a FIXED canonical SPMHRF (delay=4.5, dispersion=0.75)
# with flexible_hrf_parameters left at its False default, so NO HRF tweak
# is needed to match the shift arm — both fix the same canonical HRF.
#
# (subject, ROI) decode from SLURM_ARRAY_TASK_ID:
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   sub_idx = idx0 // N_ROIS ;  roi_idx = idx0 % N_ROIS
#   subject = SUB_IDS[sub_idx]
#
# PILOT (3 subjects x 11 ROIs = 33 tasks; default SUB_IDS below):
#   sbatch --array=1-33 fit_gain_cv.sh
#
# FULL (30 subjects x 11 ROIs = 330 tasks):
#   PILOT=0 sbatch --array=1-330%150 fit_gain_cv.sh

set -eo pipefail

LOGFILE="$HOME/logs/gain_cv_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
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

sub_pad=$(printf "%02d" "$subject")
scontrol update jobid="${SLURM_JOB_ID}" \
    name="gain_cv_sub-${sub_pad}" 2>/dev/null || true

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

echo "Running fit_af_prf_cv_v2 (GAIN arm, all gains free, all 4 folds) for sub-${subject}, roi=${roi}"

"$PYTHON" -u -m retsupp.modeling.fit_af_prf_cv_v2 \
    "$subject" \
    --roi "$roi" \
    --bids-folder "$bids_folder" \
    --sus-hp-sign free \
    --sus-lp-sign free \
    --dyn-hp-sign free \
    --dyn-lp-sign free \
    --target-gain free \
    --resolution 50 \
    --max-voxels 500 \
    --p-signal-thr 0.5 \
    --max-n-iterations 1500 \
    --output-subdir af_prf_cv_shiftvsgain/gain

echo "Finished:       $(date)"
