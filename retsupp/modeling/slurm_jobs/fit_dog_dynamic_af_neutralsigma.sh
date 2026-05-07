#!/bin/bash
#SBATCH --job-name=dog_dyn_neut
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# σ-init robustness test for DoG dyn-v3.
#
# Refits a small subset of subjects with NEUTRAL inits:
#   sigma_AF  = 2.0
#   sigma_dyn = 2.0   (matches sigma_AF; original default was 0.5)
#
# Original fits used sigma_AF=2.0 and sigma_dyn=0.5, which biased the
# optimizer toward σ_dyn < σ_AF. With both inits equal at 2.0, the
# optimizer must choose σ_AF vs σ_dyn from the data alone.
#
# If the converged σ values still show σ_AF >> σ_dyn, the original
# finding is robust to init. If they end up similar, the original
# σ_AF >> σ_dyn was driven by the init, not the data.
#
# Output goes to a SEPARATE derivatives directory so the original fits
# are untouched:
#   derivatives/af_prf_joint_dynamic_v3_dog_neutralsigma/sub-XX/...
#
# Submission (5 subjects x 8 ROIs = 40 array tasks):
#   sbatch --array=1-40 retsupp/modeling/slurm_jobs/fit_dog_dynamic_af_neutralsigma.sh
#
# Subject set is hard-coded below: SUB_IDS=(2 5 11 18 25). Edit if you
# want a different sample.
#
# The array task id maps to (subject_idx, roi):
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = SUB_IDS[ idx0 // 8 ]
#   roi     = ROIS[    idx0 %  8 ]

set -euo pipefail

# --- Logging. ---
LOGFILE="$HOME/logs/dog_dyn_neutralsigma_${SLURM_JOB_NAME:-swap}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        sigma init robustness — sigma_AF=2.0, sigma_dyn=2.0 (neutral)"

# --- Subject + ROI decoding from SLURM_ARRAY_TASK_ID. ---
SUB_IDS=(2 5 11 18 25)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-40." >&2
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

echo "Subject:     ${subject}"
echo "ROI:         ${roi}"

# --- Conda env (CUDA env, forced to CPU). ---
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_dog_dynamic_af_braincoder.py"

echo "Running neutral-sigma robustness fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --output-subdir af_prf_joint_dynamic_v3_dog_neutralsigma

echo "Finished:    $(date)"
