#!/bin/bash
#SBATCH --job-name=gauss_dyn_target_allShS
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# v3 + target + ALL-sharedSigma joint AF + GAUSSIAN-PRF braincoder fit.
# Uses model 1 (Gaussian PRF, fixed HRF) parameters as the per-voxel
# encoder. Stricter variant of the sharedSigma fit: ALL three Gaussian
# widths (sigma_AF, sigma_dyn, sigma_T_dyn) are tied to a single shared
# σ (= sigma_dyn slot 8). Reduces unidentifiability and should sharpen
# the gain estimates (g_HP, g_HP_dyn, g_T_dyn).
#
# Phantom filter: voxels with R² >= 0.999 are dropped from the
# candidate pool.
#
# 8 nominal shared parameters; effective 6 (σ_AF and σ_T_dyn are
# overridden to σ_dyn every iteration):
#   sigma_AF (:= sigma_dyn), g_HP, g_LP,
#   sigma_dyn, g_HP_dyn, g_LP_dyn,
#   g_T_dyn, sigma_T_dyn (:= sigma_dyn).
# Per-voxel: x, y, sd, baseline, amplitude (5 params; no DoG surround).
#
# Output:
#   derivatives/af_prf_joint_dynamic_v3_gaussian_with_target_allSharedSigma/sub-XX/...
#
# Submission: 25 working subjects x 8 ROIs = 200 array tasks.
# Excludes:
#   - subjects 19, 22, 24, 27, 30 (broken neuropythy)
#   sbatch --array=1-200%80 retsupp/modeling/slurm_jobs/fit_gaussian_dyn_v3_target_allSharedSigma.sh
#
# Note: subjects 6, 8 were excluded earlier (broken m1 chunks); now recovered.
# Their positions in SUB_IDS:
#   sub-06 at sub_idx=5 (0-idx) -> array tasks 41..48
#   sub-08 at sub_idx=7 (0-idx) -> array tasks 57..64
#
# The array task id maps to (subject_idx, roi):
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = SUB_IDS[ idx0 // N_ROIS ]
#   roi     = ROIS[    idx0 %  N_ROIS ]

set -euo pipefail

# --- Logging. ---
LOGFILE="$HOME/logs/gauss_dyn_v3_target_allSharedSigma_${SLURM_JOB_NAME:-target_allShS}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        Gaussian (m1) + v3 + target + allSharedSigma"

# --- Subject + ROI decoding from SLURM_ARRAY_TASK_ID. ---
# Excludes broken subjects: 19, 22, 24, 27, 30 (neuropythy).
SUB_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-${N_SUBS}*${N_ROIS}." >&2
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

# Rename job in squeue for clarity.
scontrol update jobid="${SLURM_JOB_ID}" \
    name="gauss_dyn_allShS_sub-$(printf %02d "$subject")_${roi}" 2>/dev/null || true

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
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_gaussian_dynamic_af_braincoder.py"

echo "Running Gaussian (m1) + v3 + target + allSharedSigma fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-label 1 \
    --r2-thr 0.05 \
    --r2-max 0.999 \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --g-t-dyn-init 0.0 \
    --all-shared-sigma

echo "Finished:    $(date)"
