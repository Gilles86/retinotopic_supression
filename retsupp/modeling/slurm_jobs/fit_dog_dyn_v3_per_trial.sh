#!/bin/bash
#SBATCH --job-name=dog_dyn_perTrial
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Per-trial dynamic-gain joint AF + DoG-PRF fit
# (sharedSigma v3+target, with g_HP_dyn / g_LP_dyn / g_T_dyn replaced
#  by per-trial vectors of length n_trials).
#
# Output:
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_perTrial/sub-XX/...
#
# Submission: 28 subjects (skip sub-06, sub-08) x 8 ROIs = 224 array tasks.
#   sbatch --array=1-224%150 retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_per_trial.sh
#
# The array task id maps to (subject_idx, roi):
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = SUB_IDS[ idx0 // N_ROIS ]
#   roi     = ROIS[    idx0 %  N_ROIS ]

set -euo pipefail

# --- Logging. ---
LOGFILE="$HOME/logs/dog_dyn_perTrial_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        per-trial dyn gains (sharedSigma v3+target)"

# Defense-in-depth: spread out NFS profile reads across a 60-s window.
sleep $(( (RANDOM % 60) + 1 ))

# --- Subject + ROI decoding from SLURM_ARRAY_TASK_ID. ---
# Skip sub-06, sub-08 per project memory note (m4 PRFs missing).
SUB_IDS=(1 2 3 4 5 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-$((N_SUBS*N_ROIS))." >&2
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

# Rename job for nicer squeue display once we know subject/roi.
scontrol update jobid="${SLURM_JOB_ID}" \
    name="perTrial_sub-$(printf %02d ${subject})_roi-${roi}" 2>/dev/null || true

echo "Subject:     ${subject}"
echo "ROI:         ${roi}"

# --- Conda env (force CPU; per-trial fit benefits from many threads). ---
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_dog_dyn_v3_per_trial.py"

echo "Running per-trial dyn-gain fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 200 \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --max-n-iterations 1500

echo "Finished:    $(date)"
