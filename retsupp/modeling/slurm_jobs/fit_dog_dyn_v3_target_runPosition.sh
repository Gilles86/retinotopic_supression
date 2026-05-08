#!/bin/bash
#SBATCH --job-name=dog_dyn_target_rP
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# v3 + target + sharedSigma joint AF + DoG-PRF braincoder fit, with
# per-run-position SUSTAINED gains. The single (g_HP, g_LP) pair is
# replaced by 6 parameters (g_{HP,LP}_pos{0,1,2}) — one pair per
# chronological position of the run within its HP block. Tests for
# learning over the 3-run HP blocks (VSS2026 talk).
#
# σ_T_dyn := σ_dyn (sharedSigma); the dynamic-distractor and
# target-onset gains are unchanged from the sharedSigma fit.
#
# Output:
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_runPosition/sub-XX/...
#
# Submission: 30 subjects x 8 ROIs = 240 array tasks.
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target_runPosition.sh
#
# The array task id maps to (subject_idx, roi):
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = SUB_IDS[ idx0 // 8 ]
#   roi     = ROIS[    idx0 %  8 ]

set -euo pipefail

# --- Logging. ---
LOGFILE="$HOME/logs/dog_dyn_v3_target_runPosition_${SLURM_JOB_NAME:-target_rP}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        v3 + target + sharedSigma + per-run-position gains"

# --- Subject + ROI decoding from SLURM_ARRAY_TASK_ID. ---
SUB_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-240." >&2
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

# --- Conda env (CUDA env, forced to CPU per fit_dog_dynamic_af_braincoder_cpu.sh). ---
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

echo "Running v3 + target + sharedSigma + runPosition fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --with-target \
    --shared-target-sigma \
    --per-run-position-gains \
    --sigma-af-init 5.0 \
    --sigma-dyn-init 5.0 \
    --sigma-t-dyn-init 5.0 \
    --g-t-dyn-init 0.0

echo "Finished:    $(date)"
