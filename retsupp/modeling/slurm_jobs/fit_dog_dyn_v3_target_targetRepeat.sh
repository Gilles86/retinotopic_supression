#!/bin/bash
#SBATCH --job-name=dog_dyn_target_targetRepeat
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00

# v3 + target + sharedSigma + target-repeat split. The dynamic TARGET
# gain g_T_dyn is split into:
#   g_T_dyn         -> g_T_dyn_switch (parent slot 13, renamed)
#                    + g_T_dyn_repeat (new slot 15)
# A trial counts as target-repeat iff its TARGET is at the SAME ring
# location as the immediately preceding trial. Distractor gains
# (g_HP_dyn, g_LP_dyn) are NOT split (kept as the parent's single
# HP/LP pair).
#
# Output:
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_targetRepeat/sub-XX/...
#
# Submission: 30 subjects x 11 ROIs = 330 array tasks.
#   sbatch --array=1-330 retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target_targetRepeat.sh

set -euo pipefail

LOGFILE="$HOME/logs/dog_dyn_v3_target_targetRepeat_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        v3 + target + sharedSigma + target-repeat split"

SUB_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-330." >&2
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

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate retsupp_cuda
set -u

export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_dog_dynamic_af_braincoder.py"

echo "Running v3 + target + sharedSigma + target-repeat split for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --with-target \
    --shared-target-sigma \
    --with-target-repeat-split \
    --max-n-iterations 1500 \
    --p-signal-thr 0.5 \
    --mode signed

echo "Finished:    $(date)"
