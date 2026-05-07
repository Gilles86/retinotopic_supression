#!/bin/bash
#SBATCH --job-name=dog_dyn_target
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# v3 + target ('phasic capture') joint AF + DoG-PRF braincoder fit.
#
# Extends the v3 model with a 5th spatial-modulation term: a phasic
# TARGET-onset gain. The contrast g_HP_dyn < 0 (suppression at the
# distractor) vs g_T_dyn > 0 (capture at the target) is a positive
# control validating the AF framework.
#
# Adds 2 new shared parameters: g_T_dyn, sigma_T_dyn (8 shared total
# instead of 6).
#
# Output:
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target/sub-XX/...
#
# Initial test set: 5 subjects x 8 ROIs = 40 array tasks.
#   sbatch --array=1-40 retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target.sh
#
# Subject set is hard-coded below: SUB_IDS=(2 5 11 18 25). Edit if
# you want a different sample.
#
# The array task id maps to (subject_idx, roi):
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = SUB_IDS[ idx0 // 8 ]
#   roi     = ROIS[    idx0 %  8 ]

set -euo pipefail

# --- Logging. ---
LOGFILE="$HOME/logs/dog_dyn_v3_target_${SLURM_JOB_NAME:-target}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"
echo "Test:        v3 + target (phasic capture) — g_T_dyn, sigma_T_dyn"

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

echo "Running v3 + target fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --with-target \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --sigma-t-dyn-init 2.0 \
    --g-t-dyn-init 0.0 \
    --output-subdir af_prf_joint_dynamic_v3_dog_with_target

echo "Finished:    $(date)"
