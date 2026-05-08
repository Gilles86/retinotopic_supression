#!/bin/bash
#SBATCH --job-name=af3models
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Three-models AF sweep on a Gaussian-PRF backbone.
#
# Takes <model-version> as a positional argument: drive | analytical |
# numerical (see fit_three_models.py for details).
#
# Output:
#   derivatives/af_three_models/{drive,analytical,numerical}/sub-XX/...
#
# Submission:
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_three_models.sh drive
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_three_models.sh analytical
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_three_models.sh numerical
#
# The array task id maps to (subject_idx, roi):
#   idx0    = SLURM_ARRAY_TASK_ID - 1
#   subject = SUB_IDS[ idx0 // 8 ]
#   roi     = ROIS[    idx0 %  8 ]

set -euo pipefail

# --- Argument parsing. ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-version>   (drive | analytical | numerical)" >&2
    exit 2
fi
MODEL_VERSION="$1"
case "$MODEL_VERSION" in
    drive|analytical|numerical) ;;
    *)
        echo "ERROR: model-version must be drive | analytical | numerical, "\
             "got '$MODEL_VERSION'" >&2
        exit 2
        ;;
esac

# --- Logging. ---
LOGFILE="$HOME/logs/af3models_${MODEL_VERSION}_${SLURM_JOB_NAME:-af3models}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:           $(hostname)"
echo "Job:            ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:        $(date)"
echo "Model version:  ${MODEL_VERSION}"

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

echo "Subject:        ${subject}"
echo "ROI:            ${roi}"

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
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_three_models.py"

echo "Running ${MODEL_VERSION} fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --model-version "$MODEL_VERSION" \
    --max-voxels 500 \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0

echo "Finished:       $(date)"
