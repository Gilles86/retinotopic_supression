#!/bin/bash
#SBATCH --job-name=af3models_pilot
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Pilot run of three-models AF sweep on sub-3, sub-17, sub-23 only.
# Identical recipe to fit_three_models.sh but with restricted SUB_IDS so
# the array size is 3 × 8 = 24 (validates an approach before scaling to
# all 30 subjects).
#
# Usage:
#   sbatch --array=1-24 retsupp/modeling/slurm_jobs/fit_three_models_pilot.sh analytical
#   sbatch --array=1-24 retsupp/modeling/slurm_jobs/fit_three_models_pilot.sh drive
#   sbatch --array=1-24 retsupp/modeling/slurm_jobs/fit_three_models_pilot.sh numerical

set -euo pipefail

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

LOGFILE="$HOME/logs/af3models_pilot_${MODEL_VERSION}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:           $(hostname)"
echo "Job:            ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:        $(date)"
echo "Model version:  ${MODEL_VERSION}"

# PILOT: only 3 subjects.
SUB_IDS=(3 17 23)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-24." >&2
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
    --sigma-dyn-init 2.0 \
    --with-target

echo "Finished:       $(date)"
