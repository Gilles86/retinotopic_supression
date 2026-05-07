#!/bin/bash
#SBATCH --job-name=cond_gauss
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00

# Fast conditionwise GAUSSIAN PRF refit — apples-to-apples voxel
# kernel match for the joint AF+PRF model. Inits from model 4 mean
# fits (already on cluster), restricts to retinotopic ROI voxels.
#
# CPU only. ~2 min per subject thanks to ROI restriction.
#
# Submit:
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_condition_gaussian_roi.sh
#
# Output:
#   derivatives/prf_conditionfit/model1/sub-XX/sub-XX_gaussian_roi.tsv

set -euo pipefail

LOGFILE="$HOME/logs/cond_gauss_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: --array required (1..30)"; exit 2
fi
subject="${SLURM_ARRAY_TASK_ID}"
echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
echo "Subject: ${subject}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

"$PYTHON" -u \
    "$HOME/git/retsupp/retsupp/modeling/fit_condition_gaussian_roi.py" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --init-model 4 \
    --resolution 50 \
    --save-nifti

echo "Finished: $(date)"
