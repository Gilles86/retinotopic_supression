#!/bin/bash
#SBATCH --job-name=prf_full_l4
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --constraint=L4
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# Whole-cortex PRF fit, all 6 model variants, smart-init pipeline,
# FULL 8-item paradigm. One job per subject (chunks all run inside).
#
# Pipeline order (each step inits from the previous):
#   1) Gaussian + fixed HRF        (grid + GD)
#   3) Gaussian + flexible HRF     (init from 1, GD only)
#   2) DoG + fixed HRF             (init from 1, GD only)
#   4) DoG + flexible HRF          (init from 3, GD only)   <- canonical
#   5) DN + fixed HRF              (init from 4, GD only)
#   6) DN + flexible HRF           (init from 4, GD only)
#
# Output:
#   derivatives/prf/model{N}/sub-XX/sub-XX_desc-{par}.nii.gz
#
# Submission: 30 subjects, one job each:
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_prf_l4.sh

set -euo pipefail

LOGFILE="$HOME/logs/prf_full_l4_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:     $(date)"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2
    exit 2
fi
subject="${SLURM_ARRAY_TASK_ID}"
echo "Subject:     ${subject}"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

# GPU: keep CUDA visible (we WANT the L4); just check it's there.
export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_prf.py"

echo "GPU sanity check:"
$PYTHON -c "import tensorflow as tf; print('  GPUs:', tf.config.list_physical_devices('GPU'))"

echo
echo "Running whole-cortex PRF (full paradigm, all 6 models) for sub-${subject}"
"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --resolution 50 \
    --voxel-chunk-size 10000 \
    --max-n-iterations 2000 \
    --paradigm-kind full

echo "Finished:    $(date)"
