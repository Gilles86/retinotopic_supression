#!/bin/bash
#SBATCH --job-name=test_gauss_af_sub01_V1
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00

# One-off test: run the new Gaussian (m1) AF + v3 + target + sharedSigma
# fit on sub-01 V1 with a reduced voxel cap so it finishes in ~10 min.
# Used to validate the script before launching the full 184-task array.

set -euo pipefail
LOGFILE="$HOME/logs/test_gauss_af_sub01_V1_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=2

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_gaussian_dynamic_af_braincoder.py"

"$PYTHON" -u "$SCRIPT" 1 \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --roi V1 \
    --resolution 50 \
    --max-voxels 100 \
    --max-n-iterations 400 \
    --model-label 1 \
    --r2-thr 0.05 \
    --r2-max 0.999

echo "Finished: $(date)"
