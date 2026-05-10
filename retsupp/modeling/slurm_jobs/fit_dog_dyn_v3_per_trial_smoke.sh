#!/bin/bash
#SBATCH --job-name=perTrial_smoke
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Smoke test: per-trial dynamic-gain fit on sub-02 V1 with 100 voxels
# and 800 GD iterations. Validates that the model converges and the
# output TSVs are well-formed before scaling to the full 28x8 array.

set -euo pipefail

LOGFILE="$HOME/logs/perTrial_smoke_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID}"
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
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_dog_dyn_v3_per_trial.py"

"$PYTHON" -u "$SCRIPT" \
    2 \
    --bids-folder "$bids_folder" \
    --roi V1 \
    --max-voxels 100 \
    --max-n-iterations 800

echo "Finished: $(date)"
