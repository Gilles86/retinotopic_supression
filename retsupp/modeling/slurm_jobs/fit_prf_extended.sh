#!/bin/bash
#SBATCH --job-name=prf_extended
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --time=60:00
#SBATCH --constraint=A100
#SBATCH --mem=64G

# Usage:
#   sbatch fit_prf_extended.sh <subject> <model>
#
# Submits one GPU fit of the extended-design PRF model for the given
# subject and model label.  Routes stdout/stderr to ~/logs by name+jobid.

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-prf_extended}_sub-${1}_model-${2}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

# Cluster-specific conda activation.  neural_priors2 has braincoder + GPU TF.
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2

export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

subject=$1
model=$2

nvidia-smi
python -c "import tensorflow as tf; print('GPU devs:', tf.config.list_physical_devices('GPU'))"

echo "Running fit_prf_extended.py for subject $subject, model $model"
"$HOME/data/conda/envs/neural_priors2/bin/python" -u \
    "$HOME/git/retsupp/retsupp/modeling/fit_prf_extended.py" \
    "$subject" --model "$model" \
    --bids_folder "$bids_folder" \
    --r2_thr 0.05 \
    --resolution 80 \
    --grid_radius 5.0 \
    --distractor_radius 0.4 \
    --max_n_iterations 4000
