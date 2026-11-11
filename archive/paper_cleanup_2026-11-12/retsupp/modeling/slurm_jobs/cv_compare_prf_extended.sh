#!/bin/bash
#SBATCH --job-name=prf_cv
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --time=90:00
#SBATCH --constraint=A100
#SBATCH --mem=64G

# Usage:
#   sbatch cv_compare_prf_extended.sh <subject> [<holdout_session> <holdout_run>]

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-prf_cv}_sub-${1}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2

export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

subject=$1
holdout_session=${2:-2}
holdout_run=${3:-6}

nvidia-smi
python -c "import tensorflow as tf; print('GPU devs:', tf.config.list_physical_devices('GPU'))"

echo "CV compare for subject $subject, holdout ses-$holdout_session run-$holdout_run"

"$HOME/data/conda/envs/neural_priors2/bin/python" -u \
    "$HOME/git/retsupp/retsupp/modeling/cv_compare_prf_extended.py" \
    "$subject" \
    --holdout_session "$holdout_session" \
    --holdout_run "$holdout_run" \
    --bids_folder "$bids_folder" \
    --resolution 60 \
    --grid_radius 5.0 \
    --distractor_radius 0.4 \
    --max_n_iterations 2000 \
    --r2_thr 0.04
