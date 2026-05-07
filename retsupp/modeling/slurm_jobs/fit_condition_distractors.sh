#!/bin/bash
#SBATCH --job-name=cond_distractors
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --constraint=L4
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Conditionwise PRF refit on the FULL paradigm (bar + distractor disks).
# Inits per voxel from the baseline mean-fit (default model 4).
#
# Usage:
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_condition_distractors.sh <model_label> [<init_model>]
#
# Positional:
#   $1 = target PRF model (1, 2, 3, 4, 6).
#   $2 = init model used for x/y/sd/baseline/amplitude init (default 4).
#
# Output:
#   derivatives/prf_conditionfit_distractors/model<N>/sub-XX/

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: --array=1-30 is required (subject ID)"; exit 2
fi
if [[ -z "${1:-}" ]]; then
    echo "ERROR: positional <model_label> required (e.g. 1 or 4)"; exit 2
fi

subject="${SLURM_ARRAY_TASK_ID}"
model="$1"
init_model="${2:-4}"

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-cond_distractors}_sub-${subject}_model-${model}_init-${init_model}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
echo "Subject:     ${subject}"
echo "Model:       ${model}"
echo "Init model:  ${init_model}"
echo "Started:     $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

nvidia-smi || true
"$PYTHON" -c "import tensorflow as tf; print('GPU devs:', tf.config.list_physical_devices('GPU'))"

"$PYTHON" -u \
    "$HOME/git/retsupp/retsupp/modeling/fit_condition_distractors.py" \
    "$subject" \
    --model "$model" \
    --init-model "$init_model" \
    --bids_folder "$bids_folder" \
    --r2_thr 0.06 \
    --resolution 50 \
    --grid_radius 5.0 \
    --distractor_radius 0.4 \
    --max_n_iterations 4000 \
    --voxel_chunk_size 5000

echo "Finished: $(date)"
