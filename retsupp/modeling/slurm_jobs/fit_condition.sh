#!/bin/bash
#SBATCH --job-name=fit_condition
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --constraint=L4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Conditionwise PRF refit — per HP-distractor condition, FULL paradigm.
# Writes to derivatives/prf_conditionfit/model{N}/sub-XX/.
# Inits per voxel from the MEAN-fit model on disk (same model_label).
#
# Required env: MODEL (default 4).
# SLURM_ARRAY_TASK_ID = subject id.

set -eo pipefail

MODEL="${MODEL:-4}"
subject="${SLURM_ARRAY_TASK_ID:?ERROR: --array required (subject id)}"
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/fit_condition_m${MODEL}_sub-${sub_pad}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" \
    name="cond_m${MODEL}_sub-${sub_pad}" 2>/dev/null || true

sleep $(( RANDOM % 30 ))

echo "Host: $(hostname) | sub-${subject} | model ${MODEL}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

"$PYTHON" -u "$HOME/git/retsupp/retsupp/modeling/fit_condition.py" \
    "$subject" --model "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --resolution 50

echo "Finished: $(date)"
