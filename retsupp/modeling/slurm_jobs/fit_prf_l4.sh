#!/bin/bash
#SBATCH --job-name=prf_l4
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
# No GPU type constraint: take whatever is free first (L4/A100/H100/...).
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Whole-cortex PRF fit, ONE model per job. Subject + model from args.
#
# Usage:
#   sbatch --array=1-30 --export=ALL,MODEL=1 .../fit_prf_l4.sh
#
# Run model 1 first; once it lands, models 2/3 (init from 1), then 4
# (init from 3), then 5/6 (init from 4) can be submitted in any order.

set -euo pipefail

LOGFILE="$HOME/logs/prf_l4_m${MODEL:-?}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL env var not set. Pass --export=ALL,MODEL=N." >&2; exit 2
fi

subject="${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) | Job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | sub-${subject} | model ${MODEL}"
echo "Started: $(date)"

# Enable lmod and load CUDA before conda. Required on V100/A100/H100
# nodes — those don't have libnvrtc.so / libcudnn in the env's bundled
# CUDA. L4 nodes work without it but loading is harmless.
source /etc/profile.d/lmod.sh 2>/dev/null
module load cuda/12.6.3 2>&1 || echo "WARN: module load cuda failed"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
$PYTHON -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

DEBUG_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then DEBUG_FLAG="--debug"; fi
CHUNK="${CHUNK:-10000}"
SUFFIX_FLAG=""
if [[ -n "${OUTPUT_SUFFIX:-}" ]]; then
    SUFFIX_FLAG="--output-suffix $OUTPUT_SUFFIX"
fi
echo "Using chunk size: $CHUNK"
echo "Output suffix: ${OUTPUT_SUFFIX:-(none)}"

$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/fit_prf.py" \
    "$subject" --model "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --resolution 50 \
    --voxel-chunk-size "$CHUNK" \
    --max-n-iterations 2000 \
    --paradigm-kind full \
    $DEBUG_FLAG $SUFFIX_FLAG

echo "Finished: $(date)"
