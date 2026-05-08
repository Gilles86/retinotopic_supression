#!/bin/bash
#SBATCH --job-name=prf_cpu
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Note: --cpus-per-task overridden via sbatch CLI for benchmarking.

# CPU twin of fit_prf_l4.sh. Same logic; just no GPU and TF runs on CPU
# threads. Useful for benchmarking GPU-vs-CPU and for nodes when L4
# queue is busy.

set -euo pipefail

LOGFILE="$HOME/logs/prf_cpu_m${MODEL:-?}_c${CHUNK:-?}_${SLURM_CPUS_PER_TASK:-?}cpu_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL env var not set." >&2; exit 2
fi

subject="${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) | Job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | sub-${subject} | model ${MODEL} | CPUs=${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=-1     # force CPU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
$PYTHON -c "import tensorflow as tf; print('GPUs (should be empty):', tf.config.list_physical_devices('GPU'))"

CHUNK="${CHUNK:-10000}"
DEBUG_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then DEBUG_FLAG="--debug"; fi
echo "Using chunk size: $CHUNK"

$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/fit_prf.py" \
    "$subject" --model "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --resolution 50 \
    --voxel-chunk-size "$CHUNK" \
    --max-n-iterations 2000 \
    --paradigm-kind full \
    $DEBUG_FLAG

echo "Finished: $(date)"
