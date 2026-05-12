#!/bin/bash
#SBATCH --job-name=prf_gpu
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
# Constraint to GPUs whose compute capability the env's CUDA 11.8 ptxas
# supports. H100/H200 are compute_9.0 (Hopper) and crash with
# "ptxas: Failed to launch ptxas" — see sub-02 m1 attempt for the trace.
#SBATCH --constraint="A100|V100|L4"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Generous walltime — measured m1 GPU is ~40 min, but m2-m6 timing
# isn't known yet; revisit once we have data.
#SBATCH --time=04:00:00

# Whole-cortex PRF fit on any GPU. ONE model per job.
#
# After MODEL is parsed, the script renames itself via scontrol so
# squeue shows e.g.  prf_m3_sub-05  instead of the generic prf_gpu.
#
# Env vars (all read via --export=ALL,KEY=VAL):
#   MODEL    int 1..6   (required)
#   KIND     full|bar   (default: full)
#   SMOKE    0|1        (1 → --debug)
#   CHUNK    int        (default: 10000)
#   OUTPUT_SUFFIX str   (optional)
#
# Usage:
#   sbatch --array=1-30 --export=ALL,MODEL=1 .../fit_prf_l4.sh
#   sbatch --array=1-30 --export=ALL,MODEL=2,KIND=bar .../fit_prf_l4.sh
#
# Chain: m1 → {m2, m3}; m2 → {m4, m5}; m5 → m6 (see MODEL_CFG.init_from
# in fit_prf.py). Submit later stages with --dependency=afterok:$PREV.

set -euo pipefail

KIND="${KIND:-full}"
LOGFILE="$HOME/logs/prf_l4_m${MODEL:-?}_${KIND}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL env var not set. Pass --export=ALL,MODEL=N." >&2; exit 2
fi
if [[ "$KIND" != "full" && "$KIND" != "bar" ]]; then
    echo "ERROR: KIND must be 'full' or 'bar' (got '$KIND')." >&2; exit 2
fi

subject="${SLURM_ARRAY_TASK_ID}"
sub_pad=$(printf "%02d" "$subject")
echo "Host: $(hostname) | Job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | sub-${subject} | model ${MODEL} | kind ${KIND}"
echo "Started: $(date)"

# Rename the job in squeue so it tells you what it is doing.
scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_m${MODEL}_${KIND}_sub-${sub_pad}" 2>/dev/null || true

# cuDNN 8 in the conda env requires libnvrtc.so at runtime, but the env
# does not bundle nvidia-cuda-nvrtc. On V100/A100/H100 nodes, lmod has
# no `cuda/*` module (MODULEPATH is just /etc/lmod/...), so the previous
# `module load cuda/12.6.3` silently failed and LD_LIBRARY_PATH stayed
# empty -> TF crashed with "libnvrtc.so: cannot open shared object file".
# Point LD_LIBRARY_PATH at the spack-installed CUDA 11.8 (matches TF 2.14
# / cuDNN 8.7 ABI). Glob handles spack-hashed dirname.
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Using system CUDA: ${SYS_CUDA_GLOB[0]}"
else
    echo "WARN: system cuda-11.8.0 not found under /apps/u24/opt/x86_64_v3/"
fi

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
    --paradigm-kind "$KIND" \
    $DEBUG_FLAG $SUFFIX_FLAG

echo "Finished: $(date)"
