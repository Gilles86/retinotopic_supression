#!/bin/bash
#SBATCH --job-name=prf_cpu_chunked
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# CPU twin of fit_prf_l4_chunked.sh.  Same chunking semantics; no GPU,
# TF forced to CPU.  Useful when the GPU queue is congested — CPU
# partitions typically have idle nodes.  Per-task wallclock is ~3-5×
# slower than GPU, but dispatch is faster.
#
# 16 CPUs / 16 GB chosen for cluster citizenship: 32 CPUs gives only
# ~1.3× speedup over 16 (TF PRF fitting plateaus past 16 cores due to
# memory bandwidth), but ties up twice the node.  16 CPUs lets ~2× as
# many tasks land per node, which matters more on a congested cluster.
#
# Required env vars (--export=ALL,...):
#   MODEL      1..6
#   N_CHUNKS   chunks per subject (10 is fine on CPU)
#   KIND       full | bar         (default full)
#   SUBJECT    pin to one subject (preferred mode)
#   N_SUBS     phase-wide mode (alternative; with no SUBJECT, array
#              index encodes both subject and chunk)
#
# Pass --time at sbatch time (priority bucket matters).

set -eo pipefail

KIND="${KIND:-full}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${MODEL:-}" || -z "${N_CHUNKS:-}" ]]; then
    echo "ERROR: MODEL and N_CHUNKS env vars required." >&2; exit 2
fi
if [[ "$KIND" != "full" && "$KIND" != "bar" ]]; then
    echo "ERROR: KIND must be 'full' or 'bar' (got '$KIND')." >&2; exit 2
fi

if [[ -n "${SUBJECT:-}" ]]; then
    subject="$SUBJECT"
    chunk_idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
else
    if [[ -z "${N_SUBS:-}" ]]; then
        echo "ERROR: either SUBJECT, or N_SUBS env var must be set." >&2
        exit 2
    fi
    idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
    sub_idx=$(( idx0 / N_CHUNKS ))
    chunk_idx=$(( idx0 % N_CHUNKS ))
    subject=$(( sub_idx + 1 ))
    if [[ "$subject" -gt "$N_SUBS" ]]; then
        echo "Array index out of range." >&2; exit 2
    fi
fi
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/prf_cpuchunk_m${MODEL}_${KIND}_sub-${sub_pad}_chunk-${chunk_idx}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_cpu_m${MODEL}_${KIND}_sub-${sub_pad}_c${chunk_idx}" 2>/dev/null || true

echo "Host: $(hostname) | sub-${subject} | model ${MODEL} | kind ${KIND}"
echo "  chunk ${chunk_idx}/${N_CHUNKS}  array_task=${SLURM_ARRAY_TASK_ID}  CPUs=${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TF_NUM_INTEROP_THREADS=2
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/fit_prf.py" \
    "$subject" --model "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --resolution 50 \
    --voxel-chunk-size 100000 \
    --paradigm-kind "$KIND" \
    --chunk-index "$chunk_idx" \
    --n-chunks "$N_CHUNKS"

echo "Finished: $(date)"
