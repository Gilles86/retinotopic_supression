#!/bin/bash
#SBATCH --job-name=prf_chunked
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100|L4"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Per-(subject, chunk) GPU PRF fit. ONE model + ONE kind per submission.
# Array task index maps to (sub_idx, chunk_idx) via the same mapping
# as fit_prf_chunked.sh:
#
#   idx0      = SLURM_ARRAY_TASK_ID - 1
#   sub_idx   = idx0 / N_CHUNKS
#   chunk_idx = idx0 % N_CHUNKS
#   subject   = sub_idx + 1
#
# Required env vars (--export=ALL,...):
#   MODEL      1..6
#   N_CHUNKS   chunks per subject (typically 10)
#   N_SUBS     total subjects     (typically 30)
#   KIND       full | bar         (default full)
#
# Submit at the tightest plausible walltime via --time at sbatch time
# (priority bucket matters). Typical per-task wallclock on A100:
#   m1: 8-12 min (includes grid)
#   m2/m3: 6-10 min
#   m4: 10-15 min (DoG + flex HRF, longer schedule)
#   m5: 8-12 min (DN, fixed HRF)
#   m6: 12-18 min (DN + flex HRF, most complex)
#
# Outputs land at derivatives/{prf|prf_bar}/model{N}/sub-XX/chunks/.
# Run merge_prf_chunks.sh afterwards to assemble final NIfTIs.

set -eo pipefail

KIND="${KIND:-full}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${MODEL:-}" || -z "${N_CHUNKS:-}" || -z "${N_SUBS:-}" ]]; then
    echo "ERROR: MODEL, N_CHUNKS, N_SUBS env vars required." >&2; exit 2
fi
if [[ "$KIND" != "full" && "$KIND" != "bar" ]]; then
    echo "ERROR: KIND must be 'full' or 'bar' (got '$KIND')." >&2; exit 2
fi

idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_idx=$(( idx0 / N_CHUNKS ))
chunk_idx=$(( idx0 % N_CHUNKS ))
subject=$(( sub_idx + 1 ))
sub_pad=$(printf "%02d" "$subject")

if [[ "$subject" -gt "$N_SUBS" ]]; then
    echo "Array index out of range." >&2; exit 2
fi

LOGFILE="$HOME/logs/prf_chunked_m${MODEL}_${KIND}_sub-${sub_pad}_chunk-${chunk_idx}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_m${MODEL}_${KIND}_sub-${sub_pad}_c${chunk_idx}" 2>/dev/null || true

echo "Host: $(hostname) | sub-${subject} | model ${MODEL} | kind ${KIND}"
echo "  chunk ${chunk_idx}/${N_CHUNKS}  array_task=${SLURM_ARRAY_TASK_ID}"
echo "Started: $(date)"

# CUDA runtime (TF 2.14 / cuDNN 8.7 ABI; CUDA 11.8 spack install).
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
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
