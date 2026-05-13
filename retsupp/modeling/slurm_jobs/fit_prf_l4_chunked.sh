#!/bin/bash
#SBATCH --job-name=prf_chunked
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
# L4-only: L4 nodes have 1 GPU each, so each job owns its node and there's
# no cuInit race condition. V100/A100/H100/H200 nodes pack 8 GPUs each;
# when 8 of our jobs land simultaneously, the driver locks up with
# CUDA_ERROR_UNKNOWN and TF falls back to CPU (60× slowdown, timeouts).
# Observed 2026-05-13 on u24-chiivm0-604. Cluster has ~16 L4 nodes
# (gpu:L4:1) which throttles us but the dispatch is reliable.
#SBATCH --constraint="L4"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Partition=lowprio: same physical L4/A100/V100 nodes as standard,
# but jobs dispatch almost immediately (5-10s) instead of sitting in
# fairshare queue for hours. Trade-off: a higher-priority standard job
# can preempt us mid-fit. For a 15-30 min chunk that's a low risk;
# preempted tasks just get resubmitted. See retsupp/CLAUDE.md
# §"Lowprio partition for GPU PRF fits".

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
if [[ -z "${MODEL:-}" || -z "${N_CHUNKS:-}" ]]; then
    echo "ERROR: MODEL and N_CHUNKS env vars required." >&2; exit 2
fi
if [[ "$KIND" != "full" && "$KIND" != "bar" ]]; then
    echo "ERROR: KIND must be 'full' or 'bar' (got '$KIND')." >&2; exit 2
fi

# Two modes:
# (a) Per-subject mode (preferred): SUBJECT is exported by the caller
#     and the array index just enumerates chunks.  N_SUBS not needed.
# (b) Phase-wide mode (legacy): SUBJECT not set; array index encodes
#     both subject and chunk via (idx-1) % N_CHUNKS.  N_SUBS required.
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

LOGFILE="$HOME/logs/prf_chunked_m${MODEL}_${KIND}_sub-${sub_pad}_chunk-${chunk_idx}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

# SLURM's Name field is array-shared — whichever task renames last wins
# for ALL tasks. Don't put chunk_idx here (it would mislead in squeue).
# The chunk index is recoverable from the array task suffix on JobID.
scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_m${MODEL}_${KIND}_sub-${sub_pad}" 2>/dev/null || true

# Random startup stagger (0-30s) to avoid two failure modes:
#  1. cuInit dogpile on multi-GPU nodes (race when 8 jobs simultaneously
#     init TF on the same node; defended at constraint level by L4-only,
#     but kept for defense-in-depth in case future jobs allow A100/V100).
#  2. NFS profile-read dogpile (the "user env retrieval failed" issue
#     SLURM uses to mark tasks held when too many start at the same
#     instant). 30s spread is much wider than either race window.
sleep $(( RANDOM % 30 ))

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
