#!/bin/bash
#SBATCH --job-name=prf_chk
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=00:30:00

# Chunked PRF fit: ONE voxel-chunk per array task, distributed across
# CPU nodes. After all tasks finish, run merge_prf_chunks.sh per
# subject to assemble the final NIfTIs.
#
# Sizing (Gaussian model 1, T=3096, V_per_chunk=5000):
#   - Base load (BOLD + paradigm + masker)        ~5 GB
#   - Per-voxel activations during GD backprop    ~5 GB
#   - Total peak                                  ~10-11 GB  -> 12G request
#   - DoG/DN models will need larger mem (use --mem on resubmit).
#
# Layout: array task ID maps to (subject_idx, chunk_idx).
#
# Usage:
#   sbatch --array=1-1620 --export=ALL,MODEL=1,N_CHUNKS=54,N_SUBS=30 \
#          retsupp/modeling/slurm_jobs/fit_prf_chunked.sh
#
# 30 subjects × 54 chunks = 1620 array tasks. Each task is small
# (16 CPU, 12 GB, ~6-10 min wall) so the cluster can schedule many
# in parallel.

set -euo pipefail

# Stagger the "user env retrieval" / conda init by a random 0-60s jitter.
# This avoids the dogpile of 1000+ array tasks hitting the NFS-mounted
# $HOME profile at the same instant, which causes "user env retrieval
# failed requeued held" SLURM errors. The wall-time cost is trivial
# compared to the time saved on manually releasing held tasks.
sleep $(( (RANDOM % 60) + 1 ))

# Logging policy:
#  KEEP_LOG=1  -> per-task log written to ~/logs/prf_chk_m{M}_{JOB}_{TASK}.txt
#  KEEP_LOG=0 (default)  -> stdout goes to /dev/null (per SBATCH directive)
#                            avoiding ~1500 log files per array on shared NFS.
# Failures are still visible via `sacct -j JOBID`. To debug a specific
# task, resubmit it with KEEP_LOG=1.
if [[ "${KEEP_LOG:-0}" == "1" ]]; then
    LOGFILE="$HOME/logs/prf_chk_m${MODEL:-?}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
    mkdir -p "$(dirname "$LOGFILE")"
    exec >"$LOGFILE" 2>&1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${MODEL:-}" || -z "${N_CHUNKS:-}" || -z "${N_SUBS:-}" ]]; then
    echo "ERROR: MODEL, N_CHUNKS, N_SUBS env vars required." >&2; exit 2
fi

idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_idx=$(( idx0 / N_CHUNKS ))
chunk_idx=$(( idx0 % N_CHUNKS ))
subject=$(( sub_idx + 1 ))

if [[ "$subject" -gt "$N_SUBS" ]]; then
    echo "Array index out of range." >&2; exit 2
fi

echo "Host: $(hostname) | Job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
echo "  sub-${subject} | model ${MODEL} | chunk ${chunk_idx}/${N_CHUNKS}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TF_NUM_INTEROP_THREADS=2

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/fit_prf.py" \
    "$subject" --model "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --resolution 50 \
    --voxel-chunk-size 100000 \
    --max-n-iterations 2000 \
    --paradigm-kind full \
    --chunk-index "$chunk_idx" \
    --n-chunks "$N_CHUNKS"
# Note: --voxel-chunk-size 100000 means "no internal batching" — the
# task processes all its voxels in one GD call (per-task voxel count
# ~5000 with N_CHUNKS=54).

echo "Finished: $(date)"
