#!/bin/bash
#SBATCH --job-name=surfbold_cache
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Per-subject vol-to-surf BOLD caching. For each subject, projects every
# (ses, run) cleaned BOLD volume onto fsnative inflated surface and saves
# (V, T) float32 arrays per hemisphere. Output files feed the paradigm-
# brain movie renderer (which then runs locally in ~30 s per subject from
# the cached arrays). Caches land in a shared dir so they can be rsynced
# back to the laptop.
#
# Usage:  sbatch --array=1-30 retsupp/visualize/vss2026/slurm_jobs/cache_surfbold.sh

set -eo pipefail

LOGFILE="$HOME/logs/surfbold_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
subject="${SLURM_ARRAY_TASK_ID}"
sub_pad=$(printf "%02d" "$subject")
echo "Host: $(hostname) | Job ${SLURM_JOB_ID}.${subject} | sub-${sub_pad}"
echo "Started: $(date)"

scontrol update jobid="${SLURM_JOB_ID}" \
    name="surfbold_sub-${sub_pad}" 2>/dev/null || true

# Stagger task start to avoid NFS profile-read storm on large arrays.
sleep $(( (RANDOM % 60) + 1 ))

# Use retsupp_cuda — per CLAUDE.md the cluster's `retsupp` env is
# incomplete (no braincoder). retsupp_cuda has the same numpy / nilearn /
# scipy and is fine for CPU work; TF falls back to CPU automatically when
# no GPU is requested.
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

CACHE_DIR=/shares/zne.uzh/gdehol/cache/retsupp/surfbold
mkdir -p "$CACHE_DIR"

cd "$HOME/git/retsupp"
$PYTHON -u -m retsupp.visualize.vss2026.make_paradigm_brain_movie \
    --cache-only \
    --subjects "$subject" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
