#!/bin/bash
#SBATCH --account=zne.uzh
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/dev/null
#SBATCH --job-name=clean_roi_cache

# Build per-subject, per-ROI cleaned-BOLD + paradigm caches.
# Loads the 12 cleaned NIfTIs ONCE per subject and writes 11 small
# .npz files (one per ROI: V1, V2, V3, V3AB, hV4, LO, TO, VO, IPS,
# SPL1, FEF).
#
# Usage:
#   sbatch --array=1-30%15 \
#       retsupp/modeling/slurm_jobs/build_cleaned_roi_cache.sh
#
# Output:
#   /shares/zne.uzh/gdehol/ds-retsupp/derivatives/cleaned_roi_cache/
#     sub-NN/sub-NN_roi-{ROI}_res-50.npz   (~5–30 MB per file)

set -euo pipefail
sleep $(( (RANDOM % 30) + 1 ))

RES="${RES:-50}"
SUB=$(printf %02d "$SLURM_ARRAY_TASK_ID")
scontrol update jobid="${SLURM_JOB_ID}" \
    name="clean_roi_cache_sub-${SUB}" 2>/dev/null || true

LOGFILE="$HOME/logs/clean_roi_cache_sub-${SUB}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "=== build_cleaned_roi_cache  res=${RES} sub-${SUB}  $(date) ==="
echo "  host: $(hostname)"

set +u
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
set -u

cd "$HOME/git/retsupp"
git rev-parse --short HEAD || true

export PYTHONUNBUFFERED=1

python -u notes/figures/talk/build_cleaned_roi_cache.py "$SLURM_ARRAY_TASK_ID" \
    --resolution "$RES" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp

echo "=== done $(date) ==="
