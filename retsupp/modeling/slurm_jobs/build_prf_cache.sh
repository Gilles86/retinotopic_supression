#!/bin/bash
#SBATCH --account=zne.uzh
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/dev/null
#SBATCH --job-name=prf_cache

# Build per-subject per-ROI BOLD+paradigm cache for warmstart fits.
# Usage:
#   ROI=V1 sbatch --array=1-30%15 retsupp/modeling/slurm_jobs/build_prf_cache.sh
#
# Output:
#   /shares/zne.uzh/gdehol/ds-retsupp/derivatives/prf_cache/
#     sub-NN/sub-NN_roi-V1_res-50.npz

set -euo pipefail
sleep $(( (RANDOM % 30) + 1 ))

ROI="${ROI:-V1}"
RES="${RES:-50}"
SUB=$(printf %02d "$SLURM_ARRAY_TASK_ID")
scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_cache_${ROI}_sub-${SUB}" 2>/dev/null || true

LOGFILE="$HOME/logs/prf_cache_${ROI}_sub-${SUB}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "=== build_prf_cache  roi=${ROI} res=${RES} sub-${SUB}  $(date) ==="
echo "  host: $(hostname)"

set +u
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
set -u

cd "$HOME/git/retsupp"
git rev-parse --short HEAD || true

export PYTHONUNBUFFERED=1

python -u notes/figures/talk/build_prf_cache.py "$SLURM_ARRAY_TASK_ID" \
    --roi "$ROI" --resolution "$RES" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp

echo "=== done $(date) ==="
