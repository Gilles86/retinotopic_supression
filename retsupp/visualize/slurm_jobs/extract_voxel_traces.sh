#!/bin/bash
#SBATCH --job-name=voxel_traces_extract
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# Per-subject voxel-trace extraction. Reads cleaned BOLD + PRF NIfTIs,
# writes a small per-(roi, hp_cond, dir_tag, t_offset) TSV under
# /shares/zne.uzh/gdehol/ds-retsupp/derivatives/voxel_traces_cache/.
#
# Usage:  sbatch --array=1-30 retsupp/visualize/slurm_jobs/extract_voxel_traces.sh

set -euo pipefail

LOGFILE="$HOME/logs/voxel_traces_extract_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
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
    name="voxel_traces_sub-${sub_pad}" 2>/dev/null || true

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp/bin/python"

OUT_DIR=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/voxel_traces_cache
mkdir -p "$OUT_DIR"

$PYTHON -u "$HOME/git/retsupp/retsupp/visualize/extract_voxel_traces.py" \
    "$subject" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --out-dir "$OUT_DIR"

echo "Finished: $(date)"
