#!/bin/bash
#SBATCH --job-name=prf_surf_sample
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# Resample volume PRF parameters to fsnative + fsaverage gii surfaces.
# One subject per array task; one model per submission.
#
# Usage:
#   sbatch --array=1-30 --export=ALL,MODEL=4 \
#       retsupp/surface/slurm_jobs/sample_prf_to_surface.sh

set -euo pipefail

LOGFILE="$HOME/logs/prf_surf_m${MODEL:-?}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${MODEL:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID and MODEL required." >&2; exit 2
fi

subject="${SLURM_ARRAY_TASK_ID}"
sub_pad=$(printf "%02d" "$subject")
echo "Host: $(hostname) | sub-${sub_pad} | model ${MODEL}"
echo "Started: $(date)"

scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_surf_m${MODEL}_sub-${sub_pad}" 2>/dev/null || true

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

# FreeSurfer is needed by nipype.SurfaceTransform (mri_surf2surf).
if [[ -z "${FREESURFER_HOME:-}" ]]; then
    if [[ -d /apps/u24/opt/x86_64_v3/freesurfer-* ]] 2>/dev/null; then
        export FREESURFER_HOME=$(ls -d /apps/u24/opt/x86_64_v3/freesurfer-* | head -1)
    fi
fi
[[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]] && \
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
echo "FREESURFER_HOME=${FREESURFER_HOME:-unset}"

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
$PYTHON -u "$HOME/git/retsupp/retsupp/surface/sample_prf_to_surface_nilearn.py" \
    "$subject" --model "$MODEL" \
    --bids_folder /shares/zne.uzh/gdehol/ds-retsupp

echo "Finished: $(date)"
