#!/bin/bash
#SBATCH --job-name=neuropythy_register
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Run neuropythy retinotopy registration for one subject. Inputs: model
# 4 surface .gii files (produced by sample_prf_to_surface.sh). Output:
# inferred_{angle,eccen,sigma,varea}.mgz under
# derivatives/fmriprep/sourcedata/freesurfer/sub-XX/mri/  (the canonical
# location read by Subject.get_retinotopic_roi etc.).
#
# Usage:
#   sbatch --array=1-30 retsupp/neuropythy/slurm_jobs/register_retinotopy.sh

set -euo pipefail

LOGFILE="$HOME/logs/neuropythy_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
subject="${SLURM_ARRAY_TASK_ID}"
sub_pad=$(printf "%02d" "$subject")
echo "Host: $(hostname) | sub-${sub_pad}"
echo "Started: $(date)"

scontrol update jobid="${SLURM_JOB_ID}" \
    name="neuropythy_sub-${sub_pad}" 2>/dev/null || true

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

if [[ -z "${FREESURFER_HOME:-}" ]]; then
    if [[ -d /apps/u24/opt/x86_64_v3/freesurfer-* ]] 2>/dev/null; then
        export FREESURFER_HOME=$(ls -d /apps/u24/opt/x86_64_v3/freesurfer-* | head -1)
    fi
fi
[[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]] && \
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
echo "FREESURFER_HOME=${FREESURFER_HOME:-unset}"

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
$PYTHON -u "$HOME/git/retsupp/retsupp/neuropythy/register_retinotopy.py" \
    "$subject" \
    --bids_dir /shares/zne.uzh/gdehol/ds-retsupp

echo "Finished: $(date)"
