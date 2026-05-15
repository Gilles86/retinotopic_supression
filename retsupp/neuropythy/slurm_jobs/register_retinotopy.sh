#!/bin/bash
#SBATCH --job-name=neuropythy_register
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
# (cpus bumped from 4 → 16: register_retinotopy uses Noah Benson's
# Java mesh-registration via JPype; the JVM side does scale with
# threads. With 16 CPUs the per-subject wall-time drops well below
# the 1h failure boundary; on a typical fsnative mesh the registration
# converges in roughly 15-30 min instead of 45-90.)

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
conda activate retsupp_neuropythy
export PYTHONUNBUFFERED=1

export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt
[[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]] && \
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fmriprep/sourcedata/freesurfer
echo "FREESURFER_HOME=${FREESURFER_HOME}"
echo "SUBJECTS_DIR=${SUBJECTS_DIR}"
echo "FS version: $(mri_surf2surf --version 2>&1 | head -1)"

PYTHON="$HOME/data/conda/envs/retsupp_neuropythy/bin/python"
$PYTHON -u "$HOME/git/retsupp/retsupp/neuropythy/register_retinotopy.py" \
    "$subject" \
    --bids_dir /shares/zne.uzh/gdehol/ds-retsupp

echo "Finished: $(date)"
