#!/bin/bash
#SBATCH --job-name=neuropythy_register_m1
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Variant of register_retinotopy.sh that uses model-1 surface files instead
# of model 4. Used for subject recovery (e.g. sub-06 / sub-08) when only
# m1 fits exist on disk.
#
# Usage:
#   sbatch --array=6,8 retsupp/neuropythy/slurm_jobs/register_retinotopy_m1.sh

set -euo pipefail

LOGFILE="$HOME/logs/neuropythy_m1_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
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
    name="neuropythy_m1_sub-${sub_pad}" 2>/dev/null || true

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
    --bids_dir /shares/zne.uzh/gdehol/ds-retsupp \
    --model 1

echo "Finished: $(date)"
