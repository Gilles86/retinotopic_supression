#!/bin/bash
#SBATCH --job-name=r2_mixture
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Fit per-(subject, ROI) 2-component Beta mixture on R² and cache
# `desc-p_signal.{nii.gz, json}` per subject. Required upstream of any
# AF / decoding step that uses mixture-FDR voxel selection.
#
# One task = one subject; the script loops over all ROIs internally.
# Required env: MODEL (default 4).
# Optional env: ROIS (space-separated; default V1..FEF).
# SLURM_ARRAY_TASK_ID = subject id.

set -eo pipefail

MODEL="${MODEL:-4}"
ROIS="${ROIS:-V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF}"
subject="${SLURM_ARRAY_TASK_ID:?ERROR: --array required (subject id)}"
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/r2_mixture_m${MODEL}_sub-${sub_pad}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" \
    name="r2_mix_m${MODEL}_sub-${sub_pad}" 2>/dev/null || true

sleep $(( RANDOM % 20 ))

echo "Host: $(hostname) | sub-${subject} | model ${MODEL} | rois: ${ROIS}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

"$PYTHON" -u "$HOME/git/retsupp/retsupp/modeling/run_r2_mixture_all.py" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --model "$MODEL" \
    --subjects "$subject" \
    --rois $ROIS

echo "Finished: $(date)"
