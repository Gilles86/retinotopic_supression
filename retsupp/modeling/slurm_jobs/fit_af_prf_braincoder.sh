#!/bin/bash
#SBATCH --job-name=af_static
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:45:00

# Joint AF + PRF braincoder fit on the cluster (CPU only).
#
# Submits one fit per (subject, ROI) pair via a SLURM array.
# 30 subjects * 8 ROIs = 240 array tasks.
#
# Usage:
#   sbatch --array=1-240 fit_af_prf_braincoder.sh                # bar paradigm
#   sbatch --array=1-240 fit_af_prf_braincoder.sh full           # full paradigm
#
# Or override via env var:
#   PARADIGM_TYPE=full sbatch --array=1-240 fit_af_prf_braincoder.sh
#
# The array task id maps to (subject, roi) as:
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = idx0 // 8 + 1                     # 1..30
#   roi     = ROIS[ idx0 %  8 ]
#
# At runtime, runs whose HP is "no distractor" or unknown are skipped
# inside the Python script.

set -euo pipefail

# --- Argument parsing: paradigm-type can come from $1 or $PARADIGM_TYPE. ---
PARADIGM_TYPE="${1:-${PARADIGM_TYPE:-bar}}"
if [[ "$PARADIGM_TYPE" != "bar" && "$PARADIGM_TYPE" != "full" ]]; then
    echo "ERROR: paradigm-type must be 'bar' or 'full', got '$PARADIGM_TYPE'" >&2
    exit 2
fi

# --- Logging. ---
LOGFILE="$HOME/logs/af_prf_braincoder_${SLURM_JOB_NAME:-af_prf}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Paradigm:    ${PARADIGM_TYPE}"
echo "Started:     $(date)"

# --- (subject, ROI) decoding from SLURM_ARRAY_TASK_ID. ---
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO)
N_ROIS=${#ROIS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-240." >&2
    exit 2
fi

idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
subject=$(( idx0 / N_ROIS + 1 ))
roi_idx=$(( idx0 % N_ROIS ))
roi="${ROIS[$roi_idx]}"

echo "Subject:     ${subject}"
echo "ROI:         ${roi}"

# --- Conda env (CPU). ---
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

# CPU-only run: hide any GPUs so TF doesn't try to register CUDA.
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
# Cap intra-op threads to fit cpus-per-task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

echo "Running fit_af_prf_braincoder.py for sub-${subject}, roi=${roi}, paradigm=${PARADIGM_TYPE}"

"$PYTHON" -u \
    "$HOME/git/retsupp/retsupp/modeling/fit_af_prf_braincoder.py" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 0 \
    --paradigm-type "$PARADIGM_TYPE"

echo "Finished:    $(date)"

# ---------------------------------------------------------------------------
# Suggested submission:
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_af_prf_braincoder.sh
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_af_prf_braincoder.sh full
# ---------------------------------------------------------------------------
