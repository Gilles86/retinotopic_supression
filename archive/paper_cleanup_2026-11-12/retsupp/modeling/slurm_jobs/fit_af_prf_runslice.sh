#!/bin/bash
#SBATCH --job-name=af_runslice
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:45:00

# Joint AF + PRF braincoder fit, run-slice variant (first/last run of
# each HP-distractor block) — CPU only.
#
# Patterned exactly like fit_af_prf_braincoder.sh: 30 subjects * 8 ROIs
# = 240 array tasks.
#
# Usage:
#   sbatch --array=1-240 fit_af_prf_runslice.sh first
#   sbatch --array=1-240 fit_af_prf_runslice.sh last
#
# Or override via env var:
#   RUN_SLICE=first sbatch --array=1-240 fit_af_prf_runslice.sh
#
# The array task id maps to (subject, roi) as:
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = idx0 // 8 + 1                     # 1..30
#   roi     = ROIS[ idx0 %  8 ]
#
# At runtime, runs whose HP is "no distractor" or unknown are skipped
# inside the Python script.

set -euo pipefail

# --- Argument parsing: run-slice can come from $1 or $RUN_SLICE. ---
RUN_SLICE="${1:-${RUN_SLICE:-first}}"
if [[ "$RUN_SLICE" != "first" && "$RUN_SLICE" != "last" ]]; then
    echo "ERROR: run-slice must be 'first' or 'last', got '$RUN_SLICE'" >&2
    exit 2
fi

# --- Logging. ---
LOGFILE="$HOME/logs/af_prf_runslice_${RUN_SLICE}_${SLURM_JOB_NAME:-af_runslice}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Run-slice:   ${RUN_SLICE}"
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

echo "Running fit_af_prf_runslice.py for sub-${subject}, roi=${roi}, run-slice=${RUN_SLICE}"

"$PYTHON" -u \
    "$HOME/git/retsupp/retsupp/modeling/fit_af_prf_runslice.py" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 0 \
    --run-slice "$RUN_SLICE"

echo "Finished:    $(date)"

# ---------------------------------------------------------------------------
# Suggested submission:
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_af_prf_runslice.sh first
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_af_prf_runslice.sh last
# ---------------------------------------------------------------------------
