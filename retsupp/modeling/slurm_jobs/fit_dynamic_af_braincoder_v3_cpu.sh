#!/bin/bash
#SBATCH --job-name=af_dyn3
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:30:00

# Joint Dynamic AF + PRF braincoder fit (v3) on the cluster (CPU only).
#
# v3 has 6 shared parameters and 5 per-voxel parameters:
#   shared:    sigma_AF, g_HP, g_LP, sigma_dyn, g_HP_dyn, g_LP_dyn
#   per-voxel: x, y, sd, baseline, amplitude
#
# Slightly more walltime than v2 because the larger shared-parameter set
# (6 vs 5) makes the optimization a bit slower to converge.
#
# Submits one fit per (subject, ROI) pair via a SLURM array.
# 30 subjects * 8 ROIs = 240 array tasks.
#
# Usage:
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_dynamic_af_braincoder_v3_cpu.sh
#
# Always uses the FULL paradigm (bar + distractor disks).
#
# The array task id maps to (subject, roi) as:
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = idx0 // 8 + 1                     # 1..30
#   roi     = ROIS[ idx0 %  8 ]
#
# Runs whose HP is "no distractor" or unknown are skipped inside the
# Python script.

set -euo pipefail

# --- Logging. ---
LOGFILE="$HOME/logs/dyn_af_prf_braincoder_${SLURM_JOB_NAME:-dyn_af_prf}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
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

# Force CPU even though the env is CUDA-enabled.
export CUDA_VISIBLE_DEVICES=-1

export PYTHONUNBUFFERED=1
# Cap intra-op threads to fit cpus-per-task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

echo "Running fit_dynamic_af_braincoder.py (v3) for sub-${subject}, roi=${roi}"

"$PYTHON" -u \
    "$HOME/git/retsupp/retsupp/modeling/fit_dynamic_af_braincoder.py" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 0 \
    --model-version v3

echo "Finished:    $(date)"

# ---------------------------------------------------------------------------
# Suggested submission:
#   sbatch --array=1-240 retsupp/modeling/slurm_jobs/fit_dynamic_af_braincoder_v3_cpu.sh
# ---------------------------------------------------------------------------
