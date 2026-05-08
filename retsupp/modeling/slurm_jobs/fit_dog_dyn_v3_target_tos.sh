#!/bin/bash
#SBATCH --job-name=dog_dyn_v3_target_tos
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# Single (subject, ROI) test of the v3 + target model with TEMPORAL
# OVERSAMPLING of the HRF convolution. Hard-coded: subject=2, ROI=V3AB.
# The script takes ONE positional argument: the oversampling factor N.
#
# Submit baseline + oversampled comparison (run separately):
#
#   sbatch retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target_tos.sh 1
#   sbatch retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target_tos.sh 4
#   sbatch retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target_tos.sh 8
#
# Each writes to a distinct subdir:
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_tos1/sub-02/...
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_tos4/sub-02/...
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_tos8/sub-02/...
#
# Memory note: paradigm + condition_indicator + dynamic_indicator +
# target_indicator + the per-batch HRF-convolved BOLD prediction tensor
# all scale ~linearly with N along the time axis. For the full retsupp
# task (12 runs * 258 TRs ~= 3100 TRs total), at N=4 the fine grid is
# ~12k samples and at N=8 ~25k samples; the dominant memory cost is the
# (B, T*N, V) prediction tensor inside braincoder. For V<=500 voxels
# and B=1 this is well within 24G even at N=8, but bump --mem if you
# scale up the voxel cap.

set -euo pipefail

# --- Positional argument: oversampling factor. ---
if [[ $# -lt 1 ]]; then
    echo "ERROR: missing positional argument <oversampling>." >&2
    echo "Usage: sbatch $0 <N>   (e.g. 1, 4, 8)" >&2
    exit 2
fi
TEMPORAL_OVERSAMPLING="$1"

# --- Logging. ---
LOGFILE="$HOME/logs/dog_dyn_v3_target_tos${TEMPORAL_OVERSAMPLING}_${SLURM_JOB_NAME:-tos}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:                  $(hostname)"
echo "Job:                   ${SLURM_JOB_ID}"
echo "Started:               $(date)"
echo "Test:                  v3 + target with --temporal-oversampling ${TEMPORAL_OVERSAMPLING}"

# --- Subject from SLURM array task ID, ROI fixed to V3AB. ---
# Submit with --array=1-30 to fan out across all 30 subjects.
roi="V3AB"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    subject="${SLURM_ARRAY_TASK_ID}"
else
    subject=2  # default for direct (non-array) test runs
fi

echo "Subject:               ${subject}"
echo "ROI:                   ${roi}"
echo "Temporal oversampling: ${TEMPORAL_OVERSAMPLING}"

# --- Conda env (CUDA env, forced to CPU per the existing v3+target script). ---
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_dog_dynamic_af_braincoder.py"

echo "Running v3 + target tos=${TEMPORAL_OVERSAMPLING} fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --with-target \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --sigma-t-dyn-init 2.0 \
    --g-t-dyn-init 0.0 \
    --temporal-oversampling "$TEMPORAL_OVERSAMPLING"

echo "Finished:              $(date)"
