#!/bin/bash
#SBATCH --job-name=dog_dyn_rect_sub02_V3AB
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# v3 + target + sharedSigma DoG-AF + DoG-PRF braincoder fit, with the
# distractor footprint rendered as an ORIENTED RECTANGLE (1.5 x 0.5 deg)
# rather than the legacy 0.4-deg disk. Single sub-2 / V3AB sanity-check
# fit to compare against the existing circle baseline at
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma/
# Output goes to
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_rect/
# so it never collides with the canonical circle results.
#
# This is NOT an array job. Submit with:
#   sbatch retsupp/modeling/slurm_jobs/fit_dog_dyn_v3_target_sharedSigma_rect_sub02_V3AB.sh

set -euo pipefail

# --- Logging. ---
LOGFILE="$HOME/logs/dog_dyn_v3_target_sharedSigma_rect_sub02_V3AB_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID}"
echo "Started:     $(date)"
echo "Test:        v3 + target + sharedSigma + RECTANGLE distractors"

subject=2
roi=V3AB
echo "Subject:     ${subject}"
echo "ROI:         ${roi}"

# --- Conda env (CUDA env, forced to CPU per fit_dog_dynamic_af_braincoder_cpu.sh). ---
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

echo "Running v3 + target + sharedSigma + RECT fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --resolution 50 \
    --max-voxels 500 \
    --model-version v3 \
    --with-target \
    --shared-target-sigma \
    --sigma-af-init 2.0 \
    --sigma-dyn-init 2.0 \
    --sigma-t-dyn-init 2.0 \
    --g-t-dyn-init 0.0 \
    --distractor-shape rectangle \
    --distractor-long-side 1.5 \
    --distractor-short-side 0.5

echo "Finished:    $(date)"
