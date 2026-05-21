#!/bin/bash
#SBATCH --job-name=klein_shift_pilot
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# Pilot run for DoGKleinShift_v3_target_6sigma — sub-3, sub-17, sub-23,
# all 11 ROIs (33 array tasks). Mirrors fit_three_models_pilot.sh.
#
# Memory: 128G. The Klein-shift forward materialises a (B, V, T, G)
# tensor via tf.map_fn over T, plus gradient buffers — same memory
# profile as the Gaussian analytical-shift model (which OOM'd at 24G
# and partially at 64G). 128G should cover the worst-case (sub-23 with
# all V1 voxels).
#
# Submission:
#   sbatch --array=1-33 retsupp/modeling/slurm_jobs/fit_klein_shift_pilot.sh

set -euo pipefail

LOGFILE="$HOME/logs/klein_shift_pilot_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:           $(hostname)"
echo "Job:            ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:        $(date)"

SUB_IDS=(3 17 23)
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-33." >&2
    exit 2
fi

idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_idx=$(( idx0 / N_ROIS ))
roi_idx=$(( idx0 % N_ROIS ))
if [[ "$sub_idx" -ge "$N_SUBS" ]]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range; max is $(( N_SUBS * N_ROIS ))." >&2
    exit 2
fi
subject="${SUB_IDS[$sub_idx]}"
roi="${ROIS[$roi_idx]}"

echo "Subject:        ${subject}"
echo "ROI:            ${roi}"

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

echo "Running DoG-Klein-shift fit for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --model-version v3 \
    --with-target \
    --shared-target-sigma \
    --klein-shift \
    --max-voxels 500 \
    --max-n-iterations 1500 \
    --p-signal-thr 0.5 \
    --mode signed

echo "Finished:       $(date)"
