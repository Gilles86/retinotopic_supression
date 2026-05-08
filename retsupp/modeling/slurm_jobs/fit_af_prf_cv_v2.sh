#!/bin/bash
#SBATCH --job-name=af_cv_v2
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Cross-validated DoG-dyn-v3 + target sharedSigma fit (CV-v2 18-class
# factorial, leave-one-condition-out CV, all 4 folds inside one job).
#
# Each array task processes ONE (subject, ROI, sign-class) and loops
# over all 4 CV folds inside the Python script. The class is encoded by
# 5 positional args (4 distractor signs + target gain).
#
# Usage:
#   sbatch --array=1-240 fit_af_prf_cv_v2.sh \\
#          <sus_hp> <sus_lp> <dyn_hp> <dyn_lp> <target_gain>
#
# Each of the 4 distractor sign args is one of {plus, zero, minus, free}.
# <target_gain> is one of {free, zero}.
#
# Submission of the full 18-cell factorial is in submit_all_cv_v2.sh.
#
# The array task id maps to (subject, roi) as:
#   idx0 = SLURM_ARRAY_TASK_ID - 1
#   subject = idx0 // 8 + 1                     # 1..30
#   roi     = ROIS[ idx0 %  8 ]
#
# Runs whose HP is "no distractor" or unknown are skipped inside the
# Python script.

set -euo pipefail

# --- Argument parsing. ---
SUS_HP="${1:-}"
SUS_LP="${2:-}"
DYN_HP="${3:-}"
DYN_LP="${4:-}"
TGT="${5:-}"
if [[ -z "$SUS_HP" || -z "$SUS_LP" || -z "$DYN_HP" || -z "$DYN_LP" \
        || -z "$TGT" ]]; then
    echo "ERROR: usage: sbatch --array=1-240 $0 <sus_hp> <sus_lp> <dyn_hp> <dyn_lp> <target_gain>" >&2
    echo "  distractor args in {plus, zero, minus, free}" >&2
    echo "  target_gain in {free, zero}" >&2
    exit 2
fi
for s in "$SUS_HP" "$SUS_LP" "$DYN_HP" "$DYN_LP"; do
    case "$s" in
        plus|zero|minus|free) ;;
        *) echo "ERROR: bad distractor sign '$s' — must be plus|zero|minus|free" >&2
           exit 2 ;;
    esac
done
case "$TGT" in
    free|zero) ;;
    *) echo "ERROR: bad target_gain '$TGT' — must be free or zero" >&2
       exit 2 ;;
esac

# --- Logging. ---
CLS="sus-${SUS_HP}-${SUS_LP}_dyn-${DYN_HP}-${DYN_LP}_tgt-${TGT}"
LOGFILE="$HOME/logs/cv-v2-${SLURM_JOB_NAME:-cv_v2}_${CLS}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Class:       ${CLS}"
echo "Signs:       sus_hp=${SUS_HP} sus_lp=${SUS_LP} dyn_hp=${DYN_HP} dyn_lp=${DYN_LP}"
echo "Target gain: ${TGT}"
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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TF_NUM_INTEROP_THREADS=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_af_prf_cv_v2.py"

echo "Running fit_af_prf_cv_v2.py (all 4 folds) for sub-${subject}, roi=${roi}, class=${CLS}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --sus-hp-sign "$SUS_HP" \
    --sus-lp-sign "$SUS_LP" \
    --dyn-hp-sign "$DYN_HP" \
    --dyn-lp-sign "$DYN_LP" \
    --target-gain "$TGT" \
    --resolution 50 \
    --max-voxels 500

echo "Finished:    $(date)"
