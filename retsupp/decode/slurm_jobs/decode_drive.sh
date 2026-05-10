#!/bin/bash
# Per-(subject, ROI) decode array. CPU-only — small linear algebra.
# Submit with explicit array length covering 28 subs x 8 ROIs = 224 cells.
#
# Build array index map (subject 1..30 minus {6, 8}, 8 ROIs):
#   idx = (sub_pos - 1) * 8 + roi_pos
# Where sub_pos enumerates 28 valid subjects in order, roi_pos = 1..8.
#
#   sbatch --array=1-224 retsupp/decode/slurm_jobs/decode_drive.sh
#
# Or limit concurrency to keep NFS happy:
#   sbatch --array=1-224%32 retsupp/decode/slurm_jobs/decode_drive.sh
#SBATCH --job-name=decode_drive
#SBATCH --output=/dev/null
#SBATCH --time=180:00
#SBATCH --account=zne.uzh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

LOGFILE="$HOME/logs/${SLURM_JOB_NAME:-decode_drive}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

# Defense-in-depth: spread out NFS profile reads.
sleep $(( (RANDOM % 30) + 1 ))

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate neural_priors2

export PYTHONUNBUFFERED=1

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"

# Subject list: 1..30 minus 6 and 8 (28 subjects).
SUBJECTS=(1 2 3 4 5 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
ROIS=(V1 V2 V3 V3AB hV4 VO LO TO)

idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
sub_pos=$(( idx / ${#ROIS[@]} ))
roi_pos=$(( idx % ${#ROIS[@]} ))

if [ "$sub_pos" -ge "${#SUBJECTS[@]}" ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID out of range for subject grid; exiting."
    exit 0
fi

SUB=${SUBJECTS[$sub_pos]}
ROI=${ROIS[$roi_pos]}

echo "[$(date)] sub-$SUB ROI=$ROI  on $(hostname)"

python -u $HOME/git/retsupp/retsupp/decode/run_decode.py "$SUB" \
    --roi "$ROI" \
    --bids-folder "$bids_folder" \
    --resolution 30 \
    --max-voxels 200 \
    --r2-min 0.0 \
    --ecc-max 6.0 \
    --learning-rate 0.5 \
    --max-n-iterations 600 \
    --resid-max-iter 300

echo "[$(date)] done"
