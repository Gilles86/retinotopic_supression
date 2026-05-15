#!/bin/bash
# One SLURM array task = one (subject, roi, session, run) decoding job.
#
# Reads the job manifest at $MANIFEST (default
# ~/git/retsupp/retsupp/decode/slurm_jobs/decode_runwise_manifest.txt);
# array task N consumes line N (space-separated:
# `subject roi session run`).
#
# Generate the manifest + submit with:
#   bash retsupp/decode/slurm_jobs/submit_decode_runwise.sh
#
# Standalone manual submit (after the manifest exists):
#   sbatch --array=1-N%150 retsupp/decode/slurm_jobs/decode_runwise.sh
#
#SBATCH --job-name=decode_rw
#SBATCH --output=/dev/null
#SBATCH --time=20:00
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

MANIFEST="${MANIFEST:-$HOME/git/retsupp/retsupp/decode/slurm_jobs/decode_runwise_manifest.txt}"
BIDS="${BIDS:-/shares/zne.uzh/gdehol/ds-retsupp}"
ENVNAME="${ENVNAME:-retsupp_cuda}"  # cluster `retsupp` is incomplete; cuda fallback is CPU-OK
MODEL="${MODEL:-4}"
RESOLUTION="${RESOLUTION:-50}"
POSTERIOR="${POSTERIOR:-0.5}"
L2_NORM="${L2_NORM:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
MAX_ITER="${MAX_ITER:-1000}"
MIN_ITER="${MIN_ITER:-200}"
RESID_MAX_ITER="${RESID_MAX_ITER:-300}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is unset; this script is array-only." >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 2
fi

LINE=$(awk -v n="$SLURM_ARRAY_TASK_ID" 'NR==n' "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "ERROR: empty manifest line $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
read -r SUBJECT ROI SESSION RUN <<<"$LINE"

JOB_TAG="sub-$(printf %02d "$SUBJECT")_roi-${ROI}_ses-${SESSION}_run-${RUN}"
LOGFILE="$HOME/logs/decode_rw_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${JOB_TAG}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

# Per-task rename — note SLURM JobName is array-shared so the LAST
# task to call this wins the displayed name (CLAUDE.md §"job-name
# convention"). The _<idx> suffix in JobID still uniquely IDs the task.
scontrol update jobid="${SLURM_JOB_ID}" name="decode_rw_${JOB_TAG}" || true

echo "[$(date)] === decode_runwise ==="
echo "  array task : $SLURM_ARRAY_TASK_ID"
echo "  job id     : $SLURM_JOB_ID"
echo "  node       : $SLURMD_NODENAME"
echo "  subject    : $SUBJECT"
echo "  roi        : $ROI"
echo "  session    : $SESSION"
echo "  run        : $RUN"
echo "  bids       : $BIDS"
echo "  model      : $MODEL"
echo "  resolution : $RESOLUTION"
echo "  posterior  : $POSTERIOR"
echo "  l2_norm    : $L2_NORM"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENVNAME"
set -u
export PYTHONUNBUFFERED=1

cd "$HOME/git/retsupp"

START=$(date +%s)
python -u -m retsupp.decode.decode_runwise \
    "$SUBJECT" \
    --session "$SESSION" \
    --run "$RUN" \
    --roi "$ROI" \
    --bids-folder "$BIDS" \
    --model "$MODEL" \
    --resolution "$RESOLUTION" \
    --posterior "$POSTERIOR" \
    --l2-norm "$L2_NORM" \
    --learning-rate "$LEARNING_RATE" \
    --max-n-iterations "$MAX_ITER" \
    --min-n-iterations "$MIN_ITER" \
    --resid-max-iter "$RESID_MAX_ITER"
END=$(date +%s)
echo "[$(date)] done in $((END - START)) s"
