#!/bin/bash
# Generate the (subject, roi, session, run) manifest, then submit one
# SLURM array job consuming it.
#
# Usage (cluster login node):
#   bash retsupp/decode/slurm_jobs/submit_decode_runwise.sh
#
# Knobs (env vars; defaults shown):
#   SUBJECTS="1..30"              # whitespace-separated subject ids
#   ROIS="V1 V2 V3 V3AB hV4 LO TO VO"
#   MAX_CONCURRENT=150            # SLURM array %N throttle
#   BIDS=/shares/zne.uzh/gdehol/ds-retsupp
#   MODEL=4 RESOLUTION=50 POSTERIOR=0.5 L2_NORM=1.0
#   LEARNING_RATE=0.01 MAX_ITER=1000 MIN_ITER=200 RESID_MAX_ITER=300
#   ENVNAME=retsupp_cuda          # CPU jobs still use the CUDA env
#                                 # because cluster `retsupp` is incomplete
#   DRY_RUN=1                     # build manifest, print sbatch line, do not submit
#
# The manifest is timestamped per call so repeat runs don't clobber a
# still-pending submission.

set -euo pipefail

ROIS="${ROIS:-V1 V2 V3 V3AB hV4 LO TO VO}"
SUBJECTS_DEFAULT=$(seq 1 30)
SUBJECTS="${SUBJECTS:-$SUBJECTS_DEFAULT}"
MAX_CONCURRENT="${MAX_CONCURRENT:-150}"
BIDS="${BIDS:-/shares/zne.uzh/gdehol/ds-retsupp}"
ENVNAME="${ENVNAME:-retsupp_cuda}"
MODEL="${MODEL:-4}"
RESOLUTION="${RESOLUTION:-50}"
POSTERIOR="${POSTERIOR:-0.5}"
L2_NORM="${L2_NORM:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
MAX_ITER="${MAX_ITER:-1000}"
MIN_ITER="${MIN_ITER:-200}"
RESID_MAX_ITER="${RESID_MAX_ITER:-300}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
SLURM_DIR="$REPO_ROOT/retsupp/decode/slurm_jobs"
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$SLURM_DIR/decode_runwise_manifest_${STAMP}.txt"

# subjects with non-standard run counts (CLAUDE.md):
#   sub-20 ses-1 -> runs 1..5    sub-24 ses-2 -> runs 1..5
# Other (sub, ses) -> runs 1..6.
runs_for() {
    local subj="$1" ses="$2"
    if [[ "$subj" -eq 20 && "$ses" -eq 1 ]]; then
        echo "1 2 3 4 5"
    elif [[ "$subj" -eq 24 && "$ses" -eq 2 ]]; then
        echo "1 2 3 4 5"
    else
        echo "1 2 3 4 5 6"
    fi
}

: > "$MANIFEST"
for subj in $SUBJECTS; do
    for roi in $ROIS; do
        for ses in 1 2; do
            for run in $(runs_for "$subj" "$ses"); do
                echo "$subj $roi $ses $run" >> "$MANIFEST"
            done
        done
    done
done

N=$(wc -l < "$MANIFEST")
echo "Wrote manifest: $MANIFEST ($N tasks)"
echo "  subjects : $SUBJECTS"
echo "  rois     : $ROIS"
echo "  model    : m$MODEL res=$RESOLUTION L2=$L2_NORM posterior=$POSTERIOR"

SBATCH_CMD=(
    sbatch
    --array="1-${N}%${MAX_CONCURRENT}"
    --export="ALL,MANIFEST=$MANIFEST,BIDS=$BIDS,ENVNAME=$ENVNAME,MODEL=$MODEL,RESOLUTION=$RESOLUTION,POSTERIOR=$POSTERIOR,L2_NORM=$L2_NORM,LEARNING_RATE=$LEARNING_RATE,MAX_ITER=$MAX_ITER,MIN_ITER=$MIN_ITER,RESID_MAX_ITER=$RESID_MAX_ITER"
    "$SLURM_DIR/decode_runwise.sh"
)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] would submit:"
    printf '  %q' "${SBATCH_CMD[@]}"; echo
    exit 0
fi

"${SBATCH_CMD[@]}"
