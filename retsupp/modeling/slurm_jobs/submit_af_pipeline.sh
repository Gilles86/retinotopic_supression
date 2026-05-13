#!/bin/bash
# Submit the attention-model (AF) pipeline per subject.
# Per subject, chains:
#   summarize_prf mean  (deps none — m4 NIfTIs already exist)
#   fit_condition       (deps none)
#   summarize_prf cond  (deps afterok fit_condition)
#   fit_attention_model (deps afterok summarize_mean + summarize_cond)
#
# All steps use MODEL=4 (DoG + flex HRF — the canonical model). ROIs
# come from neuropythy/model4 atlas via Subject.get_retinotopic_roi
# (loader fallback: canonical freesurfer dir if neuropythy snapshot
# absent).
#
# Skips subjects whose m4 NIfTI doesn't exist yet — re-run later.
#
# Usage:
#   bash submit_af_pipeline.sh               # all 30 subjects
#   SUBJECTS="1 5 10" bash submit_af_pipeline.sh
#   MODEL=4 bash submit_af_pipeline.sh

set -eo pipefail

MODEL="${MODEL:-4}"
SUBJECTS="${SUBJECTS:-$(seq 1 30)}"
BIDS="/shares/zne.uzh/gdehol/ds-retsupp"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_COND="$SCRIPT_DIR/fit_condition.sh"
S_SUMM="$SCRIPT_DIR/summarize_prf.sh"
S_AF="$SCRIPT_DIR/fit_attention_model.sh"

sb() { sbatch "$@" | awk '{print $4}'; }

n_submitted=0
n_skipped=0
for sub in $SUBJECTS; do
    sp=$(printf "%02d" "$sub")
    m4_nii="${BIDS}/derivatives/prf/model${MODEL}/sub-${sp}/sub-${sp}_desc-r2.nii.gz"
    if [[ ! -f "$m4_nii" ]]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Summarize mean (no deps — m4 NIfTI already there)
    J_MEAN=$(sb --array=$sub --export=ALL,MODEL=$MODEL,KIND=mean "$S_SUMM")

    # Conditionwise PRF refit (no deps — uses m4 mean as init)
    J_COND=$(sb --array=$sub --export=ALL,MODEL=$MODEL "$S_COND")

    # Summarize conditionwise (after fit_condition)
    J_SUMM_C=$(sb --array=$sub --dependency=afterok:$J_COND \
        --export=ALL,MODEL=$MODEL,KIND=conditionwise "$S_SUMM")

    # Fit attention model (after both summaries)
    J_AF=$(sb --array=$sub \
        --dependency=afterok:$J_MEAN,afterok:$J_SUMM_C \
        --export=ALL,MODEL=$MODEL "$S_AF")

    echo "sub-${sp}: summ_mean=$J_MEAN cond=$J_COND summ_c=$J_SUMM_C af=$J_AF"
    n_submitted=$((n_submitted + 1))
done

echo
echo "Submitted: $n_submitted subjects"
echo "Skipped (no m${MODEL} NIfTI yet): $n_skipped subjects"
