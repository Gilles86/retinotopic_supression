#!/bin/bash
# Submit CV PRF fits per (subject, model, fold) as a chain:
#   chunks (N_CHUNKS array tasks) → merge.
# One independent chain per (sub × model × fold). Failure in one
# chain doesn't block any other — matches the per-subject pattern
# from submit_prf_sweep_persub.sh.
#
# Usage:
#   bash retsupp/modeling/cv/slurm_jobs/submit_cv_persub.sh
#
# Defaults: all 30 subjects × all 6 models × 3 folds = 540 chains
# = 540 × 10 chunks + 540 merges = 5940 SLURM tasks.
#
# Override via env vars:
#   SUBJECTS="3 5 10"             # subset of subjects
#   MODELS="4 6"                  # subset of models
#   FOLDS="0 1 2"                 # subset of folds
#   N_CHUNKS=10                   # chunks per (sub, model, fold)
#   N_FOLDS=3                     # number of folds total
#   KIND=full                     # paradigm kind
#   T_CHUNK=00:25:00              # per-chunk walltime
#   SKIP_DONE=1                   # skip blocks where r2_test.nii.gz exists
#   DRY_RUN=1                     # print sbatch commands without submitting

set -eo pipefail

SUBJECTS="${SUBJECTS:-$(seq 1 30)}"
MODELS="${MODELS:-0 1 2 3 4 5 6}"
N_FOLDS="${N_FOLDS:-3}"
FOLDS="${FOLDS:-$(seq 0 $((N_FOLDS - 1)))}"
N_CHUNKS="${N_CHUNKS:-10}"
KIND="${KIND:-full}"
T_CHUNK="${T_CHUNK:-00:25:00}"
T_MERGE="${T_MERGE:-00:05:00}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
BIDS="${BIDS:-/shares/zne.uzh/gdehol/ds-retsupp}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CHUNK="$SCRIPT_DIR/fit_prf_cv_l4_chunked.sh"
S_MERGE="$SCRIPT_DIR/merge_prf_cv_chunks.sh"

block_already_done() {
    local sub=$1 model=$2 fold=$3
    local sp=$(printf "%02d" "$sub")
    [[ -f "${BIDS}/derivatives/prf_cv/model${model}/fold-${fold}/sub-${sp}/sub-${sp}_fold-${fold}_desc-r2_test.nii.gz" ]]
}

BASE_ENV=$(env | sort)
sb() {
    if [[ "$DRY_RUN" == "1" ]]; then
        # Variables passed as VAR=x in front of sb: whatever differs from the start env.
        local vars
        vars=$(comm -13 <(echo "$BASE_ENV") <(env | sort) | grep -v '^_=' | paste -sd' ' -)
        echo "DRY: ${vars:+$vars }sbatch $*" >&2
        echo "$$_dry_$RANDOM"
    else
        sbatch "$@" | awk '{print $4}'
    fi
}

total_submitted=0
total_skipped=0
for sub in $SUBJECTS; do
    sp=$(printf "%02d" "$sub")
    for model in $MODELS; do
        for fold in $FOLDS; do
            if [[ "$SKIP_DONE" == "1" ]] && block_already_done "$sub" "$model" "$fold"; then
                total_skipped=$((total_skipped + 1))
                continue
            fi
            J_CHUNK=$(SUBJECT=$sub MODEL=$model FOLD=$fold N_FOLDS=$N_FOLDS N_CHUNKS=$N_CHUNKS KIND=$KIND \
                      sb --array=1-$N_CHUNKS --time=$T_CHUNK "$S_CHUNK")
            J_MERGE=$(MODEL=$model FOLD=$fold sb --array=$sub --time=$T_MERGE \
                         --dependency=afterok:$J_CHUNK \
                         "$S_MERGE")
            echo "sub-${sp} m${model} fold${fold}: chunks=$J_CHUNK merge=$J_MERGE"
            total_submitted=$((total_submitted + 1))
        done
    done
done

echo
echo "submitted $total_submitted chains, skipped $total_skipped done"
