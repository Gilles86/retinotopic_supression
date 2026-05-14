#!/bin/bash
# Fill in missing m3 (Gaussian + flex HRF), m5 (DN, fixed HRF), m6 (DN + flex HRF)
# PRF fits for any subject that doesn't have them on disk. m3 is required for
# Gaussian-AF; m5/m6 for the future DN-AF work.
#
# Dependency chains (each per-subject):
#   m3 needs m1                 (init from m1 PRF NIfTI)
#   m5 needs m2                 (init from m2 PRF NIfTI)
#   m6 needs m5                 (init from m5 PRF NIfTI)
#
# Strategy: per subject, check what's on disk and submit chunks+merge for
# what's missing, with afterok on the queued merge of the upstream model if
# the upstream isn't already on disk.
#
# Usage:
#   bash submit_prf_m3_m6_fill.sh
#   SUBJECTS="3 5 8 10 13 17 20 22 26 27 30" bash submit_prf_m3_m6_fill.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CHUNK="$SCRIPT_DIR/fit_prf_l4_chunked.sh"
S_MERGE="$SCRIPT_DIR/merge_prf_chunks.sh"
BIDS="/shares/zne.uzh/gdehol/ds-retsupp"
KIND="full"
N_CHUNKS=10

SUBJECTS="${SUBJECTS:-$(seq 1 30)}"

sb() { sbatch "$@" | awk '{print $4}'; }

has_nifti() {
    local sub=$1 model=$2
    local sp=$(printf "%02d" "$sub")
    [[ -f "${BIDS}/derivatives/prf/model${model}/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]]
}

# Find this subject's already-queued prf_merge job ID for a given model.
# We rely on the renamed JobName (set via scontrol update inside merge script)
# OR fall back to the merge job's MODEL env var via scontrol -d.
# Simpler: caller can pass dep_jid as the merge ID returned by the previous
# block. We track via local variables.

submit_model_block() {
    # Submit chunks + merge for one (subject, model) with optional afterok.
    # Returns the merge job ID.
    local sub=$1 model=$2 dep=$3
    local DEP=""
    [[ -n "$dep" ]] && DEP="--dependency=afterok:$dep"
    local JC=$(sb --array=1-$N_CHUNKS --time=00:35:00 $DEP \
        --export=ALL,SUBJECT=$sub,MODEL=$model,N_CHUNKS=$N_CHUNKS,KIND=$KIND \
        "$S_CHUNK")
    local JM=$(sb --array=$sub --time=00:03:00 --dependency=afterok:$JC \
        --export=ALL,MODEL=$model,KIND=$KIND \
        "$S_MERGE")
    echo "$JM"
}

n_m3=0; n_m5=0; n_m6=0
for sub in $SUBJECTS; do
    sp=$(printf "%02d" "$sub")
    has_m1=$([[ -f "${BIDS}/derivatives/prf/model1/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]] && echo 1 || echo 0)
    has_m2=$([[ -f "${BIDS}/derivatives/prf/model2/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]] && echo 1 || echo 0)
    has_m3=$([[ -f "${BIDS}/derivatives/prf/model3/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]] && echo 1 || echo 0)
    has_m5=$([[ -f "${BIDS}/derivatives/prf/model5/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]] && echo 1 || echo 0)
    has_m6=$([[ -f "${BIDS}/derivatives/prf/model6/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]] && echo 1 || echo 0)

    out=""
    # m3 fill — needs m1 (already done for 29/30).
    if [[ "$has_m3" == "0" && "$has_m1" == "1" ]]; then
        JM3=$(submit_model_block "$sub" 3 "")
        out="$out m3-merge=$JM3"
        n_m3=$((n_m3 + 1))
    fi

    # m5 fill — needs m2.
    # If m2 not on disk, we'd need afterok on the queued m2 merge for this
    # sub (submit_minimal_dog_hrf_path.sh handles m2 elsewhere). For
    # simplicity here, skip m5 if m2 is missing (resubmit when m2 lands).
    JM5=""
    if [[ "$has_m5" == "0" && "$has_m2" == "1" ]]; then
        JM5=$(submit_model_block "$sub" 5 "")
        out="$out m5-merge=$JM5"
        n_m5=$((n_m5 + 1))
    fi

    # m6 fill — needs m5 (just submitted, or already on disk).
    if [[ "$has_m6" == "0" ]]; then
        # Need m5 first. If just submitted, depend on it; else require disk.
        if [[ -n "$JM5" ]]; then
            JM6=$(submit_model_block "$sub" 6 "$JM5")
            out="$out m6-merge=$JM6"
            n_m6=$((n_m6 + 1))
        elif [[ "$has_m5" == "1" ]]; then
            JM6=$(submit_model_block "$sub" 6 "")
            out="$out m6-merge=$JM6"
            n_m6=$((n_m6 + 1))
        else
            out="$out m6-SKIPPED-needs-m5"
        fi
    fi

    [[ -n "$out" ]] && echo "sub-${sp}:$out"
done

echo
echo "Submitted: m3=$n_m3   m5=$n_m5   m6=$n_m6"
echo "Note: subs missing both m2 and m{5,6} need m2 to finish first; rerun this submitter when m2 lands."
