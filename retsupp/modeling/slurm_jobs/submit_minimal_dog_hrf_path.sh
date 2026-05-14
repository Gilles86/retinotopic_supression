#!/bin/bash
# Minimal critical-path submitter: get DoG+HRF (m4) AF for every subject
# ASAP. Per subject, only refit what's NOT already on disk. Built after
# the 2026-05-14 audit found ~220 GPU-h of redundant refits queued via
# the full-sweep submitter when most subjects already had m1-m6 fresh.
#
# Categories (decision tree, top to bottom — first match wins):
#
#   [A] m4 NIfTI on disk           -> mixture + AF only           (19 subs)
#   [B] m2 NIfTI on disk           -> m4 chunks/merge + AF        (2 subs)
#   [C] m1 NIfTI on disk           -> m2/m4 chunks/merges + AF    (8 subs)
#   [D] m1 chunks on disk          -> m1 merge + m2 + m4 + AF     (1 sub: sub-30)
#   [E] nothing                    -> cache + full chain + AF     (0 subs today)
#
# All AF runs use mixture-FDR (USE_FDR=1) and model 4 (DoG + HRF).

set -eo pipefail

BIDS="/shares/zne.uzh/gdehol/ds-retsupp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CACHE="$SCRIPT_DIR/build_cleaned_bold_cache.sh"
S_CHUNK="$SCRIPT_DIR/fit_prf_l4_chunked.sh"
S_MERGE="$SCRIPT_DIR/merge_prf_chunks.sh"
S_MIX="$SCRIPT_DIR/run_r2_mixture.sh"
S_AF="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"

KIND="full"
N_CHUNKS=10
N_ROIS=11   # V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF

T_CHUNK="00:35:00"
T_MERGE="00:03:00"
T_CACHE="00:15:00"

sb() { sbatch "$@" | awk '{print $4}'; }

has_nifti() {
    local sub=$1 model=$2
    local sp=$(printf "%02d" "$sub")
    [[ -f "${BIDS}/derivatives/prf/model${model}/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]]
}

has_chunks() {
    local sub=$1 model=$2
    local sp=$(printf "%02d" "$sub")
    local d="${BIDS}/derivatives/prf/model${model}/sub-${sp}/chunks"
    [[ -d "$d" ]] && [[ $(ls "$d"/chunk-*.npz 2>/dev/null | wc -l) -ge "$N_CHUNKS" ]]
}

submit_chunks() {
    local sub=$1 model=$2 dep=$3
    local DEP=""
    [[ -n "$dep" ]] && DEP="--dependency=afterok:$dep"
    sb --array=1-$N_CHUNKS --time=$T_CHUNK $DEP \
        --export=ALL,SUBJECT=$sub,MODEL=$model,N_CHUNKS=$N_CHUNKS,KIND=$KIND \
        "$S_CHUNK"
}

submit_merge() {
    local sub=$1 model=$2 dep=$3
    local DEP=""
    [[ -n "$dep" ]] && DEP="--dependency=afterok:$dep"
    sb --array=$sub --time=$T_MERGE $DEP \
        --export=ALL,MODEL=$model,KIND=$KIND \
        "$S_MERGE"
}

submit_model_block() {
    local sub=$1 model=$2 dep=$3
    local JC=$(submit_chunks "$sub" "$model" "$dep")
    submit_merge "$sub" "$model" "$JC"
}

submit_mix_and_af() {
    local sub=$1 dep=$2
    local DEP=""
    [[ -n "$dep" ]] && DEP="--dependency=afterok:$dep"
    local J_MIX=$(sb --array=$sub $DEP "$S_MIX")
    local base=$(( (sub - 1) * N_ROIS + 1 ))
    local end=$(( base + N_ROIS - 1 ))
    local J_AF=$(sb --array=${base}-${end} --dependency=afterok:$J_MIX \
        --export=ALL,USE_FDR=1,MODEL=4 "$S_AF")
    echo "mix=$J_MIX af=$J_AF"
}

n_a=0; n_b=0; n_c=0; n_d=0; n_e=0
for sub in $(seq 1 30); do
    sp=$(printf "%02d" "$sub")
    if has_nifti "$sub" 4; then
        out=$(submit_mix_and_af "$sub" "")
        echo "sub-${sp} [A]: $out"
        n_a=$((n_a + 1))
    elif has_nifti "$sub" 2; then
        JM4=$(submit_model_block "$sub" 4 "")
        out=$(submit_mix_and_af "$sub" "$JM4")
        echo "sub-${sp} [B]: m4-merge=$JM4  $out"
        n_b=$((n_b + 1))
    elif has_nifti "$sub" 1; then
        JM2=$(submit_model_block "$sub" 2 "")
        JM4=$(submit_model_block "$sub" 4 "$JM2")
        out=$(submit_mix_and_af "$sub" "$JM4")
        echo "sub-${sp} [C]: m2=$JM2 m4=$JM4 $out"
        n_c=$((n_c + 1))
    elif has_chunks "$sub" 1; then
        JM1=$(submit_merge "$sub" 1 "")
        JM2=$(submit_model_block "$sub" 2 "$JM1")
        JM4=$(submit_model_block "$sub" 4 "$JM2")
        out=$(submit_mix_and_af "$sub" "$JM4")
        echo "sub-${sp} [D]: m1-merge=$JM1 m2=$JM2 m4=$JM4 $out"
        n_d=$((n_d + 1))
    else
        JCACHE=$(sb --array=$sub --time=$T_CACHE --export=ALL,KIND=$KIND "$S_CACHE")
        JM1=$(submit_model_block "$sub" 1 "$JCACHE")
        JM2=$(submit_model_block "$sub" 2 "$JM1")
        JM4=$(submit_model_block "$sub" 4 "$JM2")
        out=$(submit_mix_and_af "$sub" "$JM4")
        echo "sub-${sp} [E]: cache=$JCACHE m1=$JM1 m2=$JM2 m4=$JM4 $out"
        n_e=$((n_e + 1))
    fi
done

echo
echo "Summary:"
echo "  A (m4 done):                  $n_a subs"
echo "  B (m2 done, m4 missing):      $n_b subs"
echo "  C (m1 done, m2 missing):      $n_c subs"
echo "  D (m1 chunks only):           $n_d subs"
echo "  E (nothing on disk):          $n_e subs"
