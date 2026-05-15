#!/bin/bash
# Submit a full per-subject PRF + downstream pipeline as one
# dependency chain per subject. Each subject runs cache → m1 (chunks +
# merge) → m2/m3 (parallel after m1) → m4/m5 (parallel after m2) → m6
# (after m5). For each model that produces NIfTIs, surface sampling +
# neuropythy registration is also chained in.
#
# Why per-subject (not phase-wide afterok): if one subject's chunk
# task fails, only that subject's downstream stops. Phase-wide deps
# block every subject's downstream. See retsupp/CLAUDE.md
# §"Per-subject dependency chains".
#
# Neuropythy uses the canonical freesurfer subject dir as scratch
# space (one location per subject), so neuropythy_mN runs MUST be
# serial within a subject. Surface sampling can run in parallel.
#
# Usage:
#   KIND=full bash submit_prf_sweep_persub.sh           # all subjects
#   SUBJECTS="1 5 10" KIND=full bash submit_prf_sweep_persub.sh   # subset
#   NICE_M3=10000 NICE_M5=10000 NICE_M6=10000 WITH_AF=1 \
#       SUBJECTS="3 5 8 10 13 17 20 26 27 30" \
#       bash submit_prf_sweep_persub.sh
#       # ↑ critical path m1→m2→m4 normal priority; m3/m5/m6 deprioritized;
#       #   AF (sharedSigma) chained per-subject after m4 neuropythy.

set -eo pipefail

KIND="${KIND:-full}"
N_CHUNKS="${N_CHUNKS:-10}"
SUBJECTS="${SUBJECTS:-$(seq 1 30)}"
BIDS="${BIDS:-/shares/zne.uzh/gdehol/ds-retsupp}"

# Per-model SLURM --nice value (higher = lower scheduling priority).
# Default 0 = no nice adjustment. Use to deprioritize non-critical-path
# models so m4 (and downstream AF) get scheduled first.
NICE_M1="${NICE_M1:-0}"
NICE_M2="${NICE_M2:-0}"
NICE_M3="${NICE_M3:-0}"
NICE_M4="${NICE_M4:-0}"
NICE_M5="${NICE_M5:-0}"
NICE_M6="${NICE_M6:-0}"

# WITH_AF=1 chains the canonical dynamic AF model
# (fit_dog_dyn_v3_target_sharedSigma) after each subject's m4 neuropythy.
WITH_AF="${WITH_AF:-0}"
N_ROIS_AF=11   # must match ROIS array in fit_dog_dyn_v3_target_sharedSigma.sh

# CANARY=1: submit chunk #1 alone, then chunks 2..N_CHUNKS gated on
# its success. If the canary fails fast (e.g., the sentinel in
# fit_prf.py catches a cuInit-race CPU fallback and exits 1) the
# other N-1 chunks don't waste a slot each. Default ON; flip to 0
# to restore the old "submit all chunks at once" behavior.
CANARY="${CANARY:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CACHE="$SCRIPT_DIR/build_cleaned_bold_cache.sh"
S_CHUNK="${S_CHUNK:-$SCRIPT_DIR/fit_prf_l4_chunked.sh}"
S_MERGE="$SCRIPT_DIR/merge_prf_chunks.sh"
S_SURF="$(cd "$SCRIPT_DIR/../../surface/slurm_jobs" && pwd)/sample_prf_to_surface.sh"
S_NEURO="$(cd "$SCRIPT_DIR/../../neuropythy/slurm_jobs" && pwd)/register_retinotopy.sh"
S_AF="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"

# Per-chunk wallclock observed on L4 lowprio:
#   m1: ~10 min (2000-iter grid+GD); fast chunks 5 min, slow 15
#   m2-m6: ~15-30 min (4500-iter two-stage schedule)
# Walltime is tight-with-headroom so preempted resubmits don't loop.
T_CHUNK_M1=00:20:00
T_CHUNK_M2=00:50:00
T_CHUNK_M3=00:50:00
T_CHUNK_M4=01:00:00
T_CHUNK_M5=01:00:00
T_CHUNK_M6=01:00:00
# (Bumped from 35 min after sub-13 m4 chunks TIMEOUT-ed at 00:35:08.
# Per-chunk iter speed varies ~10× across runs on the same data:
# fast paths finish 4500 iter in 7 min, slow paths take 35+ min.
# Likely XLA recompile + GPU type variance. 60 min has headroom.)
T_MERGE=00:03:00   # actual elapsed is 10-30s; tight window helps backfill
T_CACHE=00:15:00
T_SURF=00:08:00   # actual elapsed ~2:30-3:00; tight window helps backfill
T_NEURO=01:00:00

sb() { sbatch "$@" | awk '{print $4}'; }

block_already_done() {
    # Treat the model block as done iff the merged r2 NIfTI exists.
    local sub=$1 model=$2
    local sp=$(printf "%02d" "$sub")
    [[ -f "${BIDS}/derivatives/prf/model${model}/sub-${sp}/sub-${sp}_desc-r2.nii.gz" ]]
}

cache_already_done() {
    local sub=$1
    local sp=$(printf "%02d" "$sub")
    compgen -G "${BIDS}/derivatives/cleaned_bold_cache/sub-${sp}/sub-${sp}_kind-${KIND}_res-50.npz" >/dev/null
}

submit_model_block() {
    # Submit chunks → merge → surface → neuropythy for one (subject, model).
    # Args: sub, model, dep_chunks (afterok for chunks; empty if m1),
    #       dep_neuro (afterok for neuropythy from previous model; empty for m1)
    # Sets globals J_CHUNK J_MERGE J_SURF J_NEURO.
    #
    # Skips the whole block (sets the four J_* globals to "") iff the
    # merged r2 NIfTI already exists. Downstream blocks pass these
    # empty strings as their afterok deps and correctly submit with no
    # dependency (the file is already on disk).
    local sub=$1 model=$2 dep_chunks=$3 dep_neuro=$4
    if block_already_done "$sub" "$model"; then
        local sp=$(printf "%02d" "$sub")
        echo "  sub-${sp} m${model}: r2 exists; skipping chunks/merge/surf/neuro"
        J_CHUNK="" J_MERGE="" J_SURF="" J_NEURO=""
        return
    fi
    local DEP_C=""
    [[ -n "$dep_chunks" ]] && DEP_C="--dependency=afterok:$dep_chunks"
    local t_chunk_var="T_CHUNK_M${model}"
    local T_CHUNK=${!t_chunk_var}
    local nice_var="NICE_M${model}"
    local NICE_VAL=${!nice_var}
    local NICE_ARG=""
    [[ "$NICE_VAL" -gt 0 ]] && NICE_ARG="--nice=$NICE_VAL"

    local CHUNK_EXPORT="ALL,SUBJECT=$sub,MODEL=$model,N_CHUNKS=$N_CHUNKS,KIND=$KIND"
    if [[ "$CANARY" == "1" && "$N_CHUNKS" -gt 1 ]]; then
        # Canary chunk #1 — runs alone so the sentinels at job start
        # (assert_gpu_available_if_expected, OOM, etc.) get exercised
        # before we sink 9 more slots into the same fault.
        local J_CANARY=$(sb $NICE_ARG --array=1 --time=$T_CHUNK $DEP_C \
            --export=$CHUNK_EXPORT "$S_CHUNK")
        J_CHUNK=$(sb $NICE_ARG --array=2-$N_CHUNKS --time=$T_CHUNK \
            --dependency=afterok:$J_CANARY \
            --export=$CHUNK_EXPORT "$S_CHUNK")
        # Merge needs BOTH the canary AND the rest. Use compound dep.
        J_MERGE=$(sb $NICE_ARG --array=$sub --time=$T_MERGE \
            --dependency=afterok:$J_CANARY:$J_CHUNK \
            --export=ALL,MODEL=$model,KIND=$KIND \
            "$S_MERGE")
    else
        J_CHUNK=$(sb $NICE_ARG --array=1-$N_CHUNKS --time=$T_CHUNK $DEP_C \
            --export=$CHUNK_EXPORT "$S_CHUNK")
        J_MERGE=$(sb $NICE_ARG --array=$sub --time=$T_MERGE \
            --dependency=afterok:$J_CHUNK \
            --export=ALL,MODEL=$model,KIND=$KIND \
            "$S_MERGE")
    fi
    J_SURF=$(sb $NICE_ARG --partition=lowprio --array=$sub --time=$T_SURF \
        --dependency=afterok:$J_MERGE \
        --export=ALL,MODEL=$model \
        "$S_SURF")
    # neuropythy must be serial within subject; depends on its surface
    # AND the previous model's neuropythy (so freesurfer scratch dir
    # is no longer in use).
    local NEURO_DEP="afterok:$J_SURF"
    [[ -n "$dep_neuro" ]] && NEURO_DEP="$NEURO_DEP,afterok:$dep_neuro"
    J_NEURO=$(sb $NICE_ARG --partition=lowprio --array=$sub --time=$T_NEURO \
        --dependency=$NEURO_DEP \
        --export=ALL,MODEL=$model \
        "$S_NEURO")
}

submit_one_subject() {
    local sub=$1
    local sub_pad=$(printf "%02d" "$sub")
    local J_CACHE=""
    if cache_already_done "$sub"; then
        echo "  sub-${sub_pad} cache: exists; skipping"
    else
        J_CACHE=$(sb --array=$sub --time=$T_CACHE \
            --export=ALL,KIND=$KIND "$S_CACHE")
    fi

    # m1 chunks gated on cache; everything else cascades.
    submit_model_block "$sub" 1 "$J_CACHE" ""
    local J1_M=$J_MERGE J1_N=$J_NEURO

    submit_model_block "$sub" 2 "$J1_M" "$J1_N"
    local J2_M=$J_MERGE J2_N=$J_NEURO

    submit_model_block "$sub" 3 "$J1_M" "$J2_N"
    local J3_N=$J_NEURO

    submit_model_block "$sub" 4 "$J2_M" "$J3_N"
    local J4_N=$J_NEURO

    # AF chain (canonical dynamic v3 + target + sharedSigma). Depends on
    # m4 neuropythy so the inferred-varea atlas exists when AF asks for
    # per-ROI voxels. One array task per ROI for this subject.
    local J_AF=""
    if [[ "$WITH_AF" == "1" ]]; then
        local sub_idx=$((sub - 1))
        local af_start=$((sub_idx * N_ROIS_AF + 1))
        local af_end=$((sub * N_ROIS_AF))
        local AF_DEP=""
        [[ -n "$J4_N" ]] && AF_DEP="--dependency=afterok:$J4_N"
        J_AF=$(sb --array=${af_start}-${af_end} $AF_DEP "$S_AF")
    fi

    submit_model_block "$sub" 5 "$J2_M" "$J4_N"
    local J5_M=$J_MERGE J5_N=$J_NEURO

    submit_model_block "$sub" 6 "$J5_M" "$J5_N"

    if [[ -n "$J_AF" ]]; then
        echo "sub-${sub_pad}: cache=$J_CACHE  m4_neuro=$J4_N  af=$J_AF  last_neuro=$J_NEURO"
    else
        echo "sub-${sub_pad}: cache=$J_CACHE chain submitted (last neuropythy job $J_NEURO)"
    fi
}

echo "=== per-subject full pipeline, kind=$KIND, ${N_CHUNKS} chunks/subject ==="
for sub in $SUBJECTS; do
    submit_one_subject "$sub"
done
echo "All per-subject chains submitted."
