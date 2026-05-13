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

set -eo pipefail

KIND="${KIND:-full}"
N_CHUNKS=10
SUBJECTS="${SUBJECTS:-$(seq 1 30)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CACHE="$SCRIPT_DIR/build_cleaned_bold_cache.sh"
S_CHUNK="$SCRIPT_DIR/fit_prf_l4_chunked.sh"
S_MERGE="$SCRIPT_DIR/merge_prf_chunks.sh"
S_SURF="$(cd "$SCRIPT_DIR/../../surface/slurm_jobs" && pwd)/sample_prf_to_surface.sh"
S_NEURO="$(cd "$SCRIPT_DIR/../../neuropythy/slurm_jobs" && pwd)/register_retinotopy.sh"

T_CHUNK=00:45:00   # per chunk; matches lowprio empirical timing
T_MERGE=00:10:00
T_CACHE=00:15:00
T_SURF=00:30:00
T_NEURO=01:00:00

sb() { sbatch "$@" | awk '{print $4}'; }

submit_model_block() {
    # Submit chunks → merge → surface → neuropythy for one (subject, model).
    # Args: sub, model, dep_chunks (afterok for chunks; empty if m1),
    #       dep_neuro (afterok for neuropythy from previous model; empty for m1)
    # Sets globals J_CHUNK J_MERGE J_SURF J_NEURO.
    local sub=$1 model=$2 dep_chunks=$3 dep_neuro=$4
    local DEP_C=""
    [[ -n "$dep_chunks" ]] && DEP_C="--dependency=afterok:$dep_chunks"

    J_CHUNK=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK $DEP_C \
        --export=ALL,SUBJECT=$sub,MODEL=$model,N_CHUNKS=$N_CHUNKS,KIND=$KIND \
        "$S_CHUNK")
    J_MERGE=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J_CHUNK \
        --export=ALL,MODEL=$model,KIND=$KIND \
        "$S_MERGE")
    J_SURF=$(sb --partition=lowprio --array=$sub --time=$T_SURF \
        --dependency=afterok:$J_MERGE \
        --export=ALL,MODEL=$model \
        "$S_SURF")
    # neuropythy must be serial within subject; depends on its surface
    # AND the previous model's neuropythy (so freesurfer scratch dir
    # is no longer in use).
    local NEURO_DEP="afterok:$J_SURF"
    [[ -n "$dep_neuro" ]] && NEURO_DEP="$NEURO_DEP,afterok:$dep_neuro"
    J_NEURO=$(sb --partition=lowprio --array=$sub --time=$T_NEURO \
        --dependency=$NEURO_DEP \
        --export=ALL,MODEL=$model \
        "$S_NEURO")
}

submit_one_subject() {
    local sub=$1
    local sub_pad=$(printf "%02d" "$sub")
    local J_CACHE=$(sb --array=$sub --time=$T_CACHE \
        --export=ALL,KIND=$KIND "$S_CACHE")

    # m1 chunks gated on cache; everything else cascades.
    submit_model_block "$sub" 1 "$J_CACHE" ""
    local J1_M=$J_MERGE J1_N=$J_NEURO

    submit_model_block "$sub" 2 "$J1_M" "$J1_N"
    local J2_M=$J_MERGE J2_N=$J_NEURO

    submit_model_block "$sub" 3 "$J1_M" "$J2_N"
    local J3_N=$J_NEURO

    submit_model_block "$sub" 4 "$J2_M" "$J3_N"
    local J4_N=$J_NEURO

    submit_model_block "$sub" 5 "$J2_M" "$J4_N"
    local J5_M=$J_MERGE J5_N=$J_NEURO

    submit_model_block "$sub" 6 "$J5_M" "$J5_N"

    echo "sub-${sub_pad}: cache=$J_CACHE chain submitted (last neuropythy job $J_NEURO)"
}

echo "=== per-subject full pipeline, kind=$KIND, ${N_CHUNKS} chunks/subject ==="
for sub in $SUBJECTS; do
    submit_one_subject "$sub"
done
echo "All per-subject chains submitted."
