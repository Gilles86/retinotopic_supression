#!/bin/bash
# Submit the full PRF model chain for all subjects, ONE paradigm kind,
# using the chunked GPU path (10 chunks/subject). Cache is built first
# so the per-task BOLD load drops from ~60-120s to ~5s.
#
# Per-phase pipeline (single kind):
#   1. cache build       (30 CPU tasks, ~5 min each, parallel)
#   2. m1 chunks         (300 GPU tasks, ~10 min each)
#   3. m1 merge          (30 CPU tasks, ~1-2 min each)
#   4. m2 chunks         (300 GPU)
#   5. m2 merge          (30 CPU)
#   6. m3 chunks         (300 GPU)
#   7. m3 merge          (30 CPU)
#   8. m4 chunks         (300 GPU)
#   9. m4 merge          (30 CPU)
#  10. m5 chunks         (300 GPU)
#  11. m5 merge          (30 CPU)
#  12. m6 chunks         (300 GPU)
#  13. m6 merge          (30 CPU)
#
# Each phase i depends on phase i-1's merge via afterok (whole array
# must succeed). Per-task walltimes are *tight* per model — priority
# bucket matters. If a chunk task fails, the merge will block and the
# chain stops; resubmit the failed chunks and rerun the merge.
#
# Usage (single kind):
#   KIND=full bash submit_prf_sweep_chunked.sh
#   KIND=bar  bash submit_prf_sweep_chunked.sh
#
# Multi-kind: just run twice in sequence (or in parallel — different
# job IDs so no conflict).

set -eo pipefail

KIND="${KIND:-full}"
N_SUBS=30
N_CHUNKS=10
N_CHUNK_TASKS=$(( N_SUBS * N_CHUNKS ))    # 300

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CACHE="$SCRIPT_DIR/build_cleaned_bold_cache.sh"
S_CHUNK="$SCRIPT_DIR/fit_prf_l4_chunked.sh"
S_MERGE="$SCRIPT_DIR/merge_prf_chunks.sh"

# Tight per-model walltimes (chunks + merge). Bumped slightly for
# more-complex models per Gilles's empirical hint.
T_CHUNK_M1=00:15:00
T_CHUNK_M2=00:15:00
T_CHUNK_M3=00:15:00
T_CHUNK_M4=00:20:00
T_CHUNK_M5=00:20:00
T_CHUNK_M6=00:25:00
T_MERGE=00:10:00
T_CACHE=00:15:00

sb() {
    sbatch "$@" | awk '{print $4}'
}

submit_chunks() {
    local model=$1 ; local wall=$2 ; local dep=$3
    local depflag=""
    [[ -n "$dep" ]] && depflag="--dependency=afterok:$dep"
    sb --array=1-$N_CHUNK_TASKS --time=$wall $depflag \
        --export=ALL,MODEL=$model,KIND=$KIND,N_CHUNKS=$N_CHUNKS,N_SUBS=$N_SUBS \
        "$S_CHUNK"
}

submit_merge() {
    local model=$1 ; local dep=$2
    sb --array=1-$N_SUBS --time=$T_MERGE \
        --dependency=afterok:$dep \
        --export=ALL,MODEL=$model,KIND=$KIND \
        "$S_MERGE"
}

echo "=== chunked sweep, kind=$KIND, ${N_CHUNKS} chunks × ${N_SUBS} subjects ==="

CACHE=$(sb --array=1-$N_SUBS --time=$T_CACHE \
        --export=ALL,KIND=$KIND \
        "$S_CACHE")
echo "  cache build -> $CACHE"

J1_C=$(submit_chunks 1 $T_CHUNK_M1 "$CACHE")
echo "  m1 chunks    -> $J1_C (after $CACHE)"
J1_M=$(submit_merge 1 "$J1_C")
echo "  m1 merge     -> $J1_M (after $J1_C)"

J2_C=$(submit_chunks 2 $T_CHUNK_M2 "$J1_M")
J2_M=$(submit_merge 2 "$J2_C")
echo "  m2 chunks    -> $J2_C  merge -> $J2_M"

J3_C=$(submit_chunks 3 $T_CHUNK_M3 "$J1_M")
J3_M=$(submit_merge 3 "$J3_C")
echo "  m3 chunks    -> $J3_C  merge -> $J3_M  (init from m1)"

J4_C=$(submit_chunks 4 $T_CHUNK_M4 "$J2_M")
J4_M=$(submit_merge 4 "$J4_C")
echo "  m4 chunks    -> $J4_C  merge -> $J4_M  (init from m2)"

J5_C=$(submit_chunks 5 $T_CHUNK_M5 "$J2_M")
J5_M=$(submit_merge 5 "$J5_C")
echo "  m5 chunks    -> $J5_C  merge -> $J5_M  (init from m2)"

J6_C=$(submit_chunks 6 $T_CHUNK_M6 "$J5_M")
J6_M=$(submit_merge 6 "$J6_C")
echo "  m6 chunks    -> $J6_C  merge -> $J6_M  (init from m5)"

echo "All phases submitted for kind=$KIND."
