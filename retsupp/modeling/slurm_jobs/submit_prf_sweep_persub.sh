#!/bin/bash
# Submit the chunked PRF sweep as PER-SUBJECT dependency chains, one
# kind. Each subject gets its own m1 → m1_merge → m2 → m2_merge → ...
# → m6 → m6_merge chain, with afterok deps between phases of that
# subject only.
#
# Why per-subject and not phase-wide afterok: if one subject's chunk
# task fails, only that subject's downstream models stop. Phase-wide
# afterok blocks the merge for the whole array, blocking every
# subject's downstream models too. See retsupp/CLAUDE.md
# §"Per-subject dependency chains".
#
# Usage:
#   KIND=full bash submit_prf_sweep_persub.sh           # all subjects
#   SUBJECTS="1 5 10" KIND=full bash submit_prf_sweep_persub.sh   # subset
#
# Submits ~12 jobs per subject (cache + 6 chunk arrays + 6 merge
# arrays), so 30 subjects ≈ 360 sbatch calls. sbatch takes ~1-2s
# each, so the submission itself takes ~5-10 min — but that's a
# one-time cost; the actual fit wallclock is unchanged vs the
# phase-wide submitter (~1.5-2h end-to-end with sufficient
# parallelism).

set -eo pipefail

KIND="${KIND:-full}"
N_CHUNKS=10
SUBJECTS="${SUBJECTS:-$(seq 1 30)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CACHE="$SCRIPT_DIR/build_cleaned_bold_cache.sh"
S_CHUNK="$SCRIPT_DIR/fit_prf_l4_chunked.sh"
S_MERGE="$SCRIPT_DIR/merge_prf_chunks.sh"

# Tight per-model walltimes (slightly more for the heavier models per
# Gilles's empirical hint that complexity costs ~bit, not crazy much).
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

submit_one_subject() {
    local sub=$1
    local sub_pad=$(printf "%02d" "$sub")

    # cache (single task = single subject)
    local J_CACHE=$(sb --array=$sub --time=$T_CACHE \
        --export=ALL,KIND=$KIND \
        "$S_CACHE")

    # m1 chunks → merge.  SUBJECT env var pins the subject; array
    # index enumerates chunks 0..N_CHUNKS-1 (offset by 1 for SLURM).
    local J1_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M1 \
        --dependency=afterok:$J_CACHE \
        --export=ALL,MODEL=1,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J1_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J1_C \
        --export=ALL,MODEL=1,KIND=$KIND \
        "$S_MERGE")

    # m2/m3 chain off m1 (per the MODEL_CFG.init_from graph)
    local J2_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M2 \
        --dependency=afterok:$J1_M \
        --export=ALL,MODEL=2,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J2_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J2_C \
        --export=ALL,MODEL=2,KIND=$KIND \
        "$S_MERGE")

    local J3_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M3 \
        --dependency=afterok:$J1_M \
        --export=ALL,MODEL=3,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J3_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J3_C \
        --export=ALL,MODEL=3,KIND=$KIND \
        "$S_MERGE")

    # m4 / m5 off m2; m6 off m5
    local J4_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M4 \
        --dependency=afterok:$J2_M \
        --export=ALL,MODEL=4,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J4_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J4_C \
        --export=ALL,MODEL=4,KIND=$KIND \
        "$S_MERGE")

    local J5_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M5 \
        --dependency=afterok:$J2_M \
        --export=ALL,MODEL=5,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J5_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J5_C \
        --export=ALL,MODEL=5,KIND=$KIND \
        "$S_MERGE")

    local J6_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M6 \
        --dependency=afterok:$J5_M \
        --export=ALL,MODEL=6,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J6_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J6_C \
        --export=ALL,MODEL=6,KIND=$KIND \
        "$S_MERGE")

    echo "sub-${sub_pad}: cache=$J_CACHE  m1=${J1_C}/${J1_M}  "\
"m2=${J2_C}/${J2_M}  m3=${J3_C}/${J3_M}  "\
"m4=${J4_C}/${J4_M}  m5=${J5_C}/${J5_M}  m6=${J6_C}/${J6_M}"
}

echo "=== per-subject chunked sweep, kind=$KIND, ${N_CHUNKS} chunks/subject ==="
for sub in $SUBJECTS; do
    submit_one_subject "$sub"
done
echo "All per-subject chains submitted."
