#!/bin/bash
# Convert an in-flight phase-wide chunked sweep to per-subject chains
# WITHOUT losing the running m1 chunks. Cancels the 11 pending
# downstream array jobs and resubmits each subject's m1_merge → m6
# chain individually, with sub-N's m1 merge depending on exactly the
# 10 m1 chunk tasks that belong to that subject.
#
# Usage:
#   M1_CHUNKS_JOB=2951327 KIND=full \
#     bash salvage_to_persub.sh \
#       2951328 2951329 2951330 2951331 2951332 2951333 \
#       2951334 2951335 2951336 2951337 2951338
#
# The trailing job IDs are the phase-wide downstream jobs to cancel.
# M1_CHUNKS_JOB is the still-running m1 chunks array whose 10
# per-subject tasks each subject's new merge will depend on.

set -eo pipefail

KIND="${KIND:-full}"
N_CHUNKS="${N_CHUNKS:-10}"
SUBJECTS="${SUBJECTS:-$(seq 1 30)}"
M1_CHUNKS_JOB="${M1_CHUNKS_JOB:?must set M1_CHUNKS_JOB (the running m1 chunks array)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_CHUNK="$SCRIPT_DIR/fit_prf_l4_chunked.sh"
S_MERGE="$SCRIPT_DIR/merge_prf_chunks.sh"

T_CHUNK_M2=00:15:00
T_CHUNK_M3=00:15:00
T_CHUNK_M4=00:20:00
T_CHUNK_M5=00:20:00
T_CHUNK_M6=00:25:00
T_MERGE=00:10:00

sb() { sbatch "$@" | awk '{print $4}'; }

# Cancel the in-flight phase-wide downstream jobs (passed as args).
if [[ "$#" -gt 0 ]]; then
    echo "Cancelling phase-wide downstream: $*"
    scancel "$@"
fi

per_sub_m1chunk_dep() {
    # Echo the afterok dependency clause for sub-N's 10 m1 chunk tasks.
    local sub=$1
    local k_start=$(( (sub - 1) * N_CHUNKS + 1 ))
    local k_end=$(( sub * N_CHUNKS ))
    local dep=""
    for k in $(seq $k_start $k_end); do
        dep="${dep}${M1_CHUNKS_JOB}_${k}:"
    done
    echo "${dep%:}"
}

submit_one_subject() {
    local sub=$1
    local sub_pad=$(printf "%02d" "$sub")

    # m1 merge: depends on this subject's 10 m1 chunk tasks only.
    local m1_dep=$(per_sub_m1chunk_dep $sub)
    local J1_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$m1_dep \
        --export=ALL,MODEL=1,KIND=$KIND \
        "$S_MERGE")

    local J2_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M2 \
        --dependency=afterok:$J1_M \
        --export=ALL,MODEL=2,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J2_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J2_C \
        --export=ALL,MODEL=2,KIND=$KIND "$S_MERGE")

    local J3_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M3 \
        --dependency=afterok:$J1_M \
        --export=ALL,MODEL=3,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J3_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J3_C \
        --export=ALL,MODEL=3,KIND=$KIND "$S_MERGE")

    local J4_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M4 \
        --dependency=afterok:$J2_M \
        --export=ALL,MODEL=4,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J4_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J4_C \
        --export=ALL,MODEL=4,KIND=$KIND "$S_MERGE")

    local J5_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M5 \
        --dependency=afterok:$J2_M \
        --export=ALL,MODEL=5,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J5_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J5_C \
        --export=ALL,MODEL=5,KIND=$KIND "$S_MERGE")

    local J6_C=$(sb --array=1-$N_CHUNKS --time=$T_CHUNK_M6 \
        --dependency=afterok:$J5_M \
        --export=ALL,MODEL=6,KIND=$KIND,SUBJECT=$sub,N_CHUNKS=$N_CHUNKS \
        "$S_CHUNK")
    local J6_M=$(sb --array=$sub --time=$T_MERGE \
        --dependency=afterok:$J6_C \
        --export=ALL,MODEL=6,KIND=$KIND "$S_MERGE")

    echo "sub-${sub_pad}: m1_merge=${J1_M} (deps ${M1_CHUNKS_JOB}_$((((sub-1)*N_CHUNKS+1)))-$((sub*N_CHUNKS)))"\
" m2=${J2_C}/${J2_M} m3=${J3_C}/${J3_M} m4=${J4_C}/${J4_M} m5=${J5_C}/${J5_M} m6=${J6_C}/${J6_M}"
}

echo "=== salvaging: pinning per-subject deps on running m1_chunks=$M1_CHUNKS_JOB ==="
for sub in $SUBJECTS; do
    submit_one_subject "$sub"
done
echo "Done. Per-subject chains now flow as each subject's m1 chunks finish."
