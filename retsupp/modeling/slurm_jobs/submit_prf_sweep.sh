#!/bin/bash
# Submit the full PRF model chain for all 30 subjects, both paradigm
# kinds. Each "stage" is an sbatch array job (one task per subject);
# downstream models wait via --dependency=afterok on the previous
# array.
#
# Chain (per kind):
#   m1 [no deps] -> m2 [after m1] -> m4 [after m2]
#                                    m5 [after m2] -> m6 [after m5]
#                   m3 [after m1]
#
# Usage:
#   bash submit_prf_sweep.sh            # both kinds (full + bar)
#   KINDS=full bash submit_prf_sweep.sh # full only
#   KINDS=bar  bash submit_prf_sweep.sh # bar only
#   SUBJECTS=1-5 KINDS=full bash submit_prf_sweep.sh   # subset
#
# Logs end up at ~/logs/prf_l4_m{N}_{kind}_*.txt as set by fit_prf_l4.sh.

set -eo pipefail

KINDS="${KINDS:-full bar}"
SUBJECTS="${SUBJECTS:-1-30}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/fit_prf_l4.sh"

# Extract the jobid from `sbatch` output. Strips " on cluster …" suffix.
sb() {
    local out
    out=$(sbatch "$@") || { echo "$out" >&2; return 1; }
    echo "$out" | awk '{print $4}'
}

submit_chain_for_kind() {
    local kind="$1"
    echo "=== submitting kind=$kind subjects=$SUBJECTS ==="

    local J1 J2 J3 J4 J5 J6

    J1=$(sb --array="$SUBJECTS" \
            --export=ALL,MODEL=1,KIND="$kind" \
            "$SLURM_SCRIPT")
    echo "  m1 -> jobid $J1"

    J2=$(sb --array="$SUBJECTS" \
            --dependency=afterok:"$J1" \
            --export=ALL,MODEL=2,KIND="$kind" \
            "$SLURM_SCRIPT")
    echo "  m2 -> jobid $J2 (after $J1)"

    J3=$(sb --array="$SUBJECTS" \
            --dependency=afterok:"$J1" \
            --export=ALL,MODEL=3,KIND="$kind" \
            "$SLURM_SCRIPT")
    echo "  m3 -> jobid $J3 (after $J1)"

    J4=$(sb --array="$SUBJECTS" \
            --dependency=afterok:"$J2" \
            --export=ALL,MODEL=4,KIND="$kind" \
            "$SLURM_SCRIPT")
    echo "  m4 -> jobid $J4 (after $J2)"

    J5=$(sb --array="$SUBJECTS" \
            --dependency=afterok:"$J2" \
            --export=ALL,MODEL=5,KIND="$kind" \
            "$SLURM_SCRIPT")
    echo "  m5 -> jobid $J5 (after $J2)"

    J6=$(sb --array="$SUBJECTS" \
            --dependency=afterok:"$J5" \
            --export=ALL,MODEL=6,KIND="$kind" \
            "$SLURM_SCRIPT")
    echo "  m6 -> jobid $J6 (after $J5)"
}

for k in $KINDS; do
    submit_chain_for_kind "$k"
done

echo "All chains submitted. Tail logs with:"
echo "  ssh sciencecluster 'tail -f ~/logs/prf_l4_m1_*.txt'"
