#!/bin/bash
# Submit the full CV-v2 18-class factorial sweep.
#
#  9 distractor classes: cross of {(zero,zero), (minus,zero),
#                                  (minus,minus)} on the SUSTAINED pair
#                       × the same 3 patterns on the DYNAMIC pair.
#  ×
#  2 target options: {free, zero}.
#
# = 18 cells.
#
# Each cell is submitted as one --array=1-240 sbatch job (30 subjects x
# 8 ROIs).  Each array task internally loops over all 4 CV folds.
#
# Total: 18 * 240 = 4320 array tasks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/fit_af_prf_cv_v2.sh"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "ERROR: cannot find $SLURM_SCRIPT" >&2
    exit 1
fi

# 3 sign-pair patterns for each of the SUSTAINED and DYNAMIC pairs.
DIST_PAIRS=("zero zero" "minus zero" "minus minus")
TGT_OPTS=("free" "zero")

n=0
for sus in "${DIST_PAIRS[@]}"; do
    for dyn in "${DIST_PAIRS[@]}"; do
        for tgt in "${TGT_OPTS[@]}"; do
            # shellcheck disable=SC2086
            echo "Submitting sus=($sus) x dyn=($dyn) x tgt=$tgt"
            sbatch --array=1-240 "$SLURM_SCRIPT" $sus $dyn "$tgt"
            n=$((n + 1))
        done
    done
done

echo "Submitted $n sbatch arrays in total (expected 18)."
