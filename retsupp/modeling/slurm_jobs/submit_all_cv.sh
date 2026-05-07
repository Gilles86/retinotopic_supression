#!/bin/bash
# Submit the full 17-class factorial CV sweep.
#
#   16 factorial cells: cross of {(zero,zero), (minus,zero), (minus,minus),
#                                  (plus,plus)} on the SUSTAINED pair
#                       × the same 4 patterns on the DYNAMIC pair.
#   +
#    1 signed-unconstrained control: (free, free, free, free).
#
# Each cell is submitted as one --array=1-240 sbatch job (30 subjects x
# 8 ROIs). Each array task internally loops over all 4 CV folds.
#
# Total: 17 * 240 = 4080 array tasks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/fit_af_prf_cv.sh"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "ERROR: cannot find $SLURM_SCRIPT" >&2
    exit 1
fi

# 4 legal sign-pair patterns for each of the SUSTAINED and DYNAMIC pairs.
# Format: "sus_hp_sign sus_lp_sign" / "dyn_hp_sign dyn_lp_sign".
PAIRS=("zero zero" "minus zero" "minus minus" "plus plus")

n=0
for sus in "${PAIRS[@]}"; do
    for dyn in "${PAIRS[@]}"; do
        # shellcheck disable=SC2086
        echo "Submitting sus=($sus) x dyn=($dyn)"
        sbatch --array=1-240 "$SLURM_SCRIPT" $sus $dyn
        n=$((n + 1))
    done
done

# Signed-unconstrained control.
echo "Submitting signed-control (free free free free)"
sbatch --array=1-240 "$SLURM_SCRIPT" free free free free
n=$((n + 1))

echo "Submitted $n sbatch arrays in total (expected 17)."
