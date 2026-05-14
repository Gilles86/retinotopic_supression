#!/bin/bash
# Submit the canonical DoG v3+target+sharedSigma AF with a LOOSER posterior
# threshold (p_signal > 0.5) for all 30 subjects. Matches the OLD FDR-α=0.05
# pool size — the diagnostic test for whether the pSig>0.95 stringency
# is what killed the V3AB / TO / hV4 sustained suppression effects.
#
# Single variant (no aperture filter). Output dir:
#   af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_pSig0.5/

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_AF="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"
N_ROIS=11
MODEL=4
PSIGNAL_THR=0.5
BIDS="/shares/zne.uzh/gdehol/ds-retsupp"

SUBJECTS="${SUBJECTS:-$(seq 1 30)}"

sb() { sbatch "$@" | awk '{print $4}'; }

mixture_dep_for() {
    local sub=$1
    squeue --me -h -n r2_mixture --format='%F %i' \
      | awk -v s="$sub" '
          { if ($2 ~ "_\\[?"s"\\]?$") { print $1; exit } }'
}

for sub in $SUBJECTS; do
    sp=$(printf "%02d" $sub)
    sidecar="$BIDS/derivatives/prf/model${MODEL}/sub-${sp}/sub-${sp}_desc-p_signal.json"
    if [[ -f "$sidecar" ]]; then
        dep_flag=""; dep_label="none (sidecar present)"
    else
        mix_jid=$(mixture_dep_for $sub)
        if [[ -z "$mix_jid" ]]; then
            echo "sub-${sp}: SKIPPED — no sidecar and no queued mixture"; continue
        fi
        dep_flag="--dependency=afterok:${mix_jid}"; dep_label="afterok:${mix_jid}"
    fi
    base=$(( (sub - 1) * N_ROIS + 1 ))
    end=$(( base + N_ROIS - 1 ))
    J=$(sb --array=${base}-${end} $dep_flag \
        --export=ALL,PSIGNAL_THR=${PSIGNAL_THR},APERTURE_MASS_THR=0,MODEL=${MODEL} \
        "$S_AF")
    echo "sub-${sp}: pSig>${PSIGNAL_THR}: $J  dep=${dep_label}"
done

echo
echo "Submitted single variant (pSig>${PSIGNAL_THR}, no aperture) for $(echo $SUBJECTS | wc -w) subs."
