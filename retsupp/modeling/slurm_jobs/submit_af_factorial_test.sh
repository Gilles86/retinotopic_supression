#!/bin/bash
# Test the 4-model factorial on the canonical DoG v3+target AF.
# Single selection variant (pSig > 0.5, no aperture filter) — the user
# decided 2026-05-14 to stop iterating over selection threshold and stick
# with pSig 0.5 (FDR-α=0.05-equivalent pool size). Aperture filter dropped.
#
# 4 model variants × N subs × 11 ROIs:
#   M1 base               (sharedSigma only; σ_T_dyn := σ_dyn)
#   M2 sharedDynGain      (+ tie g_LP_dyn := g_HP_dyn)
#   M3 allSharedSigma     (+ tie σ_AF := σ_dyn := σ_T_dyn)
#   M4 allSharedSigma + sharedDynGain
#
# Total per subject: 4 × 11 = 44 array tasks.
#
# Usage:
#   bash submit_af_factorial_test.sh                  # sub-02 + sub-07
#   SUBJECTS="3 11" bash submit_af_factorial_test.sh  # specific subjects

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_AF="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"
N_ROIS=11
MODEL=4
PSIGNAL_THR=0.5
BIDS="/shares/zne.uzh/gdehol/ds-retsupp"

SUBJECTS="${SUBJECTS:-2 7}"

sb() { sbatch "$@" | awk '{print $4}'; }

# Find this subject's queued r2_mixture array-job ID (if any).
mixture_dep_for() {
    local sub=$1
    squeue --me -h -n r2_mixture --format='%F %i' \
      | awk -v s="$sub" '
          { if ($2 ~ "_\\[?"s"\\]?$") { print $1; exit } }'
}

# Submit one (model-variant) AF array for a subject.
# Args: sub  shared_dyn_gain(0|1)  all_shared_sigma(0|1)  dep_flag
submit_variant() {
    local sub=$1 sdg=$2 ass=$3 dep_flag=$4
    local base=$(( (sub - 1) * N_ROIS + 1 ))
    local end=$(( base + N_ROIS - 1 ))
    sb --array=${base}-${end} $dep_flag \
        --export=ALL,PSIGNAL_THR=${PSIGNAL_THR},APERTURE_MASS_THR=0,MODEL=${MODEL},SHARED_DYN_GAIN=${sdg},ALL_SHARED_SIGMA=${ass} \
        "$S_AF"
}

for sub in $SUBJECTS; do
    sp=$(printf "%02d" $sub)
    sidecar="$BIDS/derivatives/prf/model${MODEL}/sub-${sp}/sub-${sp}_desc-p_signal.json"
    if [[ -f "$sidecar" ]]; then
        dep_flag=""
        dep_label="none (sidecar present)"
    else
        mix_jid=$(mixture_dep_for $sub)
        if [[ -z "$mix_jid" ]]; then
            echo "sub-${sp}: SKIPPED — no sidecar and no queued mixture"
            continue
        fi
        dep_flag="--dependency=afterok:${mix_jid}"
        dep_label="afterok:${mix_jid}"
    fi

    echo "sub-${sp}: dep=${dep_label}"
    J=$(submit_variant $sub 0 0 "$dep_flag"); echo "  M1 (base):              $J"
    J=$(submit_variant $sub 1 0 "$dep_flag"); echo "  M2 sharedDynGain:       $J"
    J=$(submit_variant $sub 0 1 "$dep_flag"); echo "  M3 allSharedSigma:      $J"
    J=$(submit_variant $sub 1 1 "$dep_flag"); echo "  M4 both ties:           $J"
done

echo
echo "Each subject: 4 array jobs of ${N_ROIS} ROI tasks (4 model variants, pSig=${PSIGNAL_THR}, no apt)."
