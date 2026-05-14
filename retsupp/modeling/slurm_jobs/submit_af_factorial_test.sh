#!/bin/bash
# Test the 4×2 model/selection factorial on the canonical DoG v3+target AF
# for a SHORT subject list (default sub-02 + sub-07, both Cat A with the
# mixture sidecar already on disk → AF can fire immediately).
#
# 4 model variants (× 2 selection variants × 11 ROIs × N subs):
#   M1 base               (sharedSigma only; σ_T_dyn := σ_dyn)
#   M2 sharedDynGain      (+ tie g_LP_dyn := g_HP_dyn)
#   M3 allSharedSigma     (+ tie σ_AF := σ_dyn := σ_T_dyn)
#   M4 allSharedSigma + sharedDynGain
#
# Each model variant is run twice:
#   selA: p_signal>0.95 only           → ..._sharedSigma{...}_pSig0.95
#   selB: p_signal>0.95 + aperture≥0.5 → ..._sharedSigma{...}_pSig0.95_apt0.5
#
# Total per subject: 4 × 2 × 11 = 88 array tasks.
#
# Usage:
#   bash submit_af_factorial_test.sh                  # sub-02 + sub-07
#   SUBJECTS="3 11" bash submit_af_factorial_test.sh  # specific subjects

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_AF="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"
N_ROIS=11
MODEL=4
PSIGNAL_THR=0.95
APERTURE_THR=0.5
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

# Submit one (model-variant, selection-variant) AF array for a subject.
# Args: sub  model_label_tag(M1|M2|M3|M4)  shared_dyn_gain(0|1)  all_shared_sigma(0|1)
#       sel_tag(selA|selB)  aperture_mass_thr(0|0.5)  dep_flag
submit_variant() {
    local sub=$1 mtag=$2 sdg=$3 ass=$4 stag=$5 apt=$6 dep_flag=$7
    local base=$(( (sub - 1) * N_ROIS + 1 ))
    local end=$(( base + N_ROIS - 1 ))
    sb --array=${base}-${end} $dep_flag \
        --export=ALL,PSIGNAL_THR=${PSIGNAL_THR},APERTURE_MASS_THR=${apt},MODEL=${MODEL},SHARED_DYN_GAIN=${sdg},ALL_SHARED_SIGMA=${ass} \
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
    # M1: base (sharedSigma only)
    J=$(submit_variant $sub M1 0 0 selA 0           "$dep_flag"); echo "  M1 selA: $J"
    J=$(submit_variant $sub M1 0 0 selB $APERTURE_THR "$dep_flag"); echo "  M1 selB: $J"
    # M2: sharedDynGain
    J=$(submit_variant $sub M2 1 0 selA 0           "$dep_flag"); echo "  M2 selA: $J"
    J=$(submit_variant $sub M2 1 0 selB $APERTURE_THR "$dep_flag"); echo "  M2 selB: $J"
    # M3: allSharedSigma
    J=$(submit_variant $sub M3 0 1 selA 0           "$dep_flag"); echo "  M3 selA: $J"
    J=$(submit_variant $sub M3 0 1 selB $APERTURE_THR "$dep_flag"); echo "  M3 selB: $J"
    # M4: allSharedSigma + sharedDynGain
    J=$(submit_variant $sub M4 1 1 selA 0           "$dep_flag"); echo "  M4 selA: $J"
    J=$(submit_variant $sub M4 1 1 selB $APERTURE_THR "$dep_flag"); echo "  M4 selB: $J"
done

echo
echo "Each subject: 8 array jobs of ${N_ROIS} ROI tasks (4 model × 2 sel)."
