#!/bin/bash
# Gaussian-PRF kernel AF factorial. Mirrors submit_af_factorial_test.sh
# but for the Gaussian (m3) voxel kernel instead of DoG (m4).
#
# 4 model variants × 1 selection (pSig > 0.5, no aperture):
#   M1 base               (sharedSigma; σ_T_dyn := σ_dyn)
#   M2 sharedDynGain      (+ tie g_LP_dyn := g_HP_dyn)
#   M3 allSharedSigma     (+ tie σ_AF := σ_dyn := σ_T_dyn)
#   M4 both ties
#
# Default subjects: 2 7 (Cat A test). Override with SUBJECTS env.
#   bash submit_gauss_af_factorial.sh                       # sub-02 + sub-07
#   SUBJECTS="$(seq 1 30)" bash submit_gauss_af_factorial.sh
#
# MODEL=3 (Gaussian + flex HRF) is the canonical "HRF optim" init.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_AF="$SCRIPT_DIR/fit_gauss_dyn_v3_target_sharedSigma.sh"
N_ROIS=11
MODEL=3
PSIGNAL_THR=0.5
BIDS="/shares/zne.uzh/gdehol/ds-retsupp"

SUBJECTS="${SUBJECTS:-2 7}"

sb() { sbatch "$@" | awk '{print $4}'; }

# Find this subject's queued r2_mixture array-job ID for the SAME MODEL
# we're about to AF on (mixtures are model-specific).
mixture_dep_for() {
    local sub=$1
    squeue --me -h -n r2_mixture --format='%F %i' \
      | awk -v s="$sub" '{ if ($2 ~ "_\\[?"s"\\]?$") { print $1; exit } }'
}

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
    # Mixture sidecar lives at derivatives/prf/model${MODEL}/sub-XX/...
    # — model-specific. For Gaussian we need the m3 mixture, not m4.
    sidecar="$BIDS/derivatives/prf/model${MODEL}/sub-${sp}/sub-${sp}_desc-p_signal.json"
    if [[ -f "$sidecar" ]]; then
        dep_flag=""; dep_label="none (m${MODEL} sidecar present)"
    else
        mix_jid=$(mixture_dep_for $sub)
        if [[ -z "$mix_jid" ]]; then
            echo "sub-${sp}: SKIPPED — no m${MODEL} sidecar and no queued mixture"
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
echo "Each subject: 4 array jobs of ${N_ROIS} ROI tasks (Gaussian, pSig=${PSIGNAL_THR})."
