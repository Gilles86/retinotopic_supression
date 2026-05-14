#!/bin/bash
# Submit the canonical AF (DoG v3 + target + sharedSigma) for all 30
# subjects in TWO variants:
#
#   A: posterior-only       → keep voxels with p_signal > 0.95
#   B: posterior + aperture → also require ≥50% PRF Gaussian mass inside
#                              the bar aperture
#
# Both variants share the same per-(subject, ROI) GMM mixture (the
# "p_signal" NIfTI). Variants are segregated by output_subdir:
#
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_pSig0.95/
#   derivatives/af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_pSig0.95_apt0.5/
#
# For each subject we afterok-chain on its r2_mixture job (if still
# queued) or fire immediately (if the per-voxel p_signal NIfTI is
# already on disk).
#
# Usage:
#   bash submit_psignal_af_variants.sh                  # all 30
#   SUBJECTS="1 5 13" bash submit_psignal_af_variants.sh

set -eo pipefail

BIDS="/shares/zne.uzh/gdehol/ds-retsupp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_AF="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"
MODEL=4
N_ROIS=11   # V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF
PSIGNAL_THR=0.95
APERTURE_THR=0.5

SUBJECTS="${SUBJECTS:-$(seq 1 30)}"

sb() { sbatch "$@" | awk '{print $4}'; }

# Find this subject's queued r2_mixture array-job ID (if any).
mixture_dep_for() {
    local sub=$1
    # squeue rows look like:  2998083_12      (single-task array, sub=12)
    # We want to match "...._<sub>" or "...._[<sub>]" exactly.
    squeue --me -h -n r2_mixture --format='%F %i' \
      | awk -v s="$sub" '
          {
              # %F is the array job ID (no _suffix). Match the task on %i.
              if ($2 ~ "_\\[?"s"\\]?$") { print $1; exit }
          }'
}

n_imm=0
n_dep=0
n_skip=0
for sub in $SUBJECTS; do
    sp=$(printf "%02d" $sub)
    sidecar="$BIDS/derivatives/prf/model${MODEL}/sub-${sp}/sub-${sp}_desc-p_signal.json"

    if [[ -f "$sidecar" ]]; then
        dep_flag=""
        dep_label="none (sidecar present)"
        n_imm=$((n_imm + 1))
    else
        mix_jid=$(mixture_dep_for $sub)
        if [[ -z "$mix_jid" ]]; then
            echo "sub-${sp}: SKIPPED — no sidecar and no queued mixture"
            n_skip=$((n_skip + 1))
            continue
        fi
        dep_flag="--dependency=afterok:${mix_jid}"
        dep_label="afterok:${mix_jid}"
        n_dep=$((n_dep + 1))
    fi

    base=$(( (sub - 1) * N_ROIS + 1 ))
    end=$(( base + N_ROIS - 1 ))

    J_A=$(sb --array=${base}-${end} $dep_flag \
        --export=ALL,PSIGNAL_THR=${PSIGNAL_THR},APERTURE_MASS_THR=0,MODEL=${MODEL} \
        "$S_AF")
    J_B=$(sb --array=${base}-${end} $dep_flag \
        --export=ALL,PSIGNAL_THR=${PSIGNAL_THR},APERTURE_MASS_THR=${APERTURE_THR},MODEL=${MODEL} \
        "$S_AF")
    echo "sub-${sp}: A(pSig)=$J_A  B(pSig+apt)=$J_B  dep=${dep_label}"
done

echo
echo "Summary: $n_imm immediate, $n_dep dep-chained, $n_skip skipped"
echo "Each subject got 2 array jobs of ${N_ROIS} ROI tasks (var A + var B)."
