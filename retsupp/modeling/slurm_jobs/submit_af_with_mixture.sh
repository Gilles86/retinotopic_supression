#!/bin/bash
# Submit per-subject [r2_mixture → 11 ROI × AF jobs] with afterok deps.
# AF tasks use --use-fdr → mixture-FDR α=0.05 thresholds per (sub, ROI).
#
# Mixture sidecar (compute_r2_mixture.py) must exist before AF runs;
# this submitter ensures so via per-subject afterok chains.
#
# Required: MODEL=4 (default — also the only sensible value, AF uses
# mean DoG+HRF as init via fit_dog_dynamic_af_braincoder.py).
#
# Usage:
#   bash submit_af_with_mixture.sh                   # all subjects with m4
#   SUBJECTS="1 4 7" bash submit_af_with_mixture.sh  # specific subjects

set -eo pipefail

MODEL="${MODEL:-4}"
BIDS="/shares/zne.uzh/gdehol/ds-retsupp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_MIX="$SCRIPT_DIR/run_r2_mixture.sh"
S_AF="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"

# 11 ROIs that the AF script SUB_IDS/ROIS arrays cover.
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}

# Default = all subjects with m4 NIfTI on disk.
if [[ -z "${SUBJECTS:-}" ]]; then
    SUBJECTS=""
    for s in $(seq 1 30); do
        sp=$(printf "%02d" "$s")
        m4="${BIDS}/derivatives/prf/model${MODEL}/sub-${sp}/sub-${sp}_desc-r2.nii.gz"
        [[ -f "$m4" ]] && SUBJECTS="$SUBJECTS $s"
    done
fi

sb() { sbatch "$@" | awk '{print $4}'; }

n_sub=0
for s in $SUBJECTS; do
    sp=$(printf "%02d" "$s")

    # Mixture job (per subject — loops 11 ROIs internally)
    J_MIX=$(sb --array=$s "$S_MIX")

    # 11 AF tasks for this subject, all afterok on this subject's mixture.
    # Compute the array task ID range from the AF script's mapping:
    #   array_id = (subject - 1) * 11 + roi_idx + 1
    base=$(( (s - 1) * N_ROIS + 1 ))
    end=$(( base + N_ROIS - 1 ))
    J_AF=$(sb --array=${base}-${end} --dependency=afterok:$J_MIX \
        --export=ALL,USE_FDR=1 "$S_AF")

    echo "sub-${sp}: mixture=$J_MIX  af=$J_AF  (tasks ${base}-${end})"
    n_sub=$((n_sub + 1))
done

echo
echo "Submitted: $n_sub subjects (mixture + 11 AF tasks each)"
