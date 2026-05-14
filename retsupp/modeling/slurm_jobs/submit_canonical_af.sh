#!/bin/bash
# Submit the canonical AF model (v3 + target + sharedSigma) on top of
# each requested base PRF model. Dispatches to the right Python /
# SLURM script per PRF model:
#
#   m1 (Gaussian)        -> fit_gaussian_dyn_v3_target_sharedSigma.sh
#   m2 (DoG no HRF)      -> fit_dog_dyn_v3_target_sharedSigma.sh  (model_label=2)
#   m3 (Gaussian + HRF)  -> fit_gaussian_dyn_v3_target_sharedSigma.sh (model_label=3)
#   m4 (DoG + HRF)       -> fit_dog_dyn_v3_target_sharedSigma.sh  (model_label=4)
#   m5 (DN)              -> fit_dn_dyn_v3_target_sharedSigma.sh    [NEW; pending]
#   m6 (DN + HRF)        -> fit_dn_dyn_v3_target_sharedSigma.sh    [NEW; pending]
#
# Each AF run depends on a per-subject r2_mixture job so --use-fdr can
# look up the mixture-FDR (α=0.05) threshold per ROI.
#
# Usage:
#   MODELS="1 2 3 4" bash submit_canonical_af.sh             # default
#   SUBJECTS="1 4 7" MODELS="1 4" bash submit_canonical_af.sh

set -eo pipefail

MODELS="${MODELS:-1 2 3 4}"   # m5/m6 added once DN-AF class lands
SUBJECTS_DEFAULT=""

BIDS="/shares/zne.uzh/gdehol/ds-retsupp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
S_MIX="$SCRIPT_DIR/run_r2_mixture.sh"
S_DOG="$SCRIPT_DIR/fit_dog_dyn_v3_target_sharedSigma.sh"
S_GAUSS="$SCRIPT_DIR/fit_gauss_dyn_v3_target_sharedSigma.sh"
S_DN="$SCRIPT_DIR/fit_dn_dyn_v3_target_sharedSigma.sh"   # may not exist yet

N_ROIS=11   # V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF

af_script_for_model() {
    case "$1" in
        1|3) echo "$S_GAUSS" ;;
        2|4) echo "$S_DOG" ;;
        5|6) echo "$S_DN" ;;
        *)   echo "" ;;
    esac
}

# Default = subjects with model-M NIfTI on disk (per requested models).
if [[ -z "${SUBJECTS:-}" ]]; then
    SUBJECTS=""
    for s in $(seq 1 30); do
        sp=$(printf "%02d" "$s")
        # Only include subjects that have ALL requested base models on disk
        ok=1
        for m in $MODELS; do
            f="${BIDS}/derivatives/prf/model${m}/sub-${sp}/sub-${sp}_desc-r2.nii.gz"
            [[ ! -f "$f" ]] && { ok=0; break; }
        done
        [[ "$ok" == "1" ]] && SUBJECTS="$SUBJECTS $s"
    done
fi

sb() { sbatch "$@" | awk '{print $4}'; }

n_sub=0
for s in $SUBJECTS; do
    sp=$(printf "%02d" "$s")
    # Shared mixture job (per subject — uses model 4 by default so
    # FDR thresholds are consistent across PRF-init choices).
    J_MIX=$(sb --array=$s "$S_MIX")
    base=$(( (s - 1) * N_ROIS + 1 ))
    end=$(( base + N_ROIS - 1 ))

    af_jobs=""
    for m in $MODELS; do
        script=$(af_script_for_model "$m")
        if [[ -z "$script" || ! -f "$script" ]]; then
            echo "  sub-${sp} m${m}: AF script not available, skipping"
            continue
        fi
        J=$(sb --array=${base}-${end} \
            --dependency=afterok:$J_MIX \
            --export=ALL,USE_FDR=1,MODEL=$m "$script")
        af_jobs="$af_jobs m${m}=$J"
    done
    echo "sub-${sp}: mix=$J_MIX$af_jobs"
    n_sub=$((n_sub + 1))
done

echo
echo "Submitted: $n_sub subjects × $(echo $MODELS | wc -w) models × ${N_ROIS} ROIs"
