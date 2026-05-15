#!/bin/bash
# Submit decode.sh array job for every (subject with m4 + cache, ROI)
# pair. ~20 subjects × 8 ROIs × 12 runs = ~1920 cells with %150 throttle.
#
# Run from cluster login node:
#   bash retsupp/decode/slurm_jobs/submit_all_decode.sh
#
# Env knobs:
#   SUBJECTS    whitespace-separated, default = all sub-NN with m4 + cache
#   ROIS        default V1 V2 V3 V3AB hV4 LO TO VO
#   MAX_VOXELS  default 200
#   NOISE_DIST  default gauss
#   THROTTLE    %N for sbatch --array; default 150
#   DRY_RUN     1 to print sbatch lines without submitting

set -euo pipefail

BIDS=/shares/zne.uzh/gdehol/ds-retsupp
ROIS="${ROIS:-V1 V2 V3 V3AB hV4 LO TO VO}"
MAX_VOXELS="${MAX_VOXELS:-2000}"
NOISE_DIST="${NOISE_DIST:-gauss}"
VOXEL_FILTER="${VOXEL_FILTER:-p_signal}"
PSIGNAL_POSTERIOR="${PSIGNAL_POSTERIOR:-0.5}"
THROTTLE="${THROTTLE:-150}"
DRY_RUN="${DRY_RUN:-0}"

# Discover subjects with BOTH a complete m4 fit AND the cleaned BOLD cache.
all_with_m4=$(ls -d ${BIDS}/derivatives/prf/model4/sub-* 2>/dev/null \
              | xargs -I{} basename {} | sed 's/sub-//' | sort -n)
SUBJECTS="${SUBJECTS:-}"
if [[ -z "$SUBJECTS" ]]; then
    for s_pad in $all_with_m4; do
        s_int=$((10#$s_pad))
        m4_x=${BIDS}/derivatives/prf/model4/sub-${s_pad}/sub-${s_pad}_desc-x.nii.gz
        cache=${BIDS}/derivatives/cleaned_bold_cache/sub-${s_pad}/sub-${s_pad}_kind-full_res-50.npz
        if [[ -f "$m4_x" && -f "$cache" ]]; then
            SUBJECTS="$SUBJECTS $s_int"
        fi
    done
    SUBJECTS=$(echo $SUBJECTS | tr ' ' '\n' | sort -nu | tr '\n' ' ')
fi
SUBJECTS=$(echo $SUBJECTS)
echo "Subjects: $SUBJECTS"
echo "ROIs    : $ROIS"
echo "Throttle: %${THROTTLE}  MaxVox: ${MAX_VOXELS}  Filter: ${VOXEL_FILTER} (psig=${PSIGNAL_POSTERIOR})  Noise: ${NOISE_DIST}"
echo

N_SUBS=$(echo $SUBJECTS | wc -w | tr -d ' ')
N_ROIS=$(echo $ROIS | wc -w | tr -d ' ')
echo "About to submit $((N_SUBS * N_ROIS)) array jobs of 12 cells each = $((N_SUBS * N_ROIS * 12)) cells total."
echo

n_submitted=0
for sub in $SUBJECTS; do
    for roi in $ROIS; do
        cmd=(sbatch --array=0-11%${THROTTLE}
             --export="ALL,SUB=$sub,ROI=$roi,MAX_VOXELS=$MAX_VOXELS,NOISE_DIST=$NOISE_DIST,VOXEL_FILTER=$VOXEL_FILTER,PSIGNAL_POSTERIOR=$PSIGNAL_POSTERIOR"
             ~/git/retsupp/retsupp/decode/slurm_jobs/decode.sh)
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[DRY] ${cmd[*]}"
        else
            "${cmd[@]}" | tail -1
            n_submitted=$((n_submitted + 1))
        fi
    done
done

echo
echo "Submitted $n_submitted array jobs."
