#!/bin/bash -l
#SBATCH --job-name=hrf_joint
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G          # DoG final2 (job 8143173): peak RSS 14.7 GB
#SBATCH --time=03:00:00    # DoG final2: 33 min - 2 h 05 on 16 CPUs
#
# One HRF per subject x ROI fitted jointly with the pRFs (cv/fit_prf_cv_roi_hrf_joint.py), the "final2" recipe:
# stage 2 and 3 with 3000 iterations each. One array task per (subject, fold): task id = 10*subject + fold.
#   cd ~/retsupp_plan && FAMILY=gauss sbatch --array=10-12,20-22,...,300-302 retsupp/modeling/cv/slurm_jobs/fit_prf_cv_roi_hrf_joint.sh
# FAMILY = dog | dn | gauss (default dog), TAG (default final2), N2 / N3 (default 3000).
# Then: python -m retsupp.modeling.cv.build_roi_hrf_table --family $FAMILY --tag final2 --fill-canonical --out ...

set -eo pipefail
FAMILY="${FAMILY:-dog}"
TAG="${TAG:-final2}"
subject=$(( SLURM_ARRAY_TASK_ID / 10 ))
fold=$(( SLURM_ARRAY_TASK_ID % 10 ))
mkdir -p "$HOME/logs"
exec >"$HOME/logs/hrf_joint_${FAMILY}_${TAG}_sub-$(printf %02d "$subject")_fold-${fold}_${SLURM_JOB_ID}.txt" 2>&1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
echo "Host: $(hostname) | sub-${subject} fold ${fold} | ${FAMILY} ${TAG} | job ${SLURM_JOB_ID} | $(date)"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"
scontrol update jobid="${SLURM_JOB_ID}" name="hrf_joint_${FAMILY}_${TAG}" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-16}" TF_NUM_INTEROP_THREADS=2
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.cv.fit_prf_cv_roi_hrf_joint "$subject" \
    --fold "$fold" --family "$FAMILY" --tag "$TAG" --n2 "${N2:-3000}" --n3 "${N3:-3000}" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
echo "Finished: $(date)"
