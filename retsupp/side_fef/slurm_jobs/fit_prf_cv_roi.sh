#!/bin/bash -l
#SBATCH --job-name=side_fef_prf
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633,u24-chaiam0-613
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4   # canary sub-03 sPCS (119 voxels): 7 min, 57% of 8 CPUs, 6.6 GB
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#
# Side project (FEF / sPCS): cross-validated pRFs in one ROI, one task per subject (array id = subject).
#   cd ~/retsupp_side_fef && REGION=sPCS VARIANT=true sbatch --array=3-30 retsupp/side_fef/slurm_jobs/fit_prf_cv_roi.sh
set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
REGION="${REGION:?}"
VARIANT="${VARIANT:-true}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/side_fef_prf_${REGION}_${VARIANT}_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt" 2>&1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
echo "Host: $(hostname) | sub-${subject} ${REGION} ${VARIANT} | job ${SLURM_JOB_ID} | code $(git -C "$CODE_DIR" rev-parse --short HEAD) | $(date)"
scontrol update jobid="${SLURM_JOB_ID}" name="side_fef_prf_${REGION}_${VARIANT}" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}" TF_NUM_INTEROP_THREADS=2
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.side_fef.fit_prf_cv_roi "$subject" "$REGION" \
    --variant "$VARIANT" ${EXTRA_ARGS:-}
echo "Finished: $(date)"
