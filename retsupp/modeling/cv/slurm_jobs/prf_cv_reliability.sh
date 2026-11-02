#!/bin/bash -l
#SBATCH --job-name=prf_cv_rel
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#
# Model-free voxel reliability per fold (prf_cv_reliability). No fits.
# Usage (from the repo root on the cluster login node):
#   sbatch --array=1-30 retsupp/modeling/cv/slurm_jobs/prf_cv_reliability.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/prf_cv_rel_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | sub-${subject} | $(date)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.cv.prf_cv_reliability \
    "$subject" --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
echo "Finished: $(date)"
