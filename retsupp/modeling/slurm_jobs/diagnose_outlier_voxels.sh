#!/bin/bash
#SBATCH --job-name=outlier_vox
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#
# Per-run raw mean and SD of the attention-field voxels (diagnose_outlier_voxels).
# Usage (from the repo root on the cluster login node):
#   sbatch --array=3,7,20 retsupp/modeling/slurm_jobs/diagnose_outlier_voxels.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/outlier_vox_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | sub-${subject} | $(date)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.diagnose_outlier_voxels \
    "$subject" --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
echo "Finished: $(date)"
