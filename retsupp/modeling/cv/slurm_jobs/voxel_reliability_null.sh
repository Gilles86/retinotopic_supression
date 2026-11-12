#!/bin/bash -l
#SBATCH --job-name=rel_null
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:40:00
#
# Cross-run reliability with a circular-shift null per voxel (voxel_reliability_null). No fits.
# Usage (from the repo root on the cluster login node):
#   sbatch --array=3-30 retsupp/modeling/cv/slurm_jobs/voxel_reliability_null.sh
#   EXTRA_ARGS=--folds sbatch --array=3-30 ...   (per CV fold, training runs only; for decode_final --selection relnull)

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/rel_null${EXTRA_ARGS:+_folds}_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | sub-${subject} | $(date)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export TMPDIR=/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$TMPDIR"
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.cv.voxel_reliability_null \
    "$subject" --bids-folder /shares/zne.uzh/gdehol/ds-retsupp ${EXTRA_ARGS:-}
echo "Finished: $(date)"
