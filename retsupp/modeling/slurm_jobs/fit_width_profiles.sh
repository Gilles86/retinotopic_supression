#!/bin/bash
#SBATCH --job-name=width_profiles
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#
# Field widths conditional on a reliable gain (retsupp.modeling.fit_width_profiles): per-run sufficient statistics
# for the gains at the fitted widths and on a width grid (sustained and transient) on the final joint fit.
# One array task per subject, nine ROIs.
#
#   cd ~/retsupp_plan && sbatch --exclude=u24-chaihm0-633 --array=3 retsupp/modeling/slurm_jobs/fit_width_profiles.sh   # canary
#   cd ~/retsupp_plan && sbatch --exclude=u24-chaihm0-633 --array=4-30 retsupp/modeling/slurm_jobs/fit_width_profiles.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/width_profiles_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | sub-${subject} | $(date)"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TF_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"

exec "$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_width_profiles
    "$subject" ${ROIS:+--rois $ROIS}
