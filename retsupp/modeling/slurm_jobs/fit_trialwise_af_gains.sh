#!/bin/bash
#SBATCH --job-name=trialwise_af
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

# Trial-wise + RT-binned phasic AF gains on top of the canonical
# pSig0.5 AF fit (all 11 ROIs per subject; data loaded once).
#
#   sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_trialwise_af_gains.sh
#   ROIS="V1 V3" sbatch --array=5 retsupp/modeling/slurm_jobs/fit_trialwise_af_gains.sh
#
# Output: derivatives/af_trialwise_gains/sub-XX/

set -eo pipefail

subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
LOGFILE="$HOME/logs/trialwise_af_sub-$(printf %02d $subject)_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=2

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  sub: ${subject}  Started: $(date)"

ROI_ARGS=""
[[ -n "${ROIS:-}" ]] && ROI_ARGS="--rois ${ROIS}"

cd "$HOME/git/retsupp"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u \
    -m retsupp.modeling.fit_trialwise_af_gains "$subject" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp $ROI_ARGS

echo "Finished: $(date)"
