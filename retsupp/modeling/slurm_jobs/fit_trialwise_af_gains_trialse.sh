#!/bin/bash -l
#SBATCH --job-name=trialse_lat
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# Per-trial gains with circular-shift placebo covariance (retsupp/modeling/fit_trialwise_af_gains_trialse.py),
# nine areas, signal voxels, on the root's m7_full fits. Addendum 6 of notes/rt_power_plan_2026-11-13.md.
#
#   LADDER_ROOT=<root> HRF_TABLE=<tsv> OUTPUT_SUBDIR=<new folder> REPO=$PWD \
#       sbatch --exclude=u24-chaihm0-633,u24-chaiam0-613 --array=3-30 retsupp/modeling/slurm_jobs/fit_trialwise_af_gains_trialse.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID:?submit with --array}
LOGFILE="$HOME/logs/trialse_lat_sub-$(printf %02d $subject)_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TF_NUM_INTEROP_THREADS=1
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  sub: ${subject}  Started: $(date)"

cd "${REPO:-$HOME/git/retsupp}"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_trialwise_af_gains_trialse "$subject" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --ladder-root "${LADDER_ROOT:?}" --hrf-table "${HRF_TABLE:?}" \
    --output-subdir "${OUTPUT_SUBDIR:?}" ${N_SHIFTS:+--n-shifts $N_SHIFTS} ${ROIS:+--rois $ROIS}
rm -rf "$TMPDIR"
echo "Finished: $(date)"
