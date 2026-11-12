#!/bin/bash
#SBATCH --job-name=baseline_terms
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --mem=24G
#
# Baseline shift vs response gain at HP (retsupp.modeling.fit_baseline_terms): additive (trial-locked, sustained)
# and multiplicative HP − LP terms on the final joint fit, held-out run 3, plus the recovery simulation.
# One array task per subject, nine ROIs.
#
#   cd ~/retsupp_plan && sbatch --exclude=u24-chaihm0-633 --array=3 retsupp/modeling/slurm_jobs/fit_baseline_terms.sh   # canary
#   cd ~/retsupp_plan && sbatch --exclude=u24-chaihm0-633 --array=4-30 retsupp/modeling/slurm_jobs/fit_baseline_terms.sh
#   N_SIM=<k> number of simulated datasets (default 10); ROIS="V1 V3" subset

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/baseline_terms_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
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

exec "$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_baseline_terms \
    "$subject" ${ROIS:+--rois $ROIS} ${N_SIM:+--n-sim $N_SIM}
