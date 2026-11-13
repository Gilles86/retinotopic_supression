#!/bin/bash
#SBATCH --job-name=hp_terms_heldout_power
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --exclude=u24-chaihm0-633,u24-chaiam0-613
#
# Leave-one-run-out held-out prediction and power simulation of the transient HP − LP terms
# (retsupp/modeling/hp_terms_heldout_power.py). One array task per subject, all nine areas.
#
# Usage (repo root on the cluster login node, after git pull):
#   sbatch --array=3-30 retsupp/modeling/slurm_jobs/hp_terms_heldout_power.sh
#   N_DRAWS=200 OUTPUT_SUBDIR=<new folder> sbatch --array=3 ...

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
LOGFILE="$HOME/logs/hp_terms_heldout_power_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | sub-${subject} | $(date)"

export TMPDIR="/scratch/$USER/hp_terms_${SLURM_JOB_ID}_${subject}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"

exec "$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.hp_terms_heldout_power \
    "$subject" --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    ${N_DRAWS:+--n-draws $N_DRAWS} ${OUTPUT_SUBDIR:+--output-subdir "$OUTPUT_SUBDIR"} ${ROIS:+--rois $ROIS}
