#!/bin/bash
#SBATCH --job-name=sim_refit
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00

# Sim+refit: simulate noiseless BOLD from our AF model + refit vanilla
# DoG per condition, for one (subject, ROI). The output TSV per job
# lives at:
#   /shares/zne.uzh/gdehol/ds-retsupp/derivatives/sim_refit/sub-XX/
#       sub-XX_roi-<ROI>_sim-refit.tsv
#
# Array index ↔ (subject, roi) is read from JOB_TABLE (newline-separated
# "subject roi" pairs, line N matches SLURM_ARRAY_TASK_ID N).
#
# Usage:
#   # Generate job table for 3 ROIs × 8 subjects = 24 lines
#   for s in 3 5 11 13 17 18 28 30; do
#     for r in hV4 TO IPS; do echo "$s $r"; done
#   done > ~/sim_refit_jobs.txt
#
#   sbatch --array=1-24 \
#       --export=ALL,JOB_TABLE=$HOME/sim_refit_jobs.txt \
#       retsupp/visualize/paper/slurm_jobs/sim_refit.sh

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${JOB_TABLE:-}" || ! -f "${JOB_TABLE}" ]]; then
    echo "ERROR: JOB_TABLE env var must point to a readable file." >&2; exit 2
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOB_TABLE}")
SUBJECT=$(echo "$LINE" | awk '{print $1}')
ROI=$(echo "$LINE" | awk '{print $2}')
if [[ -z "$SUBJECT" || -z "$ROI" ]]; then
    echo "ERROR: empty subject/roi on line ${SLURM_ARRAY_TASK_ID} of ${JOB_TABLE}" >&2
    exit 2
fi
sub_pad=$(printf "%02d" "$SUBJECT")

LOGFILE="$HOME/logs/sim_refit_sub-${sub_pad}_${ROI}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="sim_refit_sub-${sub_pad}_${ROI}" 2>/dev/null || true

echo "Host: $(hostname) | Job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | sub-${sub_pad} | ROI ${ROI}"
echo "Started: $(date)"

# conda activation under set -u tripwire (some pkgs' activate-*.d
# scripts reference unbound vars).
set +u
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
set -u

cd "$HOME/git/retsupp"

python -u -m retsupp.visualize.paper.sim_refit_one_roi \
    "${SUBJECT}" \
    --roi "${ROI}" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --max-voxels 500

echo "Finished: $(date)"
