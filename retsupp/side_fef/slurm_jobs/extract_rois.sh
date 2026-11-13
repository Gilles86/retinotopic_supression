#!/bin/bash -l
#SBATCH --job-name=side_fef_extract
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633,u24-chaiam0-613
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G            # canary sub-03: 2 min, peak 16 GB
#SBATCH --time=00:15:00
#
# Side project (FEF / sPCS): ROIs in BOLD space + cleaned BOLD + selection table, one task per subject.
#   cd ~/retsupp_side_fef && sbatch --array=3-30 retsupp/side_fef/slurm_jobs/extract_rois.sh
set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
mkdir -p "$HOME/logs"
exec >"$HOME/logs/side_fef_extract${LOSO:+_loso}_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt" 2>&1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
echo "Host: $(hostname) | sub-${subject} | job ${SLURM_JOB_ID} | code $(git -C "$CODE_DIR" rev-parse --short HEAD) | $(date)"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
if [[ -n "${LOSO:-}" ]]; then   # LOSO=1: leave-one-participant-out group ROIs (loso_rois.py)
    "$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.side_fef.loso_rois "$subject"
else
    "$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.side_fef.extract_rois "$subject" \
        --group-rois "${CODE_DIR}/notes/side_fef/data/fsaverage_rois.npz"
fi
echo "Finished: $(date)"
