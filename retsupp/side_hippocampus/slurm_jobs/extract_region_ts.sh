#!/bin/bash -l
#SBATCH --job-name=side_hpc_extract
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#
# Side project (hippocampus): cleaned BOLD of the FreeSurfer regions, one task per subject.
#   cd ~/retsupp_side_hpc && sbatch --array=3-30 retsupp/side_hippocampus/slurm_jobs/extract_region_ts.sh
set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
mkdir -p "$HOME/logs"
exec >"$HOME/logs/side_hpc_extract_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt" 2>&1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
echo "Host: $(hostname) | sub-${subject} | job ${SLURM_JOB_ID} | code $(git -C "$CODE_DIR" rev-parse --short HEAD) | $(date)"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.side_hippocampus.extract_region_ts "$subject"
echo "Finished: $(date)"
