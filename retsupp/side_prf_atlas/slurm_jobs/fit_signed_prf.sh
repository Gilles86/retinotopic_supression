#!/bin/bash -l
#SBATCH --job-name=side_prf_atlas_signed
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633,u24-chaiam0-613
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/home/gdehol/logs/side_prf_atlas_signed_%A_%a.txt

# Side analysis: whole-brain signed Gaussian pRF (CV over sessions, shift null) + reliability,
# resampled to fsaverage. One subject per array task.
#   sbatch --array=3-30 retsupp/side_prf_atlas/slurm_jobs/fit_signed_prf.sh
#   CANARY=5000 sbatch --array=3 retsupp/side_prf_atlas/slurm_jobs/fit_signed_prf.sh
set -euo pipefail
sub=${SLURM_ARRAY_TASK_ID}
scontrol update jobid="${SLURM_JOB_ID}" name="side_signedprf_sub-$(printf %02d "$sub")" 2>/dev/null || true
export TMPDIR=/scratch/$USER/side_prf_atlas_${SLURM_JOB_ID}_${sub}
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "$HOME/retsupp_plan"
echo "host $(hostname) commit $(git rev-parse --short HEAD) sub $sub canary ${CANARY:-none}"
extra=()
[[ -n "${CANARY:-}" ]] && extra=(--canary-voxels "$CANARY")
exec "$HOME/data/conda/envs/retsupp_cuda/bin/python" -m retsupp.side_prf_atlas.fit_signed_prf "$sub" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp "${extra[@]}"
