#!/bin/bash -l
#SBATCH --job-name=side_hpc_prf
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G           # canary sub-03 (LO, hippocampus, 1444-1500 voxels): OOM at 8 GB after 2 of 3 folds
#SBATCH --time=02:30:00      # canary: ~25 min per fold, CPU efficiency 70 % on 8 CPUs
#
# Side project (hippocampus): cross-validated pRFs in one region, one task per subject (array id = subject).
#   cd ~/retsupp_side_hpc && REGION=hippocampus VARIANT=true sbatch --array=3-30 retsupp/side_hippocampus/slurm_jobs/fit_prf_cv_region.sh
set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
REGION="${REGION:?}"
VARIANT="${VARIANT:-true}"
mkdir -p "$HOME/logs"
exec >"$HOME/logs/side_hpc_prf_${REGION}_${VARIANT}_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt" 2>&1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
echo "Host: $(hostname) | sub-${subject} ${REGION} ${VARIANT} | job ${SLURM_JOB_ID} | code $(git -C "$CODE_DIR" rev-parse --short HEAD) | $(date)"
scontrol update jobid="${SLURM_JOB_ID}" name="side_hpc_prf_${REGION}_${VARIANT}" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}" TF_NUM_INTEROP_THREADS=2
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.side_hippocampus.fit_prf_cv_region "$subject" "$REGION" \
    --variant "$VARIANT" ${EXTRA_ARGS:-}
echo "Finished: $(date)"
