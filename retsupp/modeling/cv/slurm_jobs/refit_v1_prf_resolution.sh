#!/bin/bash -l
#SBATCH --job-name=refit_v1_prf_res
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --exclude=u24-chaihm0-633
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G          # canary sub-23 fold 0: 16.8 GB (paper50), 10.9 GB (free100)
#SBATCH --time=01:30:00    # canary: 18 min (paper50), 50 min (free100), 16 CPUs
#
# V1 pRFs refitted under one fitting condition (cv/refit_v1_prf_resolution.py). One array task per (subject, fold):
# task id = 10*subject + fold; CONDITION (default paper50) names the condition in sim_prf_size_floor.CONDITIONS.
#   cd ~/retsupp_plan && CONDITION=free100 sbatch --array=230 retsupp/modeling/cv/slurm_jobs/refit_v1_prf_resolution.sh

set -eo pipefail
COND="${CONDITION:-paper50}"
subject=$(( SLURM_ARRAY_TASK_ID / 10 ))
fold=$(( SLURM_ARRAY_TASK_ID % 10 ))
mkdir -p "$HOME/logs"
exec >"$HOME/logs/refit_v1_prf_res_${COND}_sub-$(printf %02d "$subject")_fold-${fold}_${SLURM_JOB_ID}.txt" 2>&1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
echo "Host: $(hostname) | sub-${subject} fold ${fold} | ${COND} | job ${SLURM_JOB_ID} | $(date)"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"
scontrol update jobid="${SLURM_JOB_ID}" name="refit_v1_prf_${COND}" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1 KERAS_BACKEND=tensorflow
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-16}" TF_NUM_INTEROP_THREADS=2
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.cv.refit_v1_prf_resolution "$subject" \
    --fold "$fold" --condition "$COND"
echo "Finished: $(date)"
