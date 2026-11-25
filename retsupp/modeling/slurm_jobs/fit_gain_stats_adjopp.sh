#!/bin/bash -l
#SBATCH --job-name=adjopp_gainstats
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#
# Per-run gain statistics of the zero-sum LP-split model (retsupp.modeling.fit_gain_stats_adjopp): one array task per
# subject, nine ROIs. Resources as fit_width_profiles / fit_baseline_terms (about 5 min, data loading dominates).
#   cd <repo> && sbatch --array=3-30 retsupp/modeling/slurm_jobs/fit_gain_stats_adjopp.sh

set -eo pipefail
subject=${SLURM_ARRAY_TASK_ID}
mkdir -p "$HOME/logs"
exec >"$HOME/logs/adjopp_gainstats_sub-$(printf %02d "$subject")_${SLURM_JOB_ID}.txt" 2>&1
echo "Host: $(hostname) | job ${SLURM_JOB_ID} | sub-${subject} | $(date)"
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES="" TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}" TF_NUM_INTEROP_THREADS=1 PYTHONUNBUFFERED=1
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"
"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u -m retsupp.modeling.fit_gain_stats_adjopp "$subject" ${ROIS:+--rois $ROIS}
rm -rf "$TMPDIR"
echo "Finished $(date)"
