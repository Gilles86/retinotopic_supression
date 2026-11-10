#!/bin/bash
#SBATCH --job-name=rt_bayes_final2
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null

# Copy of trialwise_rt_bayes.sh for the final fits (CODE_DIR defaults to the repo clone).
# One hierarchical RT ~ trial-wise AF gain model per ROI (array 1-11) or ROI group (12-14: early, dorsal,
# occipitotemporal; from fit_trialwise_af_gains --groups).
#   sbatch --array=1-11 trialwise_rt_bayes.sh
# Env: TRIALS (trial table), OUT_DIR, CODE_DIR (dir containing trialwise_rt_bayes.py).

set -eo pipefail
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF early dorsal occipitotemporal pooled)
roi="${ROIS[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"
TRIALS="${TRIALS:-$HOME/retsupp_bayes/trialwise_af_trial_table.tsv.gz}"
OUT_DIR="${OUT_DIR:-$HOME/retsupp_bayes/out}"
CODE_DIR="${CODE_DIR:-$HOME/retsupp_plan/retsupp/visualize/paper}"

mkdir -p "$HOME/logs" "$OUT_DIR"
exec >"$HOME/logs/rt_bayes_${roi}_${SLURM_JOB_ID}.txt" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" name="rt_bayes_${roi}" 2>/dev/null || true
echo "Host $(hostname)  ROI ${roi}  started $(date)"

sleep $(( RANDOM % 10 ))
export PYTHONUNBUFFERED=1
"$HOME/data/conda/envs/bauer/bin/python" "$CODE_DIR/trialwise_rt_bayes.py" \
    --trials "$TRIALS" --roi "$roi" --out-dir "$OUT_DIR" ${EXTRA_ARGS:-}
echo "Finished $(date)"
