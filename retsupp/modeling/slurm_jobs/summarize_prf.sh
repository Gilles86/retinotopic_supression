#!/bin/bash
#SBATCH --job-name=summarize_prf
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

# Aggregate per-voxel PRF parameters into a tidy long-format TSV per
# subject. Writes to either:
#   derivatives/prf_summaries/model{N}/sub-XX/         (mean)
#   derivatives/prf_summaries.conditionwise/model{N}/  (conditionwise)
#
# Required env: MODEL (default 4), KIND (mean | conditionwise).
# SLURM_ARRAY_TASK_ID = subject id.

set -eo pipefail

MODEL="${MODEL:-4}"
KIND="${KIND:-mean}"
subject="${SLURM_ARRAY_TASK_ID:?ERROR: --array required (subject id)}"
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/summarize_prf_${KIND}_m${MODEL}_sub-${sub_pad}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" \
    name="summ_${KIND}_m${MODEL}_sub-${sub_pad}" 2>/dev/null || true

sleep $(( RANDOM % 20 ))

echo "Host: $(hostname) | sub-${subject} | model ${MODEL} | kind ${KIND}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

EXTRA=""
if [[ "$KIND" == "conditionwise" ]]; then EXTRA="--conditionwise"; fi
if [[ "$KIND" == "runwise" ]]; then EXTRA="--runwise"; fi

"$PYTHON" -u "$HOME/git/retsupp/retsupp/modeling/summarize_prf.py" \
    "$subject" --model "$MODEL" \
    --bids_folder /shares/zne.uzh/gdehol/ds-retsupp \
    $EXTRA

echo "Finished: $(date)"
