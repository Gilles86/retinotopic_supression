#!/bin/bash
#SBATCH --job-name=fit_attention
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Fit the 2-parameter precision-weighted attention model per ROI.
# Reads conditionwise + mean-model PRF summary TSVs; writes to
# derivatives/attention_model/model-{N}/sub-XX/.
#
# Required env: MODEL (default 4).
# SLURM_ARRAY_TASK_ID = subject id.

set -eo pipefail

MODEL="${MODEL:-4}"
subject="${SLURM_ARRAY_TASK_ID:?ERROR: --array required (subject id)}"
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/fit_attention_m${MODEL}_sub-${sub_pad}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1
scontrol update jobid="${SLURM_JOB_ID}" \
    name="af_m${MODEL}_sub-${sub_pad}" 2>/dev/null || true

echo "Host: $(hostname) | sub-${subject} | model ${MODEL}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

"$PYTHON" -u -c "
from pathlib import Path
from retsupp.modeling.fit_attention_model import process_subject
process_subject($subject, Path('/shares/zne.uzh/gdehol/ds-retsupp'),
                model_label=$MODEL)
"

echo "Finished: $(date)"
