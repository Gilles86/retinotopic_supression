#!/bin/bash
#SBATCH --job-name=cache_bold
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:15:00

# Build per-subject cleaned-BOLD + paradigm cache for fast downstream
# chunked PRF fits. One subject per array task.
#
# Usage:
#   sbatch --array=1-30 --export=ALL,KIND=full \
#          retsupp/modeling/slurm_jobs/build_cleaned_bold_cache.sh
#   sbatch --array=1-30 --export=ALL,KIND=bar \
#          retsupp/modeling/slurm_jobs/build_cleaned_bold_cache.sh
#
# Output: /shares/zne.uzh/gdehol/ds-retsupp/derivatives/cleaned_bold_cache/sub-XX/sub-XX_kind-{full,bar}_res-50.npz

set -eo pipefail

KIND="${KIND:-full}"
subject="${SLURM_ARRAY_TASK_ID}"
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/cache_bold_sub-${sub_pad}_${KIND}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="cache_bold_${KIND}_sub-${sub_pad}" 2>/dev/null || true

echo "Host: $(hostname) | sub-${subject} | kind ${KIND}"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/build_cleaned_bold_cache.py" \
    "$subject" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --paradigm-kind "$KIND"

echo "Finished: $(date)"
