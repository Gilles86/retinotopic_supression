#!/bin/bash
#SBATCH --account=zne.uzh
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null
#SBATCH --job-name=prf_warm

# Multi-stage warm-start PRF refit on V1 only, one subject per array task.
#
# Usage (submit ONE model at a time):
#   MODEL=4 sbatch --array=1-30 retsupp/modeling/slurm_jobs/fit_prf_warmstart.sh
#
# Output: notes/data/prf_warmstart_m{model}_V1_sub-{NN}.tsv
# Logs:   ~/logs/prf_warm_m{model}_sub-{NN}_{jobid}.txt

set -euo pipefail

# Random startup jitter to spread NFS profile reads.
sleep $(( (RANDOM % 30) + 1 ))

MODEL="${MODEL:?must set MODEL=2|3|4|5|6}"
SUB=$(printf %02d "$SLURM_ARRAY_TASK_ID")

# Rename job for readability in squeue/sacct.
scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_warm_m${MODEL}_sub-${SUB}" 2>/dev/null || true

LOGFILE="$HOME/logs/prf_warm_m${MODEL}_sub-${SUB}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "=== prf_warmstart  model ${MODEL}  sub-${SUB}  $(date) ==="
echo "  host: $(hostname)"

# Activate conda env. We use `retsupp_cuda` rather than `retsupp`
# because the cluster's `retsupp` install never finished — braincoder
# isn't pip-installed there. `retsupp_cuda` has braincoder; TF will
# automatically fall back to CPU since this job doesn't request a GPU.
# conda's package-specific activate.d hooks reference unbound variables
# (ADDR2LINE, AR, ...) so relax `set -u` during activation.
set +u
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
set -u

cd "$HOME/git/retsupp"
git rev-parse --short HEAD || true

export PYTHONUNBUFFERED=1

python -u notes/figures/talk/fit_prf_warmstart.py "$MODEL" \
    --subjects "$SLURM_ARRAY_TASK_ID" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp

echo "=== done $(date) ==="
