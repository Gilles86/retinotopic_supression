#!/bin/bash
#SBATCH --job-name=prf_conditionfit
#SBATCH --account=hare.econ.uzh
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Per-subject conditionwise PRF fit (per HP-condition runs grouped).
#
# Usage:
#   sbatch --array=1-30 --export=ALL,MODEL=4 \
#        retsupp/modeling/fit_condition.sh
#
# Reads MODEL from env, subject from SLURM_ARRAY_TASK_ID. Logs to
# ~/logs/retsupp_prf_conditionfit_sub-XX_model-Y_<jobid>.txt
set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${MODEL:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID and MODEL required." >&2; exit 2
fi
subject="${SLURM_ARRAY_TASK_ID}"
sub_pad=$(printf "%02d" "$subject")
model="${MODEL}"

LOGFILE="$HOME/logs/retsupp_prf_conditionfit_sub-${sub_pad}_m${model}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_cond_m${model}_sub-${sub_pad}" 2>/dev/null || true

echo "Host: $(hostname) | sub-${sub_pad} | model ${model}"
echo "Started: $(date)"

# Use system CUDA + retsupp_cuda env (matches fit_prf_l4.sh path).
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/fit_condition.py" \
    "$subject" --model "$model" \
    --bids_folder /shares/zne.uzh/gdehol/ds-retsupp \
    --max_n_iterations 4000 --r2_thr 0.06

echo "Finished: $(date)"
