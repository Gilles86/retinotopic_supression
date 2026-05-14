#!/bin/bash
#SBATCH --job-name=v100_smoke
#SBATCH --account=hare.econ.uzh
#SBATCH --partition=lowprio
#SBATCH --constraint=V100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null

# 8 tasks pinned to ONE V100 node (set via sbatch --nodelist=...).
# Each task grabs gpu:1 on that node, so we simulate the worst case for
# the cuInit race that hit us 2026-05-13 (8 parallel CUDA inits on a
# single 8-GPU box). The 30s random stagger in this script body is what
# should defuse the race; if 8/8 print SMOKE_OK we can lift the L4-only
# constraint in fit_prf_l4_chunked.sh.

set -eo pipefail

LOGFILE="$HOME/logs/v100_smoke_task-${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

sleep $(( RANDOM % 30 ))

echo "Host: $(hostname)  array_task=${SLURM_ARRAY_TASK_ID}  job=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started: $(date)"

source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1

"$HOME/data/conda/envs/retsupp_cuda/bin/python" -u \
    "$HOME/git/retsupp/notes/scripts/v100_smoke.py"

echo "Finished: $(date)"
