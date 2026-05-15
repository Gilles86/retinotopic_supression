#!/bin/bash
# Single-task sanity check for decode_runwise at the PLAN params
# (resolution=50, posterior=0.5, L2=1.0, lr=0.01). DEVICE selects
# CPU vs GPU; GPU adds --gres / --partition / --constraint on the
# sbatch command line (the body just calls module load gpu).
#
#   DEVICE=cpu  sbatch --export=ALL,DEVICE=cpu \
#       retsupp/decode/slurm_jobs/sanity_decode_runwise.sh
#   DEVICE=gpu  sbatch --export=ALL,DEVICE=gpu \
#       --partition=lowprio --gres=gpu:1 --constraint='L4|V100|A100' \
#       retsupp/decode/slurm_jobs/sanity_decode_runwise.sh
#
# Both run sub-02 V1 ses-1 run-1 by default; override with SUBJECT, ROI,
# SESSION, RUN. Writes to derivatives/decoded_paradigm/model4/sub-XX/.
#
#SBATCH --job-name=decode_sanity
#SBATCH --output=/dev/null
#SBATCH --time=45:00
#SBATCH --account=zne.uzh
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -eo pipefail

DEVICE="${DEVICE:-cpu}"
SUBJECT="${SUBJECT:-2}"
ROI="${ROI:-V1}"
SESSION="${SESSION:-1}"
RUN="${RUN:-1}"
BIDS="${BIDS:-/shares/zne.uzh/gdehol/ds-retsupp}"
MODEL="${MODEL:-4}"
RESOLUTION="${RESOLUTION:-50}"
POSTERIOR="${POSTERIOR:-0.5}"
L2_NORM="${L2_NORM:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
MAX_ITER="${MAX_ITER:-1000}"
MIN_ITER="${MIN_ITER:-200}"
RESID_MAX_ITER="${RESID_MAX_ITER:-300}"

LOGFILE="$HOME/logs/decode_sanity_${DEVICE}_sub-$(printf %02d $SUBJECT)_${ROI}_ses-${SESSION}_run-${RUN}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" name="decode_sanity_${DEVICE}_sub-$(printf %02d $SUBJECT)_${ROI}" || true

echo "[$(date)] === decode_runwise sanity test (${DEVICE}) ==="
echo "  sub-$(printf %02d $SUBJECT) ${ROI} ses-${SESSION} run-${RUN}"
echo "  model=${MODEL} res=${RESOLUTION} posterior=${POSTERIOR}"
echo "  L2=${L2_NORM} lr=${LEARNING_RATE} iter=${MAX_ITER}"

if [[ "$DEVICE" == "gpu" ]]; then
    # u24 nodes have no `module load gpu`; instead point LD_LIBRARY_PATH
    # at the system CUDA 11.8 (matches TF 2.14 + cuDNN 8.7 in
    # retsupp_cuda). Mirrors retsupp/modeling/slurm_jobs/fit_prf_l4.sh.
    source /etc/profile.d/lmod.sh 2>/dev/null || true
    SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
    if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
        export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "Using system CUDA: ${SYS_CUDA_GLOB[0]}"
    else
        echo "WARN: system cuda-11.8.0 not found"
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate retsupp_cuda
set -u
export PYTHONUNBUFFERED=1

cd "$HOME/git/retsupp"

START=$(date +%s)
python -u -m retsupp.decode.decode_runwise \
    "$SUBJECT" \
    --session "$SESSION" \
    --run "$RUN" \
    --roi "$ROI" \
    --bids-folder "$BIDS" \
    --model "$MODEL" \
    --resolution "$RESOLUTION" \
    --posterior "$POSTERIOR" \
    --l2-norm "$L2_NORM" \
    --learning-rate "$LEARNING_RATE" \
    --max-n-iterations "$MAX_ITER" \
    --min-n-iterations "$MIN_ITER" \
    --resid-max-iter "$RESID_MAX_ITER" \
    --progressbar \
    --force
END=$(date +%s)
echo "[$(date)] done in $((END - START)) s (${DEVICE})"
