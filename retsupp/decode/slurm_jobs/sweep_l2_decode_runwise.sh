#!/bin/bash
# L2-sweep for the new decode_runwise on a single (sub, ROI, run).
#
# One SBATCH job = one (subject, roi, session, run, L2) cell. Writes
# to a per-L2-suffixed npz path so the cells don't clobber each other.
# Pair with submit_sweep_l2.sh for the parallel launch.
#
#   sbatch --export=ALL,SUBJECT=3,L2=0.5 \
#       retsupp/decode/slurm_jobs/sweep_l2_decode_runwise.sh
#
#SBATCH --job-name=decode_l2sweep
#SBATCH --output=/dev/null
#SBATCH --time=30:00
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:1
#SBATCH --constraint=L4|V100|A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -eo pipefail

SUBJECT="${SUBJECT:-3}"
ROI="${ROI:-V1}"
SESSION="${SESSION:-1}"
RUN="${RUN:-1}"
L2="${L2:?must set L2 via --export=ALL,L2=...}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
BIDS="${BIDS:-/shares/zne.uzh/gdehol/ds-retsupp}"
MODEL="${MODEL:-4}"
RESOLUTION="${RESOLUTION:-50}"
POSTERIOR="${POSTERIOR:-0.5}"
MAX_ITER="${MAX_ITER:-1000}"
MIN_ITER="${MIN_ITER:-200}"
RESID_MAX_ITER="${RESID_MAX_ITER:-300}"

TAG="sub-$(printf %02d $SUBJECT)_${ROI}_ses-${SESSION}_run-${RUN}_l2-${L2}_lr-${LEARNING_RATE}"
LOGFILE="$HOME/logs/decode_l2sweep_${TAG}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" name="decode_l2sweep_${TAG}" || true

echo "[$(date)] === L2 sweep cell ==="
echo "  ${TAG}  model=${MODEL} res=${RESOLUTION} posterior=${POSTERIOR}"

# u24 GPU nodes lack `module load gpu`; point LD_LIBRARY_PATH at system CUDA 11.8.
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Using system CUDA: ${SYS_CUDA_GLOB[0]}"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate retsupp_cuda
set -u
export PYTHONUNBUFFERED=1

cd "$HOME/git/retsupp"

# Write to a per-L2 path under derivatives/decoded_paradigm_sweep/.
OUT="${BIDS}/derivatives/decoded_paradigm_sweep/model${MODEL}/sub-$(printf %02d $SUBJECT)/${TAG}_desc-decoded.npz"
mkdir -p "$(dirname "$OUT")"

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
    --l2-norm "$L2" \
    --learning-rate "$LEARNING_RATE" \
    --max-n-iterations "$MAX_ITER" \
    --min-n-iterations "$MIN_ITER" \
    --resid-max-iter "$RESID_MAX_ITER" \
    --output-path "$OUT" \
    --force
END=$(date +%s)
echo "[$(date)] done in $((END - START)) s"
