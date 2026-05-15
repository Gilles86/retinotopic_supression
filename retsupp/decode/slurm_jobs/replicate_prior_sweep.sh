#!/bin/bash
# Replicate the prior working decode sweep (the "much better" GIFs at
# notes/figures/decode_sweep/m4/) on sub-23 V1 ses-1 run-1. Uses
# smoke_test_sweep.py as-is with its original voxel filters:
#   sd >= 0.05, r2 >= 0.1, ecc <= 4.5, top-200 by r2, resolution=30
# at the prior winner cell L2=0.5, lr=0.05.
#
#   sbatch retsupp/decode/slurm_jobs/replicate_prior_sweep.sh
#
#SBATCH --job-name=decode_replicate
#SBATCH --output=/dev/null
#SBATCH --time=15:00
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:1
#SBATCH --constraint=L4|V100|A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -eo pipefail

SUBJECT="${SUBJECT:-23}"
ROI="${ROI:-V1}"
SESSION="${SESSION:-1}"
RUN="${RUN:-1}"
BIDS="${BIDS:-/shares/zne.uzh/gdehol/ds-retsupp}"
VOXEL_R2_MIN="${VOXEL_R2_MIN:-0.1}"
VOXEL_ECC_MAX="${VOXEL_ECC_MAX:-4.5}"
VOXEL_MAX="${VOXEL_MAX:-200}"
L2="${L2:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
RESIDUAL_METHOD="${RESIDUAL_METHOD:-gauss}"

LOGFILE="$HOME/logs/decode_replicate_sub-$(printf %02d $SUBJECT)_${ROI}_r2-${VOXEL_R2_MIN}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "[$(date)] === replicate sweep on sub-$(printf %02d $SUBJECT) $ROI ==="
echo "  filters: sd>=0.05, r2>=${VOXEL_R2_MIN}, ecc<=${VOXEL_ECC_MAX}, top-${VOXEL_MAX}, res=30"
echo "  cell: L2=${L2}, lr=${LEARNING_RATE}"

source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
nvidia-smi --query-gpu=name --format=csv,noheader || true

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate retsupp_cuda
set -u
export PYTHONUNBUFFERED=1

cd "$HOME/git/retsupp"

START=$(date +%s)
python -u -m retsupp.decode.smoke_test_sweep \
    --bids-folder "$BIDS" \
    --subject "$SUBJECT" --roi "$ROI" \
    --session "$SESSION" --run "$RUN" \
    --model 4 --resolution 30 \
    --l2-norms "$L2" --learning-rates "$LEARNING_RATE" \
    --voxel-r2-min "$VOXEL_R2_MIN" \
    --voxel-ecc-max "$VOXEL_ECC_MAX" \
    --voxel-max "$VOXEL_MAX" \
    --residual-method "$RESIDUAL_METHOD" \
    --max-n-iterations 1000 --resid-max-iter 300

# smoke_test_sweep names outputs only by (l2, lr) — they collide across
# cells that share L2/lr but differ in voxel filters / residual method.
# Rename to a unique tag so the cells don't clobber each other.
TAG="sub-$(printf %02d $SUBJECT)_${ROI}_ses-${SESSION}_run-${RUN}"
DATA_DIR="$HOME/git/retsupp/notes/data/decode_sweep/m4/${TAG}"
SRC="${DATA_DIR}/l2-${L2}_lr-${LEARNING_RATE}.npz"
DST="${DATA_DIR}/l2-${L2}_lr-${LEARNING_RATE}_r2-${VOXEL_R2_MIN}_n-${VOXEL_MAX}_method-${RESIDUAL_METHOD}.npz"
if [[ -f "$SRC" ]]; then
    mv "$SRC" "$DST"
    mv "${SRC%.npz}.tsv" "${DST%.npz}.tsv" 2>/dev/null || true
    echo "Renamed output: $DST"
fi
END=$(date +%s)
echo "[$(date)] done in $((END - START)) s"
