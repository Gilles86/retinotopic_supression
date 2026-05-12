#!/bin/bash
# Per-HP V1 decode from the m4 handoff NPZ (CPU job, one HP per array task).
#
# Submit (one HP per array task; HP picked by SLURM_ARRAY_TASK_ID):
#   sbatch --array=0-3 --export=ALL,SUB=15 \
#       retsupp/decode/slurm_jobs/decode_v1_handoff.sh
#
#   sbatch --array=0-3 --export=ALL,SUB=15,MAX_VOXELS=100 \
#       retsupp/decode/slurm_jobs/decode_v1_handoff.sh
#
# Optional env overrides:
#   MAX_VOXELS    cap on # voxels after FDR (default: unset = all that pass)
#   MAX_ITERS     StimulusFitter max iters (default: script default 1000)
#   L2_NORM       L2 penalty on decoded stimulus (default: 0.01)
#   LR            learning rate (default: 0.5)
#
#SBATCH --job-name=decode_v1
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -eo pipefail

SUB="${SUB:?missing SUB env var}"
sub_pad=$(printf "%02d" "$SUB")

HP_LIST=(upper_right upper_left lower_left lower_right)
TASK_ID="${SLURM_ARRAY_TASK_ID:?this script must run as a SLURM array}"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -gt 3 ]]; then
    echo "ERROR: --array index must be 0..3 (got $TASK_ID)" >&2; exit 2
fi
HP="${HP_LIST[$TASK_ID]}"

tag="sub-${sub_pad}_hp-${HP}"
[[ -n "${MAX_VOXELS:-}" ]] && tag="${tag}_vox${MAX_VOXELS}"

LOGFILE="$HOME/logs/decode_v1_${tag}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" name="decode_v1_${tag}" \
    2>/dev/null || true

echo "Host: $(hostname) | Job ${SLURM_JOB_ID} | ${tag}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}  Mem: ${SLURM_MEM_PER_NODE:-?}MB"
echo "Started: $(date)"

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TF_NUM_INTEROP_THREADS=2

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

EXTRA=""
[[ -n "${MAX_VOXELS:-}" ]] && EXTRA="$EXTRA --max-voxels $MAX_VOXELS"
[[ -n "${MAX_ITERS:-}" ]]  && EXTRA="$EXTRA --max-n-iterations $MAX_ITERS"
[[ -n "${L2_NORM:-}" ]]    && EXTRA="$EXTRA --l2-norm $L2_NORM"
[[ -n "${LR:-}" ]]         && EXTRA="$EXTRA --learning-rate $LR"

OUT_DIR="$HOME/git/retsupp/notes/data/v1_decode/sub-${sub_pad}"
out_tag="hp-${HP}"
[[ -n "${MAX_VOXELS:-}" ]] && out_tag="${out_tag}_vox${MAX_VOXELS}"
OUT="${OUT_DIR}/decoded_${out_tag}.npz"

$PYTHON -u -m retsupp.decode.decode_v1_handoff \
    --subject "$SUB" \
    --hp "$HP" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --out "$OUT" \
    $EXTRA

echo "Finished: $(date)"
