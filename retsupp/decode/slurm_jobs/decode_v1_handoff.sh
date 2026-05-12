#!/bin/bash
# Per-run V1 decode from the m4 handoff NPZ (CPU job, one (ses,run) per task).
#
# Array task → (session, run): tasks 0..5 are session 1 runs 1..6,
# tasks 6..11 are session 2 runs 1..6. Subjects with fewer runs
# (sub-20 ses-1, sub-24 ses-2 → 5 runs) skip the trailing task.
#
# Submit:
#   sbatch --array=0-11 --export=ALL,SUB=15,MAX_VOXELS=100 \
#       retsupp/decode/slurm_jobs/decode_v1_handoff.sh
#   sbatch --array=0-11 --export=ALL,SUB=15,MAX_VOXELS=200 \
#       retsupp/decode/slurm_jobs/decode_v1_handoff.sh
#
# Optional env overrides:
#   MAX_VOXELS    # top-N by r² (default 200; required at submit time
#                 # if you want a different cap)
#   NOISE_DIST    # 'gauss' (default) or 't' — Student-t residual noise
#   MAX_ITERS     # StimulusFitter max iters (default: script default 1000)
#   L2_NORM       # L2 penalty on decoded stimulus (default: 0.01)
#   LR            # learning rate (default: 0.5)
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
TASK_ID="${SLURM_ARRAY_TASK_ID:?this script must run as a SLURM array}"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -gt 11 ]]; then
    echo "ERROR: --array index must be 0..11 (got $TASK_ID)" >&2; exit 2
fi
SES=$(( TASK_ID / 6 + 1 ))
RUN=$(( TASK_ID % 6 + 1 ))
MAX_VOXELS="${MAX_VOXELS:-200}"
NOISE_DIST="${NOISE_DIST:-gauss}"

tag="sub-${sub_pad}_ses-${SES}_run-${RUN}_vox${MAX_VOXELS}"
[[ "$NOISE_DIST" == "t" ]] && tag="${tag}_t"
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

EXTRA="--noise-dist $NOISE_DIST"
[[ -n "${MAX_ITERS:-}" ]] && EXTRA="$EXTRA --max-n-iterations $MAX_ITERS"
[[ -n "${L2_NORM:-}" ]]   && EXTRA="$EXTRA --l2-norm $L2_NORM"
[[ -n "${LR:-}" ]]        && EXTRA="$EXTRA --learning-rate $LR"

OUT_DIR="$HOME/git/retsupp/notes/data/v1_decode/sub-${sub_pad}"
out_tag="ses-${SES}_run-${RUN}_vox${MAX_VOXELS}"
[[ "$NOISE_DIST" == "t" ]] && out_tag="${out_tag}_t"
OUT="${OUT_DIR}/decoded_${out_tag}.npz"

$PYTHON -u -m retsupp.decode.decode_v1_handoff \
    --subject "$SUB" \
    --session "$SES" \
    --run "$RUN" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --max-voxels "$MAX_VOXELS" \
    --out "$OUT" \
    $EXTRA

echo "Finished: $(date)"
