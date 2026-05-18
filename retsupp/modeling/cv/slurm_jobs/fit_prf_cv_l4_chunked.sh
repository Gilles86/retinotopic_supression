#!/bin/bash
#SBATCH --job-name=prf_cv_chunked
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --constraint="L4"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Per-(subject, model, fold, chunk) GPU CV PRF fit. Mirrors
# fit_prf_l4_chunked.sh exactly (lowprio L4 + cuInit flock + 32G);
# only the python entry point and output path change.
#
# Required env (--export=ALL,...):
#   MODEL      1..6
#   FOLD       0..(N_FOLDS-1)        held-out fold index
#   N_FOLDS    total folds           (default 3)
#   SUBJECT    int                   subject id
#   N_CHUNKS   chunks per (sub,model,fold) (typically 10)
#   KIND       full | bar            (default full)
#
# Output:
#   derivatives/prf_cv/model{N}/fold-{K}/sub-XX/chunks/
#     chunk-NNNN-of-MMMM.npz
# After all chunks done, run merge_prf_cv_chunks.py to assemble
# the per-(sub, model, fold) r2_test.nii.gz.

set -eo pipefail

KIND="${KIND:-full}"
N_FOLDS="${N_FOLDS:-3}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
for v in MODEL FOLD SUBJECT N_CHUNKS; do
    if [[ -z "${!v:-}" ]]; then
        echo "ERROR: $v env var required." >&2; exit 2
    fi
done

subject="$SUBJECT"
chunk_idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/prf_cv_m${MODEL}_f${FOLD}_sub-${sub_pad}_chunk-${chunk_idx}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_cv_m${MODEL}_f${FOLD}_sub-${sub_pad}" 2>/dev/null || true

# NFS dogpile stagger
sleep $(( RANDOM % 15 ))

echo "Host: $(hostname) | sub-${subject} | model ${MODEL} | fold ${FOLD}/${N_FOLDS} | kind ${KIND}"
echo "  chunk ${chunk_idx}/${N_CHUNKS}  array_task=${SLURM_ARRAY_TASK_ID}"
echo "Started: $(date)"

# CUDA runtime (TF 2.14 / cuDNN 8.7 ABI; CUDA 11.8 spack install)
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda
export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

# cuInit race defense — see retsupp/modeling/slurm_jobs/fit_prf_l4_chunked.sh
# for the deep rationale (flock on /dev/shm, not /tmp).
LOCK="/dev/shm/cuinit_warm_$(hostname -s).flock"
echo "cuInit warm-up under flock $LOCK ..."
(
    flock -w 60 -x 200 || {
        echo "WARN: couldn't acquire $LOCK in 60s; skipping warm-up";
        exit 0;
    }
    "$PYTHON" -c "
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'cuInit warm-up: {len(gpus)} GPU(s) visible to TF', flush=True)
sys.exit(0 if gpus else 1)
"
) 200>"$LOCK"
WARMUP_RC=$?
if [[ $WARMUP_RC -ne 0 ]]; then
    echo "FATAL: cuInit warm-up returned $WARMUP_RC (no GPU after lock-protected init). Exiting fast."
    exit 1
fi

$PYTHON -u -m retsupp.modeling.cv.fit_prf_cv \
    "$subject" "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --fold "$FOLD" \
    --n-folds "$N_FOLDS" \
    --resolution 50 \
    --voxel-chunk-size 100000 \
    --paradigm-kind "$KIND" \
    --chunk-index "$chunk_idx" \
    --n-chunks "$N_CHUNKS"

echo "Finished: $(date)"
