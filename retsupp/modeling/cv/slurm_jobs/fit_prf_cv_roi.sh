#!/bin/bash -l
#SBATCH --job-name=prf_cv_roi
#SBATCH --account=zne.uzh
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G   # peak RSS 22 GB (sub-03 fold 0, models 13/14/16, 2026-11-06)
#SBATCH --time=02:00:00   # 77 min for models 13/14/16 on 16 CPUs (90% CPU use)
#
# Cross-validated m0..m6 pRF comparison inside the ROIs (fit_prf_cv_roi.py).
# One array task per (subject, fold): task id = 10*subject + fold,
# e.g. sub-03 all folds: --array=30-32; everyone: --array=10-12,20-22,...,300-302.
#
# CPU (default; partition standard):
#   sbatch --partition=standard --array=30-32 retsupp/modeling/cv/slurm_jobs/fit_prf_cv_roi.sh
# GPU (add a GPU on the command line; the script notices it):
#   sbatch --partition=lowprio --gres=gpu:1 --constraint="L4|A100" --cpus-per-task=4 \
#       --array=30-32 retsupp/modeling/cv/slurm_jobs/fit_prf_cv_roi.sh
# Optional, in front of sbatch: MODELS="0 1 2" (subset; default all),
# OUT_SUBDIR=name (derivatives folder; default prf_cv_roi).
# ENTRY=retsupp.modeling.cv.fit_prf_cv_roi_hrf: one HRF per subject x ROI, from existing fits.

set -eo pipefail
subject=$(( SLURM_ARRAY_TASK_ID / 10 ))
fold=$(( SLURM_ARRAY_TASK_ID % 10 ))
sub_pad=$(printf "%02d" "$subject")
LOGFILE="$HOME/logs/prf_cv_roi_sub-${sub_pad}_fold-${fold}_${SLURM_JOB_ID}.txt"
mkdir -p "$HOME/logs"
exec >"$LOGFILE" 2>&1

PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"
CODE_DIR="${CODE_DIR:-$(cd "${SLURM_SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    DEVICE=gpu
    SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
    if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
        export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # cuInit race defense (see retsupp/modeling/slurm_jobs/fit_prf_l4_chunked.sh)
    LOCK="/dev/shm/cuinit_warm_$(hostname -s).flock"
    (
        flock -w 60 -x 200 || exit 0
        "$PYTHON" -c "import tensorflow as tf, sys; sys.exit(0 if tf.config.list_physical_devices('GPU') else 1)"
    ) 200>"$LOCK" || { echo "FATAL: no GPU visible after warm-up"; exit 1; }
else
    DEVICE=cpu
    export CUDA_VISIBLE_DEVICES=""
    export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-16}"
    export TF_NUM_INTEROP_THREADS=2
fi
echo "Host: $(hostname) | $DEVICE | sub-${sub_pad} fold ${fold} | job ${SLURM_JOB_ID} | $(date)"
echo "Code: ${CODE_DIR} @ $(git -C "$CODE_DIR" rev-parse --short HEAD)"

"$PYTHON" -u -m "${ENTRY:-retsupp.modeling.cv.fit_prf_cv_roi}" "$subject" --fold "$fold" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp ${MODELS:+--models $MODELS} \
    ${OUT_SUBDIR:+--output-subdir $OUT_SUBDIR}

echo "Finished: $(date)"
