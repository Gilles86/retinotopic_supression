#!/bin/bash
#SBATCH --job-name=klein_shift_gpu
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
# L4 is the verified default. After the separable-Gaussian forward rewrite
# (local_models.py: DoGKleinShift), measured peak VRAM at V=500 is 642 MB and
# a fit step is ~13x faster than the old dense forward — so a 24 GB L4 has
# huge headroom and the previous A100/recompute-grad concerns are moot. L4
# nodes are single-GPU, so the cuInit race is structurally impossible, and
# they dispatch fastest on lowprio. The fit is small/fast now; 30 min walltime
# is generous (XLA compile + warmstart load + ~1500 iters). Override at submit
# time for a bigger GPU if ever needed:
#     sbatch --constraint=A100 --mem=32G --array=1-33  fit_klein_shift_pilot_gpu.sh
# H100/H200 are excluded by the env (sm_90 needs CUDA 12; we're on 11.8).
#SBATCH --constraint="L4"
#SBATCH --mem=16G

# GPU non-CV (full-data) fit for DoGKleinShift_v3_target_6sigma.
# Subject set is configurable via the KLEIN_SUBS env var; default = ALL 30.
#   Full run (30 subs × 11 ROIs = 330 tasks):
#     sbatch --array=1-330 retsupp/modeling/slurm_jobs/fit_klein_shift_pilot_gpu.sh
#   3-subject pilot (03/17/23 × 11 = 33 tasks):
#     sbatch --array=1-33 --export=ALL,KLEIN_SUBS="3 17 23" \
#            retsupp/modeling/slurm_jobs/fit_klein_shift_pilot_gpu.sh
# GPU type / host RAM are submit-time flags (see the SBATCH block above).

set -eo pipefail

LOGFILE="$HOME/logs/klein_shift_gpu_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:           $(hostname)"
echo "Job:            ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:        $(date)"

# Default to all 30 subjects; override with KLEIN_SUBS="3 17 23" for a pilot.
SUB_IDS=(${KLEIN_SUBS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30})
ROIS=(V1 V2 V3 V3AB hV4 LO TO VO IPS SPL1 FEF)
N_ROIS=${#ROIS[@]}
N_SUBS=${#SUB_IDS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-33." >&2
    exit 2
fi

idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
sub_idx=$(( idx0 / N_ROIS ))
roi_idx=$(( idx0 % N_ROIS ))
if [[ "$sub_idx" -ge "$N_SUBS" ]]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range; max is $(( N_SUBS * N_ROIS ))." >&2
    exit 2
fi
subject="${SUB_IDS[$sub_idx]}"
roi="${ROIS[$roi_idx]}"

echo "Subject:        ${subject}"
echo "ROI:            ${roi}"

# SLURM Name field is array-shared; encode only sub (not roi, which
# would mislead in squeue — roi is recoverable from the _N array suffix).
sub_pad=$(printf "%02d" "$subject")
scontrol update jobid="${SLURM_JOB_ID}" \
    name="klein_gpu_sub-${sub_pad}" 2>/dev/null || true

# NFS profile-read dogpile defense (distinct from the cuInit race).
sleep $(( RANDOM % 15 ))

# CUDA runtime (TF 2.14 / cuDNN 8.7 ABI; CUDA 11.8 spack install).
source /etc/profile.d/lmod.sh 2>/dev/null || true
SYS_CUDA_GLOB=( /apps/u24/opt/x86_64_v3/cuda-11.8.0-* )
if [[ -d "${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib" ]]; then
    export LD_LIBRARY_PATH="${SYS_CUDA_GLOB[0]}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_cuda

export PYTHONUNBUFFERED=1
PYTHON="$HOME/data/conda/envs/retsupp_cuda/bin/python"

# cuInit-race defense: per-node flock on /dev/shm (shared across all
# job-steps on the same physical node — /tmp is per-job bind-mounted
# and would NOT serialize). First job under the lock warms the driver;
# subsequent jobs hit the now-warm driver and release. Fail fast if no
# GPU appears after the lock-protected init, so a degraded node fails
# visibly instead of silently falling back to 25× slower CPU.
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
    echo "FATAL: cuInit warm-up returned $WARMUP_RC (no GPU after lock-protected init)." \
         " Node driver may be in a bad state from a prior cuInit race; exiting fast."
    exit 1
fi

bids_folder="/shares/zne.uzh/gdehol/ds-retsupp"
SCRIPT="$HOME/git/retsupp/retsupp/modeling/fit_dog_dynamic_af_braincoder.py"

echo "Running DoG-Klein-shift fit (GPU) for sub-${subject}, roi=${roi}"

"$PYTHON" -u "$SCRIPT" \
    "$subject" \
    --bids-folder "$bids_folder" \
    --roi "$roi" \
    --model-version v3 \
    --with-target \
    --shared-target-sigma \
    --klein-shift \
    --max-voxels 500 \
    --max-n-iterations 1500 \
    --p-signal-thr 0.5 \
    --mode signed

echo "Finished:       $(date)"
