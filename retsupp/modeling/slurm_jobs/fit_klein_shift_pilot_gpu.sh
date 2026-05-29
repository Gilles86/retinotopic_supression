#!/bin/bash
#SBATCH --job-name=klein_shift_gpu
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
# GPU type and host RAM are set at submit time via --constraint / --mem
# so the same script probes both A100 and L4:
#
#   A100 (safe default): 40/80 GB VRAM. The Klein-shift forward chunks a
#   (B,V,Tc,G) buffer over T; the c600c97 commit dropped recompute_grad,
#   so peak tape residency is ~64 GB at V=500 *if* it sits on the tape.
#   But that buffer lives inside the jit_compile=True XLA region, so XLA
#   may rematerialise it in backward instead of parking it in VRAM — in
#   which case L4 fits too. Only way to know is to try both.
#     sbatch --constraint=A100 --mem=64G --array=1-33  fit_klein_shift_pilot_gpu.sh
#
#   L4 probe: 24 GB VRAM, 1-GPU nodes (cuInit race structurally
#   impossible). If the tape really holds ~64 GB this OOMs with
#   ResourceExhausted (1:0); if XLA remats, it runs. Probe a few tasks
#   before committing the full array:
#     sbatch --constraint=L4 --mem=32G --array=1-3   fit_klein_shift_pilot_gpu.sh
#
# H100/H200 are excluded by the env (sm_90 needs CUDA 12; we're on 11.8).
# Defaults below apply if neither flag is given at submit time.
#SBATCH --constraint="A100"
#SBATCH --mem=64G

# GPU pilot for DoGKleinShift_v3_target_6sigma — sub-3, sub-17, sub-23,
# all 11 ROIs (33 array tasks). GPU twin of fit_klein_shift_pilot.sh.
# GPU type / host RAM are submit-time flags (see the SBATCH block above).

set -eo pipefail

LOGFILE="$HOME/logs/klein_shift_gpu_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "Host:           $(hostname)"
echo "Job:            ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Started:        $(date)"

SUB_IDS=(3 17 23)
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
