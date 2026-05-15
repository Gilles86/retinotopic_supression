#!/bin/bash
#SBATCH --job-name=prf_chunked
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
# GPU constraint: L4|V100|A100. The cuInit race (multiple jobs on the
# same 8-GPU node init'ing CUDA simultaneously) is defused by the
# 30s random stagger below. H100/H200 excluded — sm_90 needs CUDA 12,
# we're on 11.8.
#SBATCH --constraint="L4"
# (Narrowed from L4|V100|A100 because V100 + A100 are 8-GPU nodes,
# which hits the cuInit race when multiple jobs concentrate on one
# host. L4 nodes have 1 GPU each — no concurrent cuInit on same
# host possible, race structurally impossible. Throughput cost:
# ~8 idle L4 GPUs on lowprio vs ~40 V100 + ~8 A100, but the race
# fix is worth it.)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# (Bumped from 24G after sub-05 / sub-08 m4 chunks OOM'd at host-RAM
# step 0:125 during stage-2 transition. Subjects with the largest
# BOLD masks have V ≈ 350k voxels, ~20% more than typical V=300k;
# combined with --voxel-chunk-size=100000 they exceeded the 24G
# budget when TF allocated gradient tape for the 7-free-param
# spatial+amplitude stage. 32G has headroom; bump again or shrink
# --voxel-chunk-size if it recurs.)

# Partition=lowprio: same physical L4/A100/V100 nodes as standard,
# but jobs dispatch almost immediately (5-10s) instead of sitting in
# fairshare queue for hours. Trade-off: a higher-priority standard job
# can preempt us mid-fit. For a 15-30 min chunk that's a low risk;
# preempted tasks just get resubmitted. See retsupp/CLAUDE.md
# §"Lowprio partition for GPU PRF fits".

# Per-(subject, chunk) GPU PRF fit. ONE model + ONE kind per submission.
# Array task index maps to (sub_idx, chunk_idx) via the same mapping
# as fit_prf_chunked.sh:
#
#   idx0      = SLURM_ARRAY_TASK_ID - 1
#   sub_idx   = idx0 / N_CHUNKS
#   chunk_idx = idx0 % N_CHUNKS
#   subject   = sub_idx + 1
#
# Required env vars (--export=ALL,...):
#   MODEL      1..6
#   N_CHUNKS   chunks per subject (typically 10)
#   N_SUBS     total subjects     (typically 30)
#   KIND       full | bar         (default full)
#
# Submit at the tightest plausible walltime via --time at sbatch time
# (priority bucket matters). Typical per-task wallclock on A100:
#   m1: 8-12 min (includes grid)
#   m2/m3: 6-10 min
#   m4: 10-15 min (DoG + flex HRF, longer schedule)
#   m5: 8-12 min (DN, fixed HRF)
#   m6: 12-18 min (DN + flex HRF, most complex)
#
# Outputs land at derivatives/{prf|prf_bar}/model{N}/sub-XX/chunks/.
# Run merge_prf_chunks.sh afterwards to assemble final NIfTIs.

set -eo pipefail

KIND="${KIND:-full}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set." >&2; exit 2
fi
if [[ -z "${MODEL:-}" || -z "${N_CHUNKS:-}" ]]; then
    echo "ERROR: MODEL and N_CHUNKS env vars required." >&2; exit 2
fi
if [[ "$KIND" != "full" && "$KIND" != "bar" ]]; then
    echo "ERROR: KIND must be 'full' or 'bar' (got '$KIND')." >&2; exit 2
fi

# Two modes:
# (a) Per-subject mode (preferred): SUBJECT is exported by the caller
#     and the array index just enumerates chunks.  N_SUBS not needed.
# (b) Phase-wide mode (legacy): SUBJECT not set; array index encodes
#     both subject and chunk via (idx-1) % N_CHUNKS.  N_SUBS required.
if [[ -n "${SUBJECT:-}" ]]; then
    subject="$SUBJECT"
    chunk_idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
else
    if [[ -z "${N_SUBS:-}" ]]; then
        echo "ERROR: either SUBJECT, or N_SUBS env var must be set." >&2
        exit 2
    fi
    idx0=$(( SLURM_ARRAY_TASK_ID - 1 ))
    sub_idx=$(( idx0 / N_CHUNKS ))
    chunk_idx=$(( idx0 % N_CHUNKS ))
    subject=$(( sub_idx + 1 ))
    if [[ "$subject" -gt "$N_SUBS" ]]; then
        echo "Array index out of range." >&2; exit 2
    fi
fi
sub_pad=$(printf "%02d" "$subject")

LOGFILE="$HOME/logs/prf_chunked_m${MODEL}_${KIND}_sub-${sub_pad}_chunk-${chunk_idx}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

# SLURM's Name field is array-shared — whichever task renames last wins
# for ALL tasks. Don't put chunk_idx here (it would mislead in squeue).
# The chunk index is recoverable from the array task suffix on JobID.
scontrol update jobid="${SLURM_JOB_ID}" \
    name="prf_m${MODEL}_${KIND}_sub-${sub_pad}" 2>/dev/null || true

# NFS profile-read dogpile defense (a different race from cuInit —
# many SLURM tasks reading $HOME/.bashrc at once cause "user env
# retrieval failed" holds). Short random stagger spreads the reads.
sleep $(( RANDOM % 15 ))

echo "Host: $(hostname) | sub-${subject} | model ${MODEL} | kind ${KIND}"
echo "  chunk ${chunk_idx}/${N_CHUNKS}  array_task=${SLURM_ARRAY_TASK_ID}"
echo "Started: $(date)"

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

# cuInit-race defense (replaces the old 30s random stagger which is
# mathematically insufficient when 8 jobs land on the same V100/A100
# node: with sleep RANDOM%30 the expected adjacent-start gap is ~4s,
# below the ~1s needed for cuInit non-overlap → ~90% chance of
# driver deadlock).
#
# Instead: per-node flock that serializes a minimal TF GPU-init across
# all jobs landing on the same hostname. The FIRST job under the lock
# warms the driver (forces a complete cuInit on its allocated GPU);
# every subsequent job grabs the lock briefly, hits the now-warm
# driver, and releases. Lock is on /tmp (node-local), not NFS, so
# semantics are correct even at scale.
#
# IMPORTANT — the lock file MUST be on a path that is shared across
# all SLURM job-steps on the same physical node. On sciencecluster,
# `/tmp` is bind-mounted to `/var/spool/slurmd/job<id>/tmp` per job,
# so a lock at `/tmp/foo.flock` is per-JOB, not per-NODE. Multiple
# jobs landing on the same V100/A100 8-GPU node each acquire their
# own /tmp lock instantly and all proceed in parallel → cuInit race
# → 0 GPUs visible to TF in each. We use `/dev/shm` (RAM-backed
# tmpfs) which IS shared across all processes on the node.
#
# Crash safety: flock is released by the kernel when the holding
# process dies (any reason). The subshell below only holds the lock
# during its own execution; if the Python warm-up segfaults / hangs /
# is OOM-killed, the subshell exits, FD closes, lock releases. -w 60
# is belt-and-suspenders: even if some pathological state prevents
# acquisition, the job proceeds after 60s rather than hanging until
# SLURM walltime.
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
)
WARMUP_RC=$?
if [[ $WARMUP_RC -ne 0 ]]; then
    echo "FATAL: cuInit warm-up returned $WARMUP_RC (no GPU after lock-protected init). " \
         "The driver on this node may be in a bad state from a prior cuInit race; exiting fast" \
         " so the afterok chain fails visibly instead of running 25x slower until walltime."
    exit 1
fi

$PYTHON -u "$HOME/git/retsupp/retsupp/modeling/fit_prf.py" \
    "$subject" --model "$MODEL" \
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \
    --resolution 50 \
    --voxel-chunk-size 100000 \
    --paradigm-kind "$KIND" \
    --chunk-index "$chunk_idx" \
    --n-chunks "$N_CHUNKS"

echo "Finished: $(date)"
