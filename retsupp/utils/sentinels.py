"""Fail-fast sanity checks for SLURM-launched fit jobs.

The pattern: catch "silent fallback to a degraded mode" before the
job sinks 20+ minutes of wallclock on it. Today's failures (cuInit
race → CPU fallback → walltime TIMEOUT → DepNeverSatisfied cascade)
all had this shape; sentinels make the failure SLURM-visible within
seconds.

Call these at the top of any fit script's ``main()``.
"""
from __future__ import annotations

import os
import sys


def assert_gpu_available_if_expected() -> None:
    """Exit fast if SLURM allocated a GPU but TF can't see one.

    The cuInit race on multi-GPU nodes occasionally drops a job to
    CPU silently — the run then plods along at ~25× the GPU speed
    and hits walltime. We'd rather the job exit-1 here so SLURM
    marks it FAILED, the downstream ``afterok`` chain is poisoned
    *visibly*, and we resubmit (the per-node flock warm-up should
    have prevented the race in the first place, but defense-in-depth).

    Behaviour:
      - ``CUDA_VISIBLE_DEVICES`` unset or empty → no expectation; pass.
      - ``CUDA_VISIBLE_DEVICES='-1'`` or ``'NoDevFiles'`` → CPU
        explicitly requested; pass without importing TF.
      - Otherwise: import TF, check that
        ``tf.config.list_physical_devices('GPU')`` returns at least
        one device. If empty, exit with a clear error.
    """
    cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_env in ('', '-1', 'NoDevFiles'):
        return
    try:
        import tensorflow as tf
    except ImportError:
        return  # no TF in env; not our problem to flag here.
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        sys.exit(
            f"FATAL: SLURM allocated GPU (CUDA_VISIBLE_DEVICES="
            f"{cuda_env!r}) but TensorFlow sees no GPU device. "
            f"Likely a cuInit race / driver hiccup. Exiting fast "
            f"so the afterok chain fails visibly instead of "
            f"running 25× slower until walltime TIMEOUT."
        )
    print(f"sentinel: cuInit OK ({len(gpus)} GPU(s) visible to TF)",
          flush=True)
