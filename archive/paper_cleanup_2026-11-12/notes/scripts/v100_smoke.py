"""V100 cuInit-race smoke test.

Submitted as an 8-task SLURM array pinned to one V100 node, each task with
gpu:1. The 30s startup stagger in the wrapper should desynchronise the
parallel cuInit calls enough that the NVIDIA driver doesn't deadlock the
way it did 2026-05-13. If 8/8 tasks print SMOKE_OK, we know we can lift
the L4-only constraint on real PRF chunks.
"""
import os
import sys
import time


def stamp(t0, msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-t0:5.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    stamp(t0, f"start host={os.uname().nodename} pid={os.getpid()}")

    import tensorflow as tf  # noqa: E402

    stamp(t0, f"tf {tf.__version__} imported")
    gpus = tf.config.list_physical_devices("GPU")
    stamp(t0, f"physical GPUs: {gpus}")
    if not gpus:
        print("SMOKE_FAIL no_gpu_visible", flush=True)
        sys.exit(1)

    with tf.device("/GPU:0"):
        a = tf.random.normal([2048, 2048])
        b = tf.matmul(a, a)
        val = float(tf.reduce_sum(b).numpy())
    stamp(t0, f"matmul ok sum={val:.2e}")

    from braincoder.models import GaussianPRF2D  # noqa: E402

    stamp(t0, "braincoder GaussianPRF2D imported")

    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
