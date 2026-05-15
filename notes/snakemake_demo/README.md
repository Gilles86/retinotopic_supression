# Snakemake proof-of-concept (retsupp PRF pipeline)

A minimal end-to-end demo of replacing `submit_prf_sweep_persub.sh` with
Snakemake. Scope: one subject (sub-01), one model (m1), three rules:
`prf_chunks` (10 SLURM tasks on lowprio GPUs) → `prf_merge` (1 CPU task)
→ `prf_surface` (1 CPU task, both hemis in one job).

Outputs land under `derivatives/prf.snakemake_demo/model1/sub-01/` —
the canonical `derivatives/prf/...` is **not** touched.

## Files

- `Snakefile` — the three rules + the input/output wiring.
- `profile/config.yaml` — SLURM executor defaults (account, partition,
  default-resources, jobs cap).
- `scripts/sample_to_surface_demo.py` — thin fork of the surface sampler
  that reads from `prf.<suffix>/` instead of hardcoded `prf/`. Lets the
  demo go end-to-end without modifying the canonical
  `retsupp/surface/sample_prf_to_surface_nilearn.py`.
- `logs/` — Snakemake's own logs + per-rule shell stdout (slurm jobs
  log into `~/logs/slurm/` via the SLURM executor by default; this
  folder gets the rule-level log: entries).

## One-time install on the cluster

A separate env keeps the install isolated from `retsupp_cuda` (which
must stay TF 2.14 / CUDA 11.8 compatible).

```bash
ssh sciencecluster
~/data/miniforge3/bin/mamba create -y -n retsupp_snake \
    -c conda-forge -c bioconda \
    "snakemake>=8" "snakemake-executor-plugin-slurm"
```

`retsupp_snake` only runs the driver — the SLURM jobs themselves
activate `retsupp_cuda` for the actual Python work.

## Run

From the cluster login node (inside `~/git/retsupp`):

```bash
~/data/conda/envs/retsupp_snake/bin/snakemake \
    --snakefile notes/snakemake_demo/Snakefile \
    --workflow-profile notes/snakemake_demo/profile
```

The driver runs on the login node and submits each rule's tasks via
`sbatch` (the profile fixes `--executor slurm`). It will sit in the
foreground, polling SLURM, until all targets are on disk. Drive it
under tmux so it survives a disconnect:

```bash
tmux new-session -d -s snake_demo "cd ~/git/retsupp && \
    ~/data/conda/envs/retsupp_snake/bin/snakemake \
        --snakefile notes/snakemake_demo/Snakefile \
        --workflow-profile notes/snakemake_demo/profile \
        2>&1 | tee notes/snakemake_demo/logs/run.log"

# Peek every minute or so:
tmux capture-pane -t snake_demo -p | tail -30
```

## What to expect

Wall time on lowprio (very rough, sub-01 / m1): each chunk finishes
~10 min on an L4 GPU, merge runs ~1 min, surface ~3 min. **Caveat:**
how fast chunks dispatch depends entirely on the lowprio L4 queue;
when the queue is busy (~80 PD jobs observed 2026-05-15) you may only
get 4 concurrent L4 slots and the rest sit at `(Priority)`. With
plenty of idle L4s, 10 chunks dispatch in under a minute. Plan for
anywhere between 15 min and several hours wall-clock.

Snakemake names every SLURM job after its run UUID, so `squeue` shows
`0866e12f-6d64-4ea1-...` instead of a human-readable name. To find
your demo jobs: `squeue --me -O jobid,name,state | grep <first-8-of-uuid>`.

Final output the driver waits on:

```
{BIDS}/derivatives/prf.snakemake_demo/model1/sub-01/sub-01_desc-r2.optim.nilearn_space-fsnative_hemi-L.func.gii
```

`{BIDS}` = `/shares/zne.uzh/gdehol/ds-retsupp`.

## Cleanup

The whole demo lives in one folder; remove with:

```bash
rm -rf /shares/zne.uzh/gdehol/ds-retsupp/derivatives/prf.snakemake_demo/
```

## Gotchas hit during the proof-of-concept

(2026-05-15, snakemake 9.20.0 + snakemake-executor-plugin-slurm 2.6.1)

- **`slurm_extra` is locked-down.** The plugin maintains a `forbidden_options`
  list (see `validation.py::get_forbidden_slurm_options`) covering
  `--gres`, `--constraint`, `--job-name`, `--account`, `--partition`,
  `--mem`, `--time`, etc. Putting any of those in `slurm_extra` raises
  `WorkflowError` inside the threadpool worker, which is then swallowed
  — symptom: snakemake logs "Submitting 10 ready jobs" and silently
  does nothing. Use first-class resource keys instead:
  `gres="gpu:1"`, `constraint="L4"`, `slurm_partition`, `slurm_account`,
  `runtime`, `mem_mb`, `cpus_per_task`.
- **SLURM job-names are UUIDs.** The plugin sets `--job-name <run-uuid>`
  itself so it can correlate `sacct/squeue` output back to the
  workflow run. It's intentional; just unfortunate for visual
  inspection in `squeue`.
- **`time.sleep(5)` per submission.** The plugin sleeps 5 seconds
  between each sbatch call (see `__init__.py:1138`). With 10 chunks
  that's a ~50s ramp-up before all are queued. Tolerable; harmless.
- **No cuInit warm-up flock.** Our Snakefile skips the per-host
  `/dev/shm` flock the canonical `fit_prf_l4_chunked.sh` does — safe
  here because we constrain to L4 (1 GPU per node), so two jobs can't
  cuInit-race on the same hardware.
- **Cleaned-BOLD cache is a hard prereq.** The demo doesn't rebuild
  it (the canonical `build_cleaned_bold_cache.sh` is a separate job);
  Snakefile errors immediately at load time if missing.
