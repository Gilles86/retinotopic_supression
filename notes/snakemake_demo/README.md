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

Wall time on lowprio (very rough, sub-01 / m1): chunks dispatch within
30s, finish ~10 min each, merge runs ~1 min, surface ~3 min. Total
~15 min from submit to all done if no preemption.

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
