# retsupp PRF + AF pipeline — Snakemake driver

Snakemake replacement for `retsupp/modeling/slurm_jobs/submit_prf_sweep_persub.sh`.
The bash submitter remains as a fallback; this Snakemake is an
alternative entry point that infers the DAG from input/output files
instead of manually chained `--dependency=afterok:` strings.

## What's in this folder

- `Snakefile` — all rules: `cleaned_bold_cache`, `prf_chunks`,
  `prf_merge`, `prf_surface`, `neuropythy_register`, `r2_mixture`,
  `af_fit`.
- `config.yaml` — subjects, N_CHUNKS, KIND, model list, AF flags,
  cluster paths. Edit this (or override via `--config`) to change scope.
- `profile/config.yaml` — SLURM executor defaults
  (account, partition, log dir, retry policy, `jobs: 150`).
- `logs/` — per-rule shell-script stdout (one file per rule
  invocation). The SLURM executor drops its own sbatch logs under
  `.snakemake/slurm_logs/`.

## DAG

Per subject, Snakemake builds:

```
cleaned_bold_cache
        |
        v
m1 chunks(N=10) -> m1 merge -> m1 surface -> m1 neuropythy
                                                  |
              +-----------------------------------+
              v                                   v
        m2 chunks -> m2 merge -> m2 surf -> m2 neuropythy
        m3 chunks -> m3 merge -> m3 surf -> m3 neuropythy (after m2 neuro)
                            |
                            v
        m4 chunks -> m4 merge -> m4 surf -> m4 neuropythy -> r2_mixture(m4)
                                                                  |
                                                                  v
                                                    af_fit x 11 ROIs
        m5 chunks -> m5 merge -> m5 surf -> m5 neuropythy (after m4 neuro)
                            |
                            v
        m6 chunks -> m6 merge -> m6 surf -> m6 neuropythy (after m5 neuro)
```

Cross-subject independence: each subject's chain is isolated by the
`{sub_pad}` wildcard — one subject failing doesn't block any other.
Matches the "per-subject dependency chain" pattern in
`retsupp/CLAUDE.md`.

Within-subject neuropythy serialization: `neuropythy_register` for
model M takes the previous model's neuropythy mgz as an extra input
(see `NEURO_PREV_MODEL` in the Snakefile). Required because neuropythy
writes into a shared per-subject freesurfer scratch dir.

## What's done vs the bash submitter

Folded IN (was missing from the bash submitter):

- **`r2_mixture` after m4 merge.** The canonical AF pipeline silently
  required `sub-XX_desc-p_signal.nii.gz` to exist before running, but
  the bash submitter didn't include `run_r2_mixture.sh` in its chain.
  Snakemake now has it as an explicit edge. See
  `~/.claude/projects/-Users-gdehol-git-retsupp/memory/feedback_af_needs_r2_mixture.md`.

Preserved:

- Per-subject DAG isolation (no phase-wide `afterok` across subjects).
- Per-model walltime in `CHUNK_RUNTIME` matches the bash submitter's
  `T_CHUNK_M{N}` after the 2026-05-13 bumps (m1: 20, m2/m3: 50, m4: 60,
  m5/m6: 90 min).
- `--mem=32G` on chunk jobs, `--cpus-per-task=16` on neuropythy (JVM
  scales with threads), `--partition=lowprio` everywhere.
- GPU constraint = `L4` only (cuInit-race-proof, 1 GPU/node).
- cuInit warm-up flock on `/dev/shm/cuinit_warm_$(hostname -s).flock`
  inside `prf_chunks` (belt-and-suspenders given L4 constraint).
- `--account=zne.uzh` everywhere (legacy `hare.econ.uzh` in some SLURM
  scripts is bypassed — Snakemake uses the profile default).
- Sentinel files match the canonical derivatives layout
  (`derivatives/prf/modelN/sub-XX/sub-XX_desc-r2.nii.gz`,
  `derivatives/{af_subdir}/sub-XX/sub-XX_roi-{roi}_mode-signed_...tsv`,
  etc.), so the Snakemake outputs are drop-in interchangeable with
  bash-submitter outputs. Resuming a partial bash-submitter run from
  Snakemake works.

Not yet done (relative to the bash submitter):

- No `NICE_M{N}` deprioritization knobs. The bash submitter supports
  `NICE_M3=10000` etc. to deprioritize non-critical-path models;
  Snakemake doesn't expose `--nice` per rule. Workaround: run two
  separate Snakemake invocations with different `models:` lists.
- No `CANARY=1` mode (submit chunk #1 alone, then 2..N gated on its
  success). With Snakemake, a failed chunk fails its own rule and the
  rest of that subject's chain stops — no slot-burning across all 10
  chunks. The per-chunk `--restart-times: 1` in the profile gives one
  retry per chunk before giving up.
- AF subdir name is hardcoded in `config.yaml` to
  `af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_pSig0.5`. To run
  the `_allSharedSigma` variant: change `af_output_subdir` and add
  `--all-shared-sigma` to the `af_fit` rule's shell block.

## One-time install (cluster)

The Snakemake driver runs in a dedicated env so it can be on snakemake>=8
without disturbing `retsupp_cuda` (TF 2.14 / CUDA 11.8).

```bash
ssh sciencecluster
~/data/miniforge3/bin/mamba create -y -n retsupp_snake \
    -c conda-forge -c bioconda \
    "snakemake>=8" "snakemake-executor-plugin-slurm"
```

The driver process runs on the cluster login node; SLURM jobs activate
`retsupp_cuda` / `retsupp_neuropythy` for the actual Python work.

## How to dispatch

From the cluster login node, inside `~/git/retsupp` (under `tmux` so
the driver survives disconnect):

```bash
tmux new-session -d -s prf_snake "cd ~/git/retsupp && \
    ~/data/conda/envs/retsupp_snake/bin/snakemake \
        --snakefile retsupp/snakemake/Snakefile \
        --workflow-profile retsupp/snakemake/profile \
        --configfile retsupp/snakemake/config.yaml \
        2>&1 | tee retsupp/snakemake/logs/run.log"

# Peek every minute or so:
tmux capture-pane -t prf_snake -p | tail -30
```

To run a subset of subjects, override `subjects` on the CLI:

```bash
~/data/conda/envs/retsupp_snake/bin/snakemake \
    --snakefile retsupp/snakemake/Snakefile \
    --workflow-profile retsupp/snakemake/profile \
    --configfile retsupp/snakemake/config.yaml \
    --config 'subjects=[3,5,8,10]'
```

To skip AF and just run PRFs:

```bash
... --config 'with_af=false with_r2_mixture=false'
```

## Dry runs (no sbatch)

On the cluster (preferred — same filesystem as targets so existence
checks resolve correctly):

```bash
ssh sciencecluster '
cd ~/git/retsupp && \
~/data/conda/envs/retsupp_snake/bin/snakemake \
    --snakefile retsupp/snakemake/Snakefile \
    --workflow-profile retsupp/snakemake/profile \
    --configfile retsupp/snakemake/config.yaml \
    -n --quiet'
```

Locally on macOS (DAG construction only — file-existence checks may
give false negatives because `/shares/...` isn't mounted):

```bash
~/mambaforge/envs/retsupp/bin/snakemake \
    --snakefile retsupp/snakemake/Snakefile \
    --configfile retsupp/snakemake/config.yaml -n
```

## Known caveats

- **SLURM job names are UUIDs.** The SLURM executor plugin sets
  `--job-name <run-uuid>` itself (see
  `notes/snakemake_demo/README.md`); `squeue` shows opaque UUIDs
  instead of human-readable names. Find your driver's jobs via
  `squeue --me -O jobid,name | grep <first-8-of-run-uuid>`. The
  Snakemake driver prints its run UUID at startup.
- **Plugin sleeps 5s between sbatch calls** (hardcoded in
  `snakemake-executor-plugin-slurm/__init__.py`). With 1800 chunk
  tasks across a fresh run, dispatch will pace at ~12/min — not the
  bottleneck, but worth knowing.
- **`slurm_extra` cannot contain `--gres`, `--constraint`, `--mem`,
  etc.** They are owned by the plugin's first-class resource keys
  (`gres=`, `constraint=`, `mem_mb=`). Putting them in `slurm_extra`
  raises `WorkflowError` swallowed by the thread pool — symptom is
  "Submitting N ready jobs" followed by silent inaction.
- **No `--nice` per rule.** If lowprio is saturated and you want to
  deprioritize m3/m5/m6, the cleanest workaround is to run two
  Snakemake invocations with disjoint `models:` lists and submit them
  with `nice` at the shell level (`nice -n 10 snakemake ...`) — but
  that only deprioritizes the driver, not the sbatch'd jobs.
- **Re-running after a manual file deletion.** If you `rm` a merged
  NIfTI to force a refit, Snakemake will refit. If you `rm` chunks
  but the merged NIfTI is still on disk, Snakemake will see the
  merged target as up-to-date and not refit. Use
  `snakemake -F <target>` (capital F) to force recompute that target
  and everything downstream regardless of mtime.
- **Surface gii sentinel is hemi-L only.** The Python sampler writes
  both hemis + fsaverage transforms atomically; if it succeeds, all
  outputs exist. But Snakemake's "did this rule's outputs all exist
  on rerun" check only inspects the declared `output:` list, so a
  manual delete of just the hemi-R gii won't trigger a rerun. If
  necessary, expand the `output:` list to include the right-hemi /
  fsaverage variants.
- **r2_mixture script's hardcoded account is `hare.econ.uzh`.** We
  invoke `run_r2_mixture_all.py` directly inside the Snakemake rule
  (not via the SLURM wrapper script), so the rule's profile-supplied
  `slurm_account=zne.uzh` takes precedence. If you ever submit the
  wrapper script directly, expect a fairshare hit on the legacy
  account.

## Local install (for dry-runs / development)

```bash
~/mambaforge/bin/mamba install -n retsupp -c bioconda -c conda-forge \
    snakemake snakemake-executor-plugin-slurm
```

Then `~/mambaforge/envs/retsupp/bin/snakemake ...` for local dry-runs.
