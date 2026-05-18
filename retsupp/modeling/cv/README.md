# Cross-validated PRF fits — model comparison

Separate from the canonical fitting pipeline (`retsupp/modeling/fit_prf.py`).
Post-hoc 3-fold CV to compare **m0 .. m6** on a parameter-penalty-free score,
instead of training-set R² (which always rewards the most flexible
model).

**m0** is the trivial null baseline: predict each test timepoint as
the per-voxel **training mean**. Test R² captures whether *any*
non-constant structure has been learned. A voxel where any m1..m6
beats m0 has signal worth modelling; a voxel where m0 wins is a
null voxel (no PRF, or test/train drift dominates).

## What's in here

| File | Purpose |
|------|---------|
| `fit_prf_cv.py` | Per-(subject, model, fold) fitter. Loads training BOLD/paradigm for a subset of runs, fits the same `MODEL_CFG` schedule as the canonical pipeline, then predicts the held-out runs' BOLD and computes per-voxel **test R²**. |
| `merge_prf_cv_chunks.py` | Merge per-chunk NPZ → `sub-XX_fold-K_desc-r2_test.nii.gz`. |
| `slurm_jobs/fit_prf_cv_l4_chunked.sh` | GPU L4 SLURM wrapper for chunked fits. Mirrors `fit_prf_l4_chunked.sh` (lowprio, 32G, cuInit flock). |
| `slurm_jobs/merge_prf_cv_chunks.sh` | SLURM wrapper for the merge step. |
| `slurm_jobs/submit_cv_persub.sh` | Per-(sub, model, fold) chain submitter. One independent chain per tuple — failure in one doesn't block any other. |

## Fold definition

3-fold, balanced across HP-distractor conditions: each fold holds out
the K-th run of each condition (so each fold drops ~4 runs, training
on the remaining ~8). Assignment lives in `make_fold_assignments` in
`fit_prf_cv.py`:

```python
groups = sub.get_runs_by_hp()        # dict[hp_label] → [(ses, run), ...]
folds = [[], [], []]
for label in sorted(groups):
    for i, run in enumerate(sorted(groups[label])):
        folds[i % 3].append(run)
```

For typical subjects (12 runs, 4 conditions × 3 runs): each fold gets
exactly 4 runs held out, 8 in training. For the few subjects with 11
runs (sub-20 ses-1, sub-24 ses-2): one condition contributes only 2
runs, so fold 2 has 3 held out instead of 4. Negligible asymmetry.

## Warm-start

Models m2 .. m6 warm-start from the **canonical mean-fit** parameters
of their `init_from` parent (m1 → m2/m3, m2 → m4/m5, m5 → m6) —
same as the canonical pipeline. This means the warm-start is fit on
ALL 12 runs (not just the training subset), which is a slight optimism
for the test score. The alternative — refitting m1 per fold first —
would triple the CV cost and was deemed not worth it. Same compromise
as the runwise / conditionwise canonical fits.

The test score is still meaningful: model **parameters** are refined
on the training subset; only the **initial seed** comes from the
mean fit. Cross-model differences should be preserved.

## Output layout

```
derivatives/prf_cv/
└── model{N}/                       N ∈ {1, ..., 6}
    └── fold-{K}/                   K ∈ {0, 1, 2}
        └── sub-XX/
            └── sub-XX_fold-{K}_desc-r2_test.nii.gz
```

Per-voxel test R², clipped at -1 (anything below means "predicted
worse than mean" — capped for NIfTI dtype + plot sanity). Compute
aggregate per-(sub, model) cv R² as the mean across folds, voxel by
voxel.

## Dispatch (cluster)

```bash
ssh sciencecluster
cd ~/git/retsupp
git pull --rebase

# Dry-run first to confirm what would be submitted:
DRY_RUN=1 bash retsupp/modeling/cv/slurm_jobs/submit_cv_persub.sh | head

# All 30 subs × 6 models × 3 folds = 540 (sub, model, fold) chains:
bash retsupp/modeling/cv/slurm_jobs/submit_cv_persub.sh

# Subset: just m1 and m4 for sub-3 and sub-5:
SUBJECTS="3 5" MODELS="1 4" \
    bash retsupp/modeling/cv/slurm_jobs/submit_cv_persub.sh

# Skip already-done blocks (default): if a run was interrupted, just
# rerun and it'll skip what already has r2_test.nii.gz on disk.
```

Each chain is `chunks (N_CHUNKS array tasks, GPU L4) → merge (CPU)`,
walltime tuned tight (25 min chunks + 5 min merge). Per-chunk runtime
on L4 is similar to the canonical fits (~10-20 min depending on
model).

Total resource estimate, full sweep:
- 540 (sub, model, fold) tuples × 10 chunks = 5400 GPU jobs × ~15 min
- 540 merge jobs × ~2 min
- ≈ 1400 GPU-hours total, dispatched at lowprio jobs:150 → ~10 days
  wall time. Cheaper if SUBJECTS / MODELS subset reduces scope.

## Aggregation (TODO — local, after fits land)

Not yet implemented. Plan:

- `aggregate_cv.py` — per (sub, model, fold), load `r2_test.nii.gz`,
  mask by neuropythy ROIs, compute median R² per ROI. Output TSV
  with columns `subject, model, fold, roi, median_r2_test,
  n_voxels`. ~5 minutes to run locally once data is rsync'd back.
- `plot_cv_comparison.py` — figure: ROIs on x-axis, model on hue,
  median cv R² ± SEM across subjects on y. Should make the "which
  model wins" claim visible at a glance. Use the scientific-figures
  skill aesthetic.

## Snakemake integration (TODO)

Not folded into `retsupp/snakemake/Snakefile` (kept separate as
requested). When ready, the obvious move is a new Snakefile at
`retsupp/snakemake/Snakefile.cv` with the same SLURM profile, three
rules (`prf_cv_chunks`, `prf_cv_merge`, `cv_aggregate`), and the
fold + base_model + subject as wildcards.
