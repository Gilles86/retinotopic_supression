# Cross-validated R² — model comparison plan

**Status**: NOT YET IMPLEMENTED — proposal for the cleanest test
of "does the AF model explain the data better than a null
no-attention PRF model?"

## Why

Every test we've shown so far (HP-vs-LP gain test, σ_dyn < σ_AF
finding, hierarchy effect, projection-vs-distance shape match) is
**within-fit** — they test whether the fitted parameters are
non-trivially structured. They do not directly answer:

> "Does adding the AF terms make the model PREDICT held-out data
>  better than a single-Gaussian PRF that ignores HP?"

Cross-validated R² answers exactly this.

## Design — leave-one-CONDITION-out (preferred)

Per (subject, ROI):

1. Pick one of the 4 HP conditions as the held-out fold.
2. Fit each model on the BOLD from the other 3 conditions:
   - **M0 null**: single Gaussian PRF per voxel (model 1, no AF).
   - **M1 AF static** (`AttentionFieldPRF2D`).
   - **M2 AF dyn-v3** (Gaussian, separate σ_dyn, split gains).
   - **M3 DoG-AF dyn-v3** (same with DoG voxel kernel).
3. Predict BOLD on the held-out condition's runs using the fitted
   parameters and that condition's paradigm.
4. Compute per-voxel CV-R² on held-out BOLD.
5. Repeat for all 4 folds. Average.
6. ΔCV-R² = R²(M_k) − R²(M0) per voxel.

Aggregate across voxels per ROI → answer "does AF help in this ROI?"

## Design — leave-one-RUN-out (alternative)

12 runs per subject. Leave run `k` out, fit on the 11 others,
predict run `k`. Higher fold count, tighter CV estimates, but
muddles up "AF explanatory power" with "general PRF explanatory
power" because conditions are still represented in training even
when run `k` is held out.

The leave-one-CONDITION-out design isolates the AF contribution
much more cleanly.

## Implementation outline

Extend `fit_af_prf_braincoder.py` (and the v3 / DoG variants)
with a `--cv-fold {0..3}` arg:

```python
hp_to_idx = {'upper_right': 0, 'upper_left': 1,
             'lower_left': 2, 'lower_right': 3}

if args.cv_fold is not None:
    # exclude runs whose HP matches the held-out fold from training
    held_out_hp = list(hp_to_idx)[args.cv_fold]
    # train_runs = [r for r in all_runs if hp_per_run[r] != held_out_hp]
    # held_out_runs = [r for r in all_runs if hp_per_run[r] == held_out_hp]
```

After fitting on the train set, run the forward pass on the
held-out condition's paradigm and BOLD, compute CV-R² per voxel,
save.

For M0 (null), use the existing `fit_prf.py`-style script restricted
to train runs, then forward-predict on held-out runs.

## Cost

- 4 folds × 4 models × 30 subjects × 8 ROIs = 3840 SLURM jobs.
- L4 GPU, ~45 min/job → roughly 30 hours of compute on a free L4
  pool. Realistic in a day on the cluster with the existing
  retsupp_cuda env.

## Proposed reporting

Per ROI:
- Bar plot of mean CV-R² per model (M0, M1, M2, M3), 30 subjects, ±SEM.
- Per-voxel ΔCV-R² histogram (M2 − M0) per ROI; show that the
  distribution shifts positive in the ROIs where the AF effect
  is real.
- Single Wilcoxon paired test per ROI: ΔCV-R² > 0?

This is THE figure for the talk's "does the model actually help?"
question.

## See also

- [issues/jensen_artifact.md](../issues/jensen_artifact.md) — why
  current within-fit tests have a magnitude-confound (CV-R² is
  immune).
- [models/af_dynamic_v3_gauss.md](../models/af_dynamic_v3_gauss.md)
  and `_dog.md` — the candidate models.
