# Cross-validated R² — model comparison plan

**Status**: design finalized 2026-05-07. Submission pending. Sign-constraint
+ fold-loop + CPU SLURM changes to staged scripts in progress.

## Why

Every test shown so far (HP-vs-LP gain, σ_dyn < σ_AF, projection-vs-distance
shape match) is **within-fit**: it tests whether the fitted shared
parameters are non-trivially structured. None of them directly answer:

> "Does adding the AF terms predict held-out data better than a PRF
>  model that ignores HP, AND which sign pattern of HP/LP gains
>  generalizes per ROI?"

Cross-validated R² with a sign-constrained joint factorial answers both.

## Folding scheme

**Leave-one-CONDITION-out**, 4 folds per subject. Each fold = one of the
4 HP locations held out:

| fold | held-out HP | train | held-out |
|---|---|---|---|
| 0 | upper_right | UL + LL + LR | UR |
| 1 | upper_left  | UR + LL + LR | UL |
| 2 | lower_left  | UR + UL + LR | LL |
| 3 | lower_right | UR + UL + LL | LR |

Standard 12-run subject: 9 train runs / 3 held-out per fold.
Sub-20 ses-1, sub-24 ses-2 (5 runs in one session, 11 total): one fold
gets 2 train or 2 held-out runs — keep, just slightly underpowered.
Sub-1, sub-2 (buggy `AA BBB CCC DDD A` order): condition counts still 3-3-3-3,
fold structure unchanged.

Why not leave-one-RUN-out: it muddles "AF explanatory power" with
"general PRF explanatory power" because all 4 conditions remain in
training. Leave-one-CONDITION-out isolates the AF contribution.

## Voxel kernel

**DoG, always.** The Gaussian-kernel "phasic capture" finding in higher
visual cortex was traced to surround-suppression leaking into AF gains
(see `models/af_dynamic_v3_dog.md`). DoG is the right voxel kernel for
hypothesis tests about AF gain signs.

## Single model class throughout: DoG dyn-v3 (hybrid)

`DoGDynamicAttentionFieldPRF2DWithHRF_v3`. Always 4 gain parameters:
- Sustained: `g_HP`, `g_LP`
- Dynamic: `g_HP_dyn`, `g_LP_dyn`

`σ_AF`, `σ_dyn`, and the per-voxel DoG params (x, y, sd_centre,
sd_surround, baseline, amplitude_centre, amplitude_surround) are always
free.

Using the hybrid throughout (instead of static-only for sustained tests)
keeps CV-R²s on the same scale across model classes.

## 17-class joint factorial

Each gain pair takes one of **4 menu values**:

| value | parameterization | meaning |
|---|---|---|
| `(0, 0)` | both fixed at 0 | no AF on this axis |
| `(−, 0)` | HP = −softplus(raw); LP = 0 | HP-only suppress |
| `(−, −)` | both = −softplus(raw) | general suppress |
| `(+, +)` | both = +softplus(raw) | general capture (naive bottom-up) |

Cross sustained × dynamic = **16 cells**:

|  | dyn=(0,0) | dyn=(−,0) | dyn=(−,−) | dyn=(+,+) |
|---|---|---|---|---|
| **sus=(0,0)** | **null** (no AF at all) | dyn HP-only suppress | dyn general suppress | dyn capture |
| **sus=(−,0)** | sus HP-only | sus HP-only + dyn HP-only | sus HP-only + dyn general | sus suppress + dyn capture |
| **sus=(−,−)** | sus general | sus general + dyn HP-only | both general suppress | sus general + dyn capture |
| **sus=(+,+)** | sus capture | sus capture + dyn HP-only | sus capture + dyn suppress | **naive both capture** |

Plus **1 signed-unconstrained control class** — all 4 gains are plain
(unsigned) TF Variables. This is the sanity-check baseline: if a ROI's
best constrained class falls well below the unconstrained class on
CV-R², something outside our menu is preferred and we should expand the
menu. If they're tied or constrained wins, the constraints are buying
generalization.

→ **17 classes**.

## What we exclude and why

- **(+, 0)**, **(0, +)**, **(0, −)** — pure HP-only-capture or LP-only
  patterns. Never observed in unconstrained DoG fits. The signed control
  class would catch a violator.
- **(+, −)**, **(−, +)** — opposite-sign cells. Theoretically these
  represent counter-balancing (proactive HP suppression + reactive LP
  capture, à la Wang & Theeuwes), but the unconstrained DoG dyn-v3
  fits show same-sign gains in every ROI. The signed control catches a
  violator.

## Job structure: one job = (subject, ROI, model_class), fold loop inside

ROI loading dominates per-job time. Per array task:

1. Load BOLD + paradigm + ROI mask once.
2. For each fold ∈ {0, 1, 2, 3}:
   a. Restrict BOLD to train runs for this fold.
   b. Fit DoG-dyn-v3 with the class's sign constraints applied to the 4 gains.
   c. Forward-predict BOLD on held-out runs using fitted params + the
      held-out condition's paradigm + condition_indicator + dynamic_indicator.
   d. CV-R² per voxel = `1 − Σ(y − ŷ)² / Σ(y − ȳ_held)²`.
3. Save one pickle with all 4 folds' results:
   ```python
   {
       'class_label': (sus_hp, sus_lp, dyn_hp, dyn_lp),
       'shared_pars_per_fold': [4 dicts],
       'cv_r2_per_fold': [4 arrays of length n_voxels],
       'train_r2_per_fold': [4 arrays],
       'voxel_pars_per_fold': [4 DataFrames],
   }
   ```
   to `derivatives/af_prf_cv_factorial/<class_label>/sub-XX/sub-XX_roi-{ROI}_cv-fits.pkl`.

## Cost

| | jobs |
|---|---|
| 16 factorial cells × 30 subj × 8 ROI | 3840 |
| 1 signed control × 30 × 8 | 240 |
| **Total** | **4080** |

17 sbatch calls of `--array=1-240`.

## SLURM partition: CPU, not GPU

Per-ROI fits are small-tensor work, ROI loading dominates anyway. CPU
pool absorbs ~100–300 simultaneous jobs vs L4 pool's ~8–16, so wall-
time-to-completion is much shorter on CPU even though each fit is 3–5×
slower per-job.

```bash
#SBATCH --account=zne.uzh
#SBATCH --partition=generic
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
# no --gres, no --constraint=L4
```

Use `retsupp_cpu` env, not `retsupp_cuda`.

Calendar: 4080 jobs / ~200 simultaneous CPU slots × ~50 min ≈ **17 hours
wall-time** (≈ 1 day on the cluster).

## Reporting

**Separate from the talk PDF**, per the figure-organization preference.

Per (subject, ROI) we get 17 CV-R² values (mean across 4 folds). Per ROI
across subjects:

- `notes/figures/cv_winning_class_per_roi.pdf` — heatmap of the
  per-(subject, ROI) winning class. The headline figure: which gain
  pattern dominates each ROI.
- `notes/figures/cv_class_means_per_roi.pdf` — bar chart per ROI
  showing mean CV-R² across subjects for each of the 17 classes,
  with null highlighted. Sorted by class.
- `notes/figures/cv_delta_vs_null.pdf` — per-voxel ΔCV-R² (best
  constrained class − null) histogram per ROI. Wilcoxon paired test.
- `notes/figures/cv_signed_vs_constrained.pdf` — per-ROI scatter of
  signed-control CV-R² vs best-constrained CV-R². If signed wins
  systematically, the menu is too narrow.

## Implementation changes vs staged scripts

The scripts staged earlier (`fit_af_prf_cv.py`, `fit_prf_cv_null.py`,
`cv_helpers.py`, `slurm_jobs/fit_af_prf_cv.sh`) implemented a 4-model,
one-fold-per-job, GPU-on-L4 design. They need:

1. **Drop `--model` arg** — model is always DoG-dyn-v3.
2. **Add `--sus-hp-sign`, `--sus-lp-sign`, `--dyn-hp-sign`,
   `--dyn-lp-sign`**, each ∈ `{plus, zero, minus, free}`. Apply softplus
   / negative-softplus / fixed-zero / unconstrained per arg.
3. **Drop `--cv-fold` arg** — loop all 4 folds inside the job.
4. **Save one pickle per (subject, ROI)** containing all 4 folds' results.
5. **Drop `fit_prf_cv_null.py`** — null is the (zero, zero, zero, zero)
   cell of the factorial.
6. **Switch SLURM submitter to CPU partition**, drop GPU directives,
   `retsupp_cpu` env, 2h walltime.
7. **Add `submit_all_cv.sh`** that submits all 17 sbatch arrays.

## See also

- `issues/jensen_artifact.md` — why current within-fit tests have a
  magnitude confound (CV-R² is immune).
- `models/af_dynamic_v3_dog.md` — DoG dyn-v3 results that motivate the
  factorial menu.
- `models/af_dynamic_v3_gauss.md` — Gaussian-kernel results that
  motivate using DoG always.
