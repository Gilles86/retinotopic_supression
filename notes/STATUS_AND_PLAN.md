# retsupp — status and plan for tomorrow

Updated: 2026-05-07 (end of evening)

## What we're trying to show

The brain's visuospatial PRFs shift **away from the high-probability
distractor location** during visual search, with the magnitude growing
**up the visual hierarchy**. We model this with a joint
attention-field × population-receptive-field model that fits BOLD
directly and lets us infer per-(subject, ROI) AF parameters
(σ_AF, g_HP, g_LP, optionally σ_dyn / g_HP_dyn / g_LP_dyn).

## State of the model fits (cluster jobs in flight or complete)

| Voxel kernel | AF type | Status | Output dir |
|---|---|---|---|
| Gaussian | static (sustained only) | ✓ done (240/240) | `af_prf_joint_full/` |
| Gaussian | dyn-v1 (σ_dyn, single g_dyn) | ✓ done | `af_prf_joint_dynamic/` |
| Gaussian | dyn-v2 (shared σ, split HP/LP) | ✓ done | `af_prf_joint_dynamic_v2/` |
| Gaussian | dyn-v3 (separate σ_dyn, split HP/LP) | running (2760816) | `af_prf_joint_dynamic_v3/` |
| DoG | static | running (2760637) | `af_prf_joint_full_dog/` |
| DoG | dyn-v2 | running (2761298) | `af_prf_joint_dynamic_v2_dog/` |
| DoG | dyn-v3 | running (2761299) | `af_prf_joint_dynamic_v3_dog/` |
| Gaussian conditionwise (model 1, ROI-restricted) | – | ✓ done | `prf_conditionfit/model1/` |
| Baseline + conditionwise PRFs with full paradigm (distractors in design) | – | test job 2762779 staged | `prf_distractors/` and `prf_conditionfit_distractors/` |

## Headline findings (Gaussian static, 30 subjects, signed mode)

- **Sustained HP-suppression** (Wilcoxon `g_HP < g_LP`, paired, alt=less):
  - V2: p = 0.0002, 80% of 30 subjects negative
  - V3: p = 0.0002, 80%
  - V3AB: p = 0.05
  - VO: p = 0.003, 80%
  - others n.s.
- **Hierarchy effect**: |g_HP − g_LP| and predicted-shift magnitude both
  grow up the visual stream (Spearman r ≈ +0.24, p = 2×10⁻⁴ on
  parameter; r ≈ +0.17 on observed shift magnitude).
- **Dynamic gains** (v2, σ < 7° subset): early visual cortex (V2/V3)
  shows phasic SUPPRESSION on distractor presentation; V3AB/VO show
  trending capture.
- **Behavioral correlations** (PCA × RT contrasts): targeted log-ratio
  × RT(HP)−RT(LP) test is null. PCA-of-g_HP × RT(HP)−RT(no_dist)
  reaches r = +0.37, p = 0.046.

## What the slides currently show (`notes/talk_docky_jan.pdf`)

Sections:
- (a) Jensen / noise inflates distance-binned shift estimates (cartoon).
- (b) The AF model formula + parameter sweeps (modulation field heatmaps + predicted vector fields, including positive-gain attraction examples).
- (c) Per-ROI parameter distributions (mean ± SEM, t-test annotations) and the HP-vs-LP test on a scale-invariant log-ratio (% ticks).
- (d) Per-ROI predicted (model) vs observed (data, mean-of-medians per quadrant in canonical rotated frame) vector fields. **Auto-scaled per panel** so PRED and OBS are visually comparable in shape; magnitude annotated as "×N".
- (d″) Hierarchy bars (|g_HP−g_LP|, |predicted shift|, |observed shift| vs ROI).
- (d′) Per-subject example pages (top-3 by predicted-vs-observed r).
- (e) Dynamic AF intro + raw-gain test (capture vs suppression per ROI) + sustained / dynamic / total HP-LP test.
- (f) Behavior correlations (per-ROI Spearman + PCA × RT contrasts).
- (g) Qualitative SHAPE-MATCH test: per-ROI projection-vs-distance curves with the predicted curve rescaled to observed peak; Spearman r between bin-medians annotated.

A second clean version `notes/talk_docky_jan_email.pdf` drops the
messy per-ROI vector fields (sections d and d′) for the email.

## The story is missing one thing

> **Showing that the data ACTUALLY follows the model.**

The qualitative SHAPE match (slide g) is suggestive but not airtight,
because magnitudes diverge (Jensen-style noise inflates the empirical
curve).

## Tomorrow — clear-cut story candidates

1. **First-vs-last runs within HP-block** *(NEW idea, agent in flight)*: refit AF
   model on the first runs of each HP-block AND on the last runs. If
   the HP-suppression gain INCREASES from first → last, that's a learning
   signature that's basically impossible to explain with Jensen.
   Tests whether subjects are still LEARNING the priority map.

2. **Voxel-level BOLD comparison** (`plot_voxel_timeseries.py`): pick voxels
   with high R²(AF) AND large ΔR² between AF and no-AF baseline; plot
   per-condition averaged BOLD with AF-fit and no-AF predictions
   overlaid. The voxels where AF-fit tracks observed but no-AF
   doesn't = direct evidence of HP-specific BOLD modulation that no
   per-voxel-PRF baseline can explain.
   **Status**: working, but local TF/ml_dtypes mismatch caused a
   crash; needs to be run on cluster (retsupp_cuda env is fine there).

3. **Apples-to-apples comparison with model-1 conditionwise**: the
   `predict_shifts_gauss.tsv` is now in place (matched single-Gaussian
   kernel). Re-run the proj-vs-distance shape match on that.

4. **DoG-AF results** (cluster running): once landed, compare DoG-AF
   sustained vs Gaussian-AF — does the DoG kernel match resolve
   the kernel-mismatch component of the magnitude gap?

5. **Distractor-in-paradigm baseline + conditionwise PRFs**: cluster
   test job 2762779 staged. Once those land, redo the entire
   downstream analysis pipeline with PRFs that model distractor
   BOLD properly.

6. **IPS / FEF AF fits**: Wang surface labels are computed for all
   30 subjects but the `cortex_to_image` projection is broken
   (used template intensities as fill). Try `mri_surf2vol` instead
   tomorrow morning; then add IPS/FEF to ROI list and re-fit.

## Priority for the meeting

If we have 2 hours tomorrow:
- 30 min: get the IPS labels into volume space cleanly.
- 30 min: launch first-vs-last-run AF refit.
- 30 min: rebuild slide deck with whatever lands first.
- 30 min: pick the cleanest qualitative pattern figure (probably the
  voxel-level BOLD comparison) for the talk.
