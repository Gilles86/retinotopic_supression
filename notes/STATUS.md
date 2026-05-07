# Status — current state of the pipeline

**Last update**: 2026-05-07 (post-meeting with Docky and Jan).

## Cluster job status (model-fitting)

| Voxel kernel | AF type | Status | Output dir |
|---|---|---|---|
| Gaussian | static | ✓ done (240/240) | `af_prf_joint_full/` |
| Gaussian | dyn-v1 (σ_dyn, 1 g_dyn) | ✓ done | `af_prf_joint_dynamic/` |
| Gaussian | dyn-v2 (shared σ, split g_dyn) | ✓ done | `af_prf_joint_dynamic_v2/` |
| Gaussian | dyn-v3 (separate σ_dyn, split g_dyn) | running (2760816), ~171/240 | `af_prf_joint_dynamic_v3/` |
| DoG | static | running (2760637) | `af_prf_joint_full_dog/` |
| DoG | dyn-v2 | running (2761298) | `af_prf_joint_dynamic_v2_dog/` |
| DoG | dyn-v3 | running (2761299) | `af_prf_joint_dynamic_v3_dog/` |
| Gaussian conditionwise (model 1, ROI-restricted) | – | ✓ done | `prf_conditionfit/model1/` |
| Baseline + conditionwise PRFs with full paradigm | – | test 2762779 PENDING | `prf_distractors/` and `prf_conditionfit_distractors/` |
| Run-slice AF (first/last per HP-block) | – | scripts staged, not submitted | `af_prf_joint_runslice_{first,last}/` |

## Other pipelines

| Pipeline | Status |
|---|---|
| GLMsingle (single-trial betas) | finished, NIfTIs pulled locally |
| Wang 2015 atlas (IPS/SPL1/FEF) | surf → vol projection done for all 30 (still need to verify volume looks right) |
| Hierarchical Bayesian GAM | done, in `notes/gam_clusters.pdf` |

## Live output documents (regenerable)

- `notes/talk_docky_jan.pdf` — full talk deck (the meeting one).
- `notes/talk_docky_jan_email.pdf` — clean email version (no per-ROI vector field grids).
- `notes/af_parameters.pdf` — diagnostic for static-AF parameters.
- `notes/af_v2_diagnostic.pdf` — v2 dynamic AF diagnostic.
- `notes/predict_shifts_cluster.pdf` — predicted vs observed shift visualizations.
- `notes/rotated_shift_heatmaps.pdf` — 2D shift heatmaps in canonical frame.
- `notes/proj_distance_curves.pdf` — projection vs distance curves.

## Headline findings so far (Gaussian static, 30 subjects)

- **Sustained HP-suppression** (Wilcoxon paired one-sided `g_HP < g_LP`):
  V2 p=0.0002 (80% of subjects), V3 p=0.0002 (80%), V3AB p=0.05, VO p=0.003 (80%).
- **Hierarchy effect**: |g_diff| and predicted/observed shift magnitude grow up the stream
  (Spearman r ≈ +0.24, p = 2×10⁻⁴ on parameter; r ≈ +0.17 on shift magnitude).
- **σ_AF vs σ_dyn (v3)**: σ_dyn ≪ σ_AF in V1/V2/V3 — phasic response is spatially
  focal (~0.7°), sustained priority map is broad (~2°). p<0.05 in V1/V2/V3.
- **Dynamic gain in early visual cortex**: opposite direction from sustained
  (g_HP_dyn > g_LP_dyn in V2/V3 → phasic enhancement at HP). Total
  (sustained + dynamic) cancels in V2/V3, adds up in V3AB/VO.

## What's missing for the story

The qualitative pattern test (slide g of the talk PDF) shows model and
data SHAPES match per ROI but magnitudes differ (Jensen + kernel
mismatch). To close this gap we need either:

1. **Voxel-level BOLD comparison** (`plot_voxel_timeseries.py`): compare
   data → AF fit at the BOLD level where the model was fitted. Already
   built; ml_dtypes upgraded so it now runs locally too.
2. **Run-position learning effect**: refit AF on first vs last run of
   each HP-block. If g_HP-suppression strengthens, that's a
   learning signature impossible to attribute to noise.
3. **DoG-AF results** to remove the kernel-mismatch confound.

See `notes/INDEX.md` for individual analysis-specific docs.
