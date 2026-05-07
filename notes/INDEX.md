# notes/ — index

Top-level entry point. Each file below is a focused document on one
analysis / model / issue. Update them in place; don't pile everything
into one log.

## Status snapshots

- [STATUS.md](STATUS.md) — current state of the cluster jobs and
  pipeline, top-level overview.

## Analyses

- [analyses/hp_vs_lp_test.md](analyses/hp_vs_lp_test.md) — the headline
  sustained-HP-suppression statistical result.
- [analyses/sigma_comparison.md](analyses/sigma_comparison.md) — σ_AF
  (sustained) vs σ_dyn (phasic) per ROI from the v3 dynamic fits.
- [analyses/hierarchy_effect.md](analyses/hierarchy_effect.md) —
  shifts grow up the visual stream.
- [analyses/behavior_correlation.md](analyses/behavior_correlation.md) —
  correlations of model parameters with behavioral RT contrasts.
- [analyses/voxel_timeseries.md](analyses/voxel_timeseries.md) — direct
  observed-vs-AF-fit BOLD comparison per voxel.
- [analyses/run_position_learning.md](analyses/run_position_learning.md) —
  first/last-run AF refit; learning over the HP block.
- [analyses/glmsingle.md](analyses/glmsingle.md) — single-trial
  GLMsingle betas, Richter-style suppression.

## Models

- [models/af_static_gauss.md](models/af_static_gauss.md)
- [models/af_dynamic_v2_gauss.md](models/af_dynamic_v2_gauss.md)
- [models/af_dynamic_v3_gauss.md](models/af_dynamic_v3_gauss.md)
- [models/af_static_dog.md](models/af_static_dog.md)
- [models/af_dynamic_v2_dog.md](models/af_dynamic_v2_dog.md)
- [models/af_dynamic_v3_dog.md](models/af_dynamic_v3_dog.md)

## Issues / caveats

- [issues/jensen_artifact.md](issues/jensen_artifact.md) — distance-binning
  noise bias.
- [issues/identifiability.md](issues/identifiability.md) — σ × |gain|
  trade-off in the AF fits.
- [issues/kernel_mismatch.md](issues/kernel_mismatch.md) — Gaussian-AF
  vs DoG-conditionwise PRF apples-to-oranges.

## Meetings

- [meetings/2026-05-07_docky_jan.md](meetings/2026-05-07_docky_jan.md) —
  in-person meeting summary + decisions.

## Reference / archive

Historical documents (kept for context, won't be rewritten):
- VSS2026.pdf — Dock's draft talk slides.
- sumiya_thesis_*.pdf — Sumiya thesis (chapter 2 = AF+ offset).
- richter2025.pdf, tuncok2025.pdf — comparison literature.
- talk_docky_jan.pdf / talk_docky_jan_email.pdf — the live talk deck.
- The `predict_shifts*.tsv`, `af_*_parameters.tsv`, etc. — generated
  by the visualize/ pipelines, regenerable.
