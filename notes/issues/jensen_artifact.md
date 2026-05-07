# Distance-binning Jensen artifact

## The problem

We bin voxels by `d_base = |base_PRF − HP|`. Voxels in the small-d
bin are EITHER (a) truly near HP, OR (b) noise pushed the base estimate
toward HP. For case (b), the conditionwise PRF estimate (independent
noise) regresses back toward truth → looks like a "shift away from HP"
even with zero real attentional effect.

So at small d, the empirical away-from-HP projection is **inflated**
by the noise + Jensen bias. The model's predicted shift has no such
inflation (predictions are deterministic given the fitted base
position).

Net consequence: model and data **shapes** can match cleanly, but
**magnitudes** systematically diverge — empirical bigger near HP.

## What we currently do about it

In the talk PDF slide (g), we **rescale the predicted curve to match
the observed peak amplitude per ROI** before the shape comparison.
Per-ROI Spearman r between bin-medians (predicted vs observed) is
annotated as the test.

## Proposed fixes (ordered by effort)

1. **Empirical null subtraction** (~30 min): for each voxel, draw N
   "no-effect" conditionwise samples = `base_xy + ε` with σ_ε estimated
   from cross-condition variance per voxel. Compute null
   distance-projection curve. Subtract from observed. Result =
   Jensen-corrected curve.

2. **Split-half reanalysis**: refit the BASE PRF on half the runs,
   refit conditionwise on the OTHER half. Independent noise on the two
   sides → no Jensen bias in the resulting shift estimate. Cost: half
   the data, so larger SE.

3. **BOLD-level comparison instead of post-hoc shifts**
   (`plot_voxel_timeseries.py`): compare observed and AF-fit BOLD
   directly. The model was fit at this level; no post-hoc PRF-shift
   computation, no Jensen. **This is the cleanest test**, currently
   blocked only by needing to be run on cluster (TF/ml_dtypes mismatch
   was fixed locally).

4. **Probabilistic PRF refit**: marginalize over per-voxel position
   uncertainty; far too heavy for now.

## Decision

For the meeting we presented the rescaled-shape comparison
(slide g) and explained the Jensen caveat in slide (a). Going forward,
the BOLD-level voxel test (option 3) is the clean answer — should be
the headline qualitative-match figure.
