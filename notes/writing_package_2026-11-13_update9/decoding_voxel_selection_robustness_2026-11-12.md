# Fig. 6 decoding: voxel-selection robustness (2026-11-12/13)

**Verdict (decision rule fixed in advance): Fig. 6 keeps its current selection.** Neither looser selection clearly
improves decoding quality or the precision (SE) of the HP effect. The V1–V3 results do not change, and neither do the
per-area V2/V3 effects. The higher-area pooled effects get weaker and noisier at r > 0.2, and are about the same at
r > 0.1.

Figure: `notes/figures/paper_2026-11/figS_decoding_voxel_selection_n28.pdf` (+ `.md` legend with every number).
Table: `notes/data/decoding_voxel_selection_2026-11.tsv`. Code: `retsupp/visualize/paper/figS_decoding_voxel_selection.py`.
n = 28 (sub-03 … sub-30).

## What was run

- Selections, all per CV fold on that fold's **training runs only** (the decoded run never influences its voxels), BOLD
  SD ≤ 20%, cap 2000 voxels per area (highest reliability first), area skipped for a subject with < 20 voxels:
  - **current** (Fig. 6): leave-one-run-out reliability R² > 0 (`prf_cv_roi_reliability`, about r > 0.3).
  - **r > 0.2** and **r > 0.1**: mean over training runs of corr(run, mean of the other training runs) > R and
    circular-shift null p ≤ .01 (100 shifts of 30–228 TRs), the gain-field robustness rule (7bbcebc), here fold-wise:
    `voxel_reliability_null.py --folds` → `derivatives/prf_cv_roi_reliability_null_folds/`.
- Decoder unchanged (`decode_final.py --prf-source jointhrf`, free + outline decoders, nine areas), new option
  `--selection relnull --min-r R`; outputs `derivatives/decoded_final_2026-11/jointhrf_relnull_r0.2|r0.1/`.
- Pooled-area decoders (`--rois mid high dorsal occtemp --out-suffix _pooled`): one decoder on the union of the
  areas' voxels, each voxel with its own fold pRF (and area HRF), the same per-voxel selection, cap 2000 per set.
  mid = V3AB + hV4, high = LO + TO + VO + IPS, dorsal = V3AB + IPS, occtemp = hV4 + VO + LO + TO; current and r > 0.2.
  Outputs `jointhrf_pooled/`, `jointhrf_relnull_r0.2_pooled/`. `fig6_collect.ALL_ROIS` now includes these names (appended,
  so the shuffle seeds of the nine areas are unchanged), so `fig6_decoding.py` can read
  `decoding_readout_jointhrf_pooled_2026-11.tsv` / `..._quality_...` like any other ROI.
- The cap never applied: 0 subject × run × area decodes in any selection, also for the pooled sets.

## Results (mean ± SEM over subjects; p one-sided Wilcoxon unless noted)

| | current | r > 0.2 | r > 0.1 |
|---|---|---|---|
| Voxels V1 / LO / VO (mean per subject) | 367 / 226 / 49 | 420 / 277 / 55 | 604 / 414 / 110 |
| Subjects with VO / IPS / V3AB | 10 / 17 / 23 | 18 / 19 / 26 | 27 / 28 / 28 |
| Quality r, V1–V3 | .449 ± .013 | .451 ± .013 | .450 ± .012 |
| Quality r, LO / hV4 / VO | .244 / .252 / .159 | .252 / .263 / .150 | .256 / .270 / .135 |
| V1–V3 HP − next (Fig. 6d) | −1.58 ± 0.53%, p = .004 | −1.57 ± 0.53%, p = .006 | −1.64 ± 0.52%, p = .003 |
| V1–V3 HP − opposite | −0.90 ± 0.53%, p = .082 | −0.88 ± 0.51%, p = .060 | −0.97 ± 0.52%, p = .045 |
| V1–V3 next − opposite (two-sided) | +0.68 ± 0.32%, p = .070 | +0.68 ± 0.33%, p = .050 | +0.67 ± 0.33%, p = .063 |
| HP − other, all areas (Fig. 6f) | −1.14 ± 0.39%, p = .003 | −0.46 ± 0.64%, p = .041 | −0.85 ± 0.38%, p = .012 |
| Early | −1.35 ± 0.51%, p = .010 | −1.34 ± 0.50%, p = .010 | −1.42 ± 0.49%, p = .006 |
| Dorsal | −0.12 ± 0.75%, p = .56 | +1.43 ± 1.55%, p = .67 | +0.10 ± 0.70%, p = .39 |
| Occipitotemporal | −1.06 ± 0.41%, p = .006 | −0.12 ± 0.87%, p = .052 | −0.89 ± 0.40%, p = .017 |
| Per area, BH q < .05 | V2, V3 (q = .033) | V2, V3 (q = .027) | V2, V3 (q = .017), VO (q = .030) |
| Sweep, HP − next segment (Fig. 6c) | −1.16 ± 0.47%, p = .018 | −1.20 ± 0.49%, p = .015 | −1.27 ± 0.48%, p = .012 |

Decoding quality: the r > 0.2 gain in V1–V3 is +0.002 (paired p = .001), negligible; r > 0.1 +0.001 (p = .90). In the
higher areas quality is about the same per area (slightly higher in LO and hV4, lower in VO and V3AB, where the
looser rule adds subjects with weak data).

**Why r > 0.2 is noisier in the higher areas:** it lets in area × subject combinations at the 20-voxel floor that the
current rule skips. sub-20 under r > 0.2 has V3AB (23 voxels) and TO (21 voxels) with a mean decoded intensity of
about 0.006 (typical 0.03–0.06). The "% of the mean of the four quadrants" normalisation then explodes (HP − other
+33% and +60%). Without sub-20, r > 0.2 gives all areas −0.98 ± 0.40% (p = .014) and occipitotemporal −0.90 ± 0.39%
(p = .019), the same as current. At r > 0.1 these areas have hundreds of voxels and the problem does not arise.

### Pooled-area decoders (one decoder per set; n = 28 except dorsal)

| Set | current: HP − other, p, quality r, voxels | r > 0.2 |
|---|---|---|
| mid (V3AB + hV4) | −0.57 ± 0.70%, p = .24, r = .281, 219 | −0.67 ± 0.64%, p = .21, r = .286, 314 |
| high (LO + TO + VO + IPS) | +0.19 ± 0.93%, p = .46, r = .232, 369 | +0.11 ± 0.77%, p = .53, r = .231, 518 |
| dorsal (V3AB + IPS) | −1.38 ± 0.81%, p = .072, r = .238, 162 (n = 24) | +0.38 ± 1.37%, p = .33, r = .226, 240 (n = 27) |
| occtemp (hV4 + VO + LO + TO) | −0.06 ± 0.74%, p = .28, r = .276, 447 | −0.20 ± 0.60%, p = .34, r = .280, 600 |

Pooling voxels into one decoder gives decoding quality as good as the best single higher area (r ≈ .23–.29). The HP
effect does not show in any pooled set, also not in occipitotemporal, where the mean of the per-area effects is
significant (−1.06%, p = .006). So the pooled-pathway result of Fig. 6f comes from averaging per-area effects and does
not appear when one decoder is fitted on all of the pathway's voxels. This argues against putting the pooled decoders in
the main figure as support for an occipitotemporal effect.

## Decision

Rule fixed in advance: a looser selection replaces the current one only if it clearly improves decoding quality and
the precision (SE) of the HP effect, never because a p value gets smaller.

- Quality: unchanged in V1–V3 (Δr ≤ .002), mixed in the higher areas.
- SE of the HP effect: V1–V3 HP − next 0.53 → 0.53 / 0.52. All areas 0.39 → 0.64 / 0.38. Occipitotemporal 0.41 → 0.87 /
  0.40. No clear gain, and r > 0.2 is worse.
- Fig. 6 therefore keeps the current selection. The V1–V3 quadrant test, the sweep and the per-area V2/V3 effects
  hold under both looser rules. The r > 0.1 rule adds VO to the BH-significant areas (n = 27 instead of 10); report this
  as robustness only.

## Cost

Cluster CPU, standard partition, account hare.econ.uzh. Fold-wise reliability: 29 tasks, ~4–8 min, ≤ 22 GB.
Decodes: about 1400 tasks (4 CPUs, 10–11 GB), 2–25 min each (residual fit grows with voxel count; sub-05, sub-09,
sub-12 and sub-17 pooled tasks needed > 10 min). About 460 vCPU-h allocated in total, roughly 1 CHF. Failures were
transient filesystem errors (sub-05 permission denied, sub-21 atlas "not found", both files present) and walltime
timeouts; all were resubmitted and completed.
