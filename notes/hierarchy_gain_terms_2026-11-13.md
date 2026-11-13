# Do the gain terms grow along the visual hierarchy? (2026-11-13, n = 28)

Reference fits: the polished root `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished` (DoG pRFs
refitted together with the gain fields; 241 of 252 participant × area cells, the other 11 have no reliable voxels).
Joint terms: `notes/data/gain_field_parameters_polished_2026-11-12.tsv` (dog/m7_full). Estimated alone (dS, dD, dT):
`notes/data/conditional_estimates_polished_2026-11-12.tsv` (`valid` only). The shared sustained gain estimated alone
(S) comes from a base that was not refitted (unchanged since update 8). Gaussian and DN rows: not refitted. Field
widths: `field_widths_reliable_polished_2026-11-12_n28_fits.tsv` (update 8). Participants 3–30. Gains in %; slopes in
percentage points per doubling of pRF size.

Code: `retsupp/visualize/paper/hierarchy_gain_terms.py` (all tests), `retsupp/modeling/fit_gain_bins.py` (per-voxel
sufficient statistics for step 4), `retsupp/visualize/paper/figS_hierarchy_gain_terms.py` (figure). One command per
stage for another reference fit (e.g. the converged/LM refit):

```bash
# cluster, ~/retsupp_plan after git pull (after regenerate_reference.sh has written that reference's tables)
REF_ROOT=<root> TAG=<tag> bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh cluster-submit
# laptop, repo root, when the 28 tasks are done
REF_ROOT=<root> TAG=<tag> bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh local
```

Outputs: `notes/data/hierarchy_gain_terms_<TAG>_n28_*.tsv`,
`notes/figures/paper_2026-11/figS_hierarchy_gain_terms_<TAG>_n28.{pdf,png,md}`. The DoG/Gaussian/DN table is
`notes/figures/paper_2026-11/table_hierarchy_gain_terms_<TAG>_n28.md` (written by the same `local` stage).

## Hierarchy axis and ordering

- **pRF size** (main axis), as in `hierarchy_regression.py`: per area, the median over participants (3–30) of the
  voxel-weighted mean pRF σ at 1–2.5° eccentricity (`fig1_prf_sigma.tsv`): V1 0.56°, V2 0.64°, V3 0.76°, hV4 0.91°,
  LO 0.94°, V3AB 1.15°, VO 1.56°, TO 1.60°, IPS 1.65°. Slope per doubling (log2).
- **Hierarchical rank** (check): V1 1, V2 2, V3 3, then V3AB, hV4 and LO 4, then IPS, VO and TO 5 (each second-tier
  area receives its main input from the first tier of its stream: IPS from V3AB dorsally, VO from hV4 ventrally, TO from
  LO laterally).
- Area sets: all nine; V1–V3 + dorsal (V3AB, IPS); V1–V3 + occipitotemporal (hV4, VO, LO, TO).
- Per participant, OLS slope of the term on the predictor across that participant's areas (at least three areas);
  across participants: median, bootstrapped 95% CI, k positive, Wilcoxon signed-rank; BH across the nine terms per
  pRF model and area set.

## 1. Slopes for every term (DoG, refitted; all nine areas, per doubling of pRF size)

| Term | Slope (pp per doubling) [95% CI] | k > 0 | p | q |
|---|---|---|---|---|
| Target gain | **+27.4 [+15.3, +68.3]** | 22/28 | .0022 | .005 |
| Distractor gain (location-weighted mean) | +11.4 [−11.2, +32.1] | 17/28 | .16 | .28 |
| Shared sustained gain, estimated alone | **−6.4 [−8.0, −5.1]** (more negative higher up) | 1/28 | 2.2 × 10⁻⁸ | < .001 |
| Sustained gain at LP, joint (sign only) | more negative higher up | 1/28 | 1.5 × 10⁻⁸ | < .001 |
| dS, estimated alone | −0.7 [−3.7, +3.3] | 12/28 | .61 | .76 |
| dS, joint | +0.8 [−10.2, +9.7] | 15/28 | .90 | .90 |
| dD, estimated alone | −3.1 [−9.3, +8.3] | 13/28 | .63 | .76 |
| dT, estimated alone | −0.0 [−26.0, +6.1] | 14/28 | .68 | .76 |

Same pattern with the rank predictor (target +16.2 per rank step, p = .0076; shared sustained −2.4, p < 10⁻⁷; dS
alone −0.5, p = .33) and in both branches (target: dorsal +20.8, p = .011; occipitotemporal +24.3, p = .003; dS alone:
+0.4, p = .76; −0.0, p = .60). Gaussian pRFs: same verdicts (target +77.0, distractor now also +30.1, p = .001; dS alone
−0.6, p = .37). DN pRFs: target +22.5 (p = .017, q = .105); dS alone −2.4 [−10.4, +0.2], p = .023, q = .105 (larger
in higher areas, not significant after correction); dD alone −5.8, p = .04 (q = .12). Full table:
`notes/figures/paper_2026-11/table_hierarchy_gain_terms_polished_2026-11-12_n28.md`.

## 2. Target gain versus dS: flat or too noisy?

`hierarchy_gain_terms_polished_2026-11-12_n28_dissociation.tsv`

- In % (absolute amount): target slope +27.4 vs dS (alone) −0.7; paired difference +34.4, target > dS in 20 of 28,
  p = .018. dS slope CI [−3.7, +3.3]; **minimum detectable slope (80% power, two-sided α = .05; 2.8 × bootstrap SE)
  = 4.6 pp per doubling**. So a growth of dS like the target's (≈ 27 pp) or even a sixth of it is excluded.
- Relative to each term's own size (each divided by its pooled median; dS −1.0%, target +17%): target +1.60 [+0.89,
  +3.98] vs dS +0.67 [−3.23, +3.64]; paired p = .49; minimum detectable slope 4.5 (i.e. 4.5 × dS's own size per
  doubling). **Too noisy**: dS is so small (≈ 1%) that a growth in proportion to its own size, as large as the
  target's, cannot be detected or excluded.
- Verdict: dS does not grow along the hierarchy in absolute amount as the target gain does (it is flat within
  ±3.7 pp per doubling); whether it grows in proportion to its own size is beyond the resolution of the data.
- The joint dS (CI ±10 pp; MDS 13.7 pp) is too noisy for either statement.

## 3. Constant amount or constant proportion of the sustained gain?

`..._proportion_per_area.tsv`, `..._proportion_regression.tsv`

- The shared sustained gain (estimated alone) becomes more negative higher up (−6.4 pp per doubling, 27 of 28), dS
  does not. Ratio of medians dS / shared sustained gain per area (robust to near-zero values of single participants,
  as both are medians over 21–28 participants): V1 0.93, V3 0.78, hV4 1.27; V2 0.09 (dS ≈ 0); V3AB 0.19, LO 0.19,
  VO 0.25, TO 0.15, IPS 0.04. **dS is closer to a constant amount (area medians 0.1–2.7%, not increasing) than to a constant
  proportion of the shared sustained gain**: the proportion falls from ~1 in V1/V3/hV4 to ~0.2 in the higher areas.
- Within participants, dS (alone) regressed on the shared sustained gain (alone) across areas: slope +0.22 [−0.28,
  +0.56], p = .63 (a constant proportion predicts a clearly positive slope); intercept −1.7 [−3.4, +2.2], p = .21.
- Ratios to the joint sustained gain at LP or HP are in the table but not interpretable: the LP gain's size trades off
  against the item weight and pRF amplitude after the refit (sign only, update 8).

## 4. Within areas: smaller vs larger pRFs

Per participant × area, the five gains re-estimated (least squares, everything else fixed at the joint fit) for the
smallest, middle and largest third of voxels by pRF size; per participant the median over areas of the slope on log2
bin pRF size. Areas with fewer than 60 voxels dropped (170 of 241 participant × area fits; mostly IPS and VO). dS here
is the joint dS (all five gains free). `..._bins.tsv`, `..._bin_slopes.tsv`.

- Target gain grows with pRF size within areas: +7.3 pp per doubling [+1.4, +18.8], 20/28, p = .0065, q = .033 (bin
  medians 2.7%, 12.2%, 16.5%).
- Distractor gain: −7.5 [−12.5, +1.1], p = .032, q = .079. Shared sustained gain (tied, re-estimated): +0.1, p = .81.
  dS: +1.9 [−1.0, +11.4], p = .18 (no dependence).
- Caveat: within an area, pRF size and eccentricity go together (bin median eccentricities 0.6°, 1.4°, 2.7°). By
  eccentricity terciles the target slope is +13.3 (p = .019) and the distractor −20.7 (p = 1 × 10⁻⁴). The within-area
  target result is therefore either a pRF-size or an eccentricity (closer to the 4° item ring) effect; this analysis
  cannot separate them.

## 5. Sustained field width against pRF size

Fits with a reliable sustained gain (update 8 widths, refitted fits). `..._width_slope.tsv`

- |gain|/SE > 2 (155 fits, 28 participants): −0.80° per doubling [−1.59, −0.13], p = .022, 8/28 positive.
- |gain|/SE > 3 (118 fits, 21 participants): −0.04° [−0.32, +0.10], p = .43.
- **Equivalence**: at the stricter threshold the slope lies within ±0.32° per doubling (12% of the median FWHM of
  2.7°); at either threshold a widening with pRF size is excluded (upper CI +0.10° and −0.13°). The narrowing at > 2
  is not robust (it rests on the V1/V2 fits, which are few and wide, as in update 8's claim (b)).
- So the sustained field does not scale with the pRF size of the area it is measured in; its width (≈ 2.8°) is
  constant across the hierarchy.

## 6. Interpretation

Each area is fitted with the stimulus as its input, so an area's gain is the gain accumulated up to that area, not
the gain added at that stage. The target gain accumulates along both streams (and within areas toward larger,
more eccentric pRFs); the HP-specific sustained reduction does not grow with it. It is no larger in the higher areas
than in V1–V3 and hV4 (largest in V3, hV4 and TO), so it is closer to a constant amount than to a fixed fraction of the shared
sustained gain, which does grow. Its field width does not scale with pRF size either. Together these favour a
spatially fixed, location-specific reduction that is present from early cortex on (or inherited from it), rather
than one built up stage by stage like the target gain. The data cannot exclude that dS grows in proportion to its own
(small) size.
