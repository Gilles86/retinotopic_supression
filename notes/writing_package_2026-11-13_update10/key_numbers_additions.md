# Key numbers added in update 10: the gain terms along the visual hierarchy (n = 28)

New numbers only; nothing earlier changes. Reference fits: the refitted (polished) DoG fits of update 8.
Gains in %; slopes in percentage points (pp) per doubling of the area's pRF size (median per-participant slope
[bootstrapped 95% CI], participants with a positive slope, Wilcoxon p; BH q across the nine terms). The LP gain is
reported by its sign only.

Sources (all `notes/data/hierarchy_gain_terms_polished_2026-11-12_n28_<x>.tsv`; `<x>` given per line):
inputs `gain_field_parameters_polished_2026-11-12.tsv` (joint), `conditional_estimates_polished_2026-11-12.tsv`
(estimated alone, `valid` only), `field_widths_reliable_polished_2026-11-12_n28_fits.tsv` (widths),
`fig1_prf_sigma.tsv` (pRF size axis), `derivatives/gain_bins_polished_2026-11-12` (within-area bins).

## Slopes along the hierarchy (`regression`; all nine areas, DoG)

- Hierarchy axis (`prf_size_axis`): pRF σ at 1–2.5° eccentricity, V1 0.56°, V2 0.64°, V3 0.76°, hV4 0.91°, LO 0.94°,
  V3AB 1.15°, VO 1.56°, TO 1.60°, IPS 1.65°.
- Target gain: +27.4 pp [+15.3, +68.3], 22/28, p = .0022, q = .005. Rank predictor: +16.2 pp per step, p = .0076.
  Dorsal branch (V1–V3, V3AB, IPS): +20.8, p = .011; occipitotemporal (V1–V3, hV4, VO, LO, TO): +24.3, p = .003.
- Distractor gain: +11.4 [−11.2, +32.1], 17/28, p = .16 (n.s.).
- Shared sustained gain (estimated alone): −6.4 [−8.0, −5.1], 1/28 positive, p = 2.2 × 10⁻⁸ (more negative higher up).
- Sustained gain at LP (joint): more negative higher up in 27/28 (sign only), p = 1.5 × 10⁻⁸.
- dS estimated alone: −0.7 [−3.7, +3.3], 12/28, p = .61. Joint: +0.8 [−10.2, +9.7], p = .90. Rank: −0.5, p = .33;
  dorsal +0.4 (p = .76), occipitotemporal −0.0 (p = .60).
- dD alone −3.1 [−9.3, +8.3], p = .63; dT alone −0.0 [−26.0, +6.1], p = .68.
- Gaussian pRFs: target +77.0, p = 7.5 × 10⁻⁹; distractor +30.1, p = .0011; dS alone −0.6, p = .37.
  DN pRFs: target +22.5, p = .017 (q = .105); dS alone −2.4 [−10.4, +0.2], p = .023 (q = .105); dD alone −5.8,
  p = .04 (q = .12). (Gaussian and DN: pRFs not refitted.) Table: `figures/table_hierarchy_gain_terms_polished_2026-11-12_n28.md`.

## Target vs dS (`dissociation`)

- Paired, in pp: target − dS (alone) slope +34.4, target larger in 20/28, p = .018.
- dS (alone) minimum detectable slope (80% power, α = .05 two-sided; 2.8 × bootstrap SE): 4.6 pp per doubling.
- Relative to each term's own size (divided by its pooled median): target +1.60 [+0.89, +3.98], dS +0.67 [−3.23,
  +3.64]; paired p = .49.

## dS relative to the sustained gain (`proportion_per_area`, `proportion_regression`)

- Ratio of area medians dS / shared sustained gain (both estimated alone): V1 0.93, V2 0.09, V3 0.78, hV4 1.27;
  V3AB 0.19, IPS 0.04, VO 0.25, LO 0.19, TO 0.15.
- Within participants, dS on the shared sustained gain across areas: slope +0.22 [−0.28, +0.56], p = .63.

## Within areas (`bin_slopes`, `bins`; terciles of pRF size; 170 participant × area fits with ≥ 60 voxels)

- Target gain: +7.3 pp per doubling of pRF size [+1.4, +18.8], 20/28, p = .0065, q = .033 (tercile medians 2.7%,
  12.2%, 16.5%).
- Distractor gain −7.5 [−12.5, +1.1], p = .032, q = .079; dS (joint) +1.9 [−1.0, +11.4], p = .18; shared sustained
  +0.1, p = .81.
- Terciles of pRF size differ in eccentricity (medians 0.6°, 1.4°, 2.7°); by eccentricity terciles: target +13.3
  (p = .019), distractor −20.7 (p = 1 × 10⁻⁴).

## Sustained field width vs pRF size (`width_slope`)

- |gain|/SE > 2: −0.80° per doubling [−1.59, −0.13], p = .022 (155 fits, 28 participants).
- |gain|/SE > 3: −0.04° [−0.32, +0.10], p = .43 (118 fits, 21 participants); equivalence bound ±0.32° per doubling
  (12% of the median FWHM, 2.7°).
