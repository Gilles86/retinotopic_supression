# Key numbers that change in update 8 (pRFs refitted together with the gain fields; n = 28)

**Provisional until the voxel-selection check is decided** (see UPDATE_NOTES.md).

Each line: earlier value (update 4_final / update 7: gain fields fitted with the pRFs held at the fit without gain
fields) → new value (pRFs refitted together with the gain fields). Gains in %; differences between gains in percentage
points (written as %). Per area: median over participants, * BH q < .05 across the nine areas (Wilcoxon). Pooled: per
participant the median over the nine areas; median [bootstrapped 95% CI], participants < 0 (or > 0), Wilcoxon p.
The complete set of numbers for the new reference, generated from the tables, is `key_numbers_reference_full.md`
(also `notes/data/reference_numbers_polished_2026-11-12_n28.{md,tsv}`).

Sources (new):
- joint fit (all terms fitted together, m7_full): `notes/data/gain_field_parameters_polished_2026-11-12.tsv`
  (rows dog/m7_full and dog/m4_TDS replaced by the refitted fits; Gaussian and DN rows unchanged);
- estimated alone (each HP − LP term added alone to the model with item weight, target, distractor and sustained
  gain, m4_TDS refitted): `notes/data/conditional_estimates_polished_2026-11-12.tsv` (rows with `valid` = True, as
  for every earlier estimated-alone number);
- field widths: `notes/data/field_widths_reliable_polished_2026-11-12_n28_*.tsv`;
- gain vs pRF size: `notes/data/hierarchy_regression_polished_2026-11-12_n28.tsv`;
- the refit itself: `notes/data/prf_refit_2026-11-12_shared.tsv`, `notes/data/prf_refit_2026-11-12_prf_shift.tsv`.

## Methods: the refit

- Training error lower in every fit: 482 of 482 participant × area × model fits (241 joint, 241 for the
  estimated-alone base); median −0.029%, smallest change −0.0019%, largest −5.3% (`prf_refit_2026-11-12_shared.tsv`).
- pRF centres move little: median 0.009° (90th percentile 0.07°; medians over fits); 0.040° (90th percentile 0.11°)
  for voxels with a centre within 2° of an item location. pRF size +0.4%, amplitude +1.3% (median relative change)
  (`prf_refit_2026-11-12_prf_shift.tsv`).

## Fig. 5c: dS, the sustained gain at HP minus at LP

- **dS estimated alone, pooled (lead statistic)**: −1.4% [−1.8, −0.6], 22/28 < 0, p = 1.1 × 10⁻⁴ →
  **−1.0% [−1.9, −0.4], 22/28 < 0, p = 2.7 × 10⁻⁴**.
- dS estimated alone per area: V3 −3.0%* (q = .046) → **−2.7%* (q = .036)**; TO −1.9%* (q = .035) → **−1.9%*
  (q = .036)**; hV4 −1.5% (n.s.) → **−2.4%* (q = .036)**. Other areas n.s. as before (V1 −0.7%, V2 −0.1%, V3AB −0.6%,
  IPS −0.3%, VO −0.7%, LO −0.7%; q ≥ .12).
- dS estimated alone per pathway (BH across the three): early −0.5% (q = .040) → **−0.5% [−2.3, +0.1], 18/28 < 0,
  p = .019, q = .057**; dorsal −0.8% (q = .90) → −0.5% [−1.4, +1.0], 15/27, q = .58; occipitotemporal −1.1% (q = .040) →
  **−1.3% [−2.1, −0.4], 20/28, p = .060, q = .089**. The optional clause "present in early and occipitotemporal areas,
  not reliable in dorsal areas" no longer holds after correction: drop it (or "most consistent in occipitotemporal
  and early areas" without a significance claim).
- **dS, joint fit, pooled**: −3.4% [−6.6, −1.5], 22/28, p = .0027 → **−3.5% [−6.7, −1.3], 22/28 < 0, p = .0044**.
  Per area: V3 −9.0%* (q = .032) → **−9.5%* (q = .034)**; hV4 −5.7%* (q = .028) → **−5.6%* (q = .024)**; TO −2.4%*
  (q = .044) → **−3.1% (q = .084), no longer significant**.

## Fig. 5b: target and distractor gain, joint fit (unchanged in direction and significance)

- Target pooled +16% [+13, +24], p = 3.2 × 10⁻⁷ → **+17% [+13, +24], 25/28 > 0, p = 3.2 × 10⁻⁷**. Per area q < .05 in
  the same six areas: V3 +30 → +33%, V3AB +33 → +29%, IPS +21 → +21%, hV4 +56 → +59%, VO +50 → +38%, TO +22 → +18%;
  LO +18 → +20% (q = .090, n.s.), V1 −1.8 → −2.3%, V2 +4.8 → +5.2% (n.s.).
- Distractor (averaged over its locations) pooled +6.5% [+3.2, +15], p = .023 → **+8.0% [+4.4, +13], 21/28 > 0,
  p = .023**. Per area q < .05 in the same three: V3AB +12 → +11%, hV4 +22 → +21%, TO +8.4 → +8.2%.
- Target gain vs pRF size: +33% per doubling [+24, +67], 82% positive, q = .0017 → **+27% [+15, +68], 79% of
  participants positive, p = .0022, q = .0089**; distractor +11% [−10, +29], q = .33 → +11% [−11, +32], q = .31 (n.s.).
  Within participants the target gain falls with the transient width: ρ −.62 → −.67 (p = 7.5 × 10⁻⁸); does not track the
  item weight: ρ +.07 → +.04 (p = .60) (`figS_gain_vs_prf_size_polished_2026-11-12_n28.md`).
- Unchanged (base models not refitted): the distractor gain on top of the target term, estimated alone (+3.1%,
  p = .019; V3AB, hV4), and the sustained gain shared by all four locations estimated alone (S, −3.4%).

## Fig. 5d: transient HP − LP

- dD estimated alone: −1.0% [−2.9, +0.5], p = .099 → **−0.8% [−3.0, +0.8], 16/28 < 0, p = .14**; LO −3.3%* (q = .022)
  → LO −3.3%* (q = .020).
- dT estimated alone: −3.4% [−6.8, +0.5], p = .018 → **−3.6% [−6.5, +0.1], 19/28 < 0, p = .032**; no area q < .05.
- dD − dT: +0.9%, p = .24 → +1.0% [−0.4, +5.6], 17/28 > 0, p = .17. Joint dD: +0.4%, p = .29 → +0.1% [−2.8, +12],
  p = .35. Panel d title unchanged ("slightly less, target and distractor alike").

## The sustained gain at all four locations, joint fit (quote its level with care)

- g_LP pooled −6.7%, 24/28 < 0, p = 2.3 × 10⁻⁶ → **−11% [−21, −7.4], 23/28 < 0, p = 4.0 × 10⁻⁶**; g_HP −15%, 25/28,
  p = 1.5 × 10⁻⁵ → **−21% [−25, −14], 26/28 < 0, p = 4.1 × 10⁻⁷**.
- Per area g_LP: V3AB −11 → −21%*, IPS −33 → −58%*, VO −15 → −33%*, LO −8.7 → −19%*, TO −35 → −67%*, hV4 −5.8 → −11%
  (n.s., q = .060), V1 +2.8 → +3.1%, V2 +1.3 → +0.6%, V3 −0.4 → −0.4% (n.s.). Larger in the higher areas.
- Why: with the pRFs free, the LP gain trades off against the item weight and the pRF amplitude. Change of g_LP
  (refitted − earlier) vs change of the item weight: Spearman ρ = −.73 across 241 participant × area fits
  (p = 2.6 × 10⁻⁴¹), within participants median ρ = −.73 (p = 4.2 × 10⁻⁶); vs change of the pRF amplitude: ρ = −.17
  (p = .0074), within participants −.19 (p = .010). Item weight (joint, ratio to the bar) pooled 3.08 → 3.39; TO 4.6 → 6.0,
  IPS 18 → 22. So the sign and the per-area significance of the shared sustained gain hold, its size does not. The HP − LP difference in the
  same fit changed little (joint dS, refitted − earlier, −0.18 points per participant, p = .025;
  `notes/prf_refit_with_gains_2026-11-12.md`).

## Fig. 5a maps (V3, hV4, TO)

- HP − LP drawn (dS estimated alone): V3 −3.0 → −2.7%; hV4 −1.5 → −2.4%; TO −1.9 → −1.9%. Under the maps, sustained
  gain at LP / at HP: V3 −0.4 / −3.5 → −0.4 / −3.1%; hV4 −5.8 / −7.3 → −11 / −13%; TO −35 / −37 → −67 / −68%.
- Sustained width (FWHM), now the median among fits with a reliable sustained gain: V3 4.6 → 2.7° (17 fits); hV4
  3.0 → 2.9° (19); TO 2.5 → 2.6° (26).
- Transient: target V3 +30 → +33%, hV4 +56 → +59%, TO +22 → +18%; distractor +3 → +6%, +22 → +21%, +8 → +8%.
- Areas significant (q < .05) in the legend: joint dS V3, hV4, TO → **V3, hV4**; dS estimated alone V3, TO →
  **V3, hV4, TO**.
- Search supplement (`figS_gain_field_map_search_v3_n28`): gain at the distractor with the distractor at HP / at LP:
  V3 +9.5 / +0.7%, hV4 +16 / +7.6%, TO −59 / −58%.

## Field widths among fits with a reliable gain (update 7 → update 8)

- Calibration of the jackknife SE (0.67 expected): 0.70 / 0.66 / 0.56 → 0.73 / 0.68 / 0.58 (sustained HP / LP /
  target).
- **Sustained FWHM, reliable fits, participant medians: 2.7° [2.6, 2.9] → 2.8° [2.7, 2.9]** (all fits 3.2° → 3.0°).
  Per area (n fits): V1 3.9° (8) → 3.1° (8); V2 5.1° (6) → 3.7° (8); V3 2.7° (12) → 2.7° (17); V3AB 2.8° (16) → 3.0° (21);
  IPS 2.8° (15) → 2.8° (17); hV4 2.6° (15) → 2.9° (19); VO 2.5° (15) → 2.7° (20); LO 2.8° (14) → 2.8° (19); TO 2.5° (22) →
  2.6° (26). More fits pass the reliability threshold.
- **Narrower than 5.7°: 25/28, p < 10⁻⁴ → 28/28 participant medians, p = 7.5 × 10⁻⁹**; per area q < .05 in V1, V3AB,
  IPS, VO, LO, TO → V1, V3AB, IPS, **hV4**, VO, LO, TO (not V2, V3).
- **Early (V1–V3) vs higher areas: 3.4° vs 2.5°, 13/20 wider in early, p = .053 → 3.4° vs 2.7°, 15/24, p = .013**; at
  the stricter |gain| / SE > 3: 6 pairs, p = .69 → 14 pairs, 2.6° vs 2.7°, 6/14 wider in early, p = .58. The early-wider
  claim now passes at |z| > 2 but not at |z| > 3 and rests on few early fits (V1 8, V2 8): still not robust; keep
  update 7's wording ("no evidence that the sustained field is broader in early than in later visual areas") or
  qualify: "wider in early areas at one reliability threshold (p = .013) but not at a stricter one (p = .58)".
- Transient width with a reliable target gain at a bound: 58% → 59% (143 fits); not interpreted.
- Profile interval of the sustained width: 17.7° / 8.0° / 4.4° / 1.9° → 17.7° / 15.5° / 4.1° / 1.5° (|z| < 1, 1–2, 2–3,
  > 3); ρ = −.74 → −.83 (closed intervals).
- Gain–width trade-off (Spearman across fits / within participants): sustained .66 / .24 → .69 / .29; target
  −.61 / −.19 → −.63 / −.18; distractor −.57 / −.09 → −.58 / −.13 (n.s.).
- Field volume g · 2πσ² (deg², participant medians): sustained at HP −1.81 → −2.95 [−3.69, −2.23], 26/28 < 0,
  p = 3.2 × 10⁻⁵; at LP −0.78 → −1.56 [−2.28, −0.98], 22/28, p = .0007; target +1.86 → +1.97 [+1.14, +5.89], 25/28,
  p = .0005; distractor +0.62 → +0.73 [+0.12, +1.16], 20/28, p = .44.

## Not changed by update 8 (and why)

See UPDATE_NOTES.md, "Not regenerated".
