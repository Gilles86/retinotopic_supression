# Item 3a: is the HP change in the item gains multiplicative or additive?

**Bottom line.** Neither form is supported by the data. Both versions predict the held-out runs slightly worse than
the model with no HP term on the item gains. The additive version loses less than the multiplicative one: 22 of 28
participants predict better with it (median ΔR² −0.28 × 10⁻⁴, p = 7.9 × 10⁻⁴). The two fit the training data equally
well (+0.05 × 10⁻⁴, 16/28, p = .15), so this difference only says that the multiplicative term generalises worse. It
does not show that the brain combines the HP change additively. The fitted terms are not different from zero:
- multiplicative, h: pooled −0.1% [−23, +10], 14/28 > 0;
- additive, a: pooled −0.4% [−7.5, +7.7], 14/28 > 0.

The data therefore do not tell whether the HP change scales with an item's gain. For the text: the HP reduction can
be described as sustained (Fig. 5c). There is no evidence that it acts on the transient item gains in either form.

## Models

All four models share one base: the ladder's m5_TDS_dS. It has the item weight, one target gain g_T, one distractor
gain g_D (the same at all four locations), and the sustained gain with its HP − LP difference (dS). Each of the other
three models adds one shared term to this base, so all three have the same number of terms:
- **multiplicative** (m8_full_hpmult, new): at the HP location the target and distractor gains are scaled by a
  common factor, g_T → g_T (1 + h) and g_D → g_D (1 + h);
- **additive** (m8_full_hpadd, new): at the HP location a common offset is added, g_T → g_T + a and g_D → g_D + a;
- **distractor HP − LP only** (m7_full, the joint fit's term set, shown for reference): g_D at HP free (dD).

With h = 0 or a = 0 the new models are the base model exactly. The prediction at the start of every fit was checked
against the base: 1,943 of 1,943 checks gave max |Δprediction| = 0. The new class is
`retsupp/modeling/hp_item_gain.py`. The fit is a ladder option in `retsupp/modeling/fit_cv_ladder_nested.py`
(`--models m5_TDS_dS m7_full m8_full_hpmult m8_full_hpadd --prf-source-root <root>`), and the defaults are unchanged.

**Fitting.**
- **pRFs:** fixed at the refitted joint fit (update 8 reference, m7_full in
  `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`). This is the same voxel set,
  241 participant × area fits, n = 28.
- **Shared terms:** the base's shared terms (gains, widths, item weight) were refitted in every fold, starting from
  that fit. Each new model was then fitted from the base's fold optimum with its term at 0. Every model also got a
  second start taken from the reference dD (as h = dD/g_D, as a = dD, or as dD itself).
- **Optimiser:** bounded L-BFGS on the shared terms, as in the ladder.

The pRFs were held fixed, not refitted with the gains, because refitting would have cost about 20 times more:
- the alternating polish takes about 50 min per model and fold, against 3–5 min here;
- that comes to about 30–40 CHF for the run, against 6.5 CHF here.

All models share the same pRFs, so the pRFs' leak into the held-out runs is the same for every model. The fits are
written to `derivatives/af_prf_cv_ladder_nested_run3_bounded_maxsd20_reliable_roihrf_polprf` (held-out) and
`..._full_..._polprf` (all runs).

**Held-out prediction.** The ladder's run CV scheme (`run3`) has three folds. Each fold leaves out one run of each
HP condition (4 of 12 runs) and fits on the other 8.
- Held-out R² per participant × area is the mean over voxels of the per-voxel R², averaged over the three folds.
- ΔR² is given in units of 10⁻⁴ (0.01 percentage points of variance), as in item 1a.
- Statistics follow key_numbers.md:
  - per area: median over participants [bootstrapped 95% CI], participants > 0, Wilcoxon p, BH q across the nine areas;
  - pooled: per participant, the median over the nine areas;
  - per pathway: per participant, the median over the pathway's areas, BH across the three pathways.
- Median held-out R² of the base: 0.2145.
- This is a different CV scheme and R² definition from item 1a, which left out one run at a time and pooled the SSE.
  Here, dD − base gives −1.17 × 10⁻⁴ (1/28 > 0); item 1a's adding dD to dS gives −0.89 × 10⁻⁴ (0/28).

## Multiplicative − additive (ΔR² = R²[mult] − R²[add], held out, × 10⁻⁴)

| Area | Median [95% CI] | mult better / n | p | q |
|---|---|---|---|---|
| V1 | −0.25 [−1.49, +0.20] | 12/28 | .25 | .28 |
| V2 | −0.57 [−1.53, −0.07] | 7/28 | .0095 | .09 |
| V3 | −0.29 [−1.52, +0.06] | 10/28 | .11 | .24 |
| V3AB | −0.54 [−1.29, −0.06] | 8/27 | .09 | .24 |
| IPS | +0.35 [−3.54, +1.48] | 14/25 | .60 | .60 |
| hV4 | −0.13 [−0.77, +0.18] | 13/28 | .23 | .28 |
| VO | −0.77 [−1.73, +0.41] | 8/21 | .16 | .24 |
| LO | −0.17 [−1.43, +0.28] | 11/28 | .14 | .24 |
| TO | −0.73 [−3.31, +0.33] | 11/28 | .03 | .15 |
| **Pooled** | −0.28 [−0.75, −0.09] | 6/28 | 7.9 × 10⁻⁴ | n/a |
| Early (pathway) | −0.36 [−1.43, −0.04] | 7/28 | .01 | .02 |
| Dorsal (pathway) | −0.85 [−3.06, +0.30] | 11/27 | .16 | .16 |
| Occipitotemporal (pathway) | −0.60 [−0.91, −0.06] | 9/28 | .0022 | .0067 |

*Summary:* the additive version predicts the held-out runs a little better than the multiplicative one in 22 of 28
participants, and in the early and occipitotemporal pathways. No single area survives BH correction.

On the training runs the difference is +0.05 × 10⁻⁴ [−0.06, +0.17], 16/28 favour mult, p = .15. So the two forms fit
equally well, and the held-out difference is a difference in how well they generalise.

## Each version vs the base without an HP term on the item gains (held out, × 10⁻⁴)

| Comparison | Pooled median [95% CI] | better / n | p | Early (q) | Dorsal (q) | Occipitotemporal (q) |
|---|---|---|---|---|---|---|
| multiplicative − base | −1.53 [−2.39, −0.91] | 2/28 | 3.7 × 10⁻⁸ | −1.47 (1.1 × 10⁻⁴) | −1.59 (.0039) | −2.15 (1.1 × 10⁻⁷) |
| additive − base | −1.14 [−1.35, −0.38] | 1/28 | 5.2 × 10⁻⁷ | −0.73 (3.1 × 10⁻⁵) | −1.35 (.02) | −1.24 (8.3 × 10⁻⁶) |
| dD only − base (reference) | −1.17 [−1.40, −0.59] | 1/28 | 2.2 × 10⁻⁸ | −1.01 (2.0 × 10⁻⁴) | −1.38 (.0013) | −1.26 (6.9 × 10⁻⁶) |

*Summary:* every way of letting the HP location change the transient item gains makes held-out prediction worse,
by about 1–1.5 × 10⁻⁴, in almost every participant. On the training runs each term gains only +0.3 to +0.5 × 10⁻⁴
(26–27/28 > 0). This agrees with item 1a.

Per-area tables for these comparisons are in `notes/data/hp_item_gain_mult_vs_add_2026-11-13_tables.md`. Every area
has a median below 0. The areas with q ≥ .05 are:
- multiplicative − base: VO;
- additive − base: VO and V3AB;
- dD only − base: V1.

## Fitted h and a (all-runs fit, in %)

| | h (multiplicative), % | k > 0 / n | p | a (additive), % | k > 0 / n | p |
|---|---|---|---|---|---|---|
| V1 | −19.9 [−126, +59] | 12/28 | .65 | +4.2 [−7.2, +12.8] | 18/28 | .33 |
| V2 | −8.0 [−107, +21] | 13/28 | .68 | +8.2 [−2.0, +12.9] | 17/28 | .25 |
| V3 | +6.6 [−57, +42] | 16/28 | .93 | +12.5 [+5.0, +22.4] | 22/28 | .0076 (q = .07) |
| V3AB | +10.5 [−29, +25] | 15/27 | .80 | +2.7 [−8.5, +9.8] | 15/27 | .95 |
| IPS | −19.1 [−56, +16] | 12/25 | .18 | −4.8 [−14.9, +6.0] | 10/25 | .51 |
| hV4 | −14.2 [−33, +18] | 12/28 | .90 | +2.9 [−10.7, +19.4] | 17/28 | .37 |
| VO | −7.9 [−30, +5] | 8/21 | .49 | −4.0 [−12.3, +5.3] | 9/21 | .43 |
| LO | −4.5 [−31, +53] | 13/28 | .65 | −4.1 [−15.7, +1.9] | 10/28 | .48 |
| TO | −19.1 [−58, +15] | 11/28 | .40 | −9.1 [−18.4, +2.8] | 11/28 | .11 |
| **Pooled** | −0.1 [−23.0, +9.8] | 14/28 | .40 | −0.4 [−7.5, +7.7] | 14/28 | .73 |
| Early | −6.1 [−69, +29] | 13/28 | .37 (q = .55) | +9.6 [+3.3, +12.8] | 20/28 | .06 (q = .18) |
| Dorsal | −7.2 [−42, +22] | 12/27 | .55 (q = .55) | −1.8 [−15.0, +7.6] | 10/27 | .31 (q = .47) |
| Occipitotemporal | −10.9 [−46, +9] | 10/28 | .52 (q = .55) | −5.1 [−10.4, +2.8] | 10/28 | .49 (q = .49) |

*Summary:* neither term differs from zero, pooled, in any pathway or in any area after BH correction. h is very
uncertain (CIs of ±50–100%) because it is a ratio of small transient gains.

Units:
- h is the fractional change of both item gains at HP: −10% means each gain is 10% smaller there.
- a is in the same units as the gains, % modulation of the stimulus drive.

For scale, the same fits give a base target gain of 21% [14, 28] (25/28 > 0) and a base distractor gain of 11%
[4.8, 16] (20/28 > 0), pooled.

Implied HP changes per item in the multiplicative fit:

| | h · g_T (target), % | h · g_D (distractor), % |
|---|---|---|
| Pooled | +0.5 [−3.1, +5.1], 16/28, p = .58 | +1.0 [−0.3, +6.7], 17/28, p = .12 |
| Early | +3.4 [+0.5, +8.8], 21/28, p = .03 (q = .08) | +7.0 [+1.1, +15.5], 20/28, p = .02 (q = .05) |
| Dorsal | −3.5 [−25, +8], 11/27 | +0.3 [−12, +4], 14/27 |
| Occipitotemporal | −5.6 [−12, +1.4], 10/28 | +0.2 [−3.3, +5.1], 14/28 |

*Summary:* the item changes at HP implied by the multiplicative fit are near zero pooled. In early areas they are
slightly positive, with q ≥ .05, which is the opposite sign to a reduction.

**Robustness.**
- **Fits at the ±10 bound.** 28 of 241 all-runs fits have a gain, h or a at its ±10 (±1000%) bound in one of the
  four models. Leaving them out changes nothing: h pooled −11% [−28, +9], p = .13; a +0.6% [−5.1, +7.2], p = .65.
- **Fold means.** The run3 fold means give the same picture: h +5.6% [−9.5, +21], p = .48; a −2.3% [−8.5, +3.1],
  p = .81.
- **Transient width.** σ_dyn sits at its 0.5° or 8° bound in about half of all fits, in every model including the
  reference. This is a known property of the transient width, and these fits were not excluded.

## Sources

- **Tables:** `notes/data/hp_item_gain_mult_vs_add_2026-11-13_*.tsv`:
  - `cv_r2`: held-out and training R² per participant × area × model;
  - `cv_delta` and `cv_stats`: ΔR² and its statistics;
  - `terms_full` / `terms_run3` and `_stats`: h, a, g_T, g_D, dD and h·g per participant × area;
  - `shared_terms`: every shared term per fold;
  - `tables.md`: all tables, per area.
- **Code:**
  - `retsupp/modeling/hp_item_gain.py` (models);
  - `retsupp/modeling/fit_cv_ladder_nested.py` (`m8_full_hpmult`, `m8_full_hpadd`, `--prf-source-root`);
  - `retsupp/modeling/slurm_jobs/fit_cv_ladder_nested.sh` (`PRF_SOURCE_ROOT`);
  - `retsupp/modeling/collect_hp_item_gain.py` (statistics).
- **Jobs:** 8336667 (run3) and 8336668 (all runs), with 8339893 and 8341187 resuming timed-out tasks. CPU, about
  1,250 CPU-hours, ≈ 6.5 CHF.
