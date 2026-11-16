# Update 13b: update-11 items and the RT analyses rerun on the zero-sum main model (16 Nov 2026)

Reruns of the analyses that update 13 (`notes/writing_package_2026-11-16_update13/UPDATE_NOTES.md`, section 5) still
quoted from the earlier refitted-polished fits. Everything now sits on the zero-sum main model (m5_TDS_dS_zs: item
weight, target gain, distractor gain and a zero-sum sustained field, pRFs and gain fields fitted jointly by LM; read as
m7_full of the view root `..._zs_lm_ref`). Same code, statistics, priors and sampler settings as the originals; only the
input fits change, except where noted under "What could not be rerun as before". n = 28 (sub-03 … sub-30).

## Lead: what changed

1. **Item 1a, dS held out.** On the zero-sum fits adding dS no longer makes held-out prediction reliably worse
   (pooled ΔR² −0.11 × 10⁻⁴, 10/28 better, p = .099, no area q < .05; earlier −0.29, 3/28, p = 6.7 × 10⁻⁶, four areas
   worse). 8 of 28 participants now predict best with dS (earlier 2), 17 with no HP term (earlier 24). The transient
   terms still make held-out prediction worse in nearly every participant. The update-11 sentence "every added term
   makes the held-out run slightly worse, including dS" must be changed for dS (see item 1a).
2. **Item 1c, the real joint-fit dD is now positive.** Linearised on the zero-sum main model, the joint estimate of
   the distractor-specific HP − LP term is +7.3% (p = .0048; with dT +6.6%, p = .032; earlier +1.2%, p = .26). The
   supplementary LM fit with dD (m7_full_zs) gives the same sign: pooled +5.6%, 17/28 > 0, p = .070; so does the
   item-3a dD-only model with the main model's pRFs frozen (all runs +2.3%, p = .082; run3 fold means +5.9%, p = .018). This is the
   opposite direction of a distractor-specific reduction, so the conclusion of 1c (no evidence for one; a reduction of
   about 10 points is ruled out) holds and is, if anything, stronger. The power figures themselves barely move. The
   positive dD should not be reported as an effect: it does not survive held-out prediction (item 1a, adding dD:
   0/28 better).
3. **Main residual-RT analysis.** Distractor and global gain both credible, as before, and slightly larger: distractor
   −2.4% [−4.3, −0.6] per SD of residual log RT (P = .994; earlier −1.6 [−3.3, −0.04], P = .977), global −2.8%
   [−5.5, −0.2] (P = .982; earlier −2.6 [−5.1, −0.04], P = .976). New: the occipitotemporal global gain alone is
   credible (−4.2 [−8.1, −0.3]; earlier −3.2 [−6.1, +0.02]). The baseline (z(log RT), separate models) still has the
   global gain non-credible (−2.2 [−5.0, +0.7], P = .944), so the plan's wording rule still applies.
4. **Joint latent model.** Global gain credible as before, slightly larger: −17 ms per SD of latent gain [−31, −4],
   P(−) = .995 (earlier −15 ms [−30, −1], P = .985); −3.4% gain per SD of log RT [−4.9, −2.1] (earlier −2.4). One
   divergence (earlier 0), min bulk ESS 318 (earlier 379).
5. **Item 3a, one sentence.** Conclusion unchanged, but the multiplicative and additive forms no longer fit the
   training runs equally well: multiplicative fits slightly better in training (+0.20 × 10⁻⁴, p = .010) and worse held
   out (−0.69 × 10⁻⁴, 23/28 favour additive, p = 1.4 × 10⁻⁴; now q < .05 in V1, V2, hV4, TO).
6. Items 2 and 5b: conclusions unchanged. In item 2, "measured − joint fit with dD, target − distractor" moves from
   p = .048 to p = .070, so write "the data do not show" rather than "contradict".

## Item 1a: held-out prediction (leave one run out)

Same code (`retsupp/modeling/hp_terms_heldout_power.py --zero-sum`), 12 folds (11 for sub-20, sub-24), linear around the
fit, pRFs and widths of the all-runs main model, identical in every model. Zero-sum term set: base = item weight,
target gain, distractor gain (no sustained level shared by the four locations); dS moves g_HP by +¾ and g_LP by −¼.
Held-out R² of the base: median 0.250 (earlier 0.250). ΔR² × 10⁻⁴, pooled = per participant the median over nine areas.

| Contrast | Earlier (update 11) | Zero-sum main model |
|---|---|---|
| dS vs no HP terms | −0.29 [−0.58, −0.21], 3/28 better, p = 6.7 × 10⁻⁶; q < .05 in V1, IPS, LO, TO | **−0.11 [−0.24, +0.00], 10/28, p = .099; no area or pathway q < .05** |
| + dD to dS | −0.89 [−1.32, −0.47], 0/28, p = 7.5 × 10⁻⁹ | −0.80 [−1.44, −0.49], 0/28, p = 7.5 × 10⁻⁹ (8 areas q < .05) |
| + dT to dS + dD | −0.81 [−1.02, −0.56], 1/28, p = 1.5 × 10⁻⁸ | −0.64 [−1.11, −0.57], 1/28, p = 1.5 × 10⁻⁸ |
| + dD + dT to dS | −1.64 [−1.97, −1.01], 1/28, p = 2.2 × 10⁻⁸ | −1.39 [−1.76, −1.01], 0/28, p = 7.5 × 10⁻⁹ |
| all three vs none | −1.91 [−2.96, −1.33], 2/28, p = 3.7 × 10⁻⁸ | −1.51 [−2.40, −1.35], 4/28, p = 3.2 × 10⁻⁷ |
| Best model per participant (none / dS / dS+dD / dS+dD+dT) | 24 / 2 / 0 / 2 | **17 / 8 / 2 / 1** |
| In-sample ΔR², dS vs none | +0.29 | +0.81 |

**Wording.** Changes for dS: "Held-out prediction does not separate the HP − LP terms: adding the sustained dS neither
improves nor reliably worsens prediction of the left-out run (10 of 28 participants better, p = .099), while each
transient term makes it worse in nearly every participant." The conclusion (held-out R² cannot decide; judge the terms
from group-level estimates and their power) stands.

Per-area tables: `notes/data/update13b_zs_lm_2026-11-16/item1a_heldout.md`.

## Item 1c: sensitivity to a planted distractor-specific HP reduction

Same design (200 phase-randomised data sets per δ, two generators, linear refits with pRFs and widths held). Joint-fit
generator = the main model with the distractor gain at HP set to g_LP_dyn + δ (its own dD is 0 by construction); the
estimated-alone generator = base + each participant's estimated-alone dS + δ. Power of the pooled test (Wilcoxon over
28 participants, p < .05 in the planted direction), group median of the estimate in brackets:

| Estimator | Generator | δ = 0 | −2% | −5% | −10% |
|---|---|---|---|---|---|
| dD, joint (with dS) | est. alone: earlier → zero-sum | 2% → 3% | 16% → 10% | 55% → 46% | 97% → 90% |
| | joint fit: earlier → zero-sum | 2% → 4% | 9% → 11% | 56% → 46% | 95% → 90% |
| dD, joint with dS and dT | est. alone | 3% → 3% | 14% → 13% | 47% → 36% | 94% → 88% |
| | joint fit | 2% → 4% | 13% → 12% | 45% → 36% | 91% → 87% |
| dD − dT, joint with dS and dT | est. alone | 4% → 4% | 13% → 8% | 40% → 32% | 87% → 76% |
| | joint fit | 4% → 4% | 12% → 9% | 40% → 32% | 89% → 75% |
| dD estimated alone (base refitted) | est. alone | 19% → **45%** (−0.8 → −1.6) | 52% → 74% | 91% → 96% | 100% → 100% |
| | joint fit | 51% → 43% (−1.5 → −1.7) | 81% → 74% | 98% → 96% | 100% → 100% |
| dD − dT estimated alone (base refitted) | est. alone | 6% → 10% | 3% → 4% | 14% → 6% | 59% → 34% |
| | joint fit | 12% → 10% | 4% → 3% | 6% → 6% | 44% → 34% |

Joint estimators stay unbiased (bias ≤ 0.6 points). Real data, same linearised estimators (pooled median):

| Estimator | Earlier | Zero-sum |
|---|---|---|
| dD, joint (with dS) | +1.15%, p = .26 | **+7.32%, p = .0048** |
| dD, joint with dS and dT | +3.15%, p = .58 | +6.63%, p = .032 |
| dD − dT, joint | −0.32%, p = .78 | +0.15%, p = .40 |
| dD estimated alone (base held) | −0.78%, p = .16 | −2.54%, p = .056 |
| dD − dT estimated alone (base held) | +1.27%, p = .16 | −1.04%, p = .88 |

On the earlier fits the linearised estimated-alone dD reproduced the reported value exactly (−0.78%). Here it does
not (−2.54% vs the reported zero-sum −1.2%, p = .31), because the linearisation now sits on the main model's pRFs and
widths, whereas the reported estimated-alone terms sit on m4_TDS_zs at the other participants' median σ_AF.

**Wording.** The power statement of update 11 stands with slightly lower numbers: "the joint test detects a −10-point
distractor-specific reduction 87–90% of the time, a −5-point one 36–46% and a −2-point one rarely (10–13%)." The
observed joint dD is positive (+7.3%, linearised; +5.6%, p = .070, in the LM fit with dD), so a reduction of about the
size of the distractor gain is ruled out. The estimated-alone dD is even more biased by the (now larger) sustained
reduction: with no planted effect it is "significant" in 43–45% of data sets (earlier 19–51%), so it remains no test of
a distractor-specific term. Full tables: `notes/data/update13b_zs_lm_2026-11-16/item1c_tables_generated.md`.

## Item 2: item responses at HP, model vs measured

Same pipeline (`predict_location_timeseries_polished.py`, Fig. 3 selection, bar removed, nideconv, 3.2–6.4 s), but the
selection, the bar-alone prediction and the measured series now come from the main model's pRFs
(`location_timeseries_zs_lm_2026-11-16`). Variants: S = main model as fitted (sustained HP − LP only); nohp = S with
the sustained field at 0; SD = the zero-sum model with dD (m7_full_zs, its own pRFs, same voxels). % signal, early
areas (V1–V3):

| Quantity | Earlier | Zero-sum |
|---|---|---|
| Measured HP − LP, items pooled | −0.045 [−0.088, −0.012], 20/28 < 0, p = .0095 | −0.043 [−0.097, −0.015], 21/28 < 0, p = .012 |
| S predicted, items pooled | −0.029 | −0.027 (63% of measured) |
| Measured − S, items pooled | −0.027, p = .14 | −0.017, p = .14 |
| S predicted, target / distractor / neutral | −0.029 / −0.028 / −0.026 | −0.029 / −0.025 / −0.024 |
| SD predicted, distractor | −0.002 | −0.005 |
| SD predicted, target − distractor interaction | −0.025, p = .043 | −0.022, p = .043 |
| Measured, target − distractor interaction | −0.001, p = .54 | +0.007, p = .39 |
| Measured − SD, distractor | −0.075, p = .025 | −0.083, p = .032 |
| Measured − SD, target − distractor | +0.032, p = .048 | +0.036, p = .070 |
| No HP terms, |median| | ≤ 0.007 | ≤ 0.007 |
| Location × item-type interaction, measured (early) | F(2,54) = 1.47, p = .24 | F = 1.53, p = .23 |
| Relative units (Fig. 3e), early pooled: measured / S / measured − S | −10.9 / −15.0 / p = .58 | −13.2 / −11.6 / p = .37 |
| V1 pooled: measured vs S | −0.057 vs −0.009 (shortfall p = .048, q = .14) | −0.056 vs −0.006 (p = .050, q = .15) |
| V1 target − distractor, measured | +0.058, p = .030, q = .089 | +0.069, p = .023, q = .070 |

**Wording.** Unchanged: the sustained field alone predicts about two thirds of the early HP reduction, uniformly over
item types, with a non-significant remainder; the model with dD predicts a target − distractor interaction the data do
not show (the "measured − SD" contrast is now p = .070 rather than .048, so say "does not show" rather than "contradicts").
Not rerun: SDT/SDT0 (Table 2d, see below). Tables: `notes/data/items_hp_model_vs_measured{,_anova}_n28_zs_lm_2026-11-16.tsv`.

## Item 5b: item-type contrasts, model vs measured

Same pipeline and statistics. "Model" is here the main model (S variant); the SD variant (with dD) gives the same
picture (`items_type_model_vs_measured_n28_zs_lm_2026-11-16.tsv`). % signal:

| Group | Contrast | Measured (earlier → zero-sum) | Model predicted (earlier → zero-sum) | Measured inside model band? | Measured − model (zero-sum) |
|---|---|---|---|---|---|
| V1 | target − neutral | +0.071 → +0.060 [+0.032, +0.098], 20/28, p < .001 | −0.004 → −0.006 | no → no (power 1%) | +0.083, p < .001 |
| V2 | target − neutral | +0.146 → +0.145 [+0.094, +0.177], 27/28, p < .001 | +0.007 → +0.005 | no → no (power 15%) | +0.097, p < .001 |
| Early | target − neutral | +0.120 → +0.127 | +0.043 → +0.033 | no → no | +0.083, p < .001 |
| Early | distractor − neutral | +0.058 → +0.061 | +0.008 → +0.016 | no → no | +0.040, p < .001 |
| Early | target − distractor | +0.038 → +0.034 | +0.031 → +0.016 | yes → yes | +0.030, p = .11 |
| Dorsal / Occ.-temporal | target − distractor | +0.128 / +0.105 → +0.109 / +0.096 | +0.105 / +0.091 → +0.120 / +0.094 | yes → yes | −0.009 / +0.008, p ≥ .76 |

MDE in V1 / V2: 0.039 / 0.054 (earlier 0.039 / 0.053). Fitted target gain of the main model: V1 −6.3%, V2 +3.1%,
V3 +47% (earlier V1 −2.3%, V2 +5.2%, V3 +33%).

**Wording.** Unchanged: the data show a target response above a neutral item in V1 and V2 at 1.5–3 × the MDE; the
model predicts none there, so "no target gain in V1 and V2" describes the fitted model, not the data. The model still
misses a response shared by target and distractor and gets target − distractor right (except V3, measured above the
band, as before). Voxel counts: `items_voxel_overlap_n28_zs_lm_2026-11-16.tsv`.

## Item 3a: multiplicative vs additive HP term on the item gains

Same ladder option (`--prf-source-root`), now with `--prf-source-zero-sum`: pRFs frozen at the all-runs zero-sum main
model; base = m5_TDS_dS with the sustained field zero-sum; multiplicative (h), additive (a) and dD-only models from the
base's fold optimum, second start from the dD of the zero-sum model with dD (m7_full_zs); bounded L-BFGS on the shared
terms with the zero-sum bounds; run3 CV (3 folds) and all runs. All 241 cells finished; start = base exactly (max
|Δprediction| 0 in the canary). Median held-out R² of the base 0.2150 (earlier 0.2145). ΔR² × 10⁻⁴, pooled:

| Comparison | Earlier | Zero-sum |
|---|---|---|
| mult − add, held out | −0.28 [−0.75, −0.09], 6/28 mult better, p = 7.9 × 10⁻⁴; no area q < .05 | **−0.69 [−1.10, −0.25], 5/28, p = 1.4 × 10⁻⁴; q < .05 in V1, V2, hV4, TO**; early (q = .0056) and occipitotemporal (q = .0024) |
| mult − add, training runs | +0.05 [−0.06, +0.17], 16/28, p = .15 | **+0.20 [+0.01, +0.28], 20/28, p = .010** |
| mult − base, held out | −1.53, 2/28, p = 3.7 × 10⁻⁸ | −1.90 [−2.65, −1.22], 0/28, p = 7.5 × 10⁻⁹ |
| add − base, held out | −1.14, 1/28, p = 5.2 × 10⁻⁷ | −1.00 [−1.81, −0.64], 3/28, p = 1.3 × 10⁻⁶ |
| dD only − base, held out | −1.17, 1/28, p = 2.2 × 10⁻⁸ | −1.17 [−2.35, −0.78], 0/28, p = 7.5 × 10⁻⁹ |
| Each term, training gain | +0.3 to +0.5 (26–27/28) | +0.38 to +0.61 (25–27/28) |

Every area has a median below 0 for all three "term − base" contrasts (q ≥ .05 only for add − base in V1 and VO).
Fitted terms (all-runs fit, %, pooled; per pathway and area none differs from 0 after BH):

| Term | Earlier | Zero-sum |
|---|---|---|
| h (multiplicative) | −0.1 [−23, +10], 14/28 > 0, p = .40 | −9.6 [−33, +8], 12/28, p = .19 |
| a (additive) | −0.4 [−7.5, +7.7], 14/28, p = .73 | +0.8 [−7.6, +8.6], 16/28, p = .48 |
| a, early pathway | +9.6 [+3.3, +12.8], p = .06 (q = .18) | +8.5 [+1.0, +13.2], p = .066 (q = .20) |
| h·g_T / h·g_D, early | +3.4 (p = .03, q = .08) / +7.0 (p = .02, q = .05) | +3.0 (p = .06, q = .18) / +3.9 (p = .07, q = .21) |
| Base target / distractor gain | +21% [14, 28] / +11% [4.8, 16] | +26% [14, 35] / +11% [3.9, 18] |
| Fits with a gain, h or a on a bound | 28 of 241 (±10 bounds) | 33 of 241 (zero-sum bounds) |
| Without them | h −11% (p = .13), a +0.6% (p = .65) | h −9.6% (p = .11), a +0.7% (p = .68) |
| Fold means (run3) | h +5.6% (p = .48), a −2.3% (p = .81) | h −10% (p = .49), a −1.5% (p = .73) |

**Wording.** The conclusion is unchanged (neither form is supported; both predict held-out runs worse than no HP term
on the item gains; h and a do not differ from 0). One sentence changes: the two forms no longer fit the training runs
equally well. The multiplicative one now fits them slightly better (+0.20 × 10⁻⁴, p = .010) and predicts held-out runs
worse (−0.69, 23 of 28 participants favour additive), so the difference is overfitting of the multiplicative term,
which says even less about how the brain combines the HP change. Tables:
`notes/data/hp_item_gain_mult_vs_add_zs_lm_2026-11-16_*` (per area in `_tables.md`).

## Main residual-RT analysis (rt_power_plan, steps 1 + 2 + 3)

Trial-wise gains regenerated with residualised RT on the zero-sum main model held fixed
(`derivatives/af_trialwise_gains_signal_se_residrt_zs_lm_2026-11-16`, signal voxels, jackknife SEs, groups).
`rt_power_meta.py`, nutpie, 4 chains, main 6000 + 6000 draws, baseline 2000 + 2000, target_accept .99, as originally.
Gain × 100 per SD of (residual) log RT, posterior mean [95% HDI], P = posterior probability of the sign shown; bold =
credible (HDI excludes 0).

| Analysis | Term | Earlier | Zero-sum |
|---|---|---|---|
| Baseline, z(log RT) | target | +0.9 [−0.9, +2.9], P = .843 | +0.8 [−1.4, +2.9], P = .787 |
| | distractor | **−2.0 [−3.7, −0.2], P = .986** | **−2.7 [−4.6, −0.7], P = .995** |
| | global | −1.9 [−4.5, +0.5], P = .930 | −2.2 [−5.0, +0.7], P = .944 |
| **Main (1 + 2 + 3)** | target | +1.0 [−0.8, +2.9], P = .863 | +0.8 [−1.2, +3.0], P = .794 |
| | distractor | **−1.6 [−3.3, −0.04], P = .977** | **−2.4 [−4.3, −0.6], P = .994** |
| | global | **−2.6 [−5.1, −0.04], P = .976** | **−2.8 [−5.5, −0.2], P = .982** |
| Main, groups: global | early / dorsal / occ.-temporal | −3.2 / −0.4 / −3.2 [−6.1, +0.02] | −3.1 / +0.1 / **−4.2 [−8.1, −0.3]** |
| Main, groups: distractor | early / dorsal / occ.-temporal | −2.1 / −2.5 / −0.8 | −2.9 [−5.9, +0.2] / −3.4 [−7.0, +0.0] / −1.5 |
| Main, groups: target | early / dorsal / occ.-temporal | +2.2 / −1.1 / +1.2 | +2.2 / −2.1 / +1.3 |

Diagnostics, main: 0 divergences, max R̂ 1.00, min bulk ESS 2125 (earlier 0, 1.00, 1386); posterior predictive 95%
intervals cover 93–98% of estimates and 9/9 area means (earlier 95–98%, 9/9). Baseline: 0 divergences, max R̂ 1.01,
min bulk ESS 1021; area means 8/9 inside for the target term.

**Wording.** Unchanged in substance: distractor credible in baseline and main; global credible only under the main
analysis (baseline P = .944), so the plan's rule still applies. The suggested sentence becomes: "With RT residualised on
the design and the three gains in one model, the global gain is also lower on slower trials (−2.8% [−5.5, −0.2] per SD;
with z(log RT) −2.2%, P = .94)." Checks 4 (quintiles) and 5 (r > 0.2 selection) were not rerun; the clause "but this
does not survive a looser voxel selection" refers to the earlier fits and should be dropped or marked as such.

Files: `notes/data/trialwise_signal_se_residrt_zs_lm_2026-11-16/`, `notes/data/rt_power_zs_lm_2026-11-16/` (baseline),
`notes/data/rt_power_zs_lm_2026-11-16_main6000/` (main); baseline input = the update-13 Fig. 7 table
`notes/data/trialwise_signal_se_zs_lm_2026-11-13/`.

## Joint latent model of single-trial log RT (addendum 6)

Per-trial gains with circular-shift placebo covariance regenerated on the zero-sum main model
(`derivatives/af_trialwise_trialse_zs_lm_2026-11-16`, 200 shifts, nine areas, signal voxels), pooled as before
(`collect_trialse`). Same model, priors and sampler (PyMC + NumPyro, 4 × 1000 draws after 1000 tuning, target_accept
0.9).

| Gain | Earlier: log RT per SD of latent gain, ms, P | Zero-sum | Earlier: gain × 100 per SD of log RT | Zero-sum |
|---|---|---|---|---|
| Target | +0.015 [−0.009, +0.042], +13 ms, P(+) = .878 | +0.016 [−0.011, +0.044], +14 ms [−9, +40], P(+) = .879 | +0.45 [−0.83, +1.73] | +0.32 [−1.06, +1.86], P(+) = .665 |
| Distractor | −0.009 [−0.036, +0.017], −8 ms, P(−) = .763 | −0.012 [−0.038, +0.015], −10 ms [−32, +13], P(−) = .814 | −0.19 [−1.46, +1.15] | −0.37 [−1.70, +1.09], P(−) = .706 |
| **Global** | **−0.017 [−0.034, −0.001], −15 ms [−30, −1], P(−) = .985** | **−0.020 [−0.036, −0.004], −17 ms [−31, −4], P(−) = .995** | **−2.40 [−3.70, −1.01]** | **−3.45 [−4.92, −2.06], P(−) > .999** |

- Condition effects (ms): HP − LP distractor −11.6 [−19.7, −4.1]; LP − absent +35.0; HP − absent +22.8; target at HP
  +14.6; target repeat −33.5; distractor repeat −10.6 (earlier −11.6, +34.9, +22.8, +14.8, −33.4, −10.7).
- Latent reliability τ²/(τ² + SE²): target 0.030, distractor 0.018, global 0.19 (earlier 0.03, 0.02, 0.15).
- Convergence: 1 divergence, max R̂ 1.01, min bulk ESS 318, min tail ESS 746 (earlier 0, 1.01, 379, 789).
- PPC: Fig. 1 contrasts all inside their 95% bands (HP − LP −26.6 observed vs −25.7 [−33.6, −16.3]); participant ×
  condition cells 97.6% inside (earlier 97.6%); the right skew of RT is missed as before (median and 90th percentile
  inside in 1/5 cells). The observed distractor-gain mean and SD per cell are inside in only 2/5 cells (target 4/5,
  global 5/5); earlier the gain moments were inside in 24 of 26 cell × moment checks.

**Wording.** Unchanged: trials with a higher global gain are answered faster (now about 17 ms per SD of latent global
gain); target and distractor gains are too unreliable per trial for this model. Not rerun: Bambi cross-check, Σ → 0
check, pathway fits and parameter recovery (they validate the model, which is unchanged).

Files: cluster `~/retsupp_plan/notes/data/rt_joint_latent_zs_lm_2026-11-16/` (trace `main.nc` and the 27 MB pooled
table, not committed); tables in `notes/data/rt_joint_latent_zs_lm_2026-11-16/`.

## What could not be rerun as before

- **Item 2, Table 2d (SDT/SDT0) and item 1a/1c's dT model in an LM fit:** there is no zero-sum fit with the target
  transient term (m8_full_dT exists only on the earlier, non-zero-sum fits). Items 1a/1c include dT through the linear
  refits, as before; Table 2d is not reproduced.
- **Items 1a/1c base:** the zero-sum model has no sustained level shared by the four locations, so the base has three
  terms, not four ("sustained gain at all four locations" is gone). The linearisation point is the main model with its
  sustained field at 0, i.e. the main model's pRFs and widths (earlier: m4_TDS's).
- **Item 3a bounds:** the shared fits keep the zero-sum constraint and use the zero-sum bounds (transient gains −1 to 5,
  item scale ≥ 0.05) instead of ±10.

## Code and runs

- `retsupp/modeling/hp_terms_heldout_power.py --zero-sum --ref-root <view root>`; stats
  `retsupp/visualize/paper/hp_terms_heldout_power_stats.py --tag zs_lm_2026-11-16 --zero-sum`. SLURM 8413875 (28 tasks).
  Data: `derivatives/hp_terms_heldout_power_zs_lm_2026-11-16`; `notes/data/hp_terms_*_zs_lm_2026-11-16.*`.
- Items 2/5b: `export_location_timeseries --root <view root>` (8413876) → `predict_location_timeseries_polished.py
  --sd-model m7_full_zs` with the env-configurable job (8413877); stats `items_model_vs_measured.py --tag
  zs_lm_2026-11-16 --no-sdt ...` (main-model version of 5b: `--tag zs_lm_mainS_2026-11-16`).
- Item 3a: `fit_cv_ladder_nested.py --prf-source-root <view root> --prf-source-zero-sum` (run3 8414003, all runs
  8414004; roots `..._polprf_zs_lm`); stats `collect_hp_item_gain.py --zero-sum`.
- RT: `fit_trialwise_af_gains` with `RT_MODULATOR=residual` (8413770), `fit_trialwise_af_gains_trialse` (8413771),
  joint latent main 8413966, report 8413967.
- All defaults of the changed scripts reproduce the earlier runs; every output has a new `zs_lm` name.
