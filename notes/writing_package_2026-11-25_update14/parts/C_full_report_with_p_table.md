# Follow-up numbers for Docky's edits (24 to 25 Nov 2026)

Source manuscript: Google Doc "Retsupp manuscript v2 (for Docky, 13 Nov 2026)", tabs v2 and "dockys edits" (read only, not edited). n = 28 (sub-03 to sub-30). Pooled values: per participant the median over the nine areas, then median [bootstrapped 95% CI], participants below/above 0, Wilcoxon p; per area BH q over nine areas. Trial-wise effects: posterior mean [95% HDI], P = posterior probability of the sign.

**Important for every item:** both manuscript tabs still quote the gain-field numbers of updates 8 to 11. Update 13 (zero-sum refit, `/Users/gdehol/git/retsupp/notes/writing_package_2026-11-16_update13/UPDATE_NOTES.md`) replaces them. Current values are given first below.

## 1. Size of the HP reduction ("the gain was reduced about 20%")

A gain is already expressed in % of the drive at the centre of the field, so dS is in % of drive as it stands. Under the zero-sum field, g_HP = ¾ dS and g_LP = −¼ dS.

| Quantity (update 13, zero-sum main model, n = 28) | Pooled: median [95% CI], k/28 < 0, p |
|---|---|
| dS, joint fit | −5.5% [−7.6, −2.1], 21/28, p = .012 |
| dS, estimated alone | −1.5% [−2.4, −0.9], 21/28, p = .0071 |
| dS / target gain, ratio of medians (joint) | −0.24 [−0.43, −0.10] (manuscript: −0.20 [−0.45, −0.07], from the earlier, non-zero-sum fits) |
| dS / target gain, per participant (cells with a reliable target gain) | −0.08 [−0.16, −0.06], 23/27, p = .001 |
| Decoded bar, HP quadrant minus the other three (all nine areas) | −1.10% [−1.52, −0.34], 21/28, p = .0033 one-sided (.0065 two-sided); mean ± SEM −1.14 ± 0.39% as in the text |

dS per area, median (BH q over nine areas):

| Area | Joint | Alone (uncorrected p) | Decoded bar, median [95% CI], n |
|---|---|---|---|
| V1 | −8.1% (q = .16) | +0.1% (p = .90) | −0.37 [−1.43, +0.54], 28 |
| V2 | −7.8% (q = .16) | −2.8% (p = .14) | −1.48 [−2.77, −0.68], 28 |
| V3 | −10.2% (q = .13) | −4.8% (p = .013) | −1.69 [−3.04, 0.00], 28 |
| V3AB | −1.8% (q = .77) | −0.8% (p = .59) | −1.99 [−4.12, +1.02], 23 |
| IPS | −1.2% (q = .77) | −0.6% (p = .37) | +1.45 [−2.01, +5.13], 17 |
| hV4 | −8.2% (q = .022) | −2.5% (p = .13) | −1.35 [−3.80, +1.82], 27 |
| VO | −2.2% (q = .62) | −0.9% (p = .32) | −1.06 [−9.20, +0.10], 10 |
| LO | −1.2% (q = .77) | +1.0% (p = .60) | +0.10 [−2.30, +2.87], 27 |
| TO | −2.8% (q = .14) | −1.2% (p = .090) | −1.09 [−2.35, +0.58], 25 |

Decoded bar per pathway: early −1.35% [−2.11, +0.02], p = .010; occipitotemporal −1.77% [−2.58, +0.37], p = .0064; dorsal −0.90% [−2.47, +3.05], n = 24, p = .56 (one-sided). Computed from `/Users/gdehol/git/retsupp/notes/data/decoding_readout_jointhrf_2026-11.tsv` with `fig6_decoding.pathway_effects` (reproduces the text's mean ± SEM exactly).

**What "about 20%" can refer to.** Not the gain. The gain at HP is 1.5% (alone) to 5.5% (joint) of the drive lower than at LP. Two quantities are near 20%:
- the ratio of the HP reduction to the target gain: "a fifth" (−0.20) on the earlier fits, now "about a quarter" (−0.24);
- the measured item response in early visual cortex, 21.6% lower at HP than opposite HP (relative units, Fig. 3e; −15.5% pooled over all areas). This is a response, not a gain, and is not from the gain-field model.

Suggested abstract wording: "the gain at the high-probability location was reduced by a few percent of the drive, about a quarter of the gain at the target". Also note: "the salient distractor received about half this gain" still holds on update 13 (target +23%, distractor +11%).

**Note for all items.** Both manuscript tabs (v2 and "dockys edits") still carry the pre-update-13 gain-field numbers (target +17%, distractor +8.0%, dS −1.0% / −3.5%, ratio −0.20, distractor-RT −2.0%). Update 13 (`/Users/gdehol/git/retsupp/notes/writing_package_2026-11-16_update13/UPDATE_NOTES.md`) replaces them. Below I give the current values first.

## 2. Divisive vs subtractive

What we have:

| Test | Result | Fits |
|---|---|---|
| Multiplicative (h) vs additive (a) HP change on the transient item gains (update 11, 3a) | Run to completion on the zero-sum model (all 241 cells; nothing to rerun). Neither form supported: h −9.6% [−33, +8], 12/28 > 0, p = .19; a +0.8% [−7.6, +8.6], 16/28, p = .48. Both predict held-out runs worse than no HP term on the item gains (mult − base −1.90 × 10⁻⁴, 0/28 better, p = 7.5 × 10⁻⁹; add − base −1.00 × 10⁻⁴, 3/28, p = 1.3 × 10⁻⁶). Mult fits training runs slightly better (p = .010) and held-out runs worse than add (23/28, p = 1.4 × 10⁻⁴): overfitting, not evidence for either. | zero-sum (update 13, part G) |
| DN baseline variants (Supp. Fig. S12): sustained field acting on DN b and/or d instead of the input | Predict held-out runs worse than the input-gain model in all participants (p = 7.5 × 10⁻⁹ at n = 28; ΔR² about −3 × 10⁻⁵ per variant). b-only and d-only barely move (median \|w\| 0.004); b+d moves along w_b = w_d (corr 0.999), which trades off with the input gain. | earlier DN fits (not zero-sum) |
| Parameter recovery of the DN variants | At the fitted effect size, simulated b or d baselines are recovered as input gain in 72% of simulations (b correct 19%, d correct 12%); only at 4× the size are they detected, and then b and d are confused. Input and null truths stay input (94%, 97%). So held-out R² cannot tell a DN-baseline effect from an input gain at our effect size. | earlier DN fits |
| Slope vs intercept (Supp. Fig. S13, step 1) | The bar-response slope is lower where the bar crosses HP: −2.1% [−4.6, −0.7], 20/28, p = .012 (vs next to HP −3.5%, p = .0011), about the size of the fitted dS. A sustained additive offset at HP is not identifiable: each run is in % of its own mean and HP is constant within a run. | earlier fits |
| Trial-locked additive term at HP (S13, steps 2 and 3) | Locked to search onset: amplitude +1.5% signal [+0.0, +2.9], 19/28 > 0, p = .010 (positive, not a suppression); held out +0.13 × 10⁻⁴, 16/28 better, p = .41. Locked to fixation: −0.7% [−1.0, +0.9], p = .85. Recovery: a matched trial-locked baseline is found in 10/10 simulated data sets and never taken for a gain; the gain at the group size is found in 6/10. | earlier fits |

**Conclusion.** The data show a gain signature (a shallower bar response at HP, matching dS) and rule out a trial-locked subtractive response at HP, but a block-sustained subtractive offset cannot be measured in this design, and divisive-normalization baseline effects of our size are recovered as input gain 72% of the time. So the data favour a gain over a trial-locked subtraction, but they do not distinguish a divisive from a sustained subtractive mechanism; "divisive rather than subtractive" overstates it, and "a change in gain, with no trial-locked subtractive component" is what we can claim.

## 3. Gradient across the LP locations (next to HP vs opposite HP)

**What exists.** The variant is `m8_full_adjopp` (`/Users/gdehol/git/retsupp/retsupp/modeling/lp_split.py`, `--models m8_full_adjopp` in `/Users/gdehol/git/retsupp/retsupp/modeling/fit_cv_ladder_nested.py`). It adds one term to the sustained field, δ = g_next − g_opp, with g_LP kept as the mean of the three LP gains (g_next = g_LP + δ/3, g_opp = g_LP − 2δ/3). So it allows HP < next < opposite, or any other order.

**Caveat: it was fitted only on the earlier fits.** It was fitted on top of the earlier m7_full (the "final2" ladder `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf`): not zero-sum, and with dD in the parent. It was fitted on all runs only (one fold), 234 of 252 subject × area fits. Its rows in `/Users/gdehol/git/retsupp/notes/data/gain_field_parameters_zs_lm_2026-11-13.tsv` are the same final2 fits copied over. There is no zero-sum version and no held-out (CV) fit. Per item 3 I did not refit.

**Estimates (earlier fits, n = 28).** Gains in %, per participant median over the nine areas, median [bootstrapped 95% CI], participants < 0, Wilcoxon p. Per area: median, BH q over the nine areas.

| Contrast | Pooled | Per area (q < .05) |
|---|---|---|
| Next − opposite (δ) | −0.6 [−3.2, +2.0], 15/28 < 0, p = .73 | none (all q ≥ .55; V1 +8.3, V2 −3.5, V3 +6.3, V3AB −0.5, IPS −4.4, hV4 −2.1, VO +0.6, LO −0.1, TO +0.1) |
| HP − next | −1.6 [−6.9, +0.5], 18/28, p = .070 | none (V3 −12.5, p = .0067, q = .061) |
| HP − opposite | −3.3 [−5.2, −1.9], 23/28, p = .0048 | hV4 −5.7, q = .0043 (V2 −8.5, q = .082; TO −2.5, q = .070) |
| dS (HP − mean LP) in this model | −2.0 [−5.9, −0.5], 19/28, p = .019 | V3 −8.4, q = .041 |

The fitted order is HP lowest. The two LP classes do not differ (next − opposite about 0, 15 of 28 participants below 0). The data therefore do not support a gradient HP < next < opposite in the sustained gain. Only HP < opposite is reliable.

**Held-out R² against the shared-LP model.** Not available: the variant was never fitted in a CV scheme.

**Does it reproduce the measured next-vs-opposite difference (−14.9 ± 6.0%, early areas, Fig. 3e)?**
- Not tested directly: the variant's predictions were never run through the Fig. 3 pipeline.
- With a fitted δ of about 0, it cannot produce a −15% item-response difference. For comparison, the shared-LP model (earlier m7_full) predicts −1.3 ± 0.8% (p = .16), and the DoG centre/surround tilt κ adds +0.27% (`/Users/gdehol/git/retsupp/notes/richter_followup_2026-11-11.md`, part C).

**Does it reproduce the decoded next-vs-opposite difference (+0.68 ± 0.32%, p = .070, Fig. 6f)?**
- Not tested: there is no decoder simulation with this variant.
- The two measured patterns point in opposite directions. In the item responses the locations next to HP are *below* opposite (−14.9%). In the decoded bar the quadrants next to HP are *above* opposite (+0.68%). A single sustained δ cannot produce both, and the fitted δ is about 0, between the two.
- Both measured next-vs-opposite effects are themselves weak (items p = .022 in early areas only, with no area q < .05; decoding p = .070).

**Answer for Docky.** The variant he asks for exists and was fitted (earlier fits). It finds no gradient between the LP locations (next − opposite −0.6%, p = .73). Its gain is lowest at HP, and only HP vs opposite is reliable (−3.3%, p = .0048). Keeping one shared LP gain is therefore justified. The Discussion paragraph that leaves "where the gradients over the LP locations come from" open stays correct.

**If you want it on the zero-sum model.** Neither option has been submitted.
1. **Cheap linear estimate** (δ added alone to the zero-sum main model, pRFs and widths held):
   `fit_fixed_group_mechanisms.py <sub> --mechanisms adjopp --base-root <zero-sum view root ..._zs_lm_ref> --hrf-table <same table> --output-subdir af_fixed_group_adjopp_zs_lm`
   via `retsupp/modeling/slurm_jobs/fit_fixed_group_mechanisms.sh`, 28 tasks. Comparable job 8262606: about 5 min per task, about 5 CPU-h in total. Needs a sub-03 canary first, because adjopp reads the view root's "m7_full", which there is the zero-sum main model without dD.
2. **Full fit with held-out R².** Needs a small code change: allow `m8_full_adjopp` in `PRF_SOURCE_MODELS` and parent it on `m5_TDS_dS` under the zero-sum bounds. It would then run like item 3a (`PRF_SOURCE_ROOT=<view root> PRF_SOURCE_ZERO_SUM=1 BOUNDED=1 MODELS="m5_TDS_dS m8_full_adjopp"`, `CV_SCHEME=run3` and `CV_SCHEME=full`, 241 tasks each). Cost scaled from item 3a, whose jobs fitted four models: 8414003 (run3) used 705 CPU-h and 8414004 (all runs) 381 CPU-h, so about 180 + 100 CPU-h. A Fig. 3 and decoder prediction from that fit would then answer both reproduction questions.

## 4. Target gain lower on location repeats: priming or repetition suppression?

Model: the update-13 trial-wise model on the zero-sum main model (signal voxels, n = 28, nine areas,
hierarchical Bayesian meta-analysis; posterior mean, 95% HDI, P = posterior probability of the sign).
Each item gain (target, distractor, global, and the neutral-item terms below) carries all trial modulators at once:
z(log RT), target repeat, distractor repeat and target at the previous distractor location.

**Reference (Fig. 7a).** Target gain on target-repeat trials, relative to the area's typical target gain:
manuscript (earlier fits) −18% [−31, −4]; current zero-sum fits **−19% [−29, −9], P > .999**
(absolute −6.4 points [−10.1, −2.7]).

**(c) RT is in the same model.** Yes: the target-repeat term is estimated together with z(log RT) and the other
history terms (`MODS` in `/Users/gdehol/git/retsupp/retsupp/modeling/fit_trialwise_af_gains.py`). The Fig. 7a
number comes from the design without the global gain; with the global gain also in the model the effect is
−14% [−24, −5], P = .998 (absolute −5.2 [−8.8, −1.4]). So the repeat effect is not an RT effect in disguise.

**(b) Target at the previous distractor location** (already in the model): no credible change.
Absolute −1.8 points [−6.5, +3.3], P(−) = .78; relative +1.1% [−11, +14]. Per pathway: early +23% [+1, +48]
(relative; absolute +4.7 [−2.4, +11.3], P(+) = .92), dorsal −10% [−31, +14], occipitotemporal −10% [−25, +4].

**(a) Neutral items at a repeated location.**
- Neutral item at the **previous distractor location** (regressor P, already in the model): gain lower,
  **−4.1 points [−7.6, −0.6], P = .987**; same sign in every area (V1 −5.9 [−10.7, −0.8], V2 −4.9 [−9.7, −0.1],
  others −2.5 to −4.8, HDIs include 0).
- Neutral item at the **previous target location** (new regressor Q, `--prev-target-neutral`, SLURM 8796023): gain
  also lower, **−6.5 points [−10.7, −2.2], P = .996** (Student-t −6.1 [−10.3, −2.1], P = .998). It is negative in
  every area, credibly so in V3 −7.7, V3AB −9.8, hV4 −11.6, LO −7.0 and TO −5.8, but not in V1 −3.1 or V2 −2.9.
  Per pathway: early −4.5 [−11.8, +2.9], dorsal −8.0 [−16.4, +0.8], occipitotemporal −7.4 [−14.2, −1.0].
  Check: the area means fall inside the predictive band in 7 of 9 areas, 0 divergences.
- With Q in the model, the target-repeat effect is larger: −23% [−33, −13], P > .999 (early −43%, dorsal −28%,
  occipitotemporal −6%). The neutral-item term at the previous distractor location stays at −4.7 [−8.1, −1.1],
  P = .994.

**Interpretation.** The pattern favours a location-level after-effect, not priming as raised priority. The
location that held the target on the previous trial gives less gain to whatever appears there next: a repeated
target loses gain, and so does a neutral item. The location of the previous distractor also gives less gain to a
neutral item. If priming had kept the previous target location at a higher priority, a neutral item there should
gain, not lose. Repetition suppression in the strict sense (adaptation to the same stimulus) also does not fit
well, because the neutral item there is a different stimulus from the target that preceded it. The best
description is a reduced gain at locations that held a singleton on the previous trial, whatever appears there now
(adaptation or inhibition of the recently selected location). That target repeats are nonetheless faster (−38.7
ms) means the behavioural benefit does not come from more gain at the repeated location. The data cannot tell
whether it comes from a lower gain needed to select the target or from processing outside visual cortex.
Suggested wording: "after a location had held the target or the distractor, it received less gain on the next
trial, whatever item appeared there".

## 5. Distractor gain and RT

**(a) Sign.** Distractor gain is lower on slower trials. Manuscript (earlier fits) −2.0% per SD of log RT
[−3.7, −0.2], P = .990; current zero-sum fits **−2.7 [−4.7, −0.7], P = .996**; residualised-RT main analysis
−2.4 [−4.3, −0.6], P = .994.
- Abstract, "distractor gain was higher on fast response trials": same direction, matches.
- Discussion, "reaction times were reliably lower when distractor gain was high": direction matches, two issues.
  (1) "Reliably" overstates it: the effect is SE-weighted and vanishes without the weighting (earlier fits:
  P = .603), weakens with looser voxel selections (5b), and in the joint latent model, which models RT from the
  gains, the distractor term is not credible (−10 ms per SD of latent distractor gain [−32, +13], P = .81); only
  the global gain is (−17 ms [−31, −4], P = .995). (2) "After regressing out the sustained suppression": the
  model does not regress out the sustained field. It holds the participant's full gain-field fit (including the
  sustained HP − LP field) fixed and z-scores log RT within session × run × distractor condition. The closest
  analysis is the residualised-RT one (RT regressed on distractor condition, target at HP, repeats, trial
  position and run; −2.4%, P = .994). Suggested wording: "distractor gain was lower on trials with slower
  responses".

**(b) Looser voxel selections.** Rerun on the zero-sum main model (SLURM 8796024 and 8796025): its gain fields
are held fixed, the voxels come from the cross-run reliability selection, and the added voxels take their pRFs from
the joint-HRF fold fits. Distractor gain per SD of log RT:

| Selection | Zero-sum (now) | Earlier fits (gain fields refitted per selection) |
|---|---|---|
| Signal voxels (Fig. 7) | −2.7 [−4.7, −0.7], P = .996, 230 estimates | −2.0 [−3.7, −0.2], P = .990 |
| r > 0.2 | **−2.5 [−4.5, −0.6], P = .994**, 236 | −1.5 [−3.0, −0.0], P = .975 |
| r > 0.1 | **−1.9 [−3.7, +0.0], P = .978**, 241 | −1.3 [−2.9, +0.1], P = .960 |

The effect weakens as the selection loosens but stays just credible at r > 0.1 on the zero-sum fits. Without the
SE weighting it is not credible in any selection (r > 0.2 +5.4, P(−) = .19; r > 0.1 +1.8, P(−) = .32). The
target-repeat effect holds in both selections (−19% and −20%, P > .999). The manuscript's clause "became weaker
with looser voxel selections" still holds in direction, but on the current fits the effect stays credible. The two
rerun columns are not identical procedures, because the earlier fits refitted the gain fields on each selection.

**(c) The earlier "opposite relations" result.** Found: October 2026 (commits 56a335e to 4e73889, 19 to 20 Oct;
script `/Users/gdehol/git/retsupp/retsupp/visualize/paper/mixed_trialwise_af_rt.R`, table
`/Users/gdehol/git/retsupp/notes/data/trialwise_af_mixed_stats.tsv`, figure
`/Users/gdehol/git/retsupp/notes/figures/trialwise_af_gains_rt_mixed.pdf`). Linear mixed models of single-trial
log RT on per-trial (ridge) gains, n = 30, 11 areas. Standardised slopes of log RT on the gain:
- Distractor gain: larger = faster in V1 (β −0.029, p = .015, q = .055) and V2 (−0.035, p = 6 × 10⁻⁴,
  q = .006); larger = slower in SPL1 (+0.040, p = .006); early group −0.028 (p = .012), dorsal group +0.015
  (p = .36).
- The robust flip was in the overall trial amplitude (mean of the gains), not the distractor gain: V1 −0.041
  (q = .003), V2 −0.032 (q = .019) versus IPS +0.085, SPL1 +0.072, FEF +0.116 (all p < 10⁻⁶).
- Caveat recorded then: in IPS, SPL1 and FEF the per-trial target, distractor and global gains correlate
  r = .92 to .98 within participants, so a distractor-specific sign there was not identifiable.

What changed since:
- Model: October used the canonical dynamic-AF fit (bar + singleton paradigm, pRFs fixed from the mean fit, shared
  σ). Now the zero-sum gain-field model (8-item paradigm, item weight, pRFs refitted jointly by LM, HRF per area).
- Areas and voxels: SPL1 and FEF (where the positive relations were) are no longer analysed; nine areas, signal
  voxels instead of the AF fit's p_signal voxels; n = 28 (sub-01, 02 out).
- Direction of the regression and estimator: RT on noisy per-trial gains (attenuated, collinear) versus the gain
  modulated by z(log RT) on the BOLD, with run-jackknife SEs and the global gain in the same model, which absorbs
  whole-display amplitude changes (the October dorsal "slower" effect was mainly that amplitude).
- Sustained gain: in both versions the sustained field is held fixed; nothing was "regressed out" in addition.

Per-area posteriors, current model (distractor gain per SD of log RT, ×100, hierarchical model):

| Area | Mean [95% HDI] | P(−) |
|---|---|---|
| V1 | −2.8 [−5.7, +0.2] | .967 |
| V2 | −3.3 [−6.2, −0.3] | .987 |
| V3 | −2.5 [−5.7, +0.7] | .933 |
| V3AB | −3.4 [−6.8, −0.5] | .984 |
| IPS | −3.7 [−7.2, +0.0] | .978 |
| hV4 | −2.4 [−5.3, +0.4] | .959 |
| LO | −2.0 [−4.8, +0.6] | .932 |
| TO | −1.8 [−4.3, +0.7] | .932 |
| VO | −2.2 [−4.8, +0.4] | .954 |

Without partial pooling (per-participant medians, Wilcoxon), only V3 has a positive median (+1.9, p = .52);
V1 −5.6 (p = .022), V2 −4.2 (p = .026), others −0.0 to −4.2 (p ≥ .099). No sign reversal between early and late
areas in the current model.

## 6. Figure 1c: absent vs LP and HP vs absent strips

Already done in Fig. 1 v6 (update 13, commits 076901b and c3e72b5), which has three within-subject strips beside c: HP − LP, LP − absent and HP − absent (plus brackets with stars over the raw RTs). The manuscript's Fig. 1 is still the older version with fewer strips, so the fix is to swap in v6, not a new render. No v7 was made, because it would be identical. Sign convention follows the existing strips (LP − absent, the distractor cost; absent − LP is the same with the sign flipped).

| Strip | Median [95% CI] | n | p (Wilcoxon) |
|---|---|---|---|
| HP − LP | −25.0 ms [−32.4, −20.0] | 28/28 faster | 7.5 × 10⁻⁹ |
| LP − absent | +42.6 ms [+27.0, +50.7] | 26/28 slower | 1.0 × 10⁻⁷ |
| HP − absent | +14.6 ms [+0.3, +21.7] | 19/28 slower | .0041 |
| Target HP − LP (panel d) | +24.3 ms [+18.5, +36.4] | 24/28 slower | 1.5 × 10⁻⁶ |

Figure: /Users/gdehol/git/retsupp/notes/figures/paper_2026-11/fig1_paradigm_behaviour_v6_n28_palA.pdf (also .png, legend .md). Script: /Users/gdehol/git/retsupp/retsupp/visualize/paper/fig1_paradigm_behaviour_v6.py.

## 7. Figure 5a with HP on a diagonal search position

New option `--hp-angle DEG` in /Users/gdehol/git/retsupp/retsupp/visualize/paper/fig5_gain_fields_v6.py (default 0 = HP at the top, output unchanged). It rotates the four ring locations (and therefore every field, which is defined relative to them) and the eight item markers together, so it is a pure display rotation: all values, scales and panels b and c are identical. With HP at the upper right (45°), the four possible target and distractor locations now sit on the diagonals as in the real display, the target is drawn opposite HP (lower left) and the distractor at the LP location next to HP (upper left). The bold dS value is placed above the HP ring in the rotated version so it stays inside the map.

- Current: /Users/gdehol/git/retsupp/notes/figures/paper_2026-11/fig5_gain_fields_v11_abc_zs_n28_palA.pdf
- Rotated: /Users/gdehol/git/retsupp/notes/figures/paper_2026-11/fig5_gain_fields_v11_abc_zs_n28_palA_hpdiag.pdf (and .png; legend .md with "HP at the upper right, one of the four diagonal positions" instead of "at the top")

Check: re-rendering with the default reproduced the published panel a pixel for pixel in the maps (only tiny anti-aliasing differences in axis labels). Fig. 4a already draws HP at a diagonal (lower right), so it needs no alternative. Fig. 6 (decoding) also puts HP at the top; it was not rotated here.

## 8. Supplementary videos

Both are feasible, and a short draft of each is made (MP4, H.264, `-preset veryfast`, yuv420p). Script:
`/Users/gdehol/git/retsupp/retsupp/visualize/paper/supp_videos.py` (`model` or `decoding`; runs locally in about 15 to 30 s).

**(a) Animated Fig. 4b.** `/Users/gdehol/git/retsupp/notes/figures/videos/suppvideo_fig4b_model_draft.mp4`
(12 s, 145 frames at 12 fps, covering 58 s of run time at 0.4 s steps, 1200 × 540).
- It animates the current schematic Fig. 4b (`fig4_model.main_hp`), with the same illustrative, exaggerated gains: one
  pRF near a ring location, in a block with HP near it and in a block with HP opposite; the same bar sweep and four
  trials in both.
- Per block: on the left the effective drive on screen (model input × gain field, with the pRF and the HP ring), on
  the right the stimulus in the pRF, the gain it sees and the predicted BOLD with and without gain, revealed up to a
  moving time cursor.
- Not done: a version with fitted values (one participant, area and voxel of the zero-sum main model). It needs the
  fitted pRF and gain terms of one cell from the view root `..._zs_lm_ref` on the cluster plus that run's design;
  the forward pass is cheap, but the code path that rebuilds the model input and HRF for one cell has to be found in
  the ladder code (`fit_cv_ladder_nested.py`/`joint_lm`). About half a day. With real gains (a few percent for dS)
  the sustained effect is invisible by eye, so the schematic is likely the better video; a fitted version would mainly
  show how well the prediction follows the measured time course.

**(b) True vs decoded bar, V1.** `/Users/gdehol/git/retsupp/notes/figures/videos/suppvideo_decoded_bar_V1_draft.mp4`
(16 s, 130 frames at 8 fps = TRs 0 to 129, about 208 s of the run, 1080 × 436).
- Three discs per frame: the true bar, the bar decoded from V1 in the held-out run of Fig. 6a (sub-03, ses-1 run-4,
  unconstrained decoder, pRFs from the other runs, `jointhrf` source), and the decoded mean over participants (V1).
- From local files only (`notes/data/decoding_frames_jointhrf_2026-11.npz`,
  `notes/data/decoding_example_run_jointhrf_2026-11.npz`), so no cluster work. The full run (258 TRs) is one flag
  (`--tr-stop 258`, about 32 s of video).

Both are drafts: no captions or scale bars yet, and colours follow Fig. 4 and Fig. 6a.

## 9. p and q values in the Results

All 69 p and q values in the Results text and figure legends of tab v2 (exact values, 13 Nov), with the threshold form, what Docky's tab currently shows, and the update-13 value where the refit changes it. Star legends in captions are left out. Notes:
- Docky's tab has one typo (row 36, "p ==< 0.05") and otherwise converted correctly to thresholds.
- Rows 44 to 50 (Gaussian and DN pRFs) and all behaviour, item-response and decoding rows are not affected by update 13.
- Fig. 7 (trial-wise) reports posterior probabilities, not p values, so it is not in the table.
- Row 16 vs row 17: the early-pathway target − distractor test is q = .015, so "all q < .001" cannot be applied to all three pathways.

| # | Section | Context (text just before the value) | Exact (v2, 13 Nov) | Threshold form | Docky tab now | Update 13 value (if it changes) |
|---|---|---|---|---|---|---|
| 1 | Behaviour | …r; median difference +42.6 ms, 95% CI [+27.0, +50.7], 26 of 28 participants | p = 1.0 × 10⁻⁷ | < .001 | p < 0.001 |  |
| 2 | Behaviour | …01 ms; −25.0 ms, 95% CI [−32.4, −20.0], faster at HP in all 28 participants | p = 7.5 × 10⁻⁹ | < .001 | p < 0.001 |  |
| 3 | Behaviour | …without a distractor (+14.6 ms, 95% CI [+0.3, +21.7], 19 of 28 participants | p = .0041 | < .01 | p < 0.01 |  |
| 4 | Behaviour | … compared with an LP location, 95% CI [+18.5, +36.4], 24 of 28 participants | p = 1.5 × 10⁻⁶ | < .001 | p < 0.001 |  |
| 5 | Behaviour | …fference +2.3 percentage points, 95% CI [+1.3, +4.4], 24 of 28 participants | p = 6.7 × 10⁻⁵ | < .001 | p < 0.001 |  |
| 6 | Behaviour | …e to no distractor (−5.2 points, 95% CI [−6.1, −3.7], 26 of 28 participants | p = 1.0 × 10⁻⁵ | < .001 | p < 0.001 |  |
| 7 | Behaviour | …ion (83.5 vs 86.2%; −2.0 points, 95% CI [−3.5, −1.3], 22 of 28 participants | p = .0011 | < .01 | p < .01 |  |
| 8 | Behaviour | … the previous trial (−38.7 ms, 95% CI [−48.8, −28.4], 26 of 28 participants | p = 3.7 × 10⁻⁸ | < .001 | p < 0.001 |  |
| 9 | Behaviour | …29.2, −6.2], compared within HP and within LP trials, 20 of 28 participants | p = .0013 | < .01 | p < .01 |  |
| 10 | pRFs | …ved the held-out R² (+.0033, 95% CI [+.0024, +.0048], 28 of 28 participants | p = 7.5 × 10⁻⁹ | < .001 | p = 7.5 × 10⁻⁹ |  |
| 11 | pRFs | …area, on top of the DoG pRF (+.0106 [+.0064, +.0196], 27 of 28 participants | p = 5.2 × 10⁻⁸ | < .001 | p = 5.2 × 10⁻⁸ |  |
| 12 | pRFs | … = 5.2 × 10⁻⁸; both together +.0179 [+.0141, +.0229], 28 of 28 participants | p = 7.5 × 10⁻⁹ | < .001 | p = 7.5 × 10⁻⁹ |  |
| 13 | pRFs | …DoG pRFs; difference +.0048, 95% CI [+.0023, +.0094], 25 of 28 participants | p = 4.8 × 10⁻⁶ | < .001 | p = 4.8 × 10⁻⁶ |  |
| 14 | Item responses | …d occipitotemporal areas (hV4, VO, LO and TO, +0.236% [+0.199, +0.277]; all | q ≤ 3 × 10⁻⁸ | < .001 | q ≤ 3 × 10⁻⁸ |  |
| 15 | Item responses | …+0.040, +0.104], +0.149% [+0.091, +0.214] and +0.156% [+0.110, +0.203], all | q ≤ 1.1 × 10⁻⁴ | < .001 | q ≤ 1.1 × 10⁻⁴ |  |
| 16 | Item responses | …sponse than the distractor in every pathway (early +0.038% [+0.004, +0.105] | q = .015 | < .05 | q = .015 |  |
| 17 | Item responses | …l +0.126% [+0.049, +0.184]; occipitotemporal +0.097% [+0.055, +0.146]; both | q < .001 (exact in Supp. Table S1) | < .001 | q < .001 |  |
| 18 | Item responses | …ponse in early visual cortex depended on the location type (F(2, 54) = 7.67 | p = .001 | < .01 | p = .001 |  |
| 19 | Item responses | …the HP location, 21.6% below the location opposite HP (95% CI [−34.9, −8.9] | p = .005 | < .01 | p = .005 |  |
| 20 | Item responses | …ons next to HP also responded less than the opposite location (−14.9 ± 6.0% | p = .022 | < .05 | p = .022 |  |
| 21 | Item responses | …hereas HP and the locations next to it did not reliably differ (−6.7 ± 3.6% | p = .095 | = .095 (n.s.) | p = .095 |  |
| 22 | Item responses | …owest at HP, then next to HP, then opposite HP, was reliable (Page's L test | p = .0016 | < .01 | p = .0016 |  |
| 23 | Item responses | …(HP vs opposite −0.11% signal, 95% CI [−0.13, −0.06], 20 of 28 participants | p = .005 | < .01 | p = .005 |  |
| 24 | Item responses | …r area, the lower response at HP than opposite it was reliable in V1 and V2 | q = .015 | < .05 | q = .015 |  |
| 25 | Item responses | …wer response at HP than opposite it was reliable in V1 and V2 (q = .015 and | q = .031 | < .05 | q = .031 |  |
| 26 | Item responses | … depend on what appeared there (location type × item type, F(4, 108) = 0.92 | p = .454 | = .454 (n.s.) | p = .454 |  |
| 27 | Item responses | …ponse at HP was also lower than opposite HP, but not reliably (dorsal −6.7% | p = .24 | = .24 (n.s.) | p = .24 |  |
| 28 | Item responses | …HP, but not reliably (dorsal −6.7%, p = .24, n = 25; occipitotemporal −9.8% | p = .090 | = .090 (n.s.) | p = .090 |  |
| 29 | Item responses | …9.8%, p = .090). Pooled over all areas, the difference was reliable (−15.5% | p = .013 | < .05 | p = .013 |  |
| 30 | Item responses | …ble S1). The effect did not differ between the pathways (pathway × location | p = .70 | = .70 (n.s.) | p = .70 |  |
| 31 | Gain fields | …edian over areas +17%, 95% CI [+13, +24], positive in 25 of 28 participants | p = 3.2 × 10⁻⁷ | < .001 | p < 0.001 | p = 4.1 × 10⁻⁷ (< .001) |
| 32 | Gain fields | …from V3 on (from +18% in TO to +59% in hV4), with the exception of LO (+20% | q = .090 | = .090 (n.s.) | q = .090 | LO +22%, q = .033 (< .05): now reliable |
| 33 | Gain fields | …ponse to the target was larger than to a neutral item in V1 (+0.071% signal | p = 2.4 × 10⁻⁴ | < .001 | p < .001 |  |
| 34 | Gain fields | …o a neutral item in V1 (+0.071% signal, p < .001) and in V2 (+0.146% signal | p = 2.2 × 10⁻⁸ | < .001 | p < .001 |  |
| 35 | Gain fields | …age points for every doubling of an area's mean pRF size (95% CI [+15, +68] | q = .0089 | < .01 | q < .01 | +32 points, p = 8.2 × 10⁻⁵, q = 5.7 × 10⁻⁴ (< .001) |
| 36 | Gain fields | …y smaller one (+8.0%, 95% CI [+4.4, +13], positive in 21 of 28 participants | p = .023 | < .05 | p ==< 0.05 (typo) | +11%, p = .0065 (< .01) |
| 37 | Gain fields | …pRFs (pooled target gain +45% and +53%, distractor gain +9.5% and +15%, all | p ≤ .032 | < .05 | p ≤ .032 |  |
| 38 | Gain fields | … in 22 of 28 participants (pooled over all areas −1.0%, 95% CI [−1.9, −0.4] | p = 2.7 × 10⁻⁴ | < .001 | p < 0.001 | −1.5%, p = .0071 (< .01) |
| 39 | Gain fields | …d the reduction was reliable in V3 (−2.7%), hV4 (−2.4%) and TO (−1.9%; each | q = .036 | < .05 | q < 0.05 | no area q < .05 (smallest q = .11, V3) |
| 40 | Gain fields | … q = .036). Pooled per pathway, it was not reliable after correction (early | q = .057 | = .057 (n.s.) | q = .057 | early q = .17 |
| 41 | Gain fields | …d per pathway, it was not reliable after correction (early q = .057, dorsal | q = .58 | = .58 (n.s.) | q = .58 | dorsal q = .50 |
| 42 | Gain fields | …reliable after correction (early q = .057, dorsal q = .58, occipitotemporal | q = .089 | = .089 (n.s.) | q = .089 | occipitotemporal q = .37 |
| 43 | Gain fields | …of the target and the distractor (the joint fit: −3.5%, 95% CI [−6.7, −1.3] | p = .0044 | < .01 | p < 0.01 | −5.5%, p = .012 (< .05); hV4 q = .022 |
| 44 | Gain fields | …it was −0.8% with Gaussian pRFs (95% CI [−2.6, −0.1], 23 of 28 participants | p = 4.2 × 10⁻⁴ | < .001 | p < 0.001 |  |
| 45 | Gain fields | … × 10⁻⁴) and −2.8% with DN pRFs (95% CI [−4.1, −0.7], 21 of 28 participants | p = .0052 | < .01 | p < 0.01 |  |
| 46 | Gain fields | …ed correction replicated only in part: hV4 did so with all three pRF models | q = .036 | < .05 | q = .036 |  |
| 47 | Gain fields | … joint fit, the pooled reduction was present with all three (Gaussian −8.8% | p = .027 | < .05 | p = .027 |  |
| 48 | Gain fields | …ed reduction was present with all three (Gaussian −8.8%, p = .027; DN −7.8% | p = .0024 | < .01 | p = .0024 |  |
| 49 | Gain fields | …4), but no single area passed correction with Gaussian or DN pRFs (smallest | q = .13 | = .13 (n.s.) | q = .13 |  |
| 50 | Gain fields | …caling the input. They predicted held-out runs worse in all 28 participants | p = 7.5 × 10⁻⁹ | < .001 | p = 7.5 × 10⁻⁹ |  |
| 51 | Gain fields | … along the hierarchy, in the dorsal as well as the occipitotemporal pathway | p = .011 | < .05 | p = .011 | p = .0061 (< .01) |
| 52 | Gain fields | …rarchy, in the dorsal as well as the occipitotemporal pathway (p = .011 and | p = .003 | < .01 | p = .003 | p = .0022 (< .01) |
| 53 | Gain fields | …F size of an area (−0.7 percentage points per doubling, 95% CI [−3.7, +3.3] | p = .61 | = .61 (n.s.) | p = .61 | alone +1.4, p = .61; joint −0.4, p = .75 |
| 54 | Gain fields | …bling, 95% CI [−3.7, +3.3], p = .61), and the target gain grew more than dS | p = .018 | < .05 | p < 0.05 | not recomputed |
| 55 | Gain fields | …at the HP location (ρ = −0.17, 95% CI [−0.33, −0.07], 24 of 28 participants | p = 6 × 10⁻⁴ | < .001 | p < 0.001 | ρ −0.20, p = .005 (< .01) |
| 56 | Decoding | …r participants: V1 .52, V2 .48, V3 .41, and .11 to .29 in higher areas; all | p < 10⁻³ (largest: VO 9.8 × 10⁻⁴; others ≤ 1.2 × 10⁻⁷) | < .001 | p < 10⁻³ |  |
| 57 | Decoding | …y in this quadrant was 1.14 ± 0.39% lower than in the three other quadrants | p = .003 | < .01 | p = .003 |  |
| 58 | Decoding | …ts (p = .003, one-sided). The reduction was present in early (−1.35 ± 0.51% | p = .010 | < .05 | p < 0.05 |  |
| 59 | Decoding | …n early (−1.35 ± 0.51%, p = .010) and occipitotemporal areas (−1.06 ± 0.41% | p = .006 | < .01 | p < 0.01 |  |
| 60 | Decoding | …ral areas (−1.06 ± 0.41%, p = .006), but not in dorsal areas (−0.12 ± 0.75% | p = .56 | = .56 (n.s.) | p = .56 |  |
| 61 | Decoding | … by area, it was reliable in V2 (−1.68 ± 0.73%) and V3 (−1.62 ± 0.59%; both | q = .033 | < .05 | q = .033 |  |
| 62 | Decoding | …e four quadrants) was below that of the quadrants next to it (+0.57 ± 0.17% | p = .004 | < .01 | p < 0.01 |  |
| 63 | Decoding | …s the difference from the opposite quadrant was not reliable (−0.12 ± 0.24% | p = .08 | = .08 (n.s.) | p = .08 |  |
| 64 | Decoding | …t of the bar was decoded more weakly (−1.33 ± 0.49% at 0.9° from the center | p = .016 | < .05 | p < 0.05 |  |
| 65 | Decoding | …he aperture, where item responses would leak in (whole aperture −1.3 ± 0.5% | p = .010 | < .05 | p < 0.05 |  |
| 66 | Decoding | …whole aperture −1.3 ± 0.5%, p = .010; the eccentricity bands did not differ | p = .51 | = .51 (n.s.) | p = .51 |  |
| 67 | Decoding | …ants with a larger predicted effect showed a larger decoded effect (r = .57 | p = .002 | < .01 | p < 0.01 |  |
| 68 | Decoding | …HP than opposite it (−0.34 ± 0.18% predicted against +0.68 ± 0.32% observed | p = .022 | < .05 | p = .022 |  |
| 69 | Decoding | … LP locations share one sustained gain, and it was not reliable in the data | p = .070 | = .070 (n.s.) | p = .070 |  |
