# Update 4 for the writing agent (25 Nov 2026): closing the VERIFY and PENDING items

Builds on update 3 (`reports/docky_followup_2026-11-25_for_writing_agent_update3.md`); where they disagree, this file wins. New and changed figures with their legend `.md` files are in `notes/writing_agent_update4_2026-11-25.zip`.

Rules (unchanged):
- Joint zero-sum fits only (main model: item weight, target gain, one distractor gain, zero-sum sustained field). No "estimated alone" terms, no unconstrained fits.
- Gain-field and trial-wise terms: Student-t hierarchical model, posterior mean [95% HDI], P = posterior probability of the sign; credible = P > .975.
- Single-area claims only from each area fitted on its own data.
- n = 28.

Source note: the revised manuscript could not be opened here (newest accessible version: v2 of 20 Nov). Items 11 and 12 are therefore checked against v2 and the update-3 handoff. Sentences are quoted as given in the request.

## Read first: three changes of substance

1. **Fig. 3: "next to HP lower than opposite HP" is a trend, not a finding** (item 13). It points the same way in every analysis, but most of it comes from a few participants with noisy estimates. With each participant weighted by the precision of their estimate (pathways pooled by voxels as in the manuscript, no pooling over areas), the early next − opposite difference is −0.030% signal [−0.070, +0.010], P = .93 (relative −4.7%, P = .84), against −0.050% (p = .038) and −14.9% (p = .022) unweighted.
   - Do not claim the HP < next < opposite ordering (Page's L) as a result; report next < opposite as a trend with this caveat.
   - Soften, rather than delete, the first "observation" of the Discussion paragraph on the LP gradient (wording in item 1). In item 6, keep the model comparison with the same caveat.
   - "Lowest at HP" holds (early −0.069% signal, P = .986).
2. **Fig. 3: the HP reduction at the item locations is credible in early visual cortex, and occipitotemporal areas point the same way** with a similar size (P = .968). Write "found in early visual cortex", not "only in early visual cortex" (item 13).
3. **The HRF is not refitted with the gains,** and the main fits' voxel selection still includes p_signal (item 10). The Methods text must say what was done. A refit on the reliability-only selection is possible (about one day on the cluster) but has not been started.

## Status overview

| Item | Status | One-line verdict |
|---|---|---|
| 1. Sustained width | done | "Local" holds at the group level (FWHM 4.7°); "in all 28 participants" and 2.8° go |
| 2. Gaussian and DN robustness | **pending** (fits done; standard-error step running on the cluster, SLURM 8890963 and 8890965) | Keep the old sentences out; mark pending; drop the DN baseline-variant sentence (S12) |
| 3. Decoding predictions | done | Predictions barely change (−1.20 vs −1.58%, r = .61); Gaussian/DN decoding holds pooled |
| 4. Item-specific HP effects | done | No distractor-specific reduction (dD +1.3 [−1.6, +4.0]); drop dT |
| 5. Gain vs baseline (S13) | done | Holds, slightly stronger; drop the 72% recovery sentence |
| 6. Model predictions of item responses | done | HP − next reproduced; the next − opposite comparison only with the trend caveat (item 13) |
| 7. LP-gradient variant | done | No reliable next − opposite difference in the sustained field (+1.0 [−0.7, +2.7]); HP below both |
| 8. Trial-wise, Student-t | done | History terms hold; new Fig. 7 panel c recommended |
| 9. Individual differences | done | No relation (ρ = −0.02); the behavioural benefit is unreliable per participant |
| 10. Methods text | done | Replacement text below; long list of sentences to delete |
| 11. Captions, Fig. 4 | done | Fig. 4 re-rendered for the zero-sum, single-distractor-gain model |
| 12. Supplement | done | Table of every cited supplement with status and new number |
| 13. Fig. 3, weighted by precision | done (new) | Item contrasts and "lowest at HP" hold; "next < opposite" is a trend only; occipitotemporal points the same way |


## 1. Width of the sustained field

**Numbers.** The width is estimated from the joint zero-sum fits by profiling. For each participant and area, everything is held at the fit except the sustained width, which is set to 17 values (FWHM 1.2 to 18.8°, log-spaced); at each value the gains are re-solved along the zero-sum direction. The profiles are then summed over participants. The best width is recomputed leaving out one participant at a time, and by resampling participants (bootstrap, 500 draws). This is the analysis of update 13, part C, and its supplementary figure is `figS_width_profiles_zs_n28_palA`.

| Unit | Group FWHM | 95% bootstrap range | At the neighbouring location (5.7° away), the field's value as % of its peak | P(FWHM < 5.7°), bootstrap |
|---|---|---|---|---|
| All areas | 4.7° | 4.0 to 5.6° | 1.8% (at most 5.9%) | .99 |
| Early (V1 to V3) | 4.7° | 4.0 to 5.6° | 1.8% (at most 5.9%) | .99 |
| Dorsal | 4.7° | 1.2 to 11.2° | 1.8% (up to 49%) | .87 |
| Occipitotemporal | 5.6° | 4.0 to 11.2° | 5.9% (up to 49%) | .61 |
| Per area | V1 4.0°, V2 4.7°, V3 4.7°, V3AB 5.6°, IPS 2.0°, hV4 6.7°, VO 4.7°, LO 5.6°, TO 4.7° | | | |

- **Per participant:** the width is not identified. 215 of 241 profiles exclude no width at 2 SE, and odd and even runs agree poorly (ρ = 0.10).
- **dS does not depend on the width:** it is negative at every width tested.

**Verdict.**
- "Local" still holds where the reduction is largest. In early visual cortex, and over all areas, the field has a FWHM of about 4.7° and has fallen to about 2% of its peak at the neighbouring locations.
- "Narrower than the 5.7° between neighbouring item locations" holds at the group level in early visual cortex and over all areas (bootstrap P = .99). It is not established in dorsal and occipitotemporal areas (P = .87 and .61).
- "In all 28 participants" and "FWHM 2.8° [2.7, 2.9]" must go. Those came from the earlier fits, from profiles with g_HP and g_LP free, and from per-participant widths that the data do not identify.

**Suggested Results sentence** (replaces "The reduction was local: ... in all 28 participants"):

> "The reduction was local: summed over participants, the sustained field was best described by a full width at half maximum of 4.7° (95% bootstrap range 4.0 to 5.6°), in early visual cortex as over all areas, narrower than the 5.7° between neighbouring item locations, so that at the neighbouring locations it had fallen to about 2% of its peak (Supplementary Fig. [width profiles]). The width was not identified in single participants, and the HP − LP difference was negative at every width tested."

**Discussion (LP gradient paragraph)**, replace "whereas the sustained field was narrower than the distance between neighbouring item locations" with:

> "whereas the sustained field around the HP location had largely fallen off at the neighbouring locations (to about 2% of its peak in early visual cortex, at the group-level width of 4.7°)"


**Addendum (item 13).** The measured "next lower than opposite" difference is only a trend: P = .93 with participants weighted by precision. So the first observation of the Discussion paragraph "Two observations are not explained by a single field ..." should be softened rather than stated as fact, for example: "the locations next to HP tended to respond less than the opposite location (Fig. 3e), which a field that has fallen to about 2% of its peak there would not produce, although this difference was not reliable once participants were weighted by the precision of their estimates." Keep the second observation, the behavioural target vs distractor pattern.

## 2. Gaussian and DN robustness: PENDING

The joint zero-sum fits for Gaussian and DN pRFs are running on the cluster (SLURM 8834744 Gaussian, 8834745 DN). About half of the 2 × 252 cells were done at the time of writing, and some failed cells are being resubmitted. The width-profile step for the standard errors follows, then the hierarchical model (`gain_meta_zs.py --family gauss|dn`).

Until then:
- Mark "Both gains replicated with Gaussian and DN pRFs (...)" and the paragraph "The reduction replicated with Gaussian and DN pRFs ..." as **pending**. Do not keep their old numbers.
- In the Limitations, replace "joint fit: DoG −3.5%, DN −7.8%, Gaussian −8.8%" and "the DoG model gave the smallest of the three estimates" with a pending marker.
- **DN baseline variants (S12, "predicted held-out runs worse in all 28 participants, p = 7.5 × 10⁻⁹"): drop.** They are earlier, non-zero-sum DN fits, n = 30. A zero-sum rerun (zero-sum DN base, three variants, recovery) costs about 1,100 CPU-h. The gain argument does not need it (item 5).

## Item 3. Decoding predictions (Fig. 6e, f) from the joint zero-sum fit

**What was rerun.** The decode simulation (`simulate_decode.py --mode fitted`) used the zero-sum view root, where m7_full is the main model m5_TDS_dS_zs (SLURM 8837804, 28 participants × 12 runs × V1 to V3, all 1,008 tasks completed). The canary log confirms zero-sum gains in the simulated signal (sub-03 V1: g_HP −0.073, g_LP +0.024). Everything else was as before: the same decoder, fold pRFs and phase-randomised residuals, and a paired control with the gains at 0. The collected tables are `notes/data/decode_simfit_zs_*` (prefix `decode_simfit_zs`).

| Quantity (V1 to V3, mean ± SEM, n = 28) | Old (polished, non-zero-sum) | Zero-sum joint fit |
|---|---|---|
| HP − next to HP, predicted (observed −1.58 ± 0.53%) | −1.09 ± 0.44% | **−1.20 ± 0.45%** (observed − predicted p = .45) |
| Per-participant correlation, predicted vs observed HP − next | r = .57, p = .002 | **r = .61, p = .001** |
| Next to HP − opposite, predicted (observed +0.68 ± 0.32%) | −0.34 ± 0.18%, difference p = .022 | **−0.39 ± 0.18%, difference p = .013** |
| HP − opposite, predicted (observed −0.90 ± 0.53%) | −1.44 ± 0.59% (r = .44, p = .019) | **−1.58 ± 0.60%** (r = .46, p = .015; difference p = .24) |
| Bar sweep, HP-quadrant segment (observed −0.65%) | −0.84% | −0.91% (observed vs predicted group means over 11 points r = .73) |

**Verdict.** The zero-sum fit predicts the decoded HP effect slightly better than the old fit (−1.20 vs −1.09%, r = .61). As before, it does not reproduce the higher intensity next to HP than opposite it.

**Suggested sentence.**
> "Without further fitting, the simulated data showed a lower intensity in the quadrant facing HP of about the observed size (HP minus next to HP: −1.20 ± 0.45% predicted against −1.58 ± 0.53% observed), and participants with a larger predicted effect showed a larger decoded effect (r = .61, p = .001; Fig. 6e, f). The model did not reproduce the higher intensity next to HP than opposite it (−0.39 ± 0.18% predicted against +0.68 ± 0.32% observed, p = .013)."

The last sentence of that paragraph stays as it is ("This difference lies outside what the model can express ... not reliable in the data (p = .070)").

**Figure.** The prediction bands move slightly, so Fig. 6 is re-rendered with the zero-sum prediction:
- `notes/figures/paper_2026-11/fig6_decoding_v13_n28_palA_zs.{pdf,png,md}`.
- It is the same command as the current Fig. 6 (`run_excluding … fig6_decoding plot --v12`) with `--prediction decode_simfit_zs`.
- The legend describes the zero-sum model.
- The data panels are unchanged.

### Fig. 6 decoding with Gaussian and DN pRFs (closes the VERIFY)

Fig. 6c statistics were recomputed with each decoder's own pRF family (joint HRF; readouts `decoding_readout_{jointhrf,gauss_jointhrf,dn_jointhrf}_2026-11.tsv`). Effect: HP quadrant minus the other three, % of the four, mean ± SEM; one-sided Wilcoxon; per-area BH over nine areas.

| | DoG (Fig. 6) | Gaussian | DN |
|---|---|---|---|
| All nine areas | −1.14 ± 0.39%, p = .0033 | −1.18 ± 0.41%, p = .0033 | −1.48 ± 0.50%, p = .0051 |
| Early (V1 to V3) | −1.35 ± 0.51%, p = .010 | −1.36 ± 0.52%, p = .011 | −1.13 ± 0.51%, p = .033 |
| Occipitotemporal | −1.06 ± 0.41%, p = .0064 | −1.06 ± 0.42%, p = .0068 | −1.29 ± 0.50%, p = .0096 |
| Dorsal | −0.12 ± 0.75%, p = .56 | −0.18 ± 0.79%, p = .44 | −2.07 ± 1.02%, p = .039 |
| Areas q < .05 | V2, V3 (q = .033) | none (V2, V3 q = .052) | V3 (q = .008) |

Through the Student-t hierarchical model (`decode_meta.py`; outputs `notes/data/decode_meta{,_gauss_jointhrf,_dn_jointhrf}_2026-11-25/`), posterior probability of a reduction:

| Decoder pRFs | All areas | Early | V3 |
|---|---|---|---|
| DoG | P = .90 | P = .968 | P = .984 |
| Gaussian | P = .92 | P = .977 | P = .987 |
| DN | P = .88 | P = .87 | P = .90 |

**Verdict.** With the Wilcoxon statistics of Fig. 6, the decoded reduction holds with all three pRF families: pooled over all areas, in early and in occipitotemporal areas. The single-area results differ: V2 and V3 with DoG, V3 alone with DN, and no single area with Gaussian pRFs. With the hierarchical model, every family gives only a trend over all areas. The early effect is credible with DoG and Gaussian pRFs but not with DN.

**Suggested sentence** (replaces the VERIFY):
> "The reduction in the quadrant facing HP was also present with decoders built on Gaussian and DN pRFs (all nine areas −1.18 ± 0.41% and −1.48 ± 0.50%, p = .0033 and .0051; early visual cortex p = .011 and .033), although which single areas passed correction depended on the pRF model (Supplementary Fig. S#)."

## 4. Item-specific HP effects (joint zero-sum)

**Numbers.** These come from the joint zero-sum main model, freeing the distractor gain separately at HP and at LP. That is one least-squares step from the fit: dS, target and both distractor gains are re-solved with everything else held. Each cell gets a leave-one-run-out jackknife SE (calibration 0.73, 0.67 expected). The effects then go through the Student-t hierarchical model. Code: `retsupp/visualize/paper/dD_meta_zs.py`; outputs: `notes/data/dD_meta_zs_2026-11-25/`.

| dD (distractor gain at HP − at LP), percentage points | Mean [95% HDI], P |
|---|---|
| All areas | +1.3 [−1.6, +4.0], P(>0) = .82 |
| Early | **+4.7 [+0.8, +8.7], P(>0) = .990** |
| Dorsal | −0.4 [−6.6, +5.7] |
| Occipitotemporal | −0.5 [−4.5, +3.0] |
| Areas | none credible on its own (V3 +5.3, P = .974; V1 +4.4, V2 +4.3, P = .96) |

Other evidence:
- Plain medians: +7.3 [+1.3, +10.1], 8/28 below 0, p = .0044, the same as update 13 item 1c.
- Full LM fit with dD (m7_full_zs): +5.6%, p = .070.
- Held out, adding dD to the main model makes prediction worse in all 28 participants (update 13 item 1a).

**Sensitivity.** A distractor-specific reduction larger than 1.6 percentage points, about a third of the distractor gain (+4.7%), lies outside the 95% HDI over all areas. In the simulation of update 13 (item 1c, zero-sum), the joint test detected a planted −10-point reduction 87 to 90% of the time, a −5-point one 36 to 46%, and a −2-point one 10 to 13%.

**dT (target-specific HP − LP).** No joint zero-sum model has it: the main model has one target gain, and the existing dT came from terms estimated alone. A zero-sum ladder model with dT needs a code change plus about 150 to 250 CPU-h. **Recommendation: drop dT from the main text.** If wanted, it can go into a supplement later.

**Verdict.** There is no distractor-specific reduction at the HP location. Over all areas dD is near zero, and if anything positive in early areas: a distractor at HP lost less gain there than the other items. That positive term does not help predict held-out runs, so it should be reported as a check, not as an effect.

**Suggested replacement paragraph:**

> "We also asked whether the HP location has an additional effect specific to the distractor, by letting the distractor gain differ between the HP and the LP locations in the same joint fit. It did not: over all areas, the distractor gain at HP minus at LP was +1.3 percentage points (95% HDI [−1.6, +4.0]), and adding this term made the prediction of held-out runs worse in every participant. A distractor-specific reduction of more than about a third of the distractor gain (1.6 points) lies outside the 95% interval, so a reduction of this size or larger is unlikely, although a smaller one cannot be excluded. In early visual cortex the term was, if anything, positive (+4.7 [+0.8, +8.7]): a distractor at HP lost less gain there than other items did. The sustained reduction alone thus accounts for the lower response at the HP location, which was the same for the target, the distractor and neutral items (Fig. 3), in line with the slower and less accurate responses to targets at HP (Fig. 1d)."

Delete the target-specific term (dT) and the sentence on what the data "could only have detected", or replace that sentence with the HDI statement above.


## Item 5. Gain versus baseline (S13) on the zero-sum joint fit

**Which parts depend on the gain-field fit.**
- **Slope of the bar response (step 1):** does not use the gain-field fit. It regresses the mean BOLD at each item location on the bar-only pRF prediction. It does depend on which pRFs select the voxels and predict the bar. I reran it with the pRFs of the joint zero-sum fit (`location_timeseries_zs_lm_2026-11-16`) so it matches the rule.
- **Trial-locked additive response and recovery (steps 2 to 4):** these are estimated on each participant's gain-field fit. They were rerun on the zero-sum main model: `fit_baseline_terms.py --zero-sum` on the zero-sum view root, SLURM 8837805, 28 participants × 9 areas, all completed.
  - dS now moves along the zero-sum direction.
  - The recovery base has no sustained field.
  - The simulated gain is dS = −3.2%, the hierarchical estimate.

**Numbers** (per participant the median over areas, median [bootstrapped 95% CI], k of 28, Wilcoxon p; no per-cell SEs exist for these terms, so no hierarchical model):

| Test | Old (manuscript) | Zero-sum |
|---|---|---|
| Bar-response slope, HP vs LP, all areas | −2.1% [−4.6, −0.7], 20/28, p = .012 | **−2.9% [−4.2, −1.1], 21/28, p = .020** |
| Bar-response slope, HP vs next to HP | −3.5% [−5.2, −0.9], 21/28, p = .0011 | **−3.4% [−5.6, −0.7], 19/28, p = .018** |
| Next vs opposite | −0.5%, p = .78 | +0.0%, p = .47 |
| Per pathway, HP vs LP | early −3.4% (p = .10), dorsal −2.2% (.55), occipitotemporal −2.5% (.063) | early −3.3% (p = .074), dorsal −2.1% (.57), occipitotemporal −3.7% (.056) |
| Trial-locked additive at HP, search onsets: amplitude | +1.5% [+0.0, +2.9], 19/28 > 0, p = .010 | **+1.8% [+0.4, +3.2], 20/28 > 0, p = .0044** (positive: no suppression) |
| Same, held-out run better | 16/28, p = .41 | **17/28, ΔR² +0.28 × 10⁻⁴ [−0.21, +1.26], p = .29** |
| Trial-locked at HP, fixation onsets | −0.7%, p = .85 | −0.2% [−0.9, +0.9], p = .90 (held-out gain 22/28, p = 3 × 10⁻⁴, from the response common to all locations) |
| Sustained additive offset | not identifiable (0.0 kept after run-mean removal) | the same |
| Recovery, 10 simulated data sets × 28 | gain found 6/10; trial-locked 10/10; never confused | **gain found as a gain 9/10; trial-locked as such 10/10; sustained offset 10/10 neither; never confused** |

**The 72% figure.** This was the DN baseline-field recovery: simulated b/d baseline effects recovered as input gain in 72% of simulations, on the earlier non-zero-sum DN fits. It cannot be kept under the rule. A rerun needs:
- the zero-sum DN base fits (running in the Gaussian/DN robustness job);
- zero-sum versions of the three DN baseline-field models;
- the recovery, about 1,100 CPU-h as before.

Recommendation: **drop the 72% sentence and the DN baseline-variant claim (S12)** rather than rerun. The gain-vs-baseline argument stands on the slope and the trial-locked tests above, which do not need it.

**Verdict.** Both pieces of evidence for a gain hold on the zero-sum fits, a little stronger than before: the shallower bar response at HP and the absence of a trial-locked suppression at HP, which recovery shows would have been found.

**Suggested sentence.**
> "The reduction at the HP location behaved as a change in gain: the response to the mapping bar was shallower where the bar crossed the HP location than where it crossed the other locations (−2.9% [−4.2, −1.1], 21 of 28 participants, p = .020), and there was no additive response at that location locked to the trials (held-out prediction better in 17 of 28 participants, p = .29; its amplitude was positive rather than negative), although simulations showed that a trial-locked response of matched size would have been found in every simulated data set (Supplementary Fig. S13)."

**Figure.** `notes/figures/paper_2026-11/figS_baseline_vs_gain_zs_n28.{pdf,png,md}`. Tables: `notes/data/baseline_vs_gain_{terms,change,confusion,confusion_per_sim}_zs_n28_2026-11-25.tsv` and `notes/data/baseline_vs_gain_step1_zs_n28_2026-11-25.tsv`.

## 6. Model predictions of the item responses (zero-sum joint fit)

Source: the noise-free prediction of the zero-sum main model (sustained HP − LP field as fitted, "S") run through the Fig. 3 deconvolution, on the main model's voxel selection (update 13 part G, item 2: `notes/data/event_deconv_zs_lm_pred_2026-11-16_relative_S`, measured on the same voxels in `event_deconv_relative_zs_lm_2026-11-16`). Relative units as Fig. 3e: % of each participant's mean response, mean ± SEM, n = 28; difference = measured − predicted per participant, Wilcoxon. Table: `notes/data/u4_item6_location_contrasts_zs_2026-11-25.tsv`.

**Location contrasts, early visual cortex (V1 to V3)**

| Contrast | Measured (same voxels) | Predicted, zero-sum model | Difference |
|---|---|---|---|
| HP − next | −9.2 ± 4.9% | −7.2 ± 3.7% | −2.0 ± 6.1, p = .94 |
| Next − opposite | −17.1 ± 6.3% | −1.5 ± 0.7% | −15.6 ± 6.5, p = .023 |
| HP − opposite | −26.3 ± 8.3% | −8.7 ± 3.7% | −17.6 ± 8.8, p = .050 |
| HP − mean of the LP locations | −14.9 ± 5.5% | −7.7 ± 3.7% | −7.2 ± 6.4, p = .37 |

Against the Fig. 3 values (Fig. 3 voxel selection; HP − next −6.7 ± 3.6%, next − opposite −14.9 ± 6.0%), the differences are p = .94 and p = .048. The model without an HP term predicts ≤ 0.7% for every contrast.

Verdict: unchanged in substance. The zero-sum model reproduces the dip at HP relative to its neighbours and about half of HP vs LP, but not the gradient between the LP locations.

> "When we analyzed the predictions of the model in the same way as the measured responses, the model reproduced the difference between HP and the locations next to it (predicted −7.2 ± 3.7% against a measured −9.2 ± 4.9%; difference p = .94), but not the difference between the locations next to HP and the location opposite it (predicted −1.5 ± 0.7% against a measured −17.1 ± 6.3%; difference p = .023)."

(If the measured values must stay those of Fig. 3: "predicted −7.2 ± 3.7% against a measured −6.7 ± 3.6%, difference p = .94; predicted −1.5 ± 0.7% against −14.9 ± 6.0%, difference p = .048". The like-for-like version above uses the same voxels for both and is preferred.)

**Item types (Fig. 3b, c), model vs measured, % signal, window 3.2 to 6.4 s** (`items_type_model_vs_measured_n28_zs_lm_mainS_2026-11-16.tsv`; median [95% CI], Wilcoxon p)

| | Measured | Predicted |
|---|---|---|
| Target − neutral, V1 | +0.060 [+0.032, +0.098], p < .001 | −0.006 [−0.012, +0.001], p = .28 |
| Target − neutral, V2 | +0.145 [+0.094, +0.177], p < .001 | +0.005 [−0.006, +0.036], p = .15 |
| Target − neutral, early / dorsal / occipitotemporal | +0.127 / +0.240 / +0.216 | +0.033 / +0.177 / +0.153 (measured − predicted p ≤ .001 in each) |
| Distractor − neutral, early / dorsal / occipitotemporal | +0.061 / +0.123 / +0.127 | +0.016 / +0.048 / +0.066 (measured − predicted p ≤ .001 in each) |
| Target − distractor, early / dorsal / occipitotemporal | +0.034 / +0.109 / +0.096 | +0.016 / +0.120 / +0.094 (difference p = .11 / .92 / .76) |

Verdict: both Results sentences still hold on the zero-sum fit.

> "...whereas the model predicted no difference there (V1 −0.006%, V2 +0.005% signal)."

> "More generally, the model predicted smaller differences between each odd item and a neutral item than we measured, in every pathway, although it captured the difference between the target and the distractor in every pathway."

**Addendum (item 13).** The measured next − opposite difference is only a trend (P = .93 with participants weighted by precision), so lead with the HP − next comparison and give next − opposite with the caveat. Suggested:

> "Analysed in the same way as the measured responses, the fitted model reproduced the lower response at HP than next to it (predicted −7.2 ± 3.7% against a measured −9.2 ± 4.9%; difference p = .94). It predicted almost no difference between the locations next to HP and the location opposite it (−1.5 ± 0.7%), less than the measured difference (−17.1 ± 6.3%; difference p = .023), although the measured difference itself was not reliable once participants were weighted by the precision of their estimates."

## Item 7. LP gradient as a zero-sum joint variant

**What was built.**
- **New model:** `m6_TDS_dS_adjopp_zs` in `retsupp/modeling/fit_cv_ladder_nested.py`. It is the main model (m5_TDS_dS_zs) plus δ = g_next − g_opp.
  - `lp_split.LPSplitDoG`: g_next = g_LP + δ/3 and g_opp = g_LP − 2δ/3, so g_LP remains the mean LP gain and the field stays zero-sum (g_HP + 2 g_next + g_opp = 0).
  - It is fitted like the main model: joint LM with every voxel's pRF free, the zero-sum constraint and the zero-sum bounds, two starts (the main model's fit with δ = 0, asserted to reproduce it, and the same with δ = dS), keeping the better.
  - The fit covers all runs only.
- **Canary:** sub-03 V1 (SLURM 8837040, 54 min).
  - The δ = 0 start reproduces the main model (max |Δpred| 7.6 × 10⁻⁶).
  - Both starts end at the same SSE (4,364,521 and 4,364,522).
  - Nothing ends on a bound; δ = −13.5% in this cell.
- **Standard errors:** `retsupp/modeling/fit_gain_stats_adjopp.py` stores per-run sufficient statistics of the gains on each fit, so jackknife SEs of δ, HP − next and HP − opposite come out as for the main model. Its canary is fine: the linear reconstruction of the fit's prediction is exact to 1.9 × 10⁻⁵.
- **Statistics:** `retsupp/visualize/paper/adjopp_meta.py` runs the Student-t hierarchical model (pathway = mean of its areas), each area on its own data, and the Wilcoxon table.

**Status: done.**
- All 241 participant × area fits finished (SLURM 8838317), followed by the gain statistics (SLURM 8840134).
- Outputs: `/data/ds-retsupp/derivatives/adjopp_gain_stats_zs_2026-11-25/`, `notes/data/adjopp_meta_zs_2026-11-25/`.
- SE calibration is 0.63 for δ and 0.69 to 0.71 for the HP contrasts (0.67 expected).
- The hierarchical model has no divergences, r-hat ≤ 1.01, and a predictive coverage of 94 to 95%.

**Numbers** (% of the drive; Student-t hierarchical model, mean [95% HDI], P = posterior probability of the sign; n = 28)

| Contrast | All areas | Early | Dorsal | Occipitotemporal |
|---|---|---|---|---|
| δ = next − opposite | +1.0 [−0.7, +2.7], P(>0) = .89 | +2.3 [−0.5, +5.1], P(>0) = .95 | −1.2 [−3.6, +1.4], P(<0) = .83 | +1.2 [−0.9, +3.3], P(>0) = .87 |
| HP − next | **−3.2 [−5.7, −0.7], P = .990** | **−5.2 [−8.8, −1.9], P = .997** | −0.8, P = .68 | −2.9 [−6.0, +0.0], P = .967 |
| HP − opposite | **−1.7 [−2.8, −0.7], P = .998** | **−2.3 [−4.2, −0.5], P = .992** | −1.2, P = .971 | **−1.5 [−2.6, −0.3], P = .994** |
| dS (HP − mean LP) | **−2.6 [−4.5, −0.7], P = .995** | **−4.2 [−6.7, −1.4], P = .999** | −1.1, P = .78 | −2.2 [−4.4, +0.1], P = .969 |

- **On their own data:** no area shows a δ. V3 has HP − next −11.0 (P = .995), and V3, V3AB and hV4 have HP − opposite (P ≥ .982).
- **Plain medians (Wilcoxon) are unstable in sign:** δ is −0.0 over all areas (p = .61) and −2.6 in early areas (p = .92). Dorsal is the only one that reaches significance (−3.9, p = .010, q = .029), and the hierarchical model does not support it (P = .83).
- **The estimates are the fit's own values.** In a third to a half of the cells, the least-squares re-solve differs slightly from the fit, by a median of 0.08 points for δ. Using the re-solved values instead changes nothing.

**Verdict.**
- The sustained field does not differ between the locations next to HP and the location opposite HP. δ has no reliable sign by any method, and in early visual cortex it is, if anything, positive (higher gain next to HP).
- This agrees with Fig. 3, where next < opposite is only a trend that does not survive weighting by precision (item 13).
- It also agrees with the decoding, where the intensity next to HP was if anything higher than opposite it (+0.68 ± 0.32%).
- The HP reduction itself holds in this model as in the main model (dS −2.6%, early −4.2%).

**Manuscript sentence.** Replace "A variant with separate sustained gains next to and opposite the HP location found no difference between them (−0.6%, p = .73)" with:

> "A variant of the model with separate sustained gains next to and opposite the HP location, fitted in the same way, found no reliable difference between them (next minus opposite +1.0% of the drive, 95% HDI [−0.7, +2.7]; early visual cortex +2.3% [−0.5, +5.1]), whereas the gain at HP remained below both (HP − next −3.2% [−5.7, −0.7]; HP − opposite −1.7% [−2.8, −0.7])."

**No held-out comparison.** The variant is fitted on all runs only, so it has no held-out R² against the main model. That would need another full set of fits in the run-wise cross-validation scheme. Do not claim that the variant is worse or better out of sample.

## 8. Trial-wise effects with the Student-t model

All rows use the trial-wise gains on the zero-sum main model and the Student-t hierarchical model; a pathway is the mean of its areas. Values are gain × 100 (percentage points of gain) unless marked %, with posterior mean [95% HDI], P, n = 28. Outputs:
- (a) `notes/data/trialwise_meta_bayes_signal_se_prevtgt_zs_lm_2026-11-25_pathavg/`
- (b) `notes/data/trialwise_meta_bayes_relnull_r0.{2,1}_zs_lm_2026-11-25_pathavg/`

### (a) History of the location

Model with the new neutral-at-previous-target term, so all terms below are estimated together, with log RT.

| Effect | All areas | Early | Dorsal | Occipitotemporal |
|---|---|---|---|---|
| Neutral item at the previous target location | **−6.1 [−10.3, −2.1], P = .998** | −4.4 [−9.2, +0.2], P = .964 | **−7.6 [−12.6, −2.6], P = .998** | **−6.6 [−11.1, −2.4], P = .999** |
| Neutral item at the previous distractor location | **−4.5 [−8.1, −1.2], P = .994** | **−5.3 [−9.5, −1.2], P = .991** | −3.9 [−9.2, +1.1], P = .933 | **−4.2 [−8.0, −0.1], P = .980** |
| Target at the previous distractor location | −1.4 [−6.7, +4.0], P = .69 | +5.1, P = .91 | −4.9, P = .89 | −4.5, P = .94 |
| Target repeat (same model), % of typical gain | **−21.7% [−31.2, −12.2], P > .999** | **−42.1%, P > .999** | **−23.6%, P = .998** | −5.3%, P = .942 |

The normal likelihood gives the same picture: −6.5, −4.7 and −1.5.

Verdict: holds. A location that held the target or the distractor gives less gain to a neutral item on the next trial; a target at the previous distractor location shows no change.

**Fig. 7, panel c: yes.** It is the evidence that decides between priming and repetition suppression (the target-repeat reduction belongs to the location, not to the repeated target), and it fits the layout of a and b.
- Rendered: `/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/fig7_compact_zs_lm_2026-11-13_n28_studentt_c.{pdf,png,md}`. Panels a and b are unchanged; c shows the three history effects over all areas and per pathway, Student-t.
- Code: `fig7_compact.py --panel-c <meta dir>`, default unchanged.
- The panel c values come from the model that contains the neutral-at-previous-target term. In that model the target-repeat effect is −21.7%, against −17.1% in panel a's model.

Suggested Results text:

> "To tell priming from repetition suppression, we also estimated the gain of a neutral item at the location that had held the target or the distractor on the previous trial (Fig. 7c). Both were lower than elsewhere (previous target location −6.1 percentage points [−10.3, −2.1], P = .998; previous distractor location −4.5 [−8.1, −1.2], P = .994), whereas a target at the previous distractor location did not change credibly (−1.4 [−6.7, +4.0], P = .69). The reduction on target repeats thus belongs to the location: whatever appears where the target was receives less gain, even a neutral item that is a different stimulus from the target before it."

Caption, panel c:

> "**c, Location history.** Gain of a neutral item at the location of the previous trial's target (orange) or distractor (green), and of a target at the previous distractor location (grey), in percentage points."

### (b) Distractor gain per SD of log RT, looser voxel selections

| Voxel selection | All areas | Early | Dorsal | Occipitotemporal |
|---|---|---|---|---|
| Signal voxels (main) | **−2.7 [−4.7, −0.8], P = .996** | **−2.9, P = .982** | **−3.8, P = .993** | −2.0, P = .967 |
| Cross-run reliability r > 0.2 | **−2.5 [−4.4, −0.7], P = .996** | −2.6, P = .975 | **−4.0, P = .995** | −1.7, P = .94 |
| r > 0.1 | **−1.8 [−3.7, −0.1], P = .978** | −2.0, P = .95 | −3.1, P = .97 | −1.0, P = .83 |

The target-repeat reduction holds in both selections: −17% (r > 0.2) and −20% (r > 0.1), P > .999. Without the SE weighting, the distractor-RT effect is not credible in any selection: +5.4 (r > 0.2), +1.8 (r > 0.1).

Verdict: the effect becomes smaller with looser selections but stays credible.

> "The lower distractor gain on slower trials became smaller with looser voxel selections but remained credible (r > 0.2: −2.5 [−4.4, −0.7], P = .996; r > 0.1: −1.8 [−3.7, −0.1], P = .978)."

Keep "we interpret it with caution". The reason is not the voxel selection but that the effect depends on weighting each estimate by its precision.

### (c) Joint latent RT model

Confirmed on the zero-sum fits. Per-trial gains were regenerated on the zero-sum main model (`derivatives/af_trialwise_trialse_zs_lm_2026-11-16`, update 13 part G). Results: `notes/data/rt_joint_latent_zs_lm_2026-11-16/main_results.tsv`. It is a model of single-trial log RT (normal likelihood on log RT); it is not one of the Student-t meta-analyses.

Results:
- Global gain: −17 ms per SD of the latent gain [−31, −4], P = .995.
- Target gain: +14 ms [−9, +40], P = .88.
- Distractor gain: −10 ms [−32, +13], P = .81.
- Checks: 1 divergence, minimum bulk ESS 318.

> "In a joint model of single-trial response times, in which each trial's gains are latent variables measured with their own error, trials with a higher global gain were answered faster (−17 ms per SD of latent gain, 95% HDI [−31, −4], P = .995); the target and distractor gains were too unreliable per trial to predict response time on their own (P = .88 and .81; Supplementary Note)."

**Short caption if Fig. 7 gets panel c** (`fig7_compact_zs_lm_2026-11-13_n28_studentt_c`):

> **Figure 7. The gain fields fluctuate with trial history and with response time.** Trial-by-trial changes in the gains, estimated per participant and area on top of the participant's gain-field fit and combined in the same hierarchical model as Fig. 5 (n = 28). **a, Trial history.** Target gain when the target repeats the location of the previous trial, and distractor gain when the distractor does, relative to the typical gain of each area. **b, Response time.** Change in the target, distractor and global gain on slower trials, per standard deviation of log response time. **c, A location that held an item gives less gain next.** Gain of a neutral item at the location of the previous trial's target or distractor, and of a target at the previous distractor's location, in percentage points. Points and bars are the posterior mean and 95% credible interval, over all nine areas and per pathway. Filled points mark effects whose sign is credible (posterior probability above .975). Stars as in Fig. 5.

The Fig. 7 legend `.md` files are regenerated with the correct fit label: the zero-sum main model, not "m7".

## 9. Individual differences: dS vs the behavioural HP benefit

dS per participant from the joint zero-sum fit (`gain_meta_zs` estimates). Benefit is the distractor-at-LP RT minus the distractor-at-HP RT, correct trials (median 25 ms). Spearman ρ, percentile bootstrap 95% CI over participants, n = 28. Code: `retsupp/visualize/paper/ds_individual_differences.py`. Output: `notes/data/ds_individual_differences_zs_2026-11-25/`.

| dS per participant | ρ [95% CI], p |
|---|---|
| **Primary: the participant's deviation in the Student-t hierarchical model** (weights areas by precision, shrinks noisy participants) | **ρ = −0.02 [−0.43, +0.39], p = .94** |
| Median over the nine areas (as before) | ρ = +0.12 [−0.27, +0.49], p = .54 |
| In the model: participant deviation regressed on z(benefit) | −0.05% of the drive per SD of benefit [−1.70, +1.51], P = .52 |

The participant deviation is the primary summary because it uses all areas with their precision. The median gives every area, however noisy, the same weight.

Reliability (odd vs even runs):
- dS per participant: split-half ρ = .57, Spearman-Brown .72.
- Behavioural benefit: split-half ρ = −.27, so the half estimates do not agree.

The benefit difference score is not reliable within participants. Any correlation with it is therefore attenuated towards 0, whatever the true relation.

Verdict: still no relation; the limitation sentence stays, with new numbers and a clearer reason.

> "It did not relate to the behavioral benefit of a distractor at HP across participants (Spearman ρ = −0.02, 95% CI [−0.43, +0.39], p = .94, with each participant's reduction taken from the hierarchical model). The reduction itself was moderately reliable across halves of the runs (split-half ρ = .57), but the behavioral benefit, a difference between two response-time conditions, was not (split-half ρ = −.27), so no between-participant relation could be detected (Hedge et al., 2018)."

## 10. Methods: the gain-field model (replacement text)

**Two corrections to earlier handoffs:**
- **The HRF is not refitted with the gains.** Each participant and area has one HRF, estimated together with the pRFs in the pRF stage (without gain fields) and held fixed in the gain-field fit. Update 3 said otherwise.
- **The voxel selection of the main fits still uses p_signal.** It combines the R² mixture p_signal > 0.5, a DoG R² > 0.05, at most 500 voxels, BOLD SD ≤ 20% and cross-run reliability > 0. This conflicts with the rule against p_signal. The selection with cross-run reliability against a shift null (r > 0.2, r > 0.1) exists only on the earlier, non-zero-sum fits (update 9 A). Rerunning the zero-sum ladder on that selection is about one day of cluster time. Decide whether this is needed. The text below describes what was done.

> "**Gain-field model.** Each voxel's predicted response was its pRF (difference of Gaussians, with the parameters of the pRF fit as starting values) applied to the effective stimulus drive, convolved with the HRF of that participant and area (estimated together with the pRFs and held fixed here). The effective drive was the model input, the mapping bar and the eight search items as identical patches with a relative item weight, multiplied by a gain field (Fig. 4a). The gain field was 1 plus a sustained field and a transient field. The sustained field consisted of Gaussians at the four possible target and distractor locations, with gain g_HP at the HP location of the run and g_LP at the three LP locations, constrained to sum to zero over the four locations (g_HP + 3 g_LP = 0). A component shared by all four locations would be present in every run and is therefore not separable from the pRFs once these are fitted together with the gain fields. The constraint removes it, so that the sustained field captures only how the gain at the HP location differs from that at the other locations (dS = g_HP − g_LP). The transient field consisted of Gaussians at the target and at the distractor while a search display was on, with gain g_T at the target and one gain g_D at the distractor, the same at HP and LP locations. The model thus had six shared parameters per participant and area (item weight, dS, g_T, g_D and the widths of the sustained and of the transient field), on top of each voxel's pRF. Bounds kept the drive non-negative (g_HP between −1 and 3, g_LP ≥ −1, transient gains between −1 and 5, item weight ≥ 0.05, field widths σ between 0.5 and 8°).
>
> All parameters, the per-voxel pRFs and the shared parameters, were fitted together by Levenberg–Marquardt on the BOLD time series of all of the area's voxels and all runs, from two starting points: the fit of the model without a sustained field, and a fit with the pRFs held at their starting values. We kept the solution with the lower error (400 iterations per start). Fits whose item weight or a gain ended on a bound (22 to 27 of 241 participant × area fits per gain, mostly in TO, LO and IPS) were kept; the conclusions are the same without them (Supplementary Table [parameters]). Voxels were selected per area as those with a reliable pRF fit [describe the current selection: p_signal > 0.5 of the R² mixture, R² > 0.05, at most 500 per area, BOLD SD ≤ 20%, and cross-run reliability > 0], 16 to 395 per area (median 131).
>
> **Statistics of the gain-field terms.** For each participant and area, we obtained the standard errors of dS, g_T and g_D by a leave-one-run-out jackknife of the gains re-solved under the model's constraints, with all other parameters held at the joint fit. Half the difference between odd and even runs, divided by this standard error, had a median of 0.73 for dS (0.67 expected). [Hierarchical model paragraph from update 2/3.] Single-area statements rest on each area fitted on its own (the same model without pooling over areas).
>
> **Variants.** As checks, we fitted (i) the same model with a distractor gain that differs between the HP and LP locations (dD), (ii) the same model with separate sustained gains next to and opposite the HP location (g_next = g_LP + δ/3, g_opp = g_LP − 2δ/3, keeping the field zero-sum), and (iii) the same model with Gaussian and divisive-normalization pRFs [pending]. The width of the sustained field was examined by profiling the fit over a grid of widths (Supplementary Fig. [width profiles])."

**Old Methods sentences that must go:**
- The nested series of models with terms added one at a time, and every "estimated alone" result.
- g_LP and g_HP both free, and "a field at HP alone is the special case g_LP = 0, which the data reject".
- The two rounds of alternating Adam / L-BFGS, and the "polish" description.
- The DoG centre-surround tilt (κ) variant.
- The DN baseline variants, unless rerun on the zero-sum DN base (see item 2).
- Leave-one-subject-out (LOSO) generalisation.
- The target-specific HP − LP term (dT).
- Any per-participant field widths and the "reliable gain (|gain| > 2 SE)" width selection.
- Wilcoxon/BH as the main statistics for gain-field terms. They stay as the supplementary check.

## 11. Figure captions, and Fig. 4 against the final model

Source note: I could not open the revised manuscript. The style follows the captions of the v2 manuscript (Google Doc "Retsupp manuscript v2 (for Docky, 13 Nov 2026)") and the update-3 handoff; numbers are from the legend .md files.

**Verdict.** Short captions for Figs. 1, 4, 5 and 7 are below. Fig. 4 did not match the final model, so it is re-rendered as a new version:
- its sustained field had LP = 0, not zero-sum;
- the legend described a "fixed" pRF and a distractor gain that differs between HP and LP.

### Figure 1 (v6, `fig1_paradigm_behaviour_v6_n28_palA`)

> **Figure 1. Participants learned to suppress the high-probability location.** **a, Search display.** Eight items on a ring at 4°, with the target and the distractor at the four diagonal positions, and the mapping bar in its 3.17° aperture. Percentages are over all trials. **b, Trial sequence and design.** Two trials to scale, the bar position every 2 s, the four sweep directions, and the HP location across blocks of runs (example order). **c, Distractor location.** Response time with the distractor at the HP location, at an LP location, or absent. Right, the differences HP − LP, LP − absent and HP − absent. **d, Target location.** Response time with the target at the HP or an LP location, on trials with a distractor. Right, the difference HP − LP. **e, Intertrial priming.** Difference in response time between trials on which the target or the distractor repeats the location of the previous trial and trials on which it does not (raw response times in Supplementary Fig. S1). c–e, n = 28, correct trials. Small dots are participants. In c and d, large points and bars are the mean ± within-participant SEM. In the difference strips and in e, diamonds and bars are the median within-participant difference and its 95% CI. Brackets over c and d mark the differences tested. \*\* p < .01, \*\*\* p < .001.

### Figure 4 (new version, `fig4_model_hp_blocks_zs_palA`)

> **Figure 4. The gain-field model.** **a, Model.** The display is reduced to the model input: the bar and the eight items as identical patches. Each voxel's pRF reads this input multiplied by a gain field. The gain field is 1 plus a sustained field at the four possible target and distractor locations, plus transient fields at the target and at the distractor while a search display is on. The sustained field sums to zero over the four locations, so less gain at the HP location (purple) comes with slightly more at each LP location (gold). Each participant and area has one gain field, shared by all voxels of that area, and the pRFs are fitted together with it. **b, Predicted responses.** A schematic stretch of a run for one pRF near a ring location, in a block with HP near it and in a block with HP at another location: the stimulus reaching the pRF, the gain it sees, and the predicted BOLD response with and without gain. Both panels are schematic, with exaggerated gains.

### Figure 5 (V3, V3AB, hV4 version, `fig5_gain_fields_v11_abc_zs_n28_palA_zsc_V3-V3AB-hV4_bayes`)

> **Figure 5. The gain fields: gain at the target and the distractor, and less sustained gain at the HP location.** Gains are the percentage change in the drive at the centre of a field. They are estimated with DoG pRFs fitted together with the gain fields and combined over participants and areas in a hierarchical model (n = 28). **a, Maps of the fitted gain fields.** The fields over the display in V3, V3AB and hV4, one area per pathway, drawn from the group estimates. These maps show the fitted fields and not data. The HP location is at the top (purple dashed ring), small circles are the eight item positions, and the grey circle is the aperture of the bar. Left, the sustained field at the HP location minus the other locations; the numbers under each map give the gain at the LP and at the HP location. Right, the transient fields at the target (orange) and at the distractor (green) while a search display is on. Their width is set for display and not interpreted. **b, Target and distractor gain.** Left of the divider, all nine areas and each pathway; right, each area. **c, Sustained gain at the HP location minus the LP locations.** Arranged as in b. b, c, Dots are participants (triangles, beyond the axis). Points and bars are the posterior mean and 95% credible interval. Stars give the posterior probability that the effect has the sign shown: \* > .975, \*\* > .995, \*\*\* > .9995. Pale points do not reach .975. Areas are estimated together, each pulled towards its pathway, so no correction across areas is applied; per-area estimates without this pooling are in Supplementary Table S⟦n⟧.

### Figure 7 (Student-t, `fig7_compact_zs_lm_2026-11-13_n28_studentt`)

> **Figure 7. The gain fields fluctuate with trial history and with response time.** Trial-by-trial changes in the gains, estimated per participant and area on top of the participant's gain-field fit and combined in the same hierarchical model as Fig. 5 (n = 28). **a, Trial history.** Target gain when the target repeats the location of the previous trial, and distractor gain when the distractor does, relative to the typical gain of each area. **b, Response time.** Change in the target, distractor and global gain on slower trials, per standard deviation of log response time. Points and bars are the posterior mean and 95% credible interval, over all nine areas and per pathway. Filled points mark effects whose sign is credible (posterior probability above .975). Stars as in Fig. 5.

**Legend .md issue (Fig. 7):** the long legend still says the trial-wise gains sit "on top of each subject's final joint gain-field fit (m7: ...)". They were estimated on the zero-sum main model (m5_TDS_dS_zs). The short caption above avoids the label; the .md should be regenerated with the right fit label (owned by `fig7_compact.py`, not changed here).

### New files

- `notes/figures/paper_2026-11/fig4_model_hp_blocks_zs_palA.{pdf,png,md}`. Option `--zero-sum` in `retsupp/visualize/paper/fig4_model.py`; the default output is unchanged. What changed:
  - the sustained field is zero-sum (illustrative HP −0.6, LP +0.2) on a signed purple/gold scale, labelled "Whole block: less at HP, more at LP";
  - "Fixed pRF" becomes "pRF";
  - panel b labels the LP-side block "Slightly more gain, whole block";
  - the legend describes one distractor gain and pRFs fitted with the gains.

## 12. Supplementary figures and tables cited in the main text

Source note: I could not open the revised manuscript. The citations come from the v2 manuscript text (Google Doc "Retsupp manuscript v2 (for Docky, 13 Nov 2026)", main tab); the order is the order of first citation there. Fit provenance follows the update-3 audit (`notes/data/gain_meta_zs_dog_2026-11-25/README_audit.md`). The revised text may cite them in a different order or drop some; renumber with the same rule (order of first citation, after drops).

Status key:
- **compliant**: no gain-field fit, or already on the joint zero-sum fits.
- **rerun**: needs a rerun or re-render on the joint zero-sum fits, or in the hierarchical-model statistics.
- **drop**: rests on estimated-alone terms or on a model the paper no longer uses.

Files are in `notes/figures/paper_2026-11/`.

### Cited in the v2 text

| Cited as | Cited for | Current file | Fit used | Status | Proposed number |
|---|---|---|---|---|---|
| S1 | Raw response times behind the priming panel (Fig. 1e) | `figS_priming_rt_v4_n28_palA` | Behaviour | compliant | **S1** |
| S10, S11 | Gain fields with Gaussian and DN pRFs (Results, pRF section and robustness paragraph) | `figS_gain_fields_gauss_n28_palA`, `figS_gain_fields_dn_n28_palA` | Gaussian/DN m7_full, not zero-sum, plus estimated-alone dS | rerun (Gaussian/DN zero-sum fits running; then the Fig. 5 layout with `--bayes-dir` per family) | **S10, S11** (become S9, S10 after drops) |
| S21 | Decoding with Gaussian and DN pRFs | `figS_decoding_prf_models_n28_palA` | Decoder pRFs without gain fields | compliant (the open VERIFY in the text is update-4 item 3) | **S20** |
| S3 | Item-location effect per area (V1, V2 q values) | `figS_item_locations_per_area_abs_n28` | Data, no model | compliant; add the precision-weighted own-data statistics (item 13) | **S2** |
| Table S1 | All Fig. 3 statistics | `tableS_item_locations_n28.md` | Data | compliant; add the precision-weighted columns of item 13 | **Table S1** |
| TODO | Including more voxels does not help at the item locations | `figS_item_locations_voxel_selection_n28_palA` | Voxel overlap with the polished m7_full | rerun (local re-render from `items_voxel_overlap_n28_zs_lm_2026-11-16.tsv`) | **S3** |
| S5 | Sum of the fields during a search display | `figS_gain_field_map_search_v3_n28_palA` (and `_zs_n28`) | Joint dD from non-zero-sum fits, estimated-alone dS | drop (Fig. 5a shows the components; the main model has no dD) | none |
| S4 | Field widths (2.8° claim) | `figS_field_widths_v3_n28_palA` | Polished fits, free g_HP/g_LP | drop; replace with `figS_width_profiles_zs_n28_palA` (group width profiles on the zero-sum fits, compliant). The 2.8° sentence is update-4 item 1. | **S4** (width profiles) |
| TODO | The model predicts smaller item-type differences than measured, gets target − distractor right | none (numbers only in `notes/data/items_type_model_vs_measured_n28_zs_lm_2026-11-16.tsv`) | Zero-sum main model | compliant data; needs a new figure or table (local) | **S5** (new) |
| S6 | Target gain vs area pRF size | `figS_gain_vs_prf_size_zs_lm_2026-11-13_zs_n28` | Joint zero-sum | rerun of the statistics only: the text now reports pathway differences from the hierarchical model, and the per-participant slope test is superseded (update 3, section 3) | **S6** |
| S12 | DN baseline variants predict held-out runs worse | `figS_dn_gain_locus_n28_palA` | DN m7_full and variants, not zero-sum, n = 30 | rerun on a zero-sum DN base, or drop (update-4 item 2) | **S12** if kept, else none |
| TODO | Item-specific HP terms (dD, dT) not reliable | `figS_transient_hp_lp_zs_n28_palA` | dD and dT estimated alone | rerun as the joint dD from m7_full_zs; dT dropped unless a zero-sum dT model is fitted (update-4 item 4) | **S7** |
| TODO | Target gain along the dorsal and occipitotemporal pathways | `figS_hierarchy_gain_terms_zs_lm_2026-11-13_zs_n28` (+ `table_hierarchy_gain_terms_zs_lm_2026-11-13_zs_n28`) | DoG joint zero-sum, plus estimated-alone rows and non-zero-sum Gaussian/DN columns | rerun: drop the alone rows, add the hierarchical pathway contrasts and slopes (update 3, section 3), Gaussian/DN columns after their zero-sum fits | **S8** |
| TODO | Decoding robust to voxel selection | `figS_decoding_voxel_selection_n28_palA` | Decoder only | compliant | **S13** |
| TODO | Decoded effect does not grow towards the aperture edge | `figS_decoding_eccentricity_n28_palA` | Data; model bands from the polished m7_full simulation | rerun the model bands (update-4 item 3) or show data only | **S14** |
| TODO (decoding figures) | Effect by eccentricity, per area with both decoders | `figS_decoding_hp_per_area_v10_n28_palA` (+ eccentricity above) | Data; V1 to V3 model lines from the polished simulation | rerun the model lines (item 3) | **S15** |
| TODO | Distractor gain and RT with looser voxel selections | `figS_fig7_relnull_r01_n28_palA`, `figS_fig7_relnull_r02_n28_palA` | Polished fits | rerun (local re-render of `fig7_compact --primary-t` from `trialwise_meta_bayes_relnull_r0.{1,2}_zs_lm_2026-11-24`, after rerunning those with the pathway-mean definition) | **S16** (one figure, both selections) |
| Table S2 | No carry-over of earlier HP locations (Discussion) | `tableS_sessions_blocks_n28.md` | a: data; b: dS per session/block estimated alone | rerun part b on the joint zero-sum fits, or drop part b and keep a | **Table S3** |
| S2 | RT by location; present in the first block | `figS_rt_by_location_n28_palA` | Behaviour | compliant | **S17** (Discussion; or move next to S1 as S1b) |
| Table S5 | Every main result with the two pilots included | `tableS_sensitivity_n30.md` | Mixed: final2 and m7 fits | rerun with the zero-sum fits and the hierarchical model (n = 30 run needs the zero-sum fits of sub-01/02: check they exist) | **Table S5** |
| Table S4 | Out-of-subject prediction (LOSO) per term | `tableS_loso_n28.md` | final2, estimated-alone steps | drop (LOSO rests on terms added alone) | none |
| S8 | LOSO per area | `figS_loso_per_roi_n28_palA` | final2 | drop | none |
| S9 | DoG centre/surround tilt κ (Methods) | `figS_dog_kappa_n28_palA` | Earlier joint fit | rerun on the zero-sum view root (small), or drop with its Methods sentence | **S18** if kept |
| S13 | Gain vs baseline shift at HP (Methods, Discussion) | `figS_baseline_vs_gain_n28_palA` | Earlier joint fit | rerun on the zero-sum view root (update-4 item 5) | **S11** |

### In the earlier numbering but not cited in the v2 text

| Old no. | File | Fit | Status |
|---|---|---|---|
| S7 | `figS_pooled_dS_dD_*` | Estimated alone and joint | drop (superseded by Fig. 5c and the hierarchical model) |
| S14 | `figS_dn_params_by_hp_n28` | DN per-condition fits (exploratory) | drop |
| S15, S16 | `figS_decoding_all_rois_n28_palA`, `figS_decoding_all_rois_examples_n28_palA` | Decoder only | compliant; cite from Fig. 6b if wanted |
| S17 to S19 | `figS_decoding_hp_per_area`, `figS_decoded_bars_rotated`, `figS_decoded_bars_per_roi` | Decoder only | compliant; S17 superseded by v10 above |
| S20 | `figS_decoding_canonical_n28_palA` | Decoder only | compliant, optional |
| S22, S23 | `figS_decode_simulation_n28_palA`, `figS_decoded_map_model_vs_measured_n28_palA` | Polished m7_full simulation | rerun with the zero-sum simulation (item 3) |
| S24 | `figS_bar_crossing_response_n28_palA` | Data | compliant, optional |
| Table S3 | `table_parameters_2026-11_zs_lm_2026-11-13_zs_n28` | DoG zero-sum; "DoG conditional" column alone; DN columns not zero-sum; LP-joint and dD-joint rows | rerun: drop the conditional column and the two rows, DN/Gaussian after their zero-sum fits; could become the per-area table of the hierarchical model |

### New supplements to cite (updates 13 to 15)

| Proposed number | Content | File | Status |
|---|---|---|---|
| S4 | Group width profiles of the sustained and transient fields (replaces the field-width figure) | `figS_width_profiles_zs_n28_palA` | compliant |
| Table S4 | Per-area estimates fitted without pooling (each area on its own data) next to the pooled hierarchical estimates, for dS, target and distractor; calibration of the SEs per area | from `notes/data/gain_meta_zs_dog_2026-11-25/no_pooling_per_area.tsv` and `calibration.tsv` | compliant data; table to be written (local) |
| Table S6 | Supplementary check: medians, Wilcoxon p and BH q next to the hierarchical estimates, per area and pathway | `notes/data/gain_meta_zs_dog_2026-11-25/wilcoxon.tsv` + `bayes.tsv` | compliant data; table to be written (local) |
| (not proposed) | dS across 19 fit versions | `figS_dS_overview_n28` | under the fixed rule this shows dropped fits; keep only as an internal provenance note, not in the paper |

### Proposed final numbering (in order of first citation in v2, after drops)

**Figures:**

| No. | Content |
|---|---|
| S1 | Priming RTs |
| S2 | Item locations per area |
| S3 | Item locations, voxel selection |
| S4 | Width profiles |
| S5 | Item types, model vs measured (new) |
| S6 | Gain vs pRF size |
| S7 | Joint dD |
| S8 | Hierarchy |
| S9, S10 | Gain fields with Gaussian and DN pRFs |
| S11 | Gain vs baseline |
| S12 | DN baseline variants, if kept |
| S13 | Decoding, voxel selection |
| S14 | Decoding by eccentricity |
| S15 | Decoding per area |
| S16 | Fig. 7 with looser voxel selections |
| S17 | RT by location |
| S18 | κ, if kept |
| S19 | Decoding with canonical HRF, if cited |
| S20 | Decoding with Gaussian and DN pRFs |

**Tables:**

| No. | Content |
|---|---|
| S1 | Item locations |
| S2 | Item locations, hierarchical (when ready) |
| S3 | Sessions and blocks |
| S4 | No-pooling per area |
| S5 | Sensitivity with n = 30 |
| S6 | Wilcoxon check |

The S10/S11 and S20 citations in the pRF paragraph come before S2 in the text. If the revised text keeps that, move the Gaussian/DN figures to S2 and S3 and shift the rest.

## 13. Fig. 3 with each participant weighted by the precision of their estimate (new)

**Inputs.** The manuscript's Fig. 3 comes from `fig3_item_locations.py` (n = 28 via `run_excluding`) on `notes/data/event_deconv_final2_2026-11/` (item × HP/LP) and `event_deconv_relative_final2_2026-11/` (item × HP / next / opposite). These were deconvolved from the cluster's `paper_2026-11/location_timeseries_final2`. Reproduced exactly:
- early target − neutral +0.125% signal (mean);
- early HP − opposite −21.6%, p = .005;
- early next − opposite −14.9 ± 6.0%, p = .022;
- all areas −15.5%, p = .013;
- V1 q = .015, V2 q = .031.

**SEs.** A new option, `deconvolve_location_responses.py --jackknife`, fits all runs, each run left out, and odd and even runs; the default is unchanged. Job script: `slurm_jobs/deconv_jackknife.sh`. SLURM 8836319, sub-03 to sub-30, about 5.5 min and 0.6 GB per task, 1 CPU. The all-run fit reproduces the final2 deconvolution exactly (max |difference| 0). Outputs: `/data/ds-retsupp/derivatives/paper_2026-11/event_deconv_final2_jk_{plain,relative}_2026-11-25/` (also on the cluster).

Per participant × area, each contrast over 3.2 to 6.4 s gets a leave-one-run-out jackknife SE. Calibration (half the odd − even difference over the SE; 0.67 expected) is 0.60 to 0.79 overall; the highest is early HP − opposite (abs) at 0.82, so its SEs are somewhat small.

**Units.** Absolute % signal is the primary unit for the location contrasts. Relative units (% of the participant's mean response, as Fig. 3e) are reported alongside. Their SEs also carry the noise of each participant's denominator, which is near zero in some cells: the V1 Wilcoxon mean is −53% and TO −111%, against medians of −30% and −25%.

**Statistics.** The primary analysis keeps the manuscript's units: each pathway, and all areas, with voxels pooled as in Fig. 3. Each unit is fitted on its own, with no pooling over areas: y ~ Student-t(ν, μ, √(SE² + τ²)) over participants, where SE is each participant's jackknife SE. This differs from the manuscript's Wilcoxon only in weighting each participant by the precision of their estimate. Single areas are fitted the same way, each on its own data. As a check, the hierarchical model over areas (pathway = mean of its areas, as for the gain fields) gives the same conclusions. No divergences in any fit.

Code: `retsupp/visualize/paper/fig3_meta.py` (`--unit-own-data`). Outputs: `notes/data/fig3_meta_units_2026-11-25/own_data_units.tsv` for pathways and all areas, and `notes/data/fig3_meta_2026-11-25/` for the hierarchical model and single areas.

### Item contrasts (% signal; n = 28, dorsal 26)

| Contrast | Unit | Wilcoxon (manuscript), mean, q | Weighted by SE, mean [95% HDI], P |
|---|---|---|---|
| Target − neutral | Early / dorsal / occ.-temporal | +0.125 / +0.264 / +0.236, all q < 10⁻⁵ | +0.115 [+0.088, +0.141] / +0.238 [+0.188, +0.288] / +0.228 [+0.188, +0.269], all P > .999 |
| | V1 (own data) | +0.060, q < .001 | +0.050, P = .997 |
| Distractor − neutral | Early / dorsal / occ.-temporal | +0.072 / +0.149 / +0.156, all q ≤ 10⁻⁴ | +0.057 [+0.021, +0.089] / +0.114 [+0.065, +0.165] / +0.125 [+0.081, +0.169], all P ≥ .999; every area credible on its own data |
| Target − distractor | Early / dorsal / occ.-temporal | +0.045 (q = .015) / +0.126 / +0.105 (q < .001) | +0.056 [+0.023, +0.090], P = .999 / +0.135 / +0.120, P > .999 |
| | V1 | −0.018, q = .57 | −0.014, P = .71 (not credible, as before) |

**Verdict, item contrasts:** unchanged. The target and the distractor are above neutral in every pathway and every area. The target is above the distractor in every pathway, but not in V1.

### Location contrasts (absolute % signal, primary; relative % in brackets)

| Contrast | Unit | Wilcoxon (manuscript) | Weighted by SE, own data |
|---|---|---|---|
| HP − opposite | All areas | −0.083, p = .0076 (rel −15.5%, p = .013) | **−0.062 [−0.126, −0.003], P = .978** (rel −11.8%, P = .971) |
| | Early | −0.092, p = .0048 (rel −21.6%, p = .0052) | **−0.069 [−0.127, −0.011], P = .986** (rel −15.2% [−29.3, −1.6], P = .984) |
| | Dorsal | −0.086, p = .089 | −0.062, P = .90 |
| | Occ.-temporal | −0.045, p = .11 (rel −9.8%, p = .090) | −0.079 [−0.159, +0.003], P = .968 (rel −9.2%, P = .948) |
| | V1 / V2 / TO | V1 −0.101, V2 −0.145 (q = .029); TO −0.147 (q = .041) | V1 −0.067, P = .962 / **V2 −0.101 [−0.165, −0.039], P = .998** / **TO −0.116 [−0.195, −0.028], P = .993** |
| Next − opposite | Early | −0.050, p = .038 (rel −14.9 ± 6.0%, p = .022) | −0.030 [−0.070, +0.010], P = .93 (rel −4.7% [−14.0, +4.4], P = .84) |
| | All areas | −0.037, p = .15 (rel −8.6%, p = .060) | −0.021, P = .84 |
| | V2 | −0.088 | −0.047, P = .968 (V1, V3 P ≤ .83) |
| HP − next | All areas | −0.047, p = .0071 (rel −6.9%, p = .018) | −0.027 [−0.062, +0.008], P = .94 (rel −5.7%, P = .953) |
| | Early / occ.-temporal | −0.042, p = .066 / −0.029, p = .22 | −0.013, P = .74 / **−0.056 [−0.110, −0.001], P = .976** |
| | TO | | **−0.084, P = .995** |

**Why the weighting matters.** On the location contrasts, the participants with the largest negative values are also the noisiest. In early areas, next − opposite has sub-25 at −0.44 and sub-20 at −0.27, each with an SE of about 0.36 against a median of 0.12; the Spearman correlation of value and SE over participants is −0.27, and −0.38 in relative units. Weighting by precision lowers these participants' influence. A standard random-effects meta-analysis gives the same answer as the Student-t model: −0.029 ± 0.018 against −0.050 unweighted. The SEs are trustworthy: half the odd − even difference over the SE is 0.65 for this contrast (0.67 expected). Pooling over areas plays no part; the hierarchical model over areas gives the same numbers (early next − opposite −0.030, P = .94).

**Verdicts, location contrasts:**
- **HP lowest: holds.** HP is below the opposite location over all areas and in early visual cortex, in both units, and on its own data in V2 and TO; V1 points the same way (P = .96).
- **Next below opposite: a trend, not a finding.** The direction is the same in every analysis and in every pathway, but weighted by precision the early difference is −0.030% signal (P = .93) and −4.7% in relative units (P = .84), against −14.9% (p = .022) unweighted. Report it with that caveat. Do not claim the HP < next < opposite ordering (Page's L) as a result.
- **HP below next:** not reliable over all areas or in early areas once weighted (P = .94 and .74). It is credible in occipitotemporal areas (P = .976), and in TO on its own data.
- **Pathways:** the HP − opposite reduction is credible in early visual cortex. Occipitotemporal areas point the same way with a similar size (P = .968, or P = .996 with areas as the units of the hierarchical model). Dorsal is not credible. So "the reduction is found in early visual cortex" is safe; "only in early visual cortex" is not.

**Consistency with Fig. 5c and 6c.** All three put the HP reduction in early visual cortex. Fig. 3 and Fig. 5c (dS, occipitotemporal −2.2%, P = .996) also point to occipitotemporal areas.

Not done: Fig. 3 re-rendered with these statistics. The figure can stay as it is: its brackets show the manuscript's tests, and the legend can point to Supplementary Table S1 for the weighted ones.

**Suggested Results text for Fig. 3 (location part):**

> "We then asked whether the response at an item location also depends on where that location lies relative to the HP location. In line with learned suppression, the response was lowest at the HP location: in early visual cortex it was 0.069% signal below the location opposite HP (95% HDI [−0.127, −0.011], P = .986, with each participant weighted by the precision of their estimate; unweighted −0.092%, p = .005), and over all areas 0.062% (P = .978). Of the single areas, V2 and TO showed the reduction on their own data, and V1 pointed the same way (P = .96). The locations next to HP tended to respond less than the location opposite HP (early, unweighted −14.9 ± 6.0%, p = .022), but this difference came mostly from participants with noisy estimates and was not reliable once these were weighted by their precision (−0.030% signal, P = .93). The location effect did not depend on what appeared there [keep the location × item type statement]."

Change: "This order, lowest at HP, then next to HP, then opposite HP, was reliable (Page's L test, p = .0016)". Either delete it or move it, with the caveat above, to Supplementary Table S1. The relative-unit and Wilcoxon statistics can stay in Table S1 next to the weighted ones.

### Why the gain-field model: the item-location responses are noisier per participant (new argument)

The same jackknife SEs let the two analyses be compared on one scale. For each participant × area cell we take the estimate divided by its SE (z), so units do not matter. If there were no effect and the SEs were right, the median |z| would be 0.67. Values are corrected for each analysis's own SE calibration (odd/even check).

| Pathway | Gain field, dS | HP − opposite (measured) | HP − next (measured) | Next − opposite (measured) |
|---|---|---|---|---|
| Early | **1.22** | 0.56 | 0.80 | 0.69 |
| Dorsal | **1.05** | 0.58 | 0.45 | 0.63 |
| Occipitotemporal | **1.22** | 0.66 | 0.72 | 0.53 |
| All areas | **1.26** | 0.60 | 0.69 | 0.61 |

(Median |z| over participant × area cells, n = 241 for dS and 219 for the measured contrasts. Without the calibration correction: dS 1.36, measured 0.63 to 0.73.)

**Reading.**
- In a single participant and area, the measured responses at the item locations cannot separate the HP effect from noise: their median |z| is at the level expected from noise alone.
- The gain-field estimate of the same effect stands about 1.2 SE from zero, about twice as far.
- The reason is that the model uses every time point of the run: the bar sweeps over the HP and LP locations throughout, not only in the few seconds after each search display. It summarises the effect in one parameter per participant and area, constrained by each voxel's pRF.
- The group-level item-location effect is still credible (HP − opposite, early P = .986), because it averages 28 participants. But its finer structure (next vs opposite, single areas) is at the edge of what these data can show. That explains the fragility found above.

**Caveats to keep in the text.**
- The precision comes from the model's assumptions: the pRF, the HRF and the shape of the field. The argument is therefore only as good as the model's fit, and the held-out tests of the model support that fit.
- It concerns precision within a participant and area. Individual differences in dS are only moderately reliable: odd and even runs agree with ρ = .57 over all areas (item 9) and ρ = .15 in early areas alone.

**Suggested sentence (Results, after Fig. 3, or Discussion):**

> "These model-free responses were noisy at the level of single participants: within a participant and area, the HP − opposite difference was typically no larger than its standard error (median |estimate / SE| 0.6, the value expected from noise alone), whereas the HP − LP difference of the gain-field model, which uses the response to the mapping bar over the whole run, was typically 1.2 standard errors from zero. This is why we estimate the effect of the HP location with the gain-field model, and use the item-location responses as a model-free check of its direction."
