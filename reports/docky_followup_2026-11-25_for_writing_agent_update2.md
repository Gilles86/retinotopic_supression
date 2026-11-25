# Update 2 for the writing agent: gain-field statistics from the hierarchical model (25 Nov 2026)

Applies on top of `reports/docky_followup_2026-11-25_for_writing_agent.md`. Where the two disagree, this file wins.

## What changed and why

Group statistics for the gain-field terms (dS, target gain, distractor gain; Fig. 5) and the trial-wise effects (Fig. 7) now come from one model: a hierarchical Bayesian model with a Student-t likelihood that weights each participant × area estimate by its precision. Before, Fig. 5 used per-area medians with Wilcoxon tests and BH correction over nine areas. That treated very noisy estimates like precise ones, so per-area significance flickered between fits. The new model makes the per-area results stable across fits and needs no correction over areas.

Reporting conventions, for every gain-field and trial-wise number:
- posterior mean, 95% highest-density interval (HDI), and P, the posterior probability of the sign shown;
- "credible" means P > .975, and the stars are * > .975, ** > .995, *** > .9995;
- the pathway value is the mean of its areas, in the same way as the all-areas value is the mean of all nine;
- n = 28.

Keep the median/Wilcoxon results as a supplementary check. Do not use the old per-area BH q values for gain-field terms anywhere in the main text.

Figures:
- Fig. 5: `notes/figures/paper_2026-11/fig5_gain_fields_v11_abc_zs_n28_palA_bayes.pdf` (with its legend `.md`). It replaces `fig5_gain_fields_v11_abc_zs_n28_palA`. Panels b and c now also show all areas and the three pathways on the left. Panel a maps draw the same estimates.
- Fig. 7: `notes/figures/paper_2026-11/fig7_compact_zs_lm_2026-11-13_n28_studentt.pdf` (with its legend `.md`). It replaces `fig7_compact_zs_lm_2026-11-13_n28`.

## Methods text (Statistics, gain-field terms)

> "For each participant and area, we re-estimated the gains around the joint fit, solving for the gains of the sustained and transient fields by least squares with the pRFs, the HRF and the field widths held at the joint fit. We obtained the standard error of each gain by a leave-one-run-out jackknife, which allows for noise that is correlated over time and between voxels. Half the difference between odd and even runs, divided by this standard error, had a median of 0.71 for dS (0.67 is expected when the standard error is correct). We combined the estimates over participants and areas in a hierarchical Bayesian model, the same as for the trial-wise gains. Each estimate followed a Student-t distribution around its true value, with a scale equal to the square root of the squared standard error plus a residual variance; the degrees of freedom were estimated (prior Gamma(2, 0.1)). The true value was the sum of a mean per pathway (early: V1, V2, V3; dorsal: V3AB, IPS; occipitotemporal: hV4, VO, LO, TO), a deviation per area and a deviation per participant, with half-normal priors on their standard deviations. Areas are thereby pooled towards their pathway, so no correction across areas is needed. We report the posterior mean, the 95% highest-density interval and the posterior probability of the sign of the effect. The overall effect is the mean of the nine area means, and the effect of a pathway is the mean of its areas' means. Models were fitted in PyMC with the nutpie sampler (4 chains × 2000 draws); there were no divergences, and all R̂ ≤ 1.01."

The Fig. 7 Methods text stays as it is, but the likelihood is now Student-t rather than normal, and a pathway is now the mean of its areas.

## Fig. 5 numbers (replace update 13's)

| | All areas | Early | Dorsal | Occipitotemporal |
|---|---|---|---|---|
| dS (HP − LP) | **−3.6% [−5.2, −1.9], P > .999** | **−6.3% [−8.9, −3.8], P > .999** | **−2.2% [−4.3, +0.0], P = .977** | **−2.2% [−4.0, −0.4], P = .995** |
| Target gain | **+22.9% [+15.0, +30.7], P > .999** | +8.4% [−0.4, +17.3], P = .974 | **+29.8% [+18.0, +41.8], P > .999** | **+30.2% [+20.4, +40.1], P > .999** |
| Distractor gain | **+4.5% [+0.5, +8.6], P = .988** | +0.9% [−4.1, +6.2], P = .635 | **+6.3% [+0.7, +12.0], P = .987** | **+6.3% [+1.7, +11.0], P = .997** |

Per area (bold = credible):

| Area | dS | Target | Distractor |
|---|---|---|---|
| V1 | **−6.1% [−9.4, −2.8], P > .999** | −7.5% [−17.7, +2.4], P = .931 | +0.9%, P = .615 |
| V2 | **−6.4% [−9.4, −3.1], P > .999** | +5.3% [−5.1, +16.5], P = .837 | −0.8%, P = .597 |
| V3 | **−6.4% [−10.1, −3.0], P > .999** | **+27.4% [+10.4, +45.8], P > .999** | +2.7%, P = .741 |
| V3AB | **−2.3% [−4.8, −0.1], P = .976** | **+34.6% [+23.0, +47.6], P > .999** | **+8.1% [+2.2, +14.0], P = .997** |
| IPS | −2.0% [−4.7, +0.8], P = .928 | **+25.1% [+7.9, +41.5], P > .999** | +4.6%, P = .890 |
| hV4 | **−4.4% [−7.7, −1.1], P = .998** | **+52.5% [+33.7, +70.4], P > .999** | **+11.3% [+3.4, +19.2], P > .999** |
| VO | −0.7% [−3.1, +1.9], P = .707 | **+24.5% [+7.3, +43.7], P = .999** | +5.9% [−0.3, +12.4], P = .971 |
| LO | −1.6% [−3.8, +0.6], P = .935 | **+23.2% [+9.9, +36.4], P > .999** | +3.8%, P = .894 |
| TO | **−2.1% [−4.2, −0.2], P = .985** | **+20.6% [+8.2, +33.7], P > .999** | +4.4%, P = .939 |

Suggested Results text for dS:

> "The sustained gain was lower at the HP location than at the LP locations: by 3.6% of the drive over all nine areas (95% HDI [−5.2, −1.9], P > .999). The reduction was largest in early visual cortex (V1 to V3, −6.3% [−8.9, −3.8]), smaller but also credible in dorsal (−2.2%) and occipitotemporal areas (−2.2%). Per area, it was credible in V1, V2, V3, V3AB, hV4 and TO."

## Sentences that change as a result

1. **Abstract, size of the reduction.** It is now about a sixth of the target gain (ratio of the posterior means −3.6 / 22.9 = −0.16), not a quarter.
   > "At the high-probability location, the gain was reduced by a few percent of the drive, about a sixth of the gain at the target, throughout the block."
2. **Abstract and Discussion: "the salient distractor received about half this gain".** It is now about a fifth (+4.5% vs +22.9%):
   > "The salient distractor received a much smaller gain, about a fifth of the target gain, in dorsal and occipitotemporal areas."
   Also change the Discussion's "about twice as much gain as the distractor" to "about five times as much".
3. **"Most clearly in V3AB, hV4 and TO" (distractor).** Now "in V3AB and hV4, and in the dorsal and occipitotemporal pathways; not in early visual cortex".
4. **Where the HP reduction is.** It is no longer "most reliable in V3, hV4 and TO" or "not best characterized as an early visual phenomenon". It is largest and credible in V1 to V3, credible in hV4, V3AB and TO, and present in all three pathways. The Discussion's localisation sentence (learned suppression "localized in areas where receptive fields begin to grow large") is not supported and should go. Suggested replacement:
   > "The reduction was present along the whole visual hierarchy and largest in early visual cortex."
   Hierarchy statement stays: dS does not grow along the hierarchy.
5. **Target gain in V1 and V2.** Still near zero (V1 −7.5%, V2 +5.3%, both not credible). The early pathway mean is +8.4% (P = .974), just short.
6. **Item 0 of the previous handoff** (target +23%, distractor +11%, dS −1.5% alone / −5.5% joint, hV4 q = .022, ratio −0.24): use the numbers above instead. "Estimated alone" vs "joint" no longer needs to appear in the main text for dS. The joint fit is the model, and these are its gains with their uncertainty.

## Fig. 7 numbers (Student-t; replace update 13's)

| Effect | All areas | Early | Dorsal | Occipitotemporal |
|---|---|---|---|---|
| Target gain on target repeats (% of typical gain) | **−17.1% [−26.4, −6.9], P > .999** | **−28.5% [−50.4, −8.6], P = .997** | **−19.4% [−36.2, −2.8], P = .989** | **−7.4% [−16.0, +0.1], P = .977** |
| Distractor gain on distractor repeats | −17.6% [−50.0, +13.1], P = .883 | −11.3%, P = .686 | −32.0%, P = .827 | −15.1%, P = .828 |
| Distractor gain per SD of log RT (× 100) | **−2.7 [−4.7, −0.8], P = .996** | **−2.9 [−5.4, −0.1], P = .982** | **−3.8 [−6.9, −0.7], P = .993** | −2.0 [−4.2, +0.3], P = .967 |
| Global gain per SD of log RT | −2.2 [−5.1, +0.6], P = .945 | −2.6, P = .965 | +0.9, P = .661 | **−3.4 [−6.4, −0.6], P = .989** |
| Target gain per SD of log RT | +0.8 [−1.3, +2.9], P = .781 | +1.9, P = .876 | −0.9, P = .717 | +0.8, P = .772 |

In absolute gain the target-repeat effect is −5.9% [−9.6, −2.3], P > .999.

Sentences that change:
- **"This decrease was present in all three pathways, although none of them excluded zero on its own"** (target repeat). It is now credible in every pathway:
  > "This decrease was present in all three pathways (early −29%, dorsal −19%, occipitotemporal −7%; each P > .975)."
- **"This decrease had the same direction in all three pathways, although none of them excluded zero on its own"** (distractor and RT). Now:
  > "This decrease was credible in early (−2.9%) and dorsal areas (−3.8%) and had the same direction in occipitotemporal areas (−2.0%, P = .967)."
- **Target repeat size.** "by 17% of the typical target gain" (was 18% or 19%).
- **Global gain and RT.** "pointed in the same direction, but not credibly (−2.2% [−5.1, +0.6], P = .945)"; it is credible in occipitotemporal areas alone (−3.4%, P = .989).
- **Looser voxel selections (previous handoff, 5b).** Those numbers came from the normal likelihood. The direction and wording there still hold.

## Unchanged

Behaviour, item responses (Fig. 3), decoding (Fig. 6) and the divisive-vs-subtractive, LP-gradient and repeat interpretations in the previous handoff are unchanged. They do not use the gain-field group statistics. The p-value table in the full report covers those sections; its gain-field rows are superseded by this file.
