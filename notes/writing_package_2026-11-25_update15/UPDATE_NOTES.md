# Update 15, 25 Nov 2026: corrected hierarchical numbers, checks, new Fig. 5

Apply on top of update 14. It replaces update 14's part B (hierarchical-model numbers) and its Fig. 5.

**Model choice, fixed from now on:** only the joint zero-sum gain-field fits are used (item weight, target gain, one distractor gain, zero-sum sustained field; pRFs and HRF fitted together with the gains). Drop "estimated alone" numbers and unconstrained fits everywhere.

**Why the numbers change slightly:** update 14's hierarchical numbers came from a re-solve that left the HP and LP sustained gains free and split the distractor gain into HP and LP. Now each estimate is the joint zero-sum fit's own gain, with a jackknife SE computed under the same constraints. The conclusions do not change.

**Statistics:** posterior mean, 95% HDI, P = posterior probability of the sign; credible means P > .975; n = 28. A pathway is the mean of its areas.

## Contents

| File | What |
|---|---|
| `figures/main/fig5_gain_fields_v11_abc_zs_n28_palA_zsc_V3-V3AB-hV4_bayes` | **Fig. 5 (proposed).** Maps for V3, V3AB and hV4 (one area per pathway, each with all three components); b and c with all areas and pathways |
| `figures/alternative/fig5_gain_fields_v11_abc_zs_n28_palA_zsc_bayes` | The same figure with the current map areas (V3, hV4, TO), in case you prefer them |
| `figures/main/fig7_compact_zs_lm_2026-11-13_n28_studentt` | Fig. 7, Student-t (unchanged since update 14, included for completeness) |
| `parts/A_update3_checks.md` | The checks: Wilcoxon vs hierarchical model, calibration, hierarchy slopes, decoding through the same model, and which figures and tables still use other fits |

## Numbers to use (replace update 14, part B)

| | All areas | Early | Dorsal | Occipitotemporal | Areas credible |
|---|---|---|---|---|---|
| dS (HP − LP) | **−3.2% [−4.8, −1.6], P > .999** | **−5.5% [−8.0, −3.0], P > .999** | **−1.9% [−3.7, −0.0], P = .977** | **−2.2% [−3.9, −0.6], P = .996** | V1 −4.7, V2 −5.8, V3 −6.0, hV4 −5.0, LO −2.2 |
| Target gain | **+23.5% [+15.3, +31.9], P > .999** | +9.4% [−0.3, +19.2], P = .974 | **+30.6% [+19.1, +43.1]** | **+30.5% [+20.4, +40.6]** | V3, V3AB, IPS, hV4, VO, LO, TO |
| Distractor gain | **+4.7% [+0.9, +8.9], P = .988** | +1.1%, P = .66 | **+6.0% [+0.4, +11.6], P = .982** | **+6.7% [+2.4, +11.2], P = .998** | V3AB +8.3, hV4 +12.8, VO +6.3 |

All per-area values, with HDIs, are in part A, section 1.

## Text changes

1. **Abstract.**
   - The HP reduction is about a seventh of the target gain (−3.2 / 23.5).
   - The distractor gain is about a fifth of the target gain (+4.7 / 23.5).
   - Suggested: "the gain at the high-probability location was reduced by a few percent of the drive, most strongly in early visual cortex".
2. **Where the reduction is.**
   > "The sustained gain was lower at the HP location than at the LP locations (−3.2% of the drive over all nine areas). The reduction was largest in early visual cortex (V1 to V3, −5.5%), smaller in dorsal (−1.9%) and occipitotemporal areas (−2.2%), and credible on their own data in V3 and hV4; V1, V2 and V3AB pointed the same way." Per-area credibility in the pooled model partly borrows from neighbouring areas (part A, section 8): do not claim LO, and base claims on all areas and the pathways.

   The early reduction is larger than the dorsal and occipitotemporal ones (+3.6 and +3.3 points, both P = .998). Remove "not best characterized as an early visual phenomenon" and "localized in areas where receptive fields begin to grow large".
3. **Hierarchy** (part A, section 3).
   - "dS did not change with pRF size" becomes "the reduction became smaller in areas with larger pRFs (+3.2 points per doubling of pRF size, P = .993)".
   - "The target gain grew along the dorsal and occipitotemporal pathways (+27 points per doubling)" becomes "the target gain was larger in dorsal and occipitotemporal areas than in early visual cortex (by 21 points, P > .999)".
   - "The target gain grew more than dS" becomes "relative to its size, the target gain grew along the hierarchy while the reduction at HP shrank (P = .979)".
   - Delete the sentence on the width of the sustained field: it is not testable per area.
   - Keep the within-participant relation with new numbers: ρ = −0.23 [−0.37, −0.07], 20 of 28 participants, p = .0055, computed across odd and even runs so that estimation errors are independent.
4. **Distractor.** "Most clearly in V3AB, hV4 and TO" becomes "in dorsal and occipitotemporal areas (credible in V3AB, hV4 and VO), not in early visual cortex".
5. **Decoding (optional).** Through the same model, the decoded-bar effect is credible in V3 (−1.40% [−2.81, −0.06], P = .984) and nearly in the early pathway (P = .968), but not over all areas (P = .90). It points to early areas, as dS does. If you report decoding with the same model, soften "the effective drive of the bar was lower" to "lower in early visual cortex".

## Still open

- **Gaussian and DN robustness.** The zero-sum joint fits are running on the cluster (jobs 8834744, 8834745). Until they are in, do not change the robustness sentences ("DoG −3.5%, DN −7.8%, Gaussian −8.8%"; "the DoG model gave the smallest estimate"), and mark them as pending.
- **Fig. 6e, f model prediction.** It still comes from the old non-zero-sum fit, so the three model-prediction numbers in the decoding text will change after a rerun.
