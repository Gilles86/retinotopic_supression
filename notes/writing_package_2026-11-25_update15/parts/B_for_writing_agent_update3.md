# Update 3 for the writing agent (25 Nov 2026)

This replaces the numbers and figures of `docky_followup_2026-11-25_for_writing_agent_update2.md`. Where the two disagree, this file wins. Everything in update 2 that is not mentioned here still applies: the statistics convention, the Methods paragraph (with the change in section 2), the Fig. 7 numbers and the abstract sentence on distractor vs target.

Background, with all checks and numbers: `reports/docky_followup_2026-11-25_update3_checks.md`. Figures and notes as a package: `notes/writing_package_2026-11-25_update15/` (zip next to it).

## 1. Model choice, fixed from now on

Use only the **joint zero-sum gain-field fits**: item weight, target gain, one distractor gain and a zero-sum sustained field (g_HP + 3 g_LP = 0), with each voxel's pRF and each area's HRF fitted together with the gains.

Do not use:
- "estimated alone" numbers;
- unconstrained (non-zero-sum) fits.

This also applies in the robustness and hierarchy sentences.

The reason is set in advance: the joint fit estimates each term with the others in the model, and the zero-sum constraint removes a sustained gain shared by all four locations that cannot be estimated (it trades off with the item weight and the pRF amplitude).

## 2. Small correction to update 2's numbers

Update 2's numbers came from a re-solve that left the HP and LP sustained gains free and split the distractor gain in two. Each estimate is now the joint zero-sum fit's own gain, with a leave-one-run-out jackknife SE computed under the same constraints. The values move slightly; the conclusions do not.

Methods paragraph from update 2: replace its first two sentences with:

> "For each participant and area, we took the three gains of the joint fit (the HP − LP difference of the sustained field, the target gain and the distractor gain) and obtained their standard errors by a leave-one-run-out jackknife. For this, the gains were re-solved by least squares under the model's constraints (zero-sum sustained field, one distractor gain), with the pRFs, the HRF, the field widths and the item weight held at the joint fit. Half the difference between odd and even runs, divided by this standard error, had a median of 0.73 for dS (0.67 is expected when the standard error is correct)."

## 3. Numbers to use

Posterior mean, 95% HDI, P = posterior probability of the sign. Credible means P > .975. n = 28. A pathway is the mean of its areas.

| | All areas | Early (V1 to V3) | Dorsal | Occipitotemporal |
|---|---|---|---|---|
| dS (HP − LP) | **−3.2% [−4.8, −1.6], P > .999** | **−5.5% [−8.0, −3.0], P > .999** | **−1.9% [−3.7, −0.0], P = .977** | **−2.2% [−3.9, −0.6], P = .996** |
| Target gain | **+23.5% [+15.3, +31.9], P > .999** | +9.4% [−0.3, +19.2], P = .974 | **+30.6% [+19.1, +43.1], P > .999** | **+30.5% [+20.4, +40.6], P > .999** |
| Distractor gain | **+4.7% [+0.9, +8.9], P = .988** | +1.1% [−4.0, +6.4], P = .66 | **+6.0% [+0.4, +11.6], P = .982** | **+6.7% [+2.4, +11.2], P = .998** |

Differences between pathways:
- dS: early is larger than dorsal by 3.6 points [+1.2, +6.1] and larger than occipitotemporal by 3.3 points [+1.1, +5.7], both P = .998.
- Target gain: dorsal and occipitotemporal are each above early by 21 points ([+9, +34] and [+11, +32]), both P > .999.

**Single areas: claim only what holds on the area's own data.** In the pooled model, an area is pulled towards its pathway and its participants' other areas, so a noisy area can look credible because its neighbours are. For single-area statements, use each area fitted on its own:

| Term | Credible on own data | Points the same way (P .94 to .96) | Do not claim |
|---|---|---|---|
| dS | V3 −9.4% [−16.3, −2.4]; hV4 −5.3% [−9.1, −1.4] | V1, V2, V3AB, VO | LO (credible only when pooled) |
| Target gain | V3, V3AB, IPS, hV4, VO, LO, TO | | |
| Distractor gain | V3AB, hV4, VO | | |

Fig. 5c keeps the stars of the pooled model. Its legend says so.

## 4. Sentences to write or change

1. **Abstract.**
   - The HP reduction is "a few percent of the drive, largest in early visual cortex", about a seventh of the target gain (−3.2 / 23.5). Update 2 said a sixth.
   - The distractor gain is about a fifth of the target gain (update 2's wording stands).
2. **Results, the HP reduction:**
   > "The sustained gain was lower at the HP location than at the LP locations, by 3.2% of the drive over all nine areas (95% HDI [−4.8, −1.6], P > .999). The reduction was largest in early visual cortex (V1 to V3, −5.5% [−8.0, −3.0]) and smaller in dorsal (−1.9%) and occipitotemporal areas (−2.2%), each credible. Of the single areas, V3 and hV4 showed it on their own data; V1, V2 and V3AB pointed the same way."
3. **Discussion: where suppression arises.** Delete:
   - "not best characterized as an early visual phenomenon";
   - "localized in areas where receptive fields begin to grow large".

   The data say the reduction is present along the hierarchy and largest early.
4. **Hierarchy** (replace each sentence):
   - "dS did not change with pRF size (−0.7 pp per doubling, p = .61)" becomes "the reduction became smaller in areas with larger pRFs (+3.2 percentage points per doubling of the area's mean pRF size, [+0.8, +5.6], P = .993)".
   - "The target gain grew along the dorsal and occipitotemporal pathways (p = .011, .003)" and "+27 pp per doubling" become "the target gain was larger in dorsal and occipitotemporal areas than in early visual cortex (by 21 percentage points, P > .999)". Do not give a slope; over nine areas it is not credible (+12 [−17, +39], P = .83).
   - "The target gain grew more with pRF size than dS (p = .018)" becomes "relative to its size, the target gain grew along the hierarchy while the reduction at HP shrank (P = .979)".
   - Delete "the width of the sustained field did not follow pRF size". It cannot be tested per area; keep the group width (FWHM 4 to 7°).
   - Keep, with new numbers: "within participants, areas with a larger target gain had a larger reduction at HP (ρ = −0.23 [−0.37, −0.07], 20 of 28 participants, p = .0055; target and dS from different halves of the runs, so that their estimation errors are independent)".
5. **Distractor gain, where.** "Most clearly in V3AB, hV4 and TO" becomes "in dorsal and occipitotemporal areas (V3AB, hV4 and VO), not in early visual cortex".
6. **Target gain in V1 and V2.** Near zero in the pooled model (V1 −5.9%, V2 +6.7%, both not credible). In V1 on its own data it is even slightly negative (−10% [−17, −3]). Keep the existing caveat that the model may miss early target gain (the measured target − neutral response in V1 and V2, Fig. 3).
7. **Decoding (Fig. 6c), optional.** Keep the current Wilcoxon statistics for Fig. 6 unless you want one model throughout. Through the hierarchical model, the decoded-bar effect is credible in V3 (−1.40% [−2.81, −0.06], P = .984) and nearly in the early pathway (P = .968), but not over all nine areas (P = .90). Either way it points to early areas, as dS does. If you add this, write "lower in early visual cortex" rather than "lower".

## 5. Figures

| Figure | File (in `notes/figures/paper_2026-11/` and the update-15 package) |
|---|---|
| **Fig. 5 (use this)** | `fig5_gain_fields_v11_abc_zs_n28_palA_zsc_V3-V3AB-hV4_bayes.{pdf,png,md}`: maps for V3, V3AB and hV4, one area per pathway, each showing all three components. Panels b and c have all areas and the three pathways left of the divider. |
| Fig. 5, alternative | `fig5_gain_fields_v11_abc_zs_n28_palA_zsc_bayes.{pdf,png,md}`: the same with the old map areas (V3, hV4, TO). |
| Fig. 7 | `fig7_compact_zs_lm_2026-11-13_n28_studentt.{pdf,png,md}`: unchanged since update 2. |

The Fig. 5 files named in update 2 (`..._palA_bayes`) are superseded.

## 6. Do not touch yet (pending)

- **Robustness across pRF models.** Gaussian and DN zero-sum joint fits are running on the cluster. Until then, leave "DoG −3.5%, DN −7.8%, Gaussian −8.8%" and "the DoG model gave the smallest of the three estimates" as they are, and mark them as pending (e.g. a comment). They will be replaced, and "largest in early visual cortex" will be checked with each pRF model.
- **Fig. 6e, f model predictions.** The decoding text's model-prediction numbers ("−1.09 ± 0.44% predicted", "r = .57", "−0.34 ± 0.18% predicted") still come from an older, non-zero-sum fit. Mark them as pending; they will be rerun.
- **Supplementary figures** built on "estimated alone" numbers or older fits (transient HP − LP terms, pooled dS/dD, LOSO, S13 baseline vs gain, S12 DN variants, the Gaussian and DN supplementary figures). Do not cite their numbers in the main text; they will be replaced or dropped.
