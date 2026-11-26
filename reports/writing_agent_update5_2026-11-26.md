# Update 5 for the writing agent (26 Nov 2026): robustness across pRF models, with the supplementary figures

Builds on update 4 (`reports/writing_agent_update4_2026-11-25.md`). Where they disagree, this file wins. It replaces update 4's item 2 sentences and closes its last open point (the Gaussian and DN supplementary figures). Figures, legends and the new table are in `notes/writing_agent_update5_2026-11-26.zip`.

Rules (unchanged):
- Joint zero-sum fits only.
- Student-t hierarchical model, posterior mean [95% HDI], P = posterior probability of the sign; credible = P > .975.
- Single-area claims only from each area fitted on its own data.
- n = 28.

## Read first

1. **The Gaussian pRF is the weakest of the three pRF models, and most clearly so in early visual cortex** (new, section 1). On the same voxels, DoG and DN pRFs explain more variance than Gaussian pRFs in every area except IPS. The advantage is largest in V1 to V3, which is where surround suppression is strongest. Use this to weigh the robustness results: the one pRF model that does not show "largest in early visual cortex" is the one that fits early visual cortex worst.
2. **The HP reduction replicates with all three pRF models** (section 2; numbers as in update 4).
3. **New supplementary figures** with Gaussian and DN pRFs, made from the zero-sum fits in the layout of Fig. 5 (section 3). They replace S10 and S11 (S9 and S10 in update 4's proposed numbering).
4. **The Fig. 5 legend has one corrected sentence** (section 3). The old sentence described a sustained level shared by all four locations, and that level is not part of the zero-sum model.

## 1. How well each pRF model fits (new)

**What was done.**
- **Data:** the in-sample R² of the joint zero-sum fits with Gaussian, DoG and DN pRFs, on the same voxels (45,405 voxels, 241 participant × area fits, all runs, about 3,100 time points per voxel).
- **Measure:** per participant and area, the median over voxels of the R² gain over the Gaussian pRF.
- **Chance level:** DoG and DN contain the Gaussian, so in-sample they never fit worse. Each gain is therefore compared with what the extra parameters per voxel (2 for DoG, 4 for DN) add by chance.
- **Code and outputs:** `retsupp/visualize/paper/prf_family_fit_quality.py`; outputs in `notes/data/prf_family_fit_quality_2026-11-26/`; table `tableS_prf_model_fit_n28.md`.

| Unit | R², Gaussian | R² gain, DoG (over chance; participants above chance) | R² gain, DN (over chance) |
|---|---|---|---|
| Early (V1 to V3) | 0.276 | +0.0081 [+0.0068, +0.0119] (18×; 28 of 28) | +0.0122 (13×; 28 of 28) |
| Occipitotemporal | 0.177 | +0.0043 (8×; 28 of 28) | +0.0059 (6×) |
| Dorsal | 0.148 | +0.0030 (6×; 24 of 27) | +0.0054 (5×) |
| V1 | 0.304 | +0.0103 (23×; 28 of 28) | +0.0148 (16×) |
| IPS | 0.120 | +0.0005 (0.9×; 11 of 25) | +0.0015 (1.3×) |
| TO | 0.139 | +0.0014 (2.7×) | +0.0025 (2.2×) |

All areas are in the supplementary table.

**What this shows.**
- **Effect size:** the surround of DoG and DN pRFs adds about 3% to the variance explained in early visual cortex. That is 13 to 23 times what the extra parameters add by chance, and holds in every participant.
- **Gradient:** the advantage falls along the hierarchy, from V1 (23×) to TO (2.7×). In IPS there is none.
- **Caveat:** "Times chance" overstates the evidence, because BOLD noise is autocorrelated. It does so by the same factor in every area, so it compares areas validly. This is an in-sample comparison; a held-out comparison has not been run.
- **DN and DoG:** DN fits slightly better than DoG in absolute R², but it has two more parameters per voxel.

**Consequences for the text.**
- **"Largest in early visual cortex":** DoG and DN show it; the Gaussian model, which fits early visual cortex worst, does not. With Gaussian pRFs, early and occipitotemporal areas are about equal. Present this as a limitation of the Gaussian model, not as evidence against the claim.
- **Early target gain:** the negative early target gain is found only with Gaussian pRFs. It is −12% [−18, −6] in early areas, and V1 and V3 are credibly negative on their own data. A pRF without a suppressive surround may push the fit to account for surround suppression elsewhere (for example, with a lower gain at the target). This is a possibility, not something we tested.
- **Dropping the Gaussian does not settle the early target gain.** With DN pRFs it is positive in V1, V2 and V3 on their own data (V1 +23 [+5, +42], V2 +62, V3 +113). With DoG pRFs it is positive only in V3 (+46), and V1 is slightly negative (−10 [−17, −3]). Keep "no conclusion about the target gain in early visual cortex".

**Suggested Methods sentence** (pRF models):

> "Gaussian pRFs fitted the data less well than DoG and DN pRFs. On the same voxels, the surround of the DoG pRF explained an additional 0.008 of the variance in V1 to V3 (about 3% of the variance explained; in every participant, 18 times what its two extra parameters add by chance). This advantage became smaller along the hierarchy and was absent in IPS (Supplementary Table [pRF model fit])."

## 2. Robustness across pRF models (replaces update 4, item 2, "Replacement sentences")

The numbers are those of update 4, item 2 (dS, % of the drive):

| | All areas | Early | Dorsal | Occipitotemporal |
|---|---|---|---|---|
| DoG (main) | −3.2 [−4.8, −1.6], P > .999 | −5.5 [−8.0, −3.0], P > .999 | −1.9, P = .977 | −2.2, P = .996 |
| Gaussian | −5.2 [−9.9, −0.2], P = .982 | −6.6 [−11.8, −1.3], P = .991 | −2.0, P = .71 | −5.8 [−11.8, +0.1], P = .971 |
| DN | −5.7 [−8.5, −3.2], P = .999 | −10.8 [−15.1, −6.4], P > .999 | −1.5, P = .82 | −4.1 [−6.8, −1.2], P = .996 |

**Early larger than the other pathways:**
- **DoG:** yes. Early exceeds dorsal by 3.6 points and occipitotemporal by 3.3, P = .998 each.
- **DN:** yes. Early exceeds dorsal by 9.2 points (P > .999) and occipitotemporal by 6.7 (P = .999).
- **Gaussian:** no. Early exceeds occipitotemporal by only 0.8 points (P = .62).

**Single areas credible on their own data:**
- DoG: V3, hV4.
- Gaussian: V3, V3AB, hV4.
- DN: V3, hV4, LO.

**Results, robustness paragraph** (replaces update 4's version):

> "The reduction at HP did not depend on the pRF model. Fitted in the same way with Gaussian or divisive-normalization (DN) pRFs, the HP − LP difference was −5.2% (95% HDI [−9.9, −0.2]) and −5.7% ([−8.5, −3.2]) over all areas, and −6.6% and −10.8% in early visual cortex; V3 and hV4 showed it on their own data with all three pRF models (Supplementary Figs. [Gaussian], [DN]). With DN pRFs, as with DoG pRFs, the reduction was larger in early visual cortex than in dorsal and occipitotemporal areas. With Gaussian pRFs, early and occipitotemporal areas did not differ; Gaussian pRFs, which lack a suppressive surround, also fitted early visual cortex worst of the three models (Supplementary Table [pRF model fit])."

**Results, target and distractor gains:**

> "The target gain in dorsal and occipitotemporal areas was present with all three pRF models. The distractor gain over all areas was present with DoG and DN pRFs (+4.7% and +19%) but not with Gaussian pRFs. In early visual cortex, the target gain depended on the pRF model: positive in V1 to V3 with DN pRFs, positive only in V3 with DoG pRFs, and negative with Gaussian pRFs. We therefore draw no conclusion about it there."

**Limitations** (unchanged from update 4):

> "joint fit: DoG −3.2%, Gaussian −5.2%, DN −5.7% over all areas; the DoG model gave the smallest of the three estimates"

**The DN baseline variants (S12) are still dropped,** as in update 4.

## 3. Figures

| Figure | File (in `notes/figures/paper_2026-11/` and the zip) | What changed |
|---|---|---|
| Supplementary, Gaussian pRFs (S10; S9 in the proposed numbering) | `figS_gain_fields_gauss_zs_n28_palA.{pdf,png,md}` | New. Zero-sum joint fits in the layout of Fig. 5: maps for V3, V3AB and hV4; target and distractor gain; dS. Same hierarchical model. Replaces `figS_gain_fields_gauss_n28_palA` (old non-zero-sum fits with estimated-alone panels). |
| Supplementary, DN pRFs (S11; S10 in the proposed numbering) | `figS_gain_fields_dn_zs_n28_palA.{pdf,png,md}` | New, as above. Replaces `figS_gain_fields_dn_n28_palA`. |
| Supplementary table, pRF model fit (new) | `tableS_prf_model_fit_n28.md` | New, section 1 |
| Fig. 5 | `fig5_gain_fields_v11_abc_zs_n28_palA_zsc_V3-V3AB-hV4_bayes.{pdf,png,md}` | Two legend sentences only (below). The figure and its numbers are unchanged. |

**Fig. 5 legend, panel a:**
- Removed: "The sustained field itself is a suppression at all four possible locations in most areas (0 of 9: ); the grey lines ... so read its sign, not its size)."
- Replaced with:

  > "Under the zero-sum constraint (g_HP + 3 g_LP = 0) the sustained gain is −dS/4 at each LP location and +3 dS/4 at HP; the grey lines under each map give these two values. A level shared by all four locations is not part of the model (with the pRFs fitted together with the gains, it cannot be separated from the item weight and the pRF amplitude)."

  If the caption of update 4, item 11 is used instead of the generated legend, nothing changes there.

**Fig. 5 legend, panel b:** "credible in ... (target) and V3AB, hV4, VO (distractor)" now gives the sign. The areas are unchanged.

**Notes on the new supplementary figures:**
- **Gaussian, panel b:** the title counts only areas with a credible positive target gain ("5 of 9"). In V1 to V3 the target gain is credibly negative, and the legend says so.
- **Both, the size of a gain:** "% of the drive" is not comparable across pRF models. Each legend says so.
- **Both, the map widths:** each area's median width among fits with a reliable sustained gain (|dS| > 2 SE), as in Fig. 5. That is only 2 to 17 fits per area, so the maps show the fitted fields and should not be read for their width.
