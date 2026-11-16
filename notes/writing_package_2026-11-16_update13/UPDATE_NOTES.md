# Update 13, 16 Nov 2026: the zero-sum refit (replaces the gain-field numbers of updates 8–11)

Every gain-field number in updates 8–11 came from fits whose sustained level shared by all four locations was not
identifiable (update 11, caveat). This update replaces them with the final model.

- **Model:** the sustained field is zero-sum over the four locations, g_HP + 3·g_LP = 0, so dS = 4/3·g_HP. Every
  parameter is fitted jointly: per-voxel DoG pRFs, HRF per area, item weight, target, distractor and dS.
- **Optimiser:** Levenberg–Marquardt, from two starts per fit.
- **Main model:** no transient HP − LP terms (dD, dT). The model with dD is kept as a supplementary check.
- **Data and statistics:** n = 28 (sub-01 and sub-02 left out), statistics as before.

Apply this on top of updates 8–12. Figures are in the update-12 colours.

**Contents:**

| File | What |
|---|---|
| `parts/A_comparison_previous_vs_zero_sum.md` | Every term, per area and pooled: earlier fits vs zero-sum, with and without on-bound fits, and the model with dD |
| `parts/B_key_numbers_zero_sum.md` | The full key-numbers file of the zero-sum reference (see "Do not use" below) |
| `parts/C_field_widths.md` | Field widths on the zero-sum fits: group widths, per-participant identifiability, the transient width along the hierarchy |
| `parts/D_update11_items_on_zero_sum.md` | Update 11 items 3b, 3c and 1b recomputed |
| `parts/G_update11_items_and_RT_on_zero_sum.md` | Update 11 items 1a, 1c, 2, 3a and 5b, the main residual-RT analysis and the joint latent RT model, rerun on the zero-sum main model (earlier and zero-sum side by side, wording changes) |
| `parts/E_zero_sum_refit_methods_note.md` | Methods record: models, starts, bounds, TO, runs |
| `parts/F_identifiability_note.md` | Why the shared level is not identifiable when the pRFs are free (for Methods) |
| `figures/main/` | Fig. 5 (a–c) and Fig. 7 on the zero-sum fits |
| `figures/supplement/` | Transient HP − LP (old Fig. 5d), field widths (new), gain vs pRF size, hierarchy, pooled dS/dD, parameter and hierarchy tables |

---

## 1. Bottom line

- **The HP reduction holds and is somewhat larger.**
  - Joint fit: dS −5.5% [−7.6, −2.1], 21/28 participants < 0, p = .012. Without on-bound fits −5.1%, p = .0061.
  - The model that also has dD gives −5.6%, p = .0089.
  - Cell for cell, dS matches the earlier fits (Spearman ρ = 0.93–0.95).
- **Estimated alone (Fig. 5c), pooled:** −1.5% [−2.4, −0.9], 21/28, p = .0071.
- **Per area, the joint fit carries the claim:** dS is reliable in hV4 (−8.2%, q = .022).
  - Estimated alone, no single area survives BH any more (V3, hV4 and TO did before, at q = .036). The per-area
    estimates are still largest in the early areas and hV4; report them with exact p (below).
- **Target and distractor gains are larger:** target +23% (7 of 9 areas), distractor +11% (p = .0065). The item
  weight is smaller (2.8 vs 3.4). This is mostly a trade-off between the item weight and the gains.
- **RT links hold and are slightly stronger** (part G):
  - main residual-RT analysis: distractor gain −2.4% per SD of residual log RT (P = .994), global gain −2.8%
    (P = .982); the occipitotemporal global gain is now credible on its own;
  - joint latent model: −17 ms per SD of latent global gain (P = .995).
- **Model comparisons (update 11) hold; four sentences change** (part G, and 3h below). The main one: adding dS no
  longer makes held-out prediction reliably worse (10/28 better, p = .099).
- **New, reportable:**
  - the sustained field has a group-level width of FWHM 4–7°;
  - the transient field is broad in V1–V3 and item-locked in hV4, V3AB and VO;
  - there is now moderate evidence against a distractor-specific HP reduction (BF01 = 3.3).

## 2. Numbers to replace in the text

All pooled values are per-participant medians over the nine areas, with median [bootstrapped 95% CI], participants
below/above 0 and Wilcoxon p.

| Quantity | Was (updates 8–11) | Now |
|---|---|---|
| dS estimated alone, pooled | −1.0%, p = 2.7 × 10⁻⁴ | **−1.5% [−2.4, −0.9], 21/28, p = .0071** |
| dS estimated alone, areas with q < .05 | V3, hV4, TO | **none**; see 3a for per-area wording |
| dS estimated alone, per pathway | – | early −2.6% (p = .056, q = .17); dorsal −0.5%; occipitotemporal −0.9% |
| dS joint, pooled | −3.5%, p = .0044 | **−5.5% [−7.6, −2.1], 21/28, p = .012** |
| dS joint, per area | hV4 q = .024 | **hV4 −8.2%, q = .022** (q = .055 without on-bound fits); V1 −8.1, V2 −7.8, V3 −10.2 (q = .13–.16) |
| Target gain, pooled | +17% [+13, +24] | **+23% [+14, +35], 25/28, p = 4.1 × 10⁻⁷**; q < .05 in V3, V3AB, IPS, hV4, VO, LO, TO |
| Distractor gain, pooled | +8.0%, p = .023 | **+11% [+1.4, +17], 20/28, p = .0065**; q ≤ .05 in V3AB, hV4 |
| Item weight | 3.4 | **2.8 [1.5, 4.2]** |
| dS / target gain | about a fifth (−0.20); per participant −0.07 | **about a quarter (−0.24 [−0.43, −0.10])**; per participant **−0.08, 23/27, p = .001** |
| Target gain vs dS, within participants | ρ −0.17, p = 6 × 10⁻⁴ | **ρ −0.20, 22/28, p = .005** (split halves p = .005) |
| Distractor gain vs dS, within participants | ρ −0.14, p = .013 | **ρ −0.52, 23/28, p = .002** (split halves p = .054) |
| dD estimated alone | −0.8%, p = .14; BF01 1.3 | **−1.2%, p = .31; BF01 3.3; within ±3.5 points (p = .020)** |
| dT estimated alone | −3.6%, p = .032 | **−2.6%, p = .019** |
| Target gain per doubling of pRF size (update 10) | +27, p = .002 | **+32% [+20, +79], 82% of participants, p = 8.2 × 10⁻⁵** |
| Distractor gain per doubling of pRF size | – | +6.5%, p = .12 |
| dS along the hierarchy (update 10) | flat | **flat** (alone +1.4 per doubling, p = .61; joint −0.4, p = .75) |
| Sustained width | FWHM 2.8° (update 8) | **FWHM 4.0–4.7° in V1–V3, 4.7–6.7° in ventral/lateral areas (group level)** |
| dS vs behavioural HP benefit, between participants | ρ −.13 / +.06, n.s. | ρ +.09 (alone), +.12 (joint), n.s. |
| Fig. 7: target gain on target-repeat trials | −18% [−31, −4] | **−19% [−29, −9]** |
| Fig. 7: distractor gain per SD log RT | −2.0% [−3.7, −0.2] | **−2.7 [−4.7, −0.7]** |
| Fig. 7: global gain per SD log RT | n.s. | −2.3 [−5.1, +0.5], n.s. |
| Main residual-RT analysis: distractor gain per SD of residual log RT | −1.6 [−3.3, −0.04], P = .977 | **−2.4 [−4.3, −0.6], P = .994** |
| Main residual-RT analysis: global gain | −2.6 [−5.1, −0.04], P = .976 | **−2.8 [−5.5, −0.2], P = .982**; occipitotemporal alone −4.2 [−8.1, −0.3] |
| Joint latent RT model: global gain | −15 ms [−30, −1] per SD, P = .985 | **−17 ms [−31, −4], P = .995** |
| Held out (update 11, 1a): dS vs no HP term | worse, 3/28 better, p = 6.7 × 10⁻⁶ | **not reliably different: −0.11 × 10⁻⁴, 10/28 better, p = .099** |
| Held out: adding dD | 0/28 better | 0/28 better |
| Sensitivity (update 11, 1c): power of the joint dD test at −10 / −5 / −2 points | 91–97% / 45–56% / 9–16% | **87–90% / 36–46% / 10–13%** |
| Real joint dD, linearised / LM fit with dD | +1.2%, p = .26 | **+7.3%, p = .0048 / +5.6%, p = .070** (opposite sign to a reduction; fails held out) |
| Multiplicative vs additive (update 11, 3a), training runs | equal (p = .15) | **multiplicative slightly better, p = .010**; held out worse, 23/28, p = 1.4 × 10⁻⁴ |
| Item responses (update 11, 2): measured − model with dD, target − distractor | p = .048 | p = .070 |
| Item-type contrasts (update 11, 5b): model's target − neutral in V1 / V2 | −0.004 / +0.007 | −0.006 / +0.005 (data +0.060 / +0.145) |

## 3. Text changes

**a. Per-area dS (Fig. 5c).** Lead with the pooled effect and give the per-area values as description.

> "Pooled over the nine areas, the sustained gain was lower at the HP location than at the LP locations (dS −1.5%,
> 21 of 28 participants, p = .0071). The reduction was largest in early visual cortex and hV4 (V3 −4.8%, p = .013;
> V2 −2.8%, p = .14; hV4 −2.5%, p = .13; TO −1.2%, p = .090; uncorrected), but no single area survived correction
> across the nine (smallest q = .11). In the joint fit, where all terms were estimated together, the reduction was
> reliable in hV4 (−8.2%, q = .022)."

Use stars with exact p as before. Uncorrected per-area p values for dS estimated alone:

| Area | dS alone | p (uncorrected) |
|---|---|---|
| V1 | +0.1% | .90 |
| V2 | −2.8% | .14 |
| V3 | −4.8% | .013 |
| V3AB | −0.8% | .59 |
| IPS | −0.6% | .37 |
| hV4 | −2.5% | .13 |
| VO | −0.9% | .32 |
| LO | +1.0% | .60 |
| TO | −1.2% | .090 |

**b. The sustained field, Methods.** Replace "The level of the gain shared by all four locations … is reported by its
sign only" (update 9, F) with:

> "Because the sustained field was present in every run, a component shared by all four locations is equivalent to a
> reshaping of every voxel's receptive field and cannot be separated from the pRF when both are fitted together. We
> therefore constrained the sustained field to sum to zero over the four locations (g_HP + 3 g_LP = 0), so that it
> captures only how the gain at the HP location differs from that at the other locations (dS = g_HP − g_LP). All
> parameters (per-voxel pRFs, the HRF of each area and the gain-field terms) were fitted jointly by
> Levenberg–Marquardt from two starting points, keeping the better fit."

**c. Remove:**
- "Sustained gain at LP (joint)": under zero-sum it is −dS/4 by construction.
- "dD joint": it is 0 in the main model.
- Any statement about the size or sign of a level shared by all four locations from the joint fit.

**d. The shared level, estimated alone.** With the pRFs held fixed, a gain shared by all four locations, added alone
to the model, is negative in every participant (−3.4% [−3.7, −2.3], 28/28, p = 7.5 × 10⁻⁹). It also becomes more
negative along the hierarchy (−6.4 points per doubling of pRF size, p = 2.2 × 10⁻⁸). It can be reported in the
supplement with the framing:

> "with the pRFs held fixed, the gain at the four possible locations was lower than elsewhere in the display
> (−3.4%), increasingly so in higher areas; this component is not separable from the pRFs once those are fitted
> together with the gain fields, and the main model therefore omits it."

**e. Widths.** See `parts/C_field_widths.md`; suggested text is there. In short:
- **Sustained field:** clear group widths (FWHM 4–7°). Not identified per participant (split-half ρ = 0.10), because
  the HP − LP modulation is only a few percent of the drive.
- **dS:** negative at every width.
- **Transient field:** broad in V1–V3, item-locked in hV4, V3AB and VO.

**f. Transient terms.** See `parts/D_update11_items_on_zero_sum.md`.

> "We found moderate evidence against a distractor-specific HP reduction (BF01 = 3.3; equivalence within ±3.5
> points)."

The sensitivity statement of update 11 holds with slightly lower power (part G, item 1c):

> "the joint test detects a −10-point distractor-specific reduction 87–90% of the time, a −5-point one 36–46% and a
> −2-point one rarely (10–13%)."

The joint dD on the real data is positive (+7.3% linearised; +5.6%, p = .070, in the LM fit with dD), so a reduction
of about the size of the distractor gain is ruled out. Do not report the positive dD as an effect: adding it makes
held-out prediction worse in every participant.

**g. Item gains and the reduction.** See `parts/D…`, 3b and 3c; suggested text is there. Within participants the
relation is stronger than before, and holds across split halves for the target.

**h. Sentences from update 11 and the RT notes that change** (details and numbers in part G):
- **1a, held out.** Replace "every added term makes the held-out run slightly worse, including dS" with:

  > "Held-out prediction does not separate the HP − LP terms: adding the sustained dS neither improves nor reliably
  > worsens prediction of the left-out run (10 of 28 participants better, p = .099), while each transient term makes
  > it worse in nearly every participant."
- **3a, multiplicative vs additive.** The two forms no longer fit the training runs equally well: the multiplicative
  one fits them slightly better (p = .010) and predicts held-out runs worse (23 of 28 participants favour additive,
  p = 1.4 × 10⁻⁴). The conclusion stands: neither form is supported, h and a do not differ from 0, and both predict
  held-out runs worse than no HP term.
- **2, item responses.** For the model with dD, write "predicts a target − distractor interaction the data do not show"
  instead of "contradict" (p = .070).
- **Main residual-RT analysis.** Drop the clause "but this does not survive a looser voxel selection", or mark it
  "(earlier fits)": that check was not rerun. Suggested sentence:

  > "With RT residualised on the design and the three gains in one model, the global gain is also lower on slower
  > trials (−2.8% [−5.5, −0.2] per SD; with z(log RT) −2.2%, P = .94)."
- **Joint latent model.** "about 17 ms per SD of latent global gain" (was 15 ms). Its checks are slightly weaker than
  before (1 divergence, minimum bulk ESS 318, and the observed distractor-gain moments inside the predictive check in
  only 2 of 5 cells); mention this if the model goes in the paper.

## 4. Figures

**Main:**
- **Fig. 1:** `fig1_paradigm_behaviour_v6_n28_palA`, replacing v5. Panel c now has a third within-subject strip,
  HP − absent (strip and a bracket
  with stars over the raw RTs): +14.6 ms [+0.3, +21.7], 19/28 slower, p = .0041. A distractor at HP still slows responses, by about a
  third of the LP cost (+42.6 ms). Behaviour only, so not affected by the refit.
- **Fig. 5:** `fig5_gain_fields_v11_abc_zs_n28_palA` (panels a–c), replacing the update 12 version.
  - Panel a's sustained maps now show the zero-sum field: under each map, the LP gain is −¼ and the HP gain +¾ of the
    HP − LP difference drawn.
  - Panel b titles now name 7 of 9 areas for the target, and V3AB and hV4 for the distractor.
  - Panel c has no per-area stars.
  - The caption is regenerated. Check two sentences against the text: the sustained field is zero-sum, and the map
    widths are FWHM 3.9–4.2°.
- **Fig. 7:** `fig7_compact_zs_lm_2026-11-13_n28`, the trial-wise gains on the zero-sum fits. Same layout and
  colours.

**Supplement:**
- `figS_transient_hp_lp_zs_n28_palA`: the old Fig. 5d, on the zero-sum fits (dD and dT estimated alone).
- `figS_width_profiles_zs_n28_palA`: **new**, the group width profiles of both fields.
- `figS_gain_vs_prf_size_zs_lm_2026-11-13_zs_n28`, `figS_hierarchy_gain_terms_zs_lm_2026-11-13_zs_n28` and
  `table_hierarchy_gain_terms_zs_lm_2026-11-13_zs_n28`: the hierarchy (update 10) on the zero-sum fits. In the table,
  ignore the row "sustained gain at the LP locations, joint fit, sign only" (see 3c). The Gaussian and DN columns are
  still the earlier, non-zero-sum fits.
- `figS_pooled_dS_dD_zs_lm_2026-11-13_zs_n28`: pooled dS and dD.
- `table_parameters_2026-11_zs_lm_2026-11-13_zs_n28`: the parameter table. Ignore its "LP (joint)" and "dD joint"
  rows.

**Not included, do not use:**
- `figS_field_widths_v3_zs_n28`: it lets g_HP and g_LP vary freely, so it is replaced by the width profiles.
- `figS_gain_field_map_search_v3_zs_n28`: it draws the joint dD, which is 0 in the main model.

## 5. Still from the earlier fits

These do not change any conclusion:
- **Update 11, item 2, Table 2d** (variants with a target-specific transient term): there is no zero-sum fit with that
  term. Everything else in items 1a, 1c, 2, 3a and 5b is rerun (part G).
- **RT side checks:** RT quintiles, the r > 0.2 voxel selection, the Bambi cross-check, pathway fits and parameter
  recovery of the joint latent model.
- **Robustness across pRF models:** the Gaussian and DN columns of the hierarchy table and the DN LOSO result.
- **Robustness to voxel selection (update 9, A):** the looser-selection fits are not refitted zero-sum. The decision
  rule and the conclusion do not depend on the shared level.

Each of these can be rerun on the zero-sum model if the paper needs it.

## 6. Bookkeeping

- **Fits:** `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm` (all 3 models × 241
  cells).
- **View root used for every paper number:** `..._zs_lm_ref`.
- **Estimated alone:** `af_fixed_group_mechanisms_zs_lm_2026-11-13`.
- **Key numbers:** `notes/data/reference_numbers_zs_lm_2026-11-13_n28.{md,tsv}`.
- **Regenerate:** `retsupp/visualize/paper/zero_sum_reference.sh`.
- **Fits on a bound** (main model, 241 cells):
  - item weight at its floor: 42 cells;
  - g_HP: 10;
  - a transient gain: 18;
  - mostly TO.

  Statistics with and without them are in part A, and the conclusions do not depend on them.
