# Update 11, 13 Nov 2026: the transient HP terms, and the HP reduction vs the item gains

This update applies on top of updates 8–10 and answers the request in `notes/request_2026-11-13_transient_terms.md`.

**Data and statistics:**
- All analyses use the refitted fits of update 8, n = 28: median, bootstrapped 95% CI, k of n, Wilcoxon, and BH across
  areas where a result is per area.
- That joint fit is not fully converged yet. dS, target and distractor estimates have been stable; any change from the
  converged fit follows in a later update.

**Where things are:**
- `parts/` holds one table file per item, with a plain-language line under each table and the source of every number.
- `figures/` holds Fig. 5 without panel d, and panel d as its own supplementary figure, both unpolished.

**Not ready:** item 3a (multiplicative vs additive) is still running and follows as an addendum.

## Bottom line for the manuscript

- **The HP reduction is one sustained field.** The sustained HP − LP gain (dS) alone explains the reduced response at
  the HP location for target, distractor and neutral items alike.
- **The transient terms don't help.** dT and dD neither improve the fit nor are shown to be zero. A distractor-specific
  reduction of 2–5 points could not have been detected; only one about as large as the whole distractor gain could.
- **Recommendation:** move dT and dD (Fig. 5d) to the supplement, with the sensitivity statement, and use the Fig. 5
  version without panel d.
- **Item gains and the reduction:** within participants, areas with a larger item gain show a larger HP reduction. The
  reduction is about a fifth of the target gain.

## Item by item: what it means for the text

- **1a, held-out prediction (`parts/item1a_heldout.md`).**
  - Leaving out one run at a time, every HP term, dS included, makes held-out prediction slightly worse (24/28
    participants predict best with no HP term).
  - Each term explains only about 0.2–0.3 × 10⁻⁴ of the variance, too little for single-run prediction to resolve.
  - Text: mention this only in Methods or the supplement as a limit of held-out prediction, not as evidence against
    any term.
- **1b, evidence for the null (`parts/item1b_null.md`).**
  - No transient term is shown to be zero. Bayes factors are inconclusive (BF01 0.5–3.7), and only dD estimated alone
    passes an equivalence test, at ±3.5 points.
  - Text: "dT and dD were not reliably different from zero, but their absence could not be established"; never "no
    difference".
- **1c, sensitivity (`parts/item1c_sensitivity.md`).**
  - In the joint fit, a planted distractor-specific HP reduction of −2 / −5 / −10 points is detected 9–16% / 55% / 95%
    of the time. The distractor gain is +8%.
  - dD estimated alone is not a clean test: it absorbs the sustained HP reduction and is attenuated.
  - Text: state that only a distractor-specific reduction about as large as the distractor gain would have been
    detected. Do not use the estimated-alone dD as evidence either way.
- **2, the sustained field explains the item responses (`parts/item2_sustained_explains_items.md`).**
  - Early areas: measured HP − LP at the item locations is −0.045% signal (p = .0095), uniform across item types.
    Location × item type is n.s.
  - The model with dS only predicts −0.029 with no significant residual (p = .14).
  - The joint fit with the transient terms fits the item responses worse: its distractor term cancels the predicted
    distractor reduction, and it predicts a target − distractor difference that the data don't show.
  - Text: "the sustained HP reduction alone accounts for the lower response at the HP location for every item type;
    adding item-specific HP terms did not improve, and partly worsened, the account of the item responses".
- **3a, multiplicative vs additive:** pending; addendum to follow.
- **3b, sizes side by side (`parts/item3b_sizes.md`).**
  - Joint fit: target +17% [+13, +24], distractor +8.0% [+4.4, +13], dS −3.5 points [−6.7, −1.3].
  - dS relative to the target gain is −0.20 as a ratio of medians, or −0.07 per participant where the target gain is
    reliable.
  - The two differ because V1 and V2 have no target gain in the model but a large dS (see 5b).
  - Text: "the HP reduction amounted to roughly a fifth of the target gain"; give the per-participant value as the
    conservative estimate.
- **3c, larger item gain, larger reduction? (`parts/item3c_correlations.md`).**
  - Within participants, across areas, yes. Target vs dS: ρ −0.17, 24/28, p = 6 × 10⁻⁴. Distractor vs dS: ρ −0.14,
    p = .013.
  - A split-half check gives the same, so it is not shared estimation noise.
  - Between participants and between areas there is no clear relation.
  - Text: "within participants, areas that weighted the items more strongly also showed a larger reduction at the HP
    location"; no between-participant claim.
- **4a, bookkeeping (`parts/item4a_bookkeeping.md`).**
  - One table maps every gain-field number to the fit it comes from.
  - Two numbers still come from fits with the pRFs held fixed: the shared sustained gain estimated alone (−3.4%), and
    the distractor gain on top of the target (+3.1%). Mark them as such, or drop them.
- **4b, Fig. 5 split (`parts/item4b_fig5_split.md`).**
  - `figures/fig5_gain_fields_v11_abc_n28_palA` (panels a–c) and `figures/figS_transient_hp_lp_n28_palA` (panel d
    alone), both unpolished; v11 is unchanged.
  - Recommended: the a–c version in the main text.
- **5a, Fig. 3d/e by pathway (`parts/item5a_fig3_cis.md`).**
  - The dorsal and occipitotemporal confidence intervals contain the early estimate, and there is no pathway ×
    location interaction (p = .56–.72).
  - Text: "the reduction at HP was weaker and less reliable in later pathways, consistent with lower precision there
    (no pathway × location interaction)"; do not claim it is absent there.
- **5b, item-type contrasts, model vs measured (`parts/item5b_itemtype_model_vs_measured.md`).**
  - Measured target − neutral is clear in V1 (+0.071% signal, p < .001) and V2 (+0.146, p < .001). The model predicts
    about zero there, outside its own noise band, and V1 and V2 are the best-sampled areas.
  - More generally, the model under-predicts the item-type contrasts (target − neutral, distractor − neutral) by
    0.05–0.10 everywhere. It also under-predicts the neutral-item response in V1 and V2. It gets target − distractor
    right everywhere.
  - Text:
    - "target gain absent in V1 and V2" must go; it is a limitation of the model, not a property of the data;
    - instead: "the item responses, and their modulation by item type, were larger in the data than the gain-field
      model predicted in early visual cortex; the target − distractor difference was captured";
    - the HP conclusions (dS, decoding) do not depend on this.

## key_numbers additions

- **dS / target gain:** ratio of medians −0.20 [−0.45, −0.07]; per participant (reliable target gain)
  −0.07 [−0.12, −0.05], 22/28, p = .034.
- **Within-participant ρ across areas:** target vs dS joint −0.17 [−0.33, −0.07], 24/28, p = 6 × 10⁻⁴; distractor vs
  dS joint −0.14 [−0.50, −0.03], p = .013.
- **Sensitivity for a distractor-specific HP reduction (joint fit):** detection 9–16% at −2, 55% at −5 and 95–97% at
  −10 points; false positives 2–4%.
- **Evidence for the null (pooled):**
  - BF01, JZS r = .707: dD alone 1.32, dT alone 0.61, dD − dT 3.69, dD joint 2.09;
  - equivalence: dD alone within ±3.5 points (p = .017).
- **Item responses at HP, early areas:** measured HP − LP −0.045% signal [−0.088, −0.012], 20/28, p = .0095; dS-only
  prediction −0.029 (residual p = .14); location × item type F(2, 54) = 1.47, p = .24.
- **Target − neutral, measured:** V1 +0.071 [+0.034, +0.098], 20/28, p < .001; V2 +0.146, 27/28, p < .001. Joint-fit
  prediction: V1 −0.004, V2 +0.007.
- **Fig. 3 by pathway, HP − opposite in % signal:** early −0.094 [−0.159, −0.031], dorsal −0.098 [−0.222, +0.030],
  occipitotemporal −0.036 [−0.101, +0.039]; pathway × location p = .70 (permutation p = .72).
- **Held-out prediction, leave one run out:** adding dS 3/28 better; adding dD 0/28; adding dD + dT 1/28.
