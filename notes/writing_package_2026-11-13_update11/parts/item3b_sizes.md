# Item 3b: target gain, distractor gain and sustained HP − LP side by side, from the same fit (n = 28)

**Fit.** All numbers come from the joint fit: all terms fitted together (m7_full), DoG pRFs refitted together with the
gain fields, root `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`.

**Units.** All three terms are a fractional change of the drive at the centre of a field, so they share one unit:
- the target and distractor gains are in %;
- dS (sustained gain at HP minus at the low-probability, LP, locations) is in percentage points of the same gain.

**Statistics.**
- **Per area:** median over participants [bootstrapped 95% CI]. Stars mark Wilcoxon against 0 with BH across the nine
  areas (* q < .05, ** q < .01, *** q < .001).
- **Pooled:** per participant the median over the nine areas, then as per area. Wilcoxon p: target 3.2 × 10⁻⁷ (25/28
  > 0); distractor .023 (21/28 > 0); dS joint .0044 (22/28 < 0).

**Ratio dS / target gain, computed two ways:**
- **(i) Ratio of medians:** median dS / median target gain over participants, with a bootstrap over participants
  (5,000 resamples).
- **(ii) Per participant:** only participant × area fits whose target gain is reliably above 0 (jackknife gain / SE > 2,
  from the per-run statistics of the same root). The ratio is computed per fit; pooled is the per-participant median
  over those areas. The number of participants (pooled) or fits (per area) is in brackets.

| Area | Target gain (%) | Distractor gain (%), averaged over its locations | dS joint (pp) | dS / target, (i) ratio of medians [95% CI] | dS / target, (ii) per participant, target reliable [95% CI] (n) |
|---|---|---|---|---|---|
| **Pooled** | **+17 [+13, +24]** | **+8.0 [+4.4, +13]** | **−3.5 [−6.7, −1.3]** | **−0.20 [−0.45, −0.07]** | **−0.07 [−0.12, −0.05] (28; 126 fits; 22/28 < 0, p = .034)** |
| V1 | −2.3 [−15, +1.5] | −0.7 [−6.9, +12] | −7.9 [−13, −0.3] | (+3.4; undefined, target ≈ 0) | — (1) |
| V2 | +5.2 [−4.1, +14] | +4.3 [−6.8, +13] | −7.7 [−11, −2.0] | (−1.5; undefined, target ≈ 0) | −0.12 [−0.23, +0.12] (6) |
| V3 | +33*** [+9.9, +52] | +6.4 [−12, +24] | −9.5* [−20, +2.1] | −0.29 [−1.09, +0.08] | −0.22 [−0.60, −0.07] (15) |
| V3AB | +29*** [+21, +48] | +11* [+4.1, +25] | −0.6 [−4.2, +1.4] | −0.02 [−0.14, +0.03] | −0.04 [−0.15, +0.01] (21) |
| IPS | +21* [+11, +103] | +1.9 [−2.0, +18] | −0.2 [−40, +1.1] | −0.01 [−0.66, +0.05] | −0.03 [−0.14, +0.05] (14) |
| hV4 | +59*** [+34, +75] | +21* [+7.9, +48] | −5.6* [−22, −3.3] | −0.10 [−0.33, −0.06] | −0.12 [−0.17, −0.06] (20) |
| VO | +38*** [+14, +73] | +8.0 [+3.0, +19] | −0.6 [−3.0, +1.7] | −0.01 [−0.11, +0.09] | −0.04 [−0.20, −0.02] (17) |
| LO | +20 [+4.1, +34] | +2.2 [−5.8, +23] | −0.5 [−3.1, +3.2] | −0.03 [−0.17, +0.20] | −0.05 [−0.12, −0.00] (14) |
| TO | +18*** [+11, +77] | +8.2** [+2.3, +48] | −3.1 [−9.0, −0.3] | −0.17 [−0.54, −0.03] | −0.05 [−0.12, −0.01] (18) |

**In plain language:**
- **Pooled, in the same fit and units:** the target gains about +17%, the distractor about +8%, and the HP location
  loses about 3.5 points of sustained gain.
- **dS relative to the target gain:** the HP reduction is about a fifth of the target gain by the ratio of medians
  (−0.20). It is about a fourteenth (−0.07) when computed per participant over the areas where the target gain is
  reliable.
- **Why the two ratios differ:** the target gain is near zero in V1 and V2, where the joint dS is largest (−8 points).
  The ratio of medians mixes those areas in. The per-participant ratio leaves them out.
- **Where the reduction is largest relative to the target:** V3 (about −0.2) and hV4 (−0.12). Elsewhere it is ≤ 0.05.

## For reference (mixed fits, not the same fit)

dS estimated alone is the lead statistic of Fig. 5c. It comes from the model with item weight, target, distractor and
sustained gain, with dS added alone (m4_TDS, refitted).
- **dS estimated alone / target gain (joint):** −0.06 [−0.12, −0.02] (ratio of medians).
- **dS joint / distractor gain:** −0.43 [−0.98, −0.15].
- **dS estimated alone / distractor gain (joint):** −0.13 [−0.27, −0.05].

The level of the sustained gain shared by all four locations (g_LP) is not in this table. Its size is not well
identified once the pRFs are refitted (update 8); only its sign is reported.

## What this means for the text

A sentence that compares sizes within the joint fit:

> "In the same fit, the sustained reduction at HP (−3.5 points) was about a fifth of the target gain (+17%; ratio of
> medians −0.20 [−0.45, −0.07]) and smaller than the distractor gain (+8.0%)."

**Use the per-participant ratio for a conservative statement.** It is −0.07 [−0.12, −0.05] in areas with a reliable
target gain. The ratio is not meaningful in V1 and V2, where the target gain is about 0.

## Sources

- **Code:** `retsupp/visualize/paper/transient_null_and_item_gains.py` (`sizes`).
- **Tables:**
  - `notes/data/update11_gain_terms_polished_2026-11-12_n28_sizes.tsv` (the three terms);
  - `notes/data/update11_gain_terms_polished_2026-11-12_n28_ratio.tsv` (both ratios).
- **Inputs:**
  - `notes/data/gain_field_parameters_polished_2026-11-12.tsv`, rows dog/m7_full: target = `g_T_dyn`; distractor =
    (`g_HP_dyn` + 3 `g_LP_dyn`)/4; dS = `g_HP` − `g_LP`;
  - reliability of the target gain: `g_T_z` in `notes/data/field_widths_reliable_polished_2026-11-12_n28_fits.tsv`;
  - sub-01 and sub-02 left out.
- **Check against key_numbers:** the target, distractor and dS joint values match update 8's
  `key_numbers_reference_full.md`.
