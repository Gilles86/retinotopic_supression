# Item 3c: does a larger target or distractor gain go with a larger reduction at HP? (n = 28)

**Question.** Across participants and across areas, does a larger target or distractor gain go with a more negative
dS (a stronger reduction of the sustained gain at HP)? A negative ρ means "larger item gain, larger HP reduction".

**Fits.**
- **Target and distractor gains:** from the joint fit (m7_full, pRFs refitted). The distractor gain is averaged over
  its locations.
- **dS:** from the same joint fit, and estimated alone (m4_TDS refitted, rows with `valid` = True).

**Levels.**
- **Between participants:** per participant the median over its areas, then Spearman across the 28 participants. CI
  from a bootstrap over participants.
- **Between areas:** the nine area medians. CI from a bootstrap over participants, recomputing the area medians each
  time.
- **Within participants across areas:** Spearman per participant over its areas (≥ 5). The table gives the median ρ
  [bootstrapped 95% CI] and the number of participants with ρ < 0. The test is Wilcoxon on Fisher-z.
- **Within participants and areas:** area medians and then participant medians removed from both terms. This takes
  out the hierarchy trend, since the target gain rises along the hierarchy and dS does not (update 10). Spearman over
  the participant × area fits; the CI comes from a bootstrap over participants. The p value treats the fits as
  independent, so read the CI.

## Correlations

| Gain | dS | Level | Spearman ρ [95% CI] | p | n | Participants with ρ < 0 |
|---|---|---|---|---|---|---|
| Target | joint | between participants | −0.20 [−0.54, +0.18] | .31 | 28 | |
| Target | joint | between areas | +0.17 [−0.37, +0.57] | .67 | 9 | |
| Target | joint | within participants, across areas | **−0.17 [−0.33, −0.07]** | 6.3 × 10⁻⁴ | 28 | 24/28 |
| Target | joint | within participants and areas | **−0.27 [−0.38, −0.16]** | (3.0 × 10⁻⁵) | 241 fits | |
| Target | estimated alone | between participants | −0.37 [−0.64, −0.02] | .054 | 28 | |
| Target | estimated alone | between areas | −0.43 [−0.83, +0.18] | .24 | 9 | |
| Target | estimated alone | within participants, across areas | **−0.32 [−0.50, −0.16]** | 2.2 × 10⁻⁴ | 28 | 22/28 |
| Target | estimated alone | within participants and areas | **−0.32 [−0.44, −0.15]** | (1.2 × 10⁻⁶) | 228 fits | |
| Distractor | joint | between participants | −0.39 [−0.70, +0.03] | .040 | 28 | |
| Distractor | joint | between areas | −0.05 [−0.65, +0.52] | .90 | 9 | |
| Distractor | joint | within participants, across areas | **−0.14 [−0.50, −0.03]** | .013 | 28 | 19/28 |
| Distractor | joint | within participants and areas | **−0.22 [−0.38, −0.03]** | (4.3 × 10⁻⁴) | 241 fits | |
| Distractor | estimated alone | between participants | **−0.56 [−0.79, −0.22]** | .0021 | 28 | |
| Distractor | estimated alone | between areas | −0.27 [−0.85, +0.12] | .49 | 9 | |
| Distractor | estimated alone | within participants, across areas | **−0.46 [−0.63, −0.22]** | 1.5 × 10⁻⁴ | 28 | 23/28 |
| Distractor | estimated alone | within participants and areas | **−0.38 [−0.53, −0.22]** | (3.5 × 10⁻⁹) | 228 fits | |

**In plain language:**
- **Within a participant:** areas (and participant × area fits) with a larger target or distractor gain show a larger
  reduction at HP.
  - With the joint dS: ρ ≈ −0.2 to −0.3.
  - With dS estimated alone: ρ ≈ −0.3 to −0.5.
- **Between participants:** clear only for the distractor gain with dS estimated alone (ρ = −0.56). With the joint dS
  it is borderline (distractor) or absent (target).
- **Between areas:** with nine points the CIs span zero. There is no evidence that areas with a larger target gain have
  a larger HP reduction; for the joint dS the sign is even positive.

## Is this shared estimation noise? Split-half check

In one fit, an error in the target or distractor term can be absorbed by dS, which would create a correlation by itself.
We therefore re-solved the five joint-fit gains (sustained at HP and at LP, target, distractor at HP and at LP) on odd
runs and on even runs separately. Each solve uses the per-run sufficient statistics of the same root, with the pRFs,
HRF, widths and item weight fixed at the joint fit.
- **Same half:** gain and dS from the same runs, so they share noise.
- **Cross half:** gain from one half and dS from the other, so their noise is independent.

| Gain | Pairing | ρ within participants and areas | Within participants: median ρ [95% CI], participants < 0, Wilcoxon p |
|---|---|---|---|
| Target | same half | −0.21 | −0.16 [−0.30, −0.08], 22/28, p = .002 |
| Target | cross half | **−0.24** | **−0.23 [−0.36, −0.05], 21/28, p = .003** |
| Distractor | same half | −0.17 | −0.15 [−0.35, −0.00], 19/28, p = .036 |
| Distractor | cross half | **−0.15** | **−0.17 [−0.29, −0.04], 20/28, p = .023** |

**Split-half reliability** (odd vs even, area and participant medians removed): target gain 0.70, distractor gain
0.45, dS 0.42.

**In plain language:** the correlation is as strong across independent halves of the data as within one half. So it is
not shared estimation noise within a fit. Within participants, where an item gain is larger, the HP reduction is
larger.

## What this means for the text

**The data support a within-participant link, not a between-participant one.** Suggested wording:

> "Within participants, the areas with a larger target or distractor gain also showed a larger reduction of the
> sustained gain at HP (target: median ρ = −0.17 [−0.33, −0.07], 24 of 28 participants; distractor: −0.14
> [−0.50, −0.03], 19 of 28; joint fit), and this held when gain and reduction were estimated from independent halves
> of the runs (ρ = −0.23 and −0.17). Across areas, the reduction did not grow with the target gain (area medians,
> ρ = +0.17 [−0.37, +0.57])."

**How this fits update 10.** It is consistent with update 10: the target gain grows along the hierarchy and dS does
not. The link is therefore one of fluctuation around each area's typical level, not of the hierarchy.

**Caveat.** The between-participant correlation of the distractor gain with dS estimated alone (ρ = −0.56) mixes two
fits (joint and estimated alone). Quote it only as secondary.

## Sources

- **Code:** `retsupp/visualize/paper/transient_null_and_item_gains.py` (`correlations`, `split_half`).
- **Tables:**
  - `notes/data/update11_gain_terms_polished_2026-11-12_n28_correlations.tsv`;
  - `notes/data/update11_gain_terms_polished_2026-11-12_n28_correlations_split_half.tsv`.
- **Inputs:**
  - `notes/data/gain_field_parameters_polished_2026-11-12.tsv` (dog/m7_full);
  - `notes/data/conditional_estimates_polished_2026-11-12.tsv` (`valid`);
  - split halves: the per-run statistics `derivatives/field_widths_reliable_polished_2026-11-12/sub-XX/*_widthprof.npz`
    (`fit_width_profiles.py` on the reference root; local copy under `/data/ds-retsupp/derivatives/`);
  - sub-01 and sub-02 left out.
