# D. Update 11 items recomputed on the zero-sum fits

Same code and statistics as update 11 (`retsupp/visualize/paper/transient_null_and_item_gains.py`), now on the
zero-sum main model (`notes/data/gain_field_parameters_zs_lm_2026-11-13.tsv`, rows m7_full of the view root =
m5_TDS_dS_zs) and the zero-sum estimated-alone terms. n = 28. Outputs: `notes/data/update13_gain_terms_zs_lm_2026-11-13_n28_*.tsv`.

## 3b. Sizes side by side (replaces update 11, 3b)

| Term | Update 11 (earlier fits) | Zero-sum main model |
|---|---|---|
| Target gain | +17% [+13, +24] | **+23% [+14, +35]**, 25/28 > 0, p = 4.1 × 10⁻⁷ |
| Distractor gain | +8.0% [+4.4, +13] | **+11% [+1.4, +17]**, 20/28 > 0, p = .0065 |
| dS (joint) | −3.5 [−6.7, −1.3] | **−5.5 [−7.6, −2.1]**, 21/28 < 0, p = .012 |
| dS / target gain, ratio of medians | −0.20 [−0.45, −0.07] | **−0.24 [−0.43, −0.10]** |
| dS / target gain, per participant, cells with a reliable target gain | −0.07 [−0.12, −0.05], 22/28, p = .034 | **−0.08 [−0.16, −0.06]**, 23/27, p = .001 |

Per area, the per-participant ratio is clearest in hV4 (−0.11, 18/23, p = .011), TO (−0.08, 15/18, p = .009) and V3
(−0.15, 14/17, p = .051).

**Text:** "the HP reduction amounted to about a quarter of the target gain (ratio of medians −0.24); per participant,
where the target gain was reliable, about a twelfth (−0.08, 23 of 27 participants, p = .001)." The two differ for the
same reason as before: V1 and V2 have a large dS but almost no target gain in the model.

## 3c. Larger item gain, larger reduction? (replaces update 11, 3c; now stronger)

Within participants, across areas (Spearman ρ per participant over ≥ 5 areas, then the median; Wilcoxon on the ρs):

| | Update 11 | Zero-sum |
|---|---|---|
| Target gain vs dS (joint) | ρ −0.17, 24/28, p = 6 × 10⁻⁴ | **ρ −0.20 [−0.47, −0.08], 22/28, p = .005** |
| Target gain vs dS (estimated alone) | – | ρ −0.24 [−0.40, −0.02], 21/28, p = .002 |
| Distractor gain vs dS (joint) | ρ −0.14, p = .013 | **ρ −0.52 [−0.60, −0.13], 23/28, p = .002** |
| Distractor gain vs dS (estimated alone) | – | ρ −0.46 [−0.65, −0.19], 23/28, p = .001 |

**Split-half check against shared estimation noise.** The gain comes from one half of the runs and dS from the other.
- **Target:** the effect survives (cross-half ρ −0.23, 20/28, p = .005; same-half −0.29).
- **Distractor:** weaker across halves (cross-half ρ −0.18, 18/28, p = .054; same-half −0.20). Part of the distractor
  correlation may therefore be shared noise.

**Between participants:**
- The distractor gain vs dS estimated alone is now nominally significant: ρ −0.38 [−0.71, +0.02], p = .044.
- Target vs dS: ρ −0.22 to −0.30, p ≥ .12.
- Between areas: n.s. (n = 9).

**Text:**

> "Within participants, areas that weighted the target more strongly also showed a larger reduction at the HP
> location (ρ = −0.20, 22 of 28 participants, p = .005; also when gain and reduction were estimated from separate
> halves of the data, p = .005). The same held for the distractor gain (ρ = −0.52, p = .002), though less clearly
> across split halves (p = .054)."

The between-participant distractor result (p = .044) can be mentioned as a trend in the same direction; it is one of
several between-participant tests and is not corrected.

## 1b. Evidence for the null of the transient terms (replaces the update 11, 1b numbers)

Estimated alone, pooled:

| Term | Median | p | BF01 (JZS r = .707) | Equivalence (TOST, Wilcoxon) |
|---|---|---|---|---|
| dD (distractor HP − LP) | −1.2% [−3.0, +1.2] | .31 | **3.3** | within ±3.5 points, p = .020 (smallest bound 3.05) |
| dT (target HP − LP) | −2.6% [−6.3, −0.1] | **.019** | 0.47 | not within ±5 |
| dD − dT | +1.5% [−0.6, +5.7] | .14 | 1.4 | not within ±3.5 |

**What changed:**
- **dD.** BF01 is now 3.3, moderate evidence for no distractor-specific HP reduction (it was 1.3), and dD is within
  ±3.5 points by the equivalence test. It is still estimated alone, so the update-11 caveat applies: it absorbs part of
  the sustained reduction.
- **dT.** Estimated alone it is −2.6%, p = .019, about as before (−3.6%, p = .032). A target-specific term estimated
  alone takes up part of the sustained HP reduction, so this is not independent evidence for a target-specific effect.
  The joint model with dT and dD did not fit the item responses better (update 11, item 2).

**Text:**

> "We found moderate evidence against a distractor-specific HP reduction (BF01 = 3.3; equivalence within ±3.5
> points), whereas a target-specific term estimated alone was negative (−2.6%, p = .019), as expected if it partly
> absorbs the sustained HP reduction."

## Items 1a, 1c, 2, 3a and 5b

Now rerun on the zero-sum main model: see `parts/G_update11_items_and_RT_on_zero_sum.md`.
