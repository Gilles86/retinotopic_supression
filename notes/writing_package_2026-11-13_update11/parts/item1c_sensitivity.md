# Item 1c: which distractor-specific HP reduction could we have detected?

**Bottom line.** The joint fit gives an unbiased estimate of a distractor-specific HP reduction. With 28 participants,
the paper's pooled test detects a −10% reduction about 90% of the time, a −5% reduction about half the time and a −2%
reduction rarely (10–16%). The observed value (joint fit: dD +1.2%, dD − dT −0.3%; both n.s.) therefore rules out a
distractor-specific reduction of about 10 points, which is about the size of the whole distractor gain (+8%). It does
not rule out one of 2–5 points. The reported estimate of dD alone is not a clean test of a distractor-specific
reduction. It takes up part of the sustained HP reduction: with no distractor-specific effect planted, it is negative
on average and comes out "significant" in 19–51% of simulated data sets. When the base model is refitted to each data
set, as in the paper's procedure, it is also attenuated to about half the planted size. Its real value (−0.8%,
p = .16) lies inside the range that no distractor-specific effect produces.

## Design

- **Participants and areas:** all 28 participants (sub-03 to sub-30), all nine areas, the current voxel selection.
- **Generators.** Each simulated data set is a noise-free prediction plus noise. Two generators, each with a planted
  distractor-specific HP − LP difference dD = δ and no target-specific term (dT = 0):
  - **Joint-fit generator:** the fitted joint model (all terms fitted together, m7_full, in the reference root of
    update 8 with the pRFs refitted together with the gain fields,
    `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`; that polish was not run to full
    convergence). Its distractor gain at HP is set to its distractor gain at LP + δ. Its sustained HP − LP term stays
    as fitted (pooled dS −3.5%).
  - **Estimated-alone generator:** the model on which the HP − LP terms are estimated alone (item weight, target,
    distractor and sustained gain, m4_TDS, same root), plus each participant × area's estimated-alone dS (pooled
    −1.0%), plus δ.
- **Unit of δ.** δ is in the unit of the reported dD: percentage points of the transient gain (gain 1 = 100%).
  δ = −5% means the distractor at HP is enhanced 5 points less than a distractor at an LP location. For scale, the
  joint-fit distractor gain is +8.0% pooled (key_numbers update 8), so δ = −2 / −5 / −10% is 25 / 63 / 125% of a
  typical distractor gain.
- **Noise.** Each participant's residuals, phase-randomised per run with one set of random phases for all voxels and
  all nine areas of the participant. This keeps each voxel's spectrum and the covariance across voxels and areas (as
  in `retsupp/decode/simulate_decode.py`). The joint-fit generator uses the joint fit's residuals; the
  estimated-alone generator uses the residuals of the linearised model with all seven terms. Both use the same
  phases.
- **Number of data sets:** 200 simulated data sets per δ (0, −2, −5, −10%) and generator.
- **Refit.** The gain terms enter the DoG model linearly (linearisation error median 2 × 10⁻⁴, largest 0.16 in one participant × area), so every refit is a
  least-squares fit around the real-data m4_TDS fit. The pRFs and field widths are held at that fit, not refitted per
  simulated data set: the alternating polish takes about 45 min per participant × area.
  - **Estimated alone, base refitted (the paper's procedure applied to new data):** the four base terms (item weight,
    target gain, distractor gain and sustained gain at all four locations) are refitted to the simulated data. Then
    dD, and separately dT, are added alone. On the real data this gives the reported values exactly (dD −0.78%,
    dT −3.64% pooled).
  - **Estimated alone, base held at the real-data fit:** shown in the full tables only.
  - **Joint:** base terms + dS + dD (the HP − LP terms of the joint fit), and base terms + dS + dD + dT (the joint fit
    with the target term split, m8_full_dT).
- **Detection.** The paper's pooled test: per participant the median over the nine areas, then a two-sided Wilcoxon
  test across the 28 participants. A data set counts as a detection at p < .05 in the planted (negative) direction;
  for dD − dT, at p < .05 in either direction. Power is the share of the 200 data sets that detect, ± the binomial
  95% CI half-width, which is at most ±7%. The full tables also give "any area at BH q < .05" and "any pathway at
  BH q < .05".

## Main table: power of the pooled test, and the estimate (group median over data sets)

| Estimator | Generator | δ = 0 (false positives) | δ = −2% | δ = −5% | δ = −10% |
|---|---|---|---|---|---|
| **dD, joint fit (with dS)** | estimated alone | 2% (est. +0.1) | 16% (−1.9) | 55% (−4.9) | 97% (−9.9) |
| | joint fit | 2% (+0.3) | 9% (−1.7) | 56% (−4.7) | 95% (−9.9) |
| **dD, joint fit with dS and dT** | estimated alone | 3% (−0.1) | 14% (−2.1) | 47% (−5.1) | 94% (−10.1) |
| | joint fit | 2% (+0.3) | 13% (−1.7) | 45% (−5.0) | 91% (−10.1) |
| **dD − dT, joint fit with dS and dT** | estimated alone | 4% (+0.3) | 13% (−1.7) | 40% (−4.7) | 87% (−9.7) |
| | joint fit | 4% (+0.1) | 12% (−1.9) | 40% (−5.0) | 89% (−10.0) |
| **dD estimated alone (base refitted)** | estimated alone | **19%** (−0.8) | 52% (−1.6) | 91% (−2.8) | 100% (−4.7) |
| | joint fit | **51%** (−1.5) | 81% (−2.3) | 98% (−3.5) | 100% (−5.5) |
| **dD − dT estimated alone (base refitted)** | estimated alone | 6% (+0.9) | 3% (−0.2) | 14% (−1.9) | 59% (−4.7) |
| | joint fit | 12% (+1.5) | 4% (+0.3) | 6% (−1.3) | 44% (−4.0) |

Real data, same estimators (pooled, median over participants):

| Estimator | Real value | p |
|---|---|---|
| dD, joint fit (with dS) | +1.15% | .26 |
| dD, joint fit with dS and dT | +3.15% | .58 |
| dD − dT, joint fit with dS and dT | −0.32% | .78 |
| dD estimated alone | −0.78% | .18 |
| dD − dT estimated alone | +1.27% | .16 |

These linearised joint values are not the paper's joint dD (+0.1%, polished m7_full): here the pRFs and widths are
held at the m4_TDS fit.

*Summary:* in the joint fit, a distractor-specific reduction at HP would have been seen reliably only if it were about
as large as the distractor gain itself (−10 points: 91–97%); at −5 points the chance is about even, and at −2 points
it is small. The estimated-alone dD is biased by the sustained reduction and should not be read as evidence for or
against a distractor-specific term.

## What this means for dD and dD − dT

- **Joint fit.** Without a planted effect the estimate is unbiased (bias ≤ 0.35 points) and the test keeps its false
  positive rate (2–4%). The spread of the group median over data sets is about ±5 points (2.5–97.5%: −4.6 to +4.4 at
  δ = 0, joint with dS, estimated-alone generator). This is the test of a distractor-specific HP reduction.
- **dD estimated alone.**
  - **It absorbs the sustained HP reduction.** A sustained reduction at HP also lowers the response to a distractor
    at HP. With dS but no planted dD, the estimate is negative: median −0.8 under the estimated-alone generator,
    −1.5 under the joint-fit generator (larger dS). It comes out significant in 19% and 51% of data sets.
  - **It is attenuated.** The refitted shared distractor gain of the base takes up part of a planted HP-specific
    change. A planted −10 is estimated as about −5, and −5 as about −3.
  - **The real value is what no distractor-specific effect produces.** The reported dD alone (−0.8%) sits inside the
    δ = 0 range: −2.3 to +0.7 under the estimated-alone generator, −3.2 to +0.03 under the joint-fit generator.
- **dD − dT estimated alone** cancels most of the sustained leak (false positives 6–12%) but is attenuated twice over.
  Its power stays below 60% even at −10.

## Full tables and sources

- **All estimators × generators,** including "any area" and "any pathway" detection and the estimators with the base
  held at the real-data fit: `item1c_tables_generated.md` (this folder).
- **Data:**
  - `notes/data/hp_terms_power_2026-11-13.tsv`: every number in the tables, one row per generator × estimator × δ;
  - `notes/data/hp_terms_power_per_draw_2026-11-13.tsv.gz`: group tests per simulated data set;
  - `notes/data/hp_terms_real_estimates_2026-11-13.tsv`: real-data estimates per participant × area;
  - `notes/data/hp_terms_linear_2026-11-13.npz`: linear sufficient statistics per participant × area (Gram
    matrices, generator and noise projections), from which every simulated estimate is recomputed.
- **Code:**
  - `retsupp/modeling/hp_terms_heldout_power.py` runs on the cluster: `retsupp/modeling/slurm_jobs/hp_terms_heldout_power.sh`,
    `--array=3-30`, 200 draws, output `derivatives/hp_terms_heldout_power_v2_2026-11-13`. It took 8 min and 23 GB per
    participant, about 4 CPU-h in total.
  - `retsupp/visualize/paper/hp_terms_heldout_power_stats.py` makes the tables: `collect` on the cluster, `stats`
    locally.
