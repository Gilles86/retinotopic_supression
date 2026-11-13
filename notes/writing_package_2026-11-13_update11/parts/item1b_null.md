# Item 1b: evidence for the null for the transient HP − LP terms (n = 28)

**Question.** Do the data support dD = 0, dT = 0 and dD − dT = 0, or are they just not significant?

**Short answer.** No transient term has positive evidence for zero. The Bayes factors are between 0.6 and 3.7 in
favour of the null, which is inconclusive to moderate and depends on the prior and on outliers. Equivalence holds only
for bounds much larger than the sustained effect:
- dD estimated alone: ±2.9 pp;
- dD − dT estimated alone: ±5.4 pp;
- dD in the joint fit: ±8.2 pp.

At the size of the sustained effect itself (dS estimated alone, ±1.0 pp), none of them is equivalent to zero. dT
estimated alone, if anything, leans away from zero.

## Definitions

**Fits.** Refitted reference of update 8. Estimated alone means each term added alone to the model with item weight,
target, distractor and sustained gain (m4_TDS, pRFs refitted). Joint means all terms fitted together (m7_full, pRFs
refitted). The joint fit has no separate target term at HP (the target gain is tied across locations), so there is no
joint dT.

**Pooling.** Per participant, the median over the nine areas. For dD − dT, the difference is taken per participant
× area first. Values are in percentage points (pp) of gain.

**Statistics.**
- Bootstrapped 95% CI: 5,000 resamples of participants.
- Wilcoxon signed-rank tests, as in key_numbers.
- **JZS Bayes factor** (Rouder et al. 2009): one-sample, with a Cauchy prior on the standardized effect.
  - r = √2/2 is the default.
  - r = √2/4 is a narrower prior, which suits small effects.
  - Also run on 10%-winsorized values, because the per-participant values are heavy-tailed.
- **Bayesian sign test:** share of participants below 0, with H0 at .5 and H1 uniform. It is robust to outliers but
  has little power.
- **BF01 > 1 favours the null.** As a guide, 1–3 is anecdotal and 3–10 moderate.

**Equivalence (TOST).** Two one-sided Wilcoxon signed-rank tests on the values shifted by ± the bound, at α = .05.
The t-based TOST gives the same picture and is in the TSV.

**Bounds, and why:**
- **±1.0 pp:** the pooled size of dS estimated alone (−1.0%), the HP effect the paper reports. A transient
  HP-specific term of that size would be as large as the sustained one.
- **±3.5 pp:** the pooled dS in the joint fit (−3.5%). This is the matching bound for joint dD.
- **±2 and ±5 pp:** two of the effect sizes simulated in item 1c (sensitivity), so the two items can be read together.
- **Smallest bound:** we also report the smallest bound at which equivalence holds. This number does not depend on
  choosing a bound.

**Calibration.** dS goes through the same tests, so the reader can see what a reliable term looks like here.

## Bayes factors and the location of each term

| Term | Fit | Median [95% CI] (pp) | Participants < 0 | Wilcoxon p | Mean (SD) | BF01, JZS r = .707 | BF01, JZS r = .354 | BF01, JZS winsorized | BF01, sign test |
|---|---|---|---|---|---|---|---|---|---|
| dD | estimated alone | −0.8 [−3.0, +0.8] | 16/28 | .14 | −2.1 (6.2) | 1.32 | 0.94 | 0.60 | 3.29 |
| dT | estimated alone | −3.6 [−6.5, +0.1] | 19/28 | .032 | −3.1 (7.4) | 0.61 | 0.49 | 0.15 | 0.75 |
| dD − dT | estimated alone | +1.0 [−0.4, +5.6] | 11/28 | .17 | +1.9 (12.2) | 3.69 | 2.21 | 0.72 | 2.32 |
| dD | joint fit | +0.1 [−2.8, +12] | 14/28 | .35 | +4.0 (15.2) | 2.09 | 1.38 | 1.70 | 4.33 |
| dS (calibration) | estimated alone | −1.0 [−1.9, −0.4] | 22/28 | 2.7 × 10⁻⁴ | −1.3 (1.5) | 0.006 (BF10 = 167) | 0.007 | 0.002 | 0.04 |
| dS (calibration) | joint fit | −3.5 [−6.7, −1.3] | 22/28 | .0044 | −3.6 (7.3) | 0.30 (BF10 = 3.3) | 0.26 | 0.005 | 0.04 |

**In plain language:**
- **dD:** neither "absent" nor "present" is supported (BF01 0.6–1.3). Only the sign test, which ignores sizes, leans
  to the null (3.3).
- **dT:** leans to a reduction at HP. The BF10 is 1.6 with the default prior and 6.5 once the outliers are winsorized,
  so dT cannot be called absent.
- **dD − dT:** gets the most support for zero (BF01 3.7, moderate), but only with the default prior and the outliers
  kept. Without the outliers it is inconclusive (0.7).
- **Joint dD:** anecdotal support for zero (1.4–2.1; sign test 4.3).
- **Calibration:** the same tests give decisive evidence for dS estimated alone (BF10 = 167).

## Equivalence tests

| Term | Fit | 90% CI of median (pp) | TOST p, ±1.0 pp | ±2 pp | ±3.5 pp | ±5 pp | Smallest bound, Wilcoxon TOST | Smallest bound, t TOST |
|---|---|---|---|---|---|---|---|---|
| dD | estimated alone | [−2.5, +0.4] | .61 | .21 | **.017** | **.0012** | ±2.9 | ±4.1 |
| dT | estimated alone | [−5.9, −0.6] | .95 | .84 | .42 | .10 | ±5.5 | ±5.5 |
| dD − dT | estimated alone | [−0.1, +4.5] | .72 | .50 | .25 | .078 | ±5.4 | ±5.8 |
| dD | joint fit | [−2.5, +10] | .76 | .69 | .54 | .33 | ±8.2 | ±8.9 |
| dS (calibration) | estimated alone | [−1.8, −0.4] | .74 | .014 | 1.6 × 10⁻⁷ | 3.7 × 10⁻⁹ | ±1.8 | ±1.8 |
| dS (calibration) | joint fit | [−6.3, −1.6] | .99 | .94 | .63 | .17 | ±6.0 | ±6.0 |

**In plain language:**
- **dD estimated alone** is the only transient term with an equivalence result: a distractor-specific HP effect larger
  than about ±3 pp is excluded. One as large as the sustained HP effect (±1 pp) is not excluded.
- **dD − dT** is not bounded below ±5 pp.
- **dT** cannot be called equivalent to zero below ±5.5 pp. Its 90% CI lies below zero.
- **Joint dD** is too imprecise to bound below ±8 pp.

## What this means for the text

**Do not write "no difference between target and distractor" or "absent".** The defensible wording is:

> "The transient HP − LP terms were not reliably different from zero, but the data do not establish that they are
> absent: Bayes factors were inconclusive (BF01 0.6–3.7 across priors), and equivalence tests excluded a
> distractor-specific HP effect only beyond about ±3 percentage points (dD estimated alone; ±5 for dD − dT), larger
> than the sustained HP effect (−1.0 points)."

**Moving panel d.** If panel d moves to the supplement, this sentence should go with it. Item 1c (sensitivity) tells
whether ±3 pp is a size we could have detected.

**Precision.** Per participant, the transient terms are 4–10 times noisier than dS estimated alone (SD 6–15 pp vs
1.5 pp).

## Sources

- **Code:** `retsupp/visualize/paper/transient_null_and_item_gains.py`. It uses `jzs_bf10` in `fig45_common.py`.
- **Table:** `notes/data/update11_gain_terms_polished_2026-11-12_n28_null_evidence.tsv` (all bounds, t-TOST, 90% CIs).
- **Inputs:**
  - `notes/data/conditional_estimates_polished_2026-11-12.tsv`, rows with `valid` = True;
  - `notes/data/gain_field_parameters_polished_2026-11-12.tsv`, rows dog/m7_full;
  - sub-01 and sub-02 left out.
