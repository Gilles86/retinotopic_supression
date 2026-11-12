# Field widths conditional on a reliable gain (2026-11-12; n = 28; 241 participant × area fits)

Request: the Results paragraph says the sustained field is 3.2° FWHM (median; 4.6–5.2° in V1–V3, 2.5–3.2° in higher
areas), narrower than the 5.7° between neighbouring item locations; the transient width is often at a bound. A width is
undefined when its gain is near zero, and the medians include such fits. Do the two claims survive?

**Fits.** Current final joint fits (m7_full; DoG pRFs; one HRF per participant × area; reliable voxels), with each
voxel's pRF held at the fit without gain fields — a refit of pRFs and gain fields together is running separately and
may shift these numbers slightly. No new model.

## Method
- **Gains and SEs.** Given the widths, the gains enter the prediction linearly (gain map 1 + Σ g·Gaussian; the
  re-solved gains equal the joint fit's to 4 decimals in every fit). Per participant × area and per run: normal
  equations for the five gains (sustained at HP, at the three LP locations; target; distractor at HP and at LP), every
  other parameter (pRFs, HRF, item scaling, widths) fixed at the joint fit → leave-one-run-out jackknife SE (12 runs).
  Distractor = mean of the HP and LP distractor gains. Calibration: median of |odd − even|/2 / SE = 0.70, 0.66, 0.56
  (sustained HP, sustained LP, target; 0.67 expected). Reliable: |gain| / SE > 2 (also > 3). Sustained field reliable
  when either sustained gain is.
- **Profile over the width.** σ on a log grid (17 values, 0.5–8°); at each σ the field's gains re-solved, **all other
  parameters fixed** at the joint fit. 95% interval of σ from the leave-one-run-out jackknife of the profile minimum
  (log scale; a χ² threshold on ΔSSE with the nominal residual variance would be far too narrow because the residuals
  are correlated over time and voxels). Minimum at a grid edge → open interval.
- Code: `retsupp/modeling/fit_width_profiles.py` (cluster jobs 8303529 canary, 8303646 sub-04–30; 7 min per
  participant), `retsupp/visualize/paper/field_widths_reliable.py`, figure `retsupp/visualize/paper/figS_field_widths_v2.py`.
  Data: `notes/data/field_widths_reliable_n28_2026-11-12_{fits,widths_thr2,widths_thr3,claims_thr2,claims_thr3,tradeoff,volume,calibration}.tsv`.

## Results
**Widths (FWHM, median [bootstrapped 95% CI], n fits), all fits vs fits with a reliable gain (|z| > 2):**

| Area | Sustained, all | Sustained, reliable | Transient, all | Transient, reliable target gain (at a bound) |
|---|---|---|---|---|
| V1 | 5.2° [3.9, 13.9], 28 | 3.9° [1.6, 4.7], 8 | 9.5°, 28 | 10.3°, 7 (43%) |
| V2 | 4.7° [4.0, 6.3], 28 | 5.1° [2.5, 12.3], 6 | 9.3°, 28 | 1.2°, 7 (71%) |
| V3 | 4.6° [3.0, 6.1], 28 | 2.7° [2.3, 3.3], 12 | 5.5°, 28 | 1.2°, 16 (63%) |
| V3AB | 3.2° [2.8, 3.8], 27 | 2.8° [2.5, 3.0], 16 | 4.4°, 27 | 1.2°, 22 (55%) |
| IPS | 3.1° [2.8, 5.0], 25 | 2.8° [2.4, 3.1], 15 | 18.3°, 25 | 18.6°, 14 (57%) |
| hV4 | 3.0° [2.6, 3.8], 28 | 2.6° [2.4, 3.2], 15 | 1.2°, 28 | 1.2°, 21 (76%) |
| VO | 2.6° [2.4, 3.1], 21 | 2.5° [2.2, 2.8], 15 | 5.0°, 21 | 2.3°, 17 (41%) |
| LO | 3.2° [2.8, 4.5], 28 | 2.8° [2.4, 3.5], 14 | 4.8°, 28 | 4.1°, 16 (50%) |
| TO | 2.5° [2.5, 2.7], 28 | 2.5° [2.3, 2.7], 22 | 1.7°, 28 | 1.2°, 18 (61%) |
| All (participant medians) | 3.2° [3.0, 3.7], 28 | **2.7° [2.6, 2.9], 28** | 5.3°, 28 | 3.3°, 28 (58%) |

Share of fits at a bound (σ 0.5° or 8°): sustained 12% of all fits, 6% of reliable fits (V1 29% → 0%).
Threshold 3: sustained, reliable fits 2.5° [2.3, 2.6] (27 participants); V1 4, V2 1, V3 4 fits left.

**Profile intervals (item 3).** Sustained: median width of the 95% interval 17.7° FWHM at |gain| / SE < 1 (30% open),
8.0° at 1–2 (20% open), 4.4° at 2–3 (9%), 1.9° at > 3 (4%); Spearman ρ(|z|, interval width) = −.74. Transient: open in
about half of the fits at every |z| (54%, 48%, 59%, 62%), median 6.6° even at |z| > 3: the transient width is not
constrained even when the target gain is.

**Trade-off (item 4).** Gain vs width, Spearman across fits / repeated-measures r within participants: sustained (mean
gain over the four locations) vs σ ρ = .66 / r = .24 (p = .0003); target ρ = −.61 / r = −.19 (p = .005); distractor
ρ = −.57 / r = −.09 (p = .20). The field volume g · 2πσ² (gain integrated over the plane, deg²) per participant (median
over areas): sustained at HP −1.81 [−2.15, −1.36], 24/28 negative, p = .0003; at LP −0.78 [−1.12, −0.58], 23/28,
p = .0009; target +1.86 [+1.02, +5.70], 25/28, p = .0003; distractor +0.62 [+0.20, +0.99], 21/28, p = .41. Per area with
BH in the volume table.

## The two claims (item 5; fits with a reliable sustained gain)
**(a) Narrower than 5.7° — survives.** 2.7° [2.6, 2.9] over participants (25 of 28 participant medians below 5.7°,
p < 10⁻⁴); per area below 5.7° in V1 (n = 8, 100%, q = .018), V3AB (q = .0003), IPS (q = .0005), VO (q = .027), LO
(q = .030), TO (q = .0005); not in V2 (n = 6, q = .84), V3 (n = 12, q = .10) or hV4 (q = .08) (Wilcoxon of FWHM − 5.7,
BH over nine areas). At |z| > 3 every higher area q ≤ .016 (too few early fits to test).
**(b) Wider in V1–V3 than in higher areas — does NOT survive.** Paired participant medians: early 3.4° vs higher 2.5°,
13 of 20 participants wider in early, p = .053 (20 participants with a reliable early fit: V1 8, V2 6, V3 12 fits);
at |z| > 3: 6 pairs, no difference (2.46° vs 2.49°, p = .69). The 4.6–5.2° early medians in the current paragraph come
largely from fits whose sustained gain is near zero (V1: 29% of all fits at a bound, 0% of reliable ones; V1 and V2 have
a sustained LP gain near zero).

## Suggested wording
- Results (width paragraph): "Where its gain was reliably different from zero (|gain| > 2 SE; 6–22 participants per
  area), the sustained field had a full width at half maximum of 2.7° [2.6, 2.9] (median over participants), narrower
  than the 5.7° between neighbouring item locations in 25 of 28 participants (p < 10⁻⁴). Its width was well determined
  only where its gain was reliable: the width's 95% interval narrowed from 18° at |gain| < 1 SE to 1.9° at > 3 SE. The
  transient width was not constrained even for reliable target gains (at a bound of the fit in 58% of them) and is not
  interpreted."
- Discussion: the data do **not** support a wider sustained field in early visual cortex once fits without a reliable
  gain are excluded (3.4° vs 2.5°, p = .053; no difference at a stricter threshold). Drop the link to Richter et al.'s
  "broad in early visual cortex, refined later", or state it as not supported: "We found no evidence that the sustained
  field is broader in early than in later visual areas once its width was estimated only where its gain was reliable."

## Verdict on Fig. 5a
Use the restricted widths for the sustained maps (V3 changes most: 4.6° → 2.7° FWHM; hV4 3.0° → 2.6°; TO 2.5° → 2.5°);
keep the fixed display width for the transient fields.
