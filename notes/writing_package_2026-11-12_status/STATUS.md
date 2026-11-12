# Status update, 12 Nov 2026, afternoon (between update 7 and update 8)

This note covers what is finished, what is still running, and how to mark the running parts in the current manuscript.
It applies on top of update 7.

- **Ready to use now:** the new Fig. 6 (section A).
- **Final but not yet in figures:** the refit that fits pRFs and gain fields together (section B). It comes formally
  with update 8.
- **Running:** three analyses (section C). Each gets a placeholder in the manuscript so that it is clear what is
  pending and which numbers may move.

---

## A. New Fig. 6, decoding (ready; replaces the current Fig. 6)

Files: `figures/fig6_decoding_v13_n28.{pdf,png,md}`. The figure is 180 mm × 4.9 in, in two rows. The `.md` is the full
caption with every number. Decoders on pooled dorsal and occipitotemporal voxels run tonight; panel letters a–f stay.

The figure's argument, row by row:

- **Row 1, "Bar decoded in every area, lower facing HP".**
  - **a: example frames.** The bar, the best run, and the mean over participants.
  - **b: decoding quality per area.** The bar is recovered from held-out runs in every area.
  - **c: the HP effect per area,** directly under b on the same area axis. HP − other three quadrants (% of the
    four-quadrant mean), mean ± SEM, BH correction across the nine areas.
    - V2 −1.68 ± 0.73% and V3 −1.62 ± 0.59% reach q = .033; all other areas have q ≥ .18.
    - V3AB, hV4 and VO are as large as V2 and V3 but noisier.
  - **d: the same contrast pooled per pathway,** on c's y-axis.
    - Early −1.35 ± 0.51% (p = .010)
    - Occipitotemporal −1.06 ± 0.41% (p = .006)
    - Dorsal −0.12 ± 0.75% (p = .56, n = 24)

    All areas pooled: −1.14 ± 0.39%, p = .003. One-sided Wilcoxon tests.
- **Row 2, "Lower intensity facing HP (V1–V3 pooled)".**
  - **e, left: two rotated maps** (HP at 12 o'clock), the data next to what the fitted gain-field model predicts for
    the same readout. Both show lower decoded intensity toward HP. The spatial correlation inside the aperture is
    r = 0.51.
  - **e, right: the bar sweep.** The difference on each segment of the bar against bar centre, for three segments:
    - the quadrant facing HP (red);
    - next to HP (dark grey);
    - opposite (light grey, dashed).

    Icons under the axis show which quadrants a bar crosses. Wherever the bar enters the quadrant facing HP, that
    segment is decoded weaker (−1.3% at +0.9°, p < .05). The segment next to HP is weaker nowhere.
  - **f: quadrant readout** with the model prediction. Observed: HP −1.02 ± 0.38%, next to HP +0.57 ± 0.17%, opposite
    −0.12 ± 0.24% (mean ± SEM), HP < next one-sided p = .004. Predicted: HP −0.91%, next +0.19%, opposite +0.53%.

**Moved to the supplement:**

- `figures/figS_decoding_eccentricity_n28`: the check against the items outside the aperture. The effect does not grow
  toward the aperture edge, and the model predicts the same smooth rise. Cite it in one sentence.
- `figures/figS_decoding_hp_per_area_v10_n28`: the per-area test with both decoders, and the per-area values with
  their exact p and q (the same values as in c).
- `figures/figS_decoding_sweep_per_area_n28`: the sweep and maps for every area. The pattern is clear in V2 and V3;
  higher areas are too noisy per area.

**How the model prediction was made (Methods).** For every participant, run and area (V1–V3), we simulated BOLD from
the participant's fitted joint model, with pRFs and gain fields fitted together and every term as fitted. We added that
run's real residuals, phase-randomised. A paired control was simulated with the same noise and all gains set to zero.
Both were decoded exactly as the data, with the gain-free, bar-only decoder. The prediction is model minus control per
participant, averaged like the data. The gain fit used all runs, including the decoded one.

**Suggested Results text:**

> "The gain fields fitted to the BOLD time series predict this readout without further fitting: decoding simulated from
> each participant's fitted model and decoded in the same way showed lower intensity in the quadrant facing HP of about
> the observed size (HP − next to HP: −1.09 ± 0.44% predicted, −1.58 ± 0.53% observed; HP − other quadrants: −1.18%
> vs −1.32%), and participants with a larger predicted effect showed a larger decoded effect (r = .57, p = .002). The
> ordering of the remaining quadrants lay outside what the model can express (one sustained gain shared by the three
> low-probability locations) and was not reliable in the data (next to HP − opposite +0.68%, p = .070)."

> "Pooled over areas, the decoded bar was also weaker in the quadrant facing HP outside early visual cortex, in
> occipitotemporal (−1.06 ± 0.41%, p = .006) but not in dorsal areas (−0.12 ± 0.75%, p = .56). Area by area, the
> effect was reliable in V2 and V3 (q = .033)."

  Do **not** add "the same split as for the HP-specific gain". After the pRF refit, the per-pathway dS (Fig. 5c) is not
  reliable in any single pathway once corrected (early q = .057, occipitotemporal q = .089); see section B."

---

## B. pRFs refitted together with the gain fields (final; formal numbers and figures in update 8)

**Methods (replaces any text saying the pRFs were kept from the fit without gain fields).**
- For the two models the paper reports, every pRF parameter was refitted together with the gain fields. Those models
  are the sustained-only model (sustained HP and LP gains plus the transient target and distractor terms, m4_TDS) and
  the joint model (m7_full).
- Fitting alternated two rounds of two steps:
  1. gradient descent on the pRFs, with the gains fixed;
  2. L-BFGS on the gains, with the pRFs fixed.
- Each step was accepted only if the training error fell. The fit improved in every participant and area (median
  −0.03% SSE), and pRF centres moved by a median of 0.009°.
- The HRF stayed at the per-area HRF.
- Items and the target, distractor and sustained gains are reported from the joint fit. The terms estimated alone (dS,
  dD, dT) were added to the refitted sustained-only model.

**What changes (medians):**

| | Before | After refit |
|---|---|---|
| dS estimated alone, pooled | −1.4% (p = 1×10⁻⁴) | −1.0% [−1.9, −0.4], 22 of 28 < 0, p = 2.7×10⁻⁴ |
| dS estimated alone, per area | V3, TO significant | V3 −2.7%, hV4 −2.4%, TO −1.9% (each q = .036) |
| dS estimated alone, per pathway | early, occipitotemporal q = .040 | not reliable in any single pathway (early q = .057, occipitotemporal q = .089) |
| dS in the joint fit | −3.4% (p = .003) | −3.5% [−6.7, −1.3], p = .0044; V3 −9.5% (q = .034), hV4 −5.6% (q = .024); TO no longer significant |
| Target gain | +16% | +17%, same areas significant |
| Distractor gain | +6.5% (p = .023) | +8.0% (p = .023), same areas |
| Sustained LP gain | −6.7% | −11% (TO −67%, IPS −58%) |

- **Claims:** the dS claim stands pooled and per area (V3, hV4, TO when estimated alone), and so do the target and
  distractor results. The per-pathway statement ("in early and occipitotemporal areas, not in dorsal") no longer holds
  after correction; describe dS by area and pooled.
- **Which fits count:** as for every earlier number, estimated-alone fits that fail the linearity check are excluded
  (13 of the dS fits on the refitted base). Without that filter the pooled value would be −1.3%; do not use it.
- **Exact values and sources** are in update 8 (`key_numbers.md`), which supersedes this table.
- **The sustained LP gain needs care.** Its sign holds, but in higher areas its size trades off against the item
  weight and the pRF amplitude, so its absolute level is not well determined. Write "negative", not a precise size.
- **Fig. 5 and the parameter table** are redrawn from these fits in update 8. Until then, keep the current Fig. 5 and
  mark its numbers as "to be updated".

---

## C. Still running: placeholders for the manuscript

Mark each in the text with **[PENDING: …]** so the numbers can be swapped in when the results arrive.

**C1. Voxel selection, gain fields** (expected tonight).
- **Why:** voxels were selected by cross-run reliability (leave-one-run-out R² > 0, about r > 0.3). That keeps few
  voxels in higher areas: a median of 11 in VO, 19 in IPS and 80–115 in TO, hV4 and V3AB.
- **What runs:**
  - Two looser, model-free selections, without p_signal or a voxel cap: each voxel's time course correlated with the
    mean of the other runs at r > 0.2 or r > 0.1, and above a circular-shift null (p ≤ .01).
  - That gives 2–4× more voxels in higher areas.
  - The joint fits are rerun, again with pRFs and gain fields free together, starting from the refitted fits.
- **Fixed decision rule:** the current selection stays main unless a looser one clearly shrinks the standard errors of
  dS and the LP gain. Otherwise the looser selections go to the supplement.
- **Placeholder (Methods/Results):** "[PENDING: robustness of the gain-field estimates to a looser, model-free voxel
  selection (r > 0.2 and r > 0.1 against a circular-shift null)]".
- **If a looser selection wins,** the Fig. 5 numbers change once more. Update 8 is built so that this is one rerun.

**C2. Voxel selection, decoding** (just started; expected tonight).
- **What runs:** the same two looser selections for the decoder, computed per fold on the training runs only, so the
  decoded run never chooses its own voxels. Readouts: decoding quality per area, the quadrant test (6d), the pooled
  and per-area effects (Fig. 6c, d and the per-area supplement), the sweep, and pooled-voxel decoders for the dorsal and occipitotemporal pathways (to fill the free slot in row 3).
- **Same decision rule:** the main figure keeps the current selection unless a looser one clearly improves decoding
  quality and the precision of the HP effect.
- **Main interest:** whether the higher areas (V3AB, hV4, VO), now large but noisy, become reliable with more voxels.
- **Placeholder (Fig. 6c/d text, Supplement):** "[PENDING: decoding with a looser voxel selection]".

**C3. The HP-specific gain along the visual hierarchy** (update 9; runs on the refitted fits).
- **Question:** does each gain term grow or shrink along the hierarchy (pRF size, area rank)? Includes target vs dS,
  and dS relative to the sustained gain.
- **Placeholder (Discussion paragraph on the hierarchy):** "[PENDING: hierarchy analysis on the refitted fits]".

---

## Decided, no further analysis planned

- **Base models without the gain-field terms (m0–m3b) are not refitted.** Items and the target, distractor and sustained
  gains come from the joint fit only.
- **No Mexican-hat field** (suppression at HP with a surrounding boost) to explain next-to-HP > opposite in the
  decoding. The model splitting the LP gain into next-to-HP and opposite found no difference (−0.6%, p = .73). The
  item-location responses (Fig. 3) show next to HP lower than opposite, the reverse of what that field predicts.
- **No retinotopic coding in hippocampus or entorhinal cortex.** The repeating response in the medial temporal lobe was
  about 100× weaker than in lateral occipital cortex. Pooled over subjects, reliable voxels showed no contralateral bias
  (hippocampus 53%, p = .16). Not reported unless asked.

## Expected order of arrival

1. Update 8 (Fig. 5 v11 and the parameter table from the refitted fits): ready, sent with this note
   (`writing_package_2026-11-12_update8.zip`). It is provisional until C1 is decided and supersedes the table in
   section B.
2. C1 and C2: tonight.
3. Update 9 (C3): after update 8.

---

## Accuracy per condition (fills the TODO in the behaviour paragraph)

Source: `notes/data/behaviour_accuracy_n28_2026-11-12.tsv`, from `retsupp/visualize/paper/behaviour_accuracy.py`.

The statistics are the same as for the response times, n = 28:
- **Accuracy:** per participant, % correct in each condition. The 499 trials without a response count as errors.
- **Effect:** the median of the within-participant differences, in percentage points (pp), with a bootstrapped 95% CI
  (10,000 resamples over participants).
- **Also shown:** the condition means with the mean difference in brackets, and the number of participants in the
  median's direction.
- **Test:** two-sided Wilcoxon signed-rank.
- **Target comparison:** on distractor-present trials only, as for the response times.

| Effect | Median difference [95% CI] | Condition means (mean difference) | Participants in that direction | p |
|---|---|---|---|---|
| Distractor at HP vs LP | +2.3 pp [+1.3, +4.4] | 86.6 vs 83.6% (+3.0) | 24/28 | 6.7 × 10⁻⁵ |
| Distractor at LP vs absent | −5.2 pp [−6.1, −3.7] | 83.6 vs 89.5% (−5.8) | 26/28 | 1.0 × 10⁻⁵ |
| Distractor at HP vs absent | −2.9 pp [−4.0, −1.6] | 86.6 vs 89.5% (−2.8) | 25/28 | 3.7 × 10⁻⁴ |
| Target at HP vs LP (distractor trials) | −2.0 pp [−3.5, −1.3] | 83.5 vs 86.2% (−2.7) | 22/28 | .0011 |

Accuracy follows the response times in every comparison, so there is no speed–accuracy trade-off:
- a distractor at HP costs less than one at LP;
- a distractor at LP costs most;
- a target at HP is found less accurately as well as more slowly.

**Suggested text, replacing the ⟦TODO⟧:**

> "Accuracy mirrored the response times, so the effects did not reflect a speed–accuracy trade-off: participants were
> more accurate with the distractor at the HP than at an LP location (86.6 vs 83.6%; median difference +2.3 percentage
> points [1.3, 4.4], 24 of 28 participants, p = 6.7 × 10⁻⁵), least accurate with a distractor at an LP location
> relative to no distractor (−5.2 points [−6.1, −3.7], 26 of 28, p = 1.0 × 10⁻⁵), and less accurate when the target
> appeared at the HP location (83.5 vs 86.2%; −2.0 points [−3.5, −1.3], 22 of 28, p = .0011)."
