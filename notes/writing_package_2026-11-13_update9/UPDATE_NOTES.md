# Update 9, 13 Nov 2026: robustness checks finished (applies on top of the status update and update 8)

All three voxel-selection checks, the RT links on the refitted fits and the V1 pRF-size check are done.

**Changes from earlier packages:**
- **Placeholders C1 and C2 of the status update:** replace them with the text below.
- **Update 8:** no longer provisional, except for the size of the sustained LP gain (section F).

**Contents:**
- `UPDATE_NOTES.md` (this file).
- The full reports the numbers come from: `voxel_selection_robustness_2026-11-12.md`,
  `decoding_voxel_selection_robustness_2026-11-12.md`, `fig3_voxel_selection_2026-11-12.md`,
  `rt_links_robustness_2026-11-12.md` and `v1_prf_size_floor_2026-11-12.md`.
- `figures/` holds:
  - the new Fig. 7, `fig7_compact_polished_n28`, on the refitted fits;
  - the new supplements `figS_item_locations_voxel_selection_n28`, `figS_decoding_voxel_selection_n28`,
    `figS_fig7_relnull_r02_n28` and `figS_fig7_relnull_r01_n28`.

**Colours.** All figures are being re-rendered in a new colour scheme. HP and the sustained field become purple; the
next-to-HP and opposite locations become lighter purples; the distractor becomes bluish green. This is a re-render
only: no numbers change. Until it arrives, refer to colours in captions generically, or not at all.

All numbers: n = 28 (sub-01 and sub-02 left out), statistics as stated.

---

## A. Voxel selection does not change the gain-field results (replaces placeholder C1)

**Methods.**
- The main analysis selects voxels whose time courses were reliable across runs.
- As a robustness check we refitted the joint model on two looser, model-free selections, again with pRFs and gain
  fields fitted together. Voxels were kept if their cross-run correlation (each run against the mean of the other
  runs) exceeded r = 0.2 or r = 0.1 and beat a circular-shift null (p ≤ .01).
- These selections contain 2–4 times as many voxels in higher areas.
- We fixed the decision rule in advance: a looser selection would replace the main one only if it clearly shrank the
  standard errors of dS and of the sustained LP gain.

**Result.** Every estimate kept its sign under both selections.

| Term (pooled over areas) | Current | r > 0.2 | r > 0.1 |
|---|---|---|---|
| dS estimated alone | −1.2% | −1.0% | −0.8% |
| dS in the joint fit | −3.6% | −4.1% | −2.5% |
| SE of dS alone vs current | 1 | 0.83 | 0.77 |
| SE of dS joint vs current | 1 | 1.15 | 0.85 |
| SE of LP gain (joint) vs current | 1 | 1.21 | 1.10 |

The standard errors did not shrink clearly, so the main analysis keeps the current selection. In this paired
comparison only cells present under all three selections are used, so the current dS-alone value differs slightly
from the paper's −1.0%.

**Suggested text (Methods or Supplement):**

> "The gain-field estimates did not depend on the voxel selection: with looser, model-free selections (cross-run
> reliability r > 0.2 or r > 0.1 against a circular-shift null; two to four times as many voxels in higher areas),
> every term kept its sign and dS remained negative (estimated alone, pooled: −1.0% and −0.8%), without a gain in
> precision."

## B. Decoding (Fig. 6) is robust to voxel selection (replaces placeholder C2)

**Method.** The same two selections were applied to the decoder. Each was computed per cross-validation fold on the
training runs only, so the decoded run never chose its own voxels.

| | Current | r > 0.2 | r > 0.1 |
|---|---|---|---|
| Decoding quality r, V1–V3 | .449 | .451 | .450 |
| HP − next to HP, V1–V3 | −1.58%, p = .004 | −1.57%, p = .006 | −1.64%, p = .003 |
| Areas with BH q < .05 | V2, V3 | V2, V3 | V2, V3, plus VO (q = .030) |
| Occipitotemporal, pooled | −1.06%, p = .006 | −0.12%, p = .052 | −0.89%, p = .017 |

- The r > 0.2 occipitotemporal value is pulled by one participant. Two of their areas sit just above the 20-voxel
  minimum and decode at almost zero intensity, so the percentage blows up. Without that participant it is −0.90%,
  p = .019.
- **Decision:** Fig. 6 keeps the current selection.

**Decoders on whole pathways (supplement only).** One decoder trained on all voxels of a pathway decodes about as well
as the best single higher area (r = .23–.29). It shows no HP effect:
- mid-level (V3AB + hV4): −0.57%, p = .24;
- high-level (LO + TO + VO + IPS): +0.19%, p = .46;
- occipitotemporal: −0.06%, p = .28;
- dorsal: −1.38%, p = .072, n = 24.

Report this in the supplement and do not interpret it. Decoding a whole pathway at once mixes very different pRF
sizes and is not a natural readout. Fig. 6d (the pathway averages of the per-area effects) stays as it is.

**Suggested text (Supplement):**

> "The decoding results did not depend on the voxel selection (fold-wise model-free selections r > 0.2 and r > 0.1;
> decoding quality .45 in V1–V3 for all three; HP − next to HP −1.57 to −1.64%, p ≤ .006)."

**Figure:** `figures/figS_decoding_voxel_selection_n28`.

## C. Fig. 3e: exact p values everywhere; the higher pathways are a trend, not a resolution limit

The re-rendered Fig. 3 gives exact p values for HP − next and HP − opposite in every pathway. The values are
two-sided Wilcoxon, % of the mean:

| Pathway | HP − next | HP − opposite |
|---|---|---|
| Early | −6.7%, p = .095 | −21.6%, p = .005 |
| Dorsal (n = 25) | −2.8%, p = .41 | −6.7%, p = .24 |
| Occipitotemporal | −5.9%, p = .28 | −9.8%, p = .090 |
| All areas | −6.9%, p = .018 | −15.5%, p = .013 |

**Why the higher pathways stay non-significant.** Two candidate explanations were tested.
- **Not pRF blur.** Blur between item locations, estimated from the fitted pRFs, keeps ≥ 97% of the HP − opposite
  contrast in every pathway.
- **More voxels did not help.** Looser selections give 2–6 times as many voxels per item location, but the added
  voxels carry a proportionally weaker effect.

**Suggested text:**

> "HP − opposite was negative in every pathway (dorsal p = .24, occipitotemporal p = .090) but reliable only in early
> visual cortex (p = .005) and pooled over all areas (p = .013). This was not a matter of spatial resolution: blur
> between item locations, estimated from the fitted pRFs, retained ≥ 97% of the HP − opposite contrast in every
> pathway. With looser, model-free voxel selections (up to six times as many voxels per item location) early visual
> cortex showed the same ordering (HP − opposite p ≤ .005), and the dorsal and occipitotemporal effects did not
> become reliable, because the added voxels carried a proportionally weaker effect."

**Figure:** `figures/figS_item_locations_voxel_selection_n28`.

## D. RT links on the refitted fits (Fig. 7 and the between-participant sentence)

Replace the numbers with those from the refitted fits. The conclusions are unchanged.

**Between participants: no link.**

| Fit | dS estimated alone vs HP benefit | dS joint vs HP benefit |
|---|---|---|
| Earlier | Spearman ρ = −.11, p = .57 | ρ = −.01, p = .97 |
| Refitted | ρ = −.13 [−.46, +.25], p = .51 | ρ = +.06 [−.34, +.44], p = .77 |

**Trial-wise gains (Fig. 7, new version `fig7_compact_polished_n28`).**
- **Target gain on target-repeat trials:** −18% [−31, −4] of the area's typical target gain, P = .995. It was −19%,
  P = .997.
- **Distractor gain per SD of log RT:** −2.0% [−3.7, −0.2], P = .990. It was −2.4%, P = .990.
- **Not credible, as before:** the global gain per SD log RT (P = .928), the target gain per SD log RT (P = .833) and
  the distractor gain on distractor-repeat trials (P = .933).

**Robustness (supplement only):**
- **Target repeat** holds under both looser voxel selections (P ≥ .996).
- **Distractor gain on slower trials** weakens: P = .975 at r > 0.2 and .960 at r > 0.1. As before, it vanishes
  without the per-estimate SE weighting.
- **Wording:** keep the distractor–RT link, and add "(sensitive to the voxel selection; Supplementary Fig. S…)".

**Figures:** `figures/figS_fig7_relnull_r02_n28` and `figures/figS_fig7_relnull_r01_n28`.

## E. V1 pRF size (already in the status update; repeated for completeness)

Replace the "floor set by the spatial resolution of the paradigm" sentence; see `v1_prf_size_floor_2026-11-12.md` and
the status update.

> "In V1, pRF size (σ) stayed between 0.54 and 0.65° across eccentricities. This is not a limit of the fitting
> procedure: in simulations with the same stimulus, noise level and fitting steps, pRFs were recovered without bias
> down to σ ≈ 0.4°, independently of the spatial resolution used in the fit, and refitting V1 at twice the resolution
> did not change it. The flat profile more likely reflects broadening that does not depend on eccentricity, such as
> small variations in eye position and the scatter of receptive fields within a voxel, on top of the small true pRFs
> of central V1."

## F. Still open: the size of the sustained LP gain

- **The problem.** The joint fit (pRFs and gain fields together) has not fully converged. Its training error still
  falls slowly after 8 rounds of the current alternating optimiser. The level of the sustained LP gain also depends on
  where the fit starts: in one test area −61% from one start and −17% from another, both unconverged.
- **What is stable.** dS (HP − LP) and its significance are much less affected.
- **What is running.** A test of whether the LP level is identified at all, and a proper joint optimiser
  (Levenberg–Marquardt with the per-voxel / shared block structure).
- **Until then:**
  - report the sustained LP gain by its sign only ("negative"), as already instructed;
  - keep all dS, target and distractor statements;
  - add to Methods: "The level of the gain shared by all four locations traded off against the item weight and the
    pRF amplitude and is reported by its sign only."
- **If the new optimiser changes any reported number,** an update follows; the reference fit regenerates with one
  command.

## G. Coming next

- **Update 10:** the hierarchy analysis on the refitted fits, i.e. whether each gain term changes along the visual
  hierarchy.
- **The re-rendered figures** in the new colour scheme.
- **The LP-gain identifiability and joint-optimiser result** (section F).
