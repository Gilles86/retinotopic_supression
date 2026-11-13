# Item 5a: Fig. 3d/e per pathway, with CIs, the pathway × location interaction and the early estimate against the later CIs (n = 28)

**Question.** In early areas Fig. 3e shows HP < next < opposite, and the effect is reliable. In the dorsal and
occipitotemporal pathways it is not reliable. Is that a difference between pathways, or only lower precision?

**Short answer.** Lower precision. In absolute units (% signal):
- HP − opposite is as large in the dorsal pathway (−0.098) as in early areas (−0.094);
- it is smaller in the occipitotemporal pathway (−0.036);
- every early estimate lies inside both later pathways' 95% CIs;
- the pathway × location interaction is far from significant (rm-ANOVA p = .70, permutation p = .72).

In relative units (% of each participant's mean response) the occipitotemporal HP − opposite is smaller than in early
areas (paired difference −14.6 [−28.6, −0.7], p = .038). Even there the interaction is not significant (Early ×
occipitotemporal p = .089; all three pathways p = .56). The data cannot say that the location effect differs between
pathways.

## Data and method

**Which arm.** The main tables use the current Fig. 3 voxels with the refitted pRFs (update 8; the "refit" arm of
`fig3_voxel_selection.py`, `notes/data/event_deconv_relative_polished_2026-11-12/`). The same numbers for Fig. 3 as
printed (`event_deconv_relative_final2_2026-11`) are in the TSVs and summarised at the end. They differ by 1–2%.

**Window means.** Each value is the mean over the window 3.2–6.4 s after search onset, from the deconvolved responses at
the item locations, relative to the run's HP location. Items are averaged over the three item types; next is the mean
of the two locations next to HP.
- **Relative:** % of the participant's mean response over all twelve item × location cells. Participant × pathway pairs
  whose mean response is ≤ 0 are left out, as in Fig. 3e; this leaves dorsal n = 26.
- **Absolute:** % signal.

**Statistics.**
- **Contrasts:** mean and median over participants, each with a bootstrapped 95% CI (10,000 resamples of
  participants); participants < 0; Wilcoxon signed-rank p (uncorrected, as in Fig. 3e).
- **Interaction:** repeated-measures ANOVA, pathway × location (HP, next, opposite), on participants with all
  pathways. The same interaction F is also tested by permutation (pathway labels shuffled within participant, 10,000
  permutations).
- **Early vs later:** the paired difference of each contrast (early minus later, mean [bootstrapped CI], Wilcoxon), and
  whether the early mean (and median) lies inside the later pathway's 95% CI.

## Contrasts per pathway: relative (% of the mean response)

| Pathway | Contrast | Mean [95% CI] | Median [95% CI] | Participants < 0 | Wilcoxon p |
|---|---|---|---|---|---|
| Early | HP − opposite | −21.5 [−34.6, −8.8] | −21.4 [−31.5, −9.9] | 20/28 | .0056 |
| Early | Next − opposite | −14.5 [−26.5, −3.6] | −10.6 [−16.3, −0.2] | 19/28 | .018 |
| Early | HP − next | −7.0 [−14.2, +0.4] | −6.9 [−16.0, −0.7] | 20/28 | .090 |
| Dorsal | HP − opposite | −10.5 [−33.7, +13.2] | −12.8 [−19.8, −4.5] | 19/26 | .14 |
| Dorsal | Next − opposite | +2.2 [−26.8, +32.4] | −2.5 [−9.4, +7.9] | 14/26 | .78 |
| Dorsal | HP − next | −12.7 [−38.1, +8.8] | −1.1 [−23.1, +4.4] | 13/26 | .29 |
| Occipitotemporal | HP − opposite | −6.9 [−17.4, +4.0] | −6.2 [−18.4, +2.0] | 18/28 | .16 |
| Occipitotemporal | Next − opposite | −3.0 [−10.2, +4.9] | −6.1 [−13.8, +1.8] | 18/28 | .28 |
| Occipitotemporal | HP − next | −4.0 [−16.3, +7.3] | −4.7 [−15.0, +7.6] | 16/28 | .51 |
| All areas | HP − opposite | −15.5 [−25.5, −5.0] | −19.7 [−24.3, −11.8] | 22/28 | .014 |
| All areas | Next − opposite | −8.3 [−17.0, +0.4] | −9.4 [−15.6, −3.1] | 20/28 | .066 |
| All areas | HP − next | −7.2 [−12.4, −1.6] | −11.0 [−14.8, −3.4] | 20/28 | .020 |

**In plain language:**
- Every contrast is negative in both later pathways, except the dorsal next − opposite mean.
- Their CIs are 2–5 times wider than in early areas and all include 0, apart from the dorsal HP − opposite median.
- Dorsal means and medians disagree (e.g. HP − next −12.7 vs −1.1) because a few participants have extreme relative
  values when their mean response is small.

## Contrasts per pathway: absolute (% signal)

| Pathway | Contrast | Mean [95% CI] | Median [95% CI] | Participants < 0 | Wilcoxon p |
|---|---|---|---|---|---|
| Early | HP − opposite | −0.094 [−0.159, −0.031] | −0.103 [−0.135, −0.055] | 20/28 | .0052 |
| Early | Next − opposite | −0.048 [−0.096, −0.004] | −0.046 [−0.071, −0.002] | 19/28 | .043 |
| Early | HP − next | −0.045 [−0.084, −0.007] | −0.024 [−0.068, −0.003] | 20/28 | .060 |
| Dorsal | HP − opposite | −0.098 [−0.222, +0.030] | −0.078 [−0.167, −0.028] | 19/26 | .056 |
| Dorsal | Next − opposite | −0.016 [−0.119, +0.082] | −0.019 [−0.073, +0.033] | 14/26 | .71 |
| Dorsal | HP − next | −0.082 [−0.170, +0.002] | −0.006 [−0.145, +0.021] | 13/26 | .21 |
| Occipitotemporal | HP − opposite | −0.036 [−0.101, +0.039] | −0.071 [−0.142, +0.013] | 18/28 | .14 |
| Occipitotemporal | Next − opposite | −0.018 [−0.070, +0.042] | −0.041 [−0.089, +0.018] | 18/28 | .26 |
| Occipitotemporal | HP − next | −0.017 [−0.090, +0.059] | −0.025 [−0.133, +0.038] | 16/28 | .40 |
| All areas | HP − opposite | −0.084 [−0.147, −0.021] | −0.098 [−0.128, −0.067] | 22/28 | .0082 |
| All areas | Next − opposite | −0.035 [−0.079, +0.009] | −0.043 [−0.063, −0.020] | 20/28 | .16 |
| All areas | HP − next | −0.050 [−0.084, −0.017] | −0.044 [−0.079, −0.016] | 20/28 | .0071 |

**In plain language:**
- In % signal, HP − opposite in the dorsal pathway is as large as in early areas, but about twice as variable.
- In the occipitotemporal pathway the mean is about a third of the early value, while the median is about two thirds.

## Pathway × location interaction (HP, next, opposite)

| Pathways | Unit | n | F | p (rm-ANOVA) | p (permutation) |
|---|---|---|---|---|---|
| Early, dorsal, occipitotemporal | relative | 26 | F(4, 100) = 0.76 | .56 | .59 |
| Early, dorsal | relative | 26 | F(2, 50) = 0.85 | .43 | .47 |
| Early, occipitotemporal | relative | 28 | F(2, 54) = 2.54 | .089 | .082 |
| Dorsal, occipitotemporal | relative | 26 | F(2, 50) = 0.19 | .83 | .82 |
| Early, dorsal, occipitotemporal | absolute | 26 | F(4, 100) = 0.55 | .70 | .72 |
| Early, dorsal | absolute | 26 | F(2, 50) = 0.33 | .72 | .73 |
| Early, occipitotemporal | absolute | 28 | F(2, 54) = 1.43 | .25 | .25 |
| Dorsal, occipitotemporal | absolute | 26 | F(2, 50) = 0.57 | .57 | .57 |

**In plain language:** the location pattern does not differ reliably between pathways in either unit. The closest is
early vs occipitotemporal in relative units (p ≈ .08).

## Early-area estimate against each later pathway's 95% CI

| Contrast | Later pathway | Unit | Early mean | Later mean [95% CI] | Early mean inside? | Early / later median: early inside? | Early − later, paired mean [95% CI] | n | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| HP − opposite | Dorsal | relative | −21.5 | −10.5 [−33.7, +13.2] | yes | −21.4 / −12.8: no | −12.0 [−38.1, +13.8] | 26 | .28 |
| Next − opposite | Dorsal | relative | −14.5 | +2.2 [−26.8, +32.4] | yes | −10.6 / −2.5: no | −18.6 [−51.6, +12.0] | 26 | .37 |
| HP − next | Dorsal | relative | −7.0 | −12.7 [−38.1, +8.8] | yes | −6.9 / −1.1: yes | +6.6 [−17.5, +34.1] | 26 | .82 |
| HP − opposite | Occipitotemporal | relative | −21.5 | −6.9 [−17.4, +4.0] | **no** | −21.4 / −6.2: no | **−14.6 [−28.6, −0.7]** | 28 | **.038** |
| Next − opposite | Occipitotemporal | relative | −14.5 | −3.0 [−10.2, +4.9] | **no** | −10.6 / −6.1: yes | −11.5 [−22.8, −0.9] | 28 | .077 |
| HP − next | Occipitotemporal | relative | −7.0 | −4.0 [−16.3, +7.3] | yes | −6.9 / −4.7: yes | −3.0 [−16.2, +12.1] | 28 | .41 |
| HP − opposite | Dorsal | absolute | −0.094 | −0.098 [−0.222, +0.030] | yes | −0.103 / −0.078: yes | +0.011 [−0.110, +0.140] | 26 | .57 |
| Next − opposite | Dorsal | absolute | −0.048 | −0.016 [−0.119, +0.082] | yes | −0.046 / −0.019: yes | −0.033 [−0.129, +0.065] | 26 | .45 |
| HP − next | Dorsal | absolute | −0.045 | −0.082 [−0.170, +0.002] | yes | −0.024 / −0.006: yes | +0.044 [−0.050, +0.148] | 26 | .88 |
| HP − opposite | Occipitotemporal | absolute | −0.094 | −0.036 [−0.101, +0.039] | yes | −0.103 / −0.071: yes | −0.058 [−0.125, +0.004] | 28 | .17 |
| Next − opposite | Occipitotemporal | absolute | −0.048 | −0.018 [−0.070, +0.042] | yes | −0.046 / −0.041: yes | −0.030 [−0.088, +0.024] | 28 | .42 |
| HP − next | Occipitotemporal | absolute | −0.045 | −0.017 [−0.090, +0.059] | yes | −0.024 / −0.025: yes | −0.028 [−0.106, +0.049] | 28 | .78 |

**In plain language:**
- **In % signal:** the early estimate lies inside every later pathway's CI, for every contrast.
- **In relative units:** the early HP − opposite and next − opposite fall outside the occipitotemporal CI. The early
  HP − opposite is also larger than the occipitotemporal one when paired (p = .038, uncorrected). This is the one
  place where the later pathway looks weaker, not just noisier.

## Fig. 3 as printed (`event_deconv_relative_final2_2026-11`)

The picture is the same:
- **Interaction:** relative p = .62 (all three pathways), early × occipitotemporal p = .19; absolute p = .83.
- **Absolute units:** the early mean lies inside every later CI.
- **Relative units:** the early mean lies outside only the occipitotemporal CI of next − opposite (−14.9 vs −3.9
  [−11.1, +3.7]). Early − occipitotemporal HP − opposite: −11.9 [−27.2, +4.3], p = .056.

## What this means for the text

**Do not write that the location effect is confined to early areas.** Suggested wording:

> "The same ordering (HP < next to HP < opposite) was present in the dorsal and occipitotemporal pathways but was not
> reliable there (Fig. 3e); the pathway × location interaction was not significant (relative p = .56; % signal
> p = .70), and in % signal the early-area estimates lay within the 95% CIs of both later pathways (e.g. HP − opposite:
> early −0.094, dorsal −0.098 [−0.222, +0.030], occipitotemporal −0.036 [−0.101, +0.039] % signal)."

**If the occipitotemporal pathway is discussed.** Add that, relative to the mean response, its HP − opposite is
smaller than in early areas (paired −14.6 points [−28.6, −0.7], p = .038 uncorrected), although the interaction is
not significant.

## Sources

- **Code:** `retsupp/visualize/paper/fig3_pathway_cis.py`. It reads the window means written by
  `retsupp/visualize/paper/fig3_voxel_selection.py`, which uses `fig3_item_locations.rel_stats` and
  `fig3_robustness.window_cells`.
- **Tables:**
  - `notes/data/fig3_pathway_cis_n28_2026-11-13_contrasts.tsv`;
  - `notes/data/fig3_pathway_cis_n28_2026-11-13_interaction.tsv`;
  - `notes/data/fig3_pathway_cis_n28_2026-11-13_early_vs_later.tsv`.
  
  Each has both arms (`selection` = `refit` / `current`).
- **Inputs:**
  - `notes/data/fig3_voxel_selection_n28_2026-11-12_cells.tsv` (relative, per item × location);
  - `notes/data/fig3_voxel_selection_n28_2026-11-12_abs_values.tsv` (% signal, items pooled);
  - sub-01 and sub-02 left out upstream.
