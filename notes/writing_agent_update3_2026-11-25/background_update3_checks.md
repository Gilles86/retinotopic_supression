# Update 3: checks before the rewrite around the hierarchical model (25 Nov 2026)

Model choice, fixed from now on: the joint zero-sum gain-field fits only (main model m5_TDS_dS_zs: item weight, target gain, one distractor gain, zero-sum sustained field g_HP + 3 g_LP = 0; per-voxel pRFs and per-area HRF fitted together with the gains by LM). Conventions: posterior mean, 95% HDI, P = posterior probability of the sign; credible = P > .975; n = 28. Code: `retsupp/visualize/paper/gain_meta_zs.py`; outputs: `notes/data/gain_meta_zs_dog_2026-11-25/`.

## 0. A correction to update 2 first

Update 2's hierarchical numbers were not computed on the main model. Its per-cell gains came from a re-solve around the joint fit that left the HP and LP sustained gains free (no zero-sum) and split the distractor gain into HP and LP. That is the unconstrained model this update drops.

Here every estimate is the joint zero-sum fit's own gain: dS = g_HP − g_LP = 4/3 g_HP, the target gain and the single distractor gain. Its SE is a leave-one-run-out jackknife of the same least-squares problem solved under the model's constraints (three free gains), with the pRFs, HRF, field widths and item weight held at the fit. On all runs this constrained solve reproduces the fitted gains (median |difference| 2 × 10⁻⁵ percentage points). The exception is 22 to 27 cells per term where the fit sits on a bound, mostly in TO, LO and IPS; those keep the fitted value.

Update 2's numbers change only slightly. The update-3 numbers below replace them.

| | Update 2 (free re-solve) | Update 3 (joint zero-sum) |
|---|---|---|
| dS, all areas | −3.6% [−5.2, −1.9] | **−3.2% [−4.8, −1.6], P > .999** |
| dS, early / dorsal / occipitotemporal | −6.3 / −2.2 / −2.2 | **−5.5 [−8.0, −3.0] / −1.9 [−3.7, −0.0] / −2.2 [−3.9, −0.6]** |
| dS, areas credible | V1, V2, V3, V3AB, hV4, TO | **V1, V2, V3, hV4, LO** (V3AB P = .970, TO .949, IPS .946) |
| Target, all areas | +22.9% | **+23.5% [+15.3, +31.9]** |
| Distractor, all areas | +4.5% | **+4.7% [+0.9, +8.9], P = .988** |

Fig. 5 is re-rendered on these numbers (section 5); the update-14 `_bayes` figures are superseded.

## 1. Statistical model check (same joint zero-sum gains, two statistics)

(a) Median over participants, bootstrap 95% CI, Wilcoxon p, BH q (nine areas; three pathways). Pooled = per participant the median over areas.
(b) Hierarchical model, Student-t.

**dS (%)**

| Unit | (a) median [95% CI], p (q) | (b) mean [95% HDI], P |
|---|---|---|
| All areas | −5.5 [−7.6, −2.1], p = .012 | −3.2 [−4.8, −1.6], P > .999 |
| Early | −8.2 [−11.0, −5.0], p = .020 (q = .061) | −5.5 [−8.0, −3.0], P > .999 |
| Dorsal | −0.8 [−7.9, +5.2], p = .75 (q = .75) | −1.9 [−3.7, −0.0], P = .977 |
| Occipitotemporal | −4.2 [−10.0, +0.2], p = .082 (q = .12) | −2.2 [−3.9, −0.6], P = .996 |
| V1 | −8.1, p = .086 (q = .16) | −4.7 [−7.9, −1.5], P = .996 |
| V2 | −7.8, p = .090 (q = .16) | −5.8 [−9.0, −2.4], P > .999 |
| V3 | −10.2, p = .028 (q = .13) | −6.0 [−9.5, −2.5], P > .999 |
| V3AB | −1.8, p = .77 (q = .77) | −1.8 [−3.8, −0.0], P = .970 |
| IPS | −1.2, p = .60 (q = .77) | −1.9 [−4.3, +0.5], P = .946 |
| hV4 | −8.2, p = .0025 (**q = .022**) | −5.0 [−8.1, −2.2], P > .999 |
| VO | −2.2, p = .41 (q = .62) | −0.2 [−2.4, +2.0], P = .57 |
| LO | −1.2, p = .76 (q = .77) | −2.2 [−4.1, −0.3], P = .989 |
| TO | −2.8, p = .048 (q = .14) | −1.5 [−3.3, +0.3], P = .949 |

**Target gain (%)**

| Unit | (a) | (b) |
|---|---|---|
| All areas | +23.4 [+13.6, +35.0], p = 4 × 10⁻⁷ | +23.5 [+15.3, +31.9], P > .999 |
| Early | +9.2, q = .15 | +9.4 [−0.3, +19.2], P = .974 |
| Dorsal | +38.2, q < .001 | +30.6 [+19.1, +43.1], P > .999 |
| Occipitotemporal | +35.7, q < .001 | +30.5 [+20.4, +40.6], P > .999 |
| Areas significant / credible | V3, V3AB, IPS, hV4, VO, LO, TO (q < .05) | the same seven (P ≥ .998); V1 −5.9, V2 +6.7 not credible |

**Distractor gain (%)**

| Unit | (a) | (b) |
|---|---|---|
| All areas | +10.5 [+1.4, +16.5], p = .0065 | +4.7 [+0.9, +8.9], P = .988 |
| Early | +1.8, q = .24 | +1.1 [−4.0, +6.4], P = .66 |
| Dorsal | +8.1, q = .11 | +6.0 [+0.4, +11.6], P = .982 |
| Occipitotemporal | +11.0, q = .011 | +6.7 [+2.4, +11.2], P = .998 |
| Areas significant / credible | V3AB, hV4 (q = .050) | V3AB, hV4, VO |

**Verdict.** On the same joint zero-sum gains, both statistics put the largest dS in V1 to V3 and hV4. The medians are −8 to −10% there and −1 to −3% elsewhere. The early-visual emphasis therefore comes from the joint fit, not from the hierarchical model. The manuscript's V3, hV4 and TO came from dS estimated alone (V1 +0.1%, V2 −2.8% there).

The hierarchical model changes how many areas are credible: five for dS against one with BH. Its estimates are smaller, because precise cells with smaller effects weigh more. Target and distractor agree in pattern, but the hierarchical distractor gain is less than half the median. (a) becomes the supplementary check.

## 2. Is early visual cortex overweighted?

(a, b) Calibration (median of half the odd − even difference over the jackknife SE; 0.67 expected), median SE (percentage points) and median voxels:

| Unit | dS calibration | dS median SE | Target calibration | Distractor calibration | Median voxels |
|---|---|---|---|---|---|
| Early | **0.82** | 9.2 | 0.53 | 0.73 | 382 |
| Dorsal | 0.66 | 6.4 | 0.49 | 0.72 | 49 |
| Occipitotemporal | 0.73 | 8.4 | 0.67 | 0.71 | 87 |
| V1 / V2 / V3 | 0.86 / 0.73 / 0.78 | 11.6 / 8.2 / 8.7 | 0.63 / 0.69 / 0.41 | 0.60 / 0.65 / 0.88 | 395 / 379 / 364 |
| V3AB / IPS | 0.78 / 0.59 | 4.0 / 21.7 | 0.52 / 0.33 | 0.66 / 0.91 | 90 / 29 |
| hV4 / VO / LO / TO | 0.77 / 0.65 / 0.72 / 0.58 | 7.3 / 6.9 / 11.4 / 9.2 | 0.63 / 0.64 / 0.83 / 0.75 | 0.96 / 0.66 / 0.81 / 0.65 | 115 / 16 / 216 / 80 |

The dS SEs in early areas are about 20% too small (0.82 vs 0.67), so early cells were somewhat overweighted. Early areas have 4 to 8 times more voxels, but their dS SEs are not smaller than elsewhere.

(c) SEs scaled per pathway so that each pathway's calibration is 0.67 (early dS SEs × 1.22):

| dS | Student-t (main) | Inflated per pathway | (d) No SE weighting |
|---|---|---|---|
| All areas | −3.2, P > .999 | −3.1, P > .999 | −4.4, P = .998 |
| Early | −5.5, P > .999 | −5.2, P > .999 | −6.9, P > .999 |
| Dorsal | −1.9, P = .977 | −1.8, P = .977 | −1.3, P = .73 |
| Occipitotemporal | −2.2, P = .996 | −2.1, P = .996 | −4.0, P = .989 |
| Areas credible | V1, V2, V3, hV4, LO | V1, V2, V3, hV4, LO | V1, V2, V3, hV4, OT pathway |

Early minus dorsal: +3.6 [+1.2, +6.1], P = .998. Early minus occipitotemporal: +3.3 [+1.1, +5.7], P = .998.

**Verdict.** "Largest in early visual cortex" survives. It holds with corrected SEs, without precision weighting (where it is larger still), and with the plain medians of section 1. The early pathway reduction is reliably larger than in the other two pathways.

## 3. Hierarchy statistics on the joint zero-sum estimates

Area-level slope inside the hierarchical model: θ = α + β·log2(mean pRF size of the area) + area + participant, Student-t. Areas are the units here: nine points, with their residual spread estimated. In the old per-participant slope test (Wilcoxon on 28 per-participant slopes), between-area scatter was treated as fixed. Slopes are in percentage points per doubling of pRF size.

| Manuscript sentence | Old (estimated alone / per-participant slopes) | Joint zero-sum, per-participant slopes (update 13) | Joint zero-sum, hierarchical | Status |
|---|---|---|---|---|
| dS did not change with pRF size (−0.7 pp, p = .61) | −0.7, p = .61 | −0.4 [−6.9, +10.6], p = .75 | **+3.2 [+0.8, +5.6], P = .993** (relative to its mean +1.27 per doubling) | **Reword**: "the reduction became smaller in areas with larger pRFs (+3.2 points per doubling)". It is the reverse of the target gain. |
| Target gain grew more with pRF size than dS (p = .018) | p = .018 | paired difference p = .0012 | Absolute slopes: +8.8 [−19, +36], P = .76. Relative to each term's size: target +0.63 vs dS (sign-flipped) −1.27, difference P = .979 | **Reword**: "relative to its size, the target gain grew along the hierarchy and the reduction at HP shrank". Do not claim a difference in absolute slopes. |
| Target gain grew along the dorsal and occipitotemporal pathways (p = .011, .003) | p = .011, .003 | p = .0061, .0022 | Slope with V1 to V3 + dorsal: +17 [−42, +74], P = .84; with V1 to V3 + occipitotemporal: +15 [−28, +59], P = .79 | **Reword as a pathway difference**: the target gain was larger in dorsal (+30.6%) and occipitotemporal areas (+30.5%) than in early areas (+9.4%), by 21 points [+9, +34] and [+11, +32], both P > .999. A slope over 5 to 7 areas is not credible. |
| Target gain +27 pp per doubling of pRF size | +27 | +32 [+20, +79], p = 8 × 10⁻⁵ | +12 [−17, +39], P = .83 | **Go** (replaced by the pathway difference above). |
| Width of the sustained field did not follow pRF size | n.s. | −2.5° per doubling [−9.9, −0.6], p = .125, but only 4 participants have a reliable width in enough areas (44 fits) | not testable | **Go**. The sustained field's width is not identified per area and participant (update 13, part C); keep only the group width (FWHM 4 to 7°). |
| Within participants, a larger target gain went with a larger HP reduction (ρ = −0.17) | −0.17, p = 6 × 10⁻⁴ | −0.20 [−0.47, −0.08], 22/28, p = .0046 | Split halves (target from odd runs, dS from even runs and vice versa, so their errors are independent): **ρ = −0.23 [−0.37, −0.07], 20/28, p = .0055** | **Holds**, with the update-3 numbers. It is not an artefact of shared estimation error. |

The distractor gain does not change with pRF size: +1.8 [−6.2, +10.1], P = .70. It is larger in occipitotemporal than in early areas (+5.4 [+0.6, +10.1], P = .986).

## 4. Robustness across pRF models (Gaussian and DN)

PENDING. Zero-sum joint fits for the Gaussian and DN families did not exist; they are being run on the cluster (canaries first, then all 28 participants × 9 areas, then the width-profile statistics for the SEs). The hierarchical model will then be run on them with `gain_meta_zs.py --family gauss|dn --root <width-profile root>`. This section will be filled in when they finish.

Until then, the robustness numbers in the Results and Limitations cannot be replaced: "DoG −3.5%, DN −7.8%, Gaussian −8.8%" and "the DoG model gave the smallest of the three estimates". Those come from non-zero-sum fits.

## 5. Fig. 5a areas

Suggestion: **V3, V3AB and hV4**, one area per pathway. Each shows all three components.
- V3: dS −6.0%, target +28%.
- V3AB: dS −1.8% (P = .970), target +35%, distractor +8%.
- hV4: dS −5.0%, target +52%, distractor +13%.

TO (current) has only a weak dS (P = .949) and no credible distractor gain. V1 and V2 have the largest dS but no target or distractor gain, so their transient maps would be blank.

Rendered on the update-3 numbers:
- Current areas (V3, hV4, TO): `/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/fig5_gain_fields_v11_abc_zs_n28_palA_zsc_bayes.pdf`
- Suggested areas (V3, V3AB, hV4): `/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/fig5_gain_fields_v11_abc_zs_n28_palA_zsc_V3-V3AB-hV4_bayes.pdf`

Both replace the update-14 `_bayes` figure, which was built on update 2's unconstrained re-solve.

## 6. Decoding through the same model

Per participant × area, the Fig. 6c effect (decoded intensity in the quadrant facing HP minus the other three, % of the four), its jackknife SE over runs and its calibration. Calibration is 0.62 overall; early 0.48 (SEs too large), dorsal 0.66, occipitotemporal 0.78. Code: `retsupp/visualize/paper/decode_meta.py`; output: `notes/data/decode_meta_2026-11-25/`.

| Unit | Hierarchical, mean [95% HDI], P | Wilcoxon, median, two-sided p (BH q) |
|---|---|---|
| All areas | −0.71 [−1.76, +0.39], P = .90 | −0.65, p = .032 |
| Early | −0.97 [−1.99, +0.05], P = .968 | −0.88, p = .025 (q = .045) |
| Dorsal | −0.11 [−2.46, +1.99], P = .53 | −0.90, p = .90 |
| Occipitotemporal | −0.82 [−2.54, +0.87], P = .83 | −1.48, p = .030 (q = .045) |
| V3 | −1.40 [−2.81, −0.06], P = .984 | −1.69, p = .0095 (q = .066) |
| V2 | −0.75, P = .88 | −1.48, p = .015 (q = .066) |
| Other areas | P ≤ .87 | q ≥ .36 |

Note: the pooled two-sided Wilcoxon here (median over areas, p = .032) differs from Fig. 6's one-sided test on the mean over areas (p = .0033).

**Verdict.** Both put the decoded effect mainly in early areas (V3 is the only credible area, V2 next), as for dS. In the hierarchical model the decoded effect is weaker overall (P = .90) than with one-sided Wilcoxon tests. Reporting decoding with the same model would lower its strength in the text: "credible in V3 and, as a trend, in early visual cortex (P = .968)". Fig. 6c and Fig. 5c then agree in where the effect is, but not in how strong it is.

## 7. Figures and tables that still use other fits

A full list is in `notes/data/gain_meta_zs_dog_2026-11-25/README_audit.md`. The main ones:
- **Fig. 5:** the update-13 version draws dS estimated alone; the update-14 `_bayes` versions use the free re-solve. Both are replaced by the `_zsc` renders above.
- **Fig. 6e, f** and the decode-simulation supplements predict from the polished non-zero-sum m7_full. They need `simulate_decode_fitted.sh` on the zero-sum view root (tens of CPU-h). The manuscript sentences "−1.09 ± 0.44% predicted", "r = .57" and "−0.34 ± 0.18% predicted" will change.
- **Gaussian and DN robustness** (S10, S11, the hierarchy and parameter table columns, S12 DN baseline variants): non-zero-sum. Section 4 is running.
- **Estimated-alone items:** figS_transient_hp_lp (dD/dT alone), the alone rows of figS_pooled_dS_dD and the hierarchy table, the "DoG conditional" column of the parameter table, and figS_loso_per_roi.
  - Drop these, or show joint dD from m7_full_zs.
  - A joint zero-sum dT needs a new ladder model: code change plus about 150 to 250 CPU-h.
- **Other earlier-fit supplements:**
  - Need reruns on the zero-sum view root: S13 (baseline vs gain) and figS_dog_kappa.
  - Need zero-sum ladder models: the LP split and Table 2d.
  - Can be re-rendered locally from existing zero-sum tables: figS_fig7_relnull and figS_item_locations_voxel_selection.
  - The Discussion's item-response predictions ("−4.9 / −1.3%") come from the polished fit.
- **Already compliant:** Fig. 7 (both versions), all trial-wise RT analyses, items 1a/1c joint rows, 2/5b, 3a, figS_width_profiles_zs, figS_gain_vs_prf_size_zs.

## 8. Does pooling make areas credible because their neighbours are?

Partly, yes. Each area was also fitted on its own: the same Student-t model with the same SEs, using only that area's 28 estimates, so no other area contributes. Output: `notes/data/gain_meta_zs_dog_2026-11-25/no_pooling_per_area.tsv`.

| dS | Pooled (main) | Own data only | Median, Wilcoxon p |
|---|---|---|---|
| V1 | −4.7, P = .996 | −4.2 [−9.7, +1.1], P = .938 | −8.1, p = .086 |
| V2 | −5.8, P > .999 | −4.6 [−10.1, +0.9], P = .949 | −7.8, p = .090 |
| V3 | −6.0, P > .999 | −9.4 [−16.3, −2.4], P = .996 | −10.2, p = .028 |
| V3AB | −1.8, P = .970 | −2.4 [−5.0, +0.3], P = .959 | −1.8, p = .77 |
| IPS | −1.9, P = .946 | −0.8, P = .73 | −1.2, p = .60 |
| hV4 | −5.0, P > .999 | −5.3 [−9.1, −1.4], P = .997 | −8.2, p = .0025 |
| VO | −0.2, P = .57 | −1.1, P = .948 | −2.2, p = .41 |
| LO | −2.2, P = .989 | −1.1 [−3.3, +1.1], P = .862 | −1.2, p = .76 |
| TO | −1.5, P = .949 | −0.9, P = .916 | −2.8, p = .048 |

On their own data, only V3 and hV4 are credible. V1, V2, V3AB and VO point the same way (P = .94 to .96).

**V1, V2 and LO become credible only through pooling.**
- V1 and V2 are pulled towards the early mean, which V3 drives.
- LO borrows most: on its own it is −1.1% (P = .86); it reaches −2.2% (P = .989) because the participants with a large reduction elsewhere are assumed to have one in LO too.

Pooling also works the other way: it pulls large estimates down (V3 −9.4 to −6.0; target in VO +75 to +25).

The same happens for the other terms:
- The target gain in V1 is −10% [−17, −3] on its own data (P = .995), but not credible when pooled.
- The distractor gain in V3 is +14% (P = .984) on its own data, but not credible when pooled.

**Verdict.** Pooled per-area credibility is only as good as the assumption that areas in a pathway are alike. With 2 to 4 areas per pathway, that assumption is weakly checked. The pathway and all-area estimates do not depend on it, because they are averages over areas.

Recommended wording:
- Base claims on all areas and the pathways.
- For single areas, name those credible on their own data: dS in V3 and hV4.
- Describe the others as pointing the same way, for example: "the reduction was credible in V3 and hV4 on their own, and V1, V2 and V3AB pointed the same way".
- Do not claim LO.
- Keep the stars in Fig. 5c from the pooled model, and say so in the legend.
