**Supplementary Table S1. Responses at the item locations (Fig. 3).** Same data, voxels and window (3.2 to 6.4 s after search onset) as Fig. 3. Section 1 gives the primary statistics: each unit fitted on its own data, every participant weighted by the precision of their estimate. Per participant and unit, the contrast in the window and its leave-one-run-out jackknife SE (half the odd − even-run difference over the SE: median 0.79 for HP − opposite and 0.70 for target − neutral over all areas, 0.67 expected); then estimate ~ Student-t(ν, μ, √(SE² + τ²)), ν ~ Gamma(2, 0.1), no pooling over areas (PyMC with nutpie, 4 × 2000 draws, target acceptance .99; 0 divergences over all fits). Pathways and all areas: voxels pooled over the pathway's areas weighted by voxel count, as in Fig. 3; single areas: each area alone. Sections 2 to 5 give the earlier statistics (Wilcoxon signed-rank tests and repeated-measures ANOVAs on the unweighted values) as checks. n = 28 (fewer where participants lack voxels at some item location).

## 1. Precision-weighted, each unit on its own data (primary)

**1a. Item types (Fig. 3b, c), % signal.** Posterior mean [95% HDI], P = posterior probability of the sign; * P > .975, ** P > .995.

| Unit | n | Target − neutral | Distractor − neutral | Target − distractor |
|---|---|---|---|---|
| Early | 28 | +0.115 [+0.088, +0.141], P > .999 ** | +0.057 [+0.021, +0.089], P = .999 ** | +0.056 [+0.023, +0.090], P = .999 ** |
| Dorsal | 26 | +0.238 [+0.187, +0.288], P > .999 ** | +0.114 [+0.065, +0.165], P > .999 ** | +0.135 [+0.077, +0.191], P > .999 ** |
| Occipitotemporal | 28 | +0.228 [+0.188, +0.269], P > .999 ** | +0.125 [+0.081, +0.169], P > .999 ** | +0.120 [+0.072, +0.166], P > .999 ** |
| All areas | 28 | +0.149 [+0.122, +0.176], P > .999 ** | +0.077 [+0.043, +0.112], P > .999 ** | +0.079 [+0.047, +0.113], P > .999 ** |
| V1 | 28 | +0.050 [+0.017, +0.083], P = .997 ** | +0.038 [+0.004, +0.073], P = .987 * | −0.011 [−0.045, +0.023], P = .744 |
| V2 | 28 | +0.124 [+0.091, +0.159], P > .999 ** | +0.056 [+0.024, +0.088], P > .999 ** | +0.071 [+0.035, +0.109], P > .999 ** |
| V3 | 28 | +0.178 [+0.143, +0.216], P > .999 ** | +0.075 [+0.030, +0.125], P = .998 ** | +0.126 [+0.070, +0.175], P > .999 ** |
| V3AB | 26 | +0.248 [+0.197, +0.302], P > .999 ** | +0.114 [+0.066, +0.167], P > .999 ** | +0.138 [+0.081, +0.199], P > .999 ** |
| IPS | 13 | +0.175 [+0.099, +0.260], P > .999 ** | +0.145 [+0.070, +0.223], P > .999 ** | +0.071 [−0.057, +0.197], P = .875 |
| hV4 | 28 | +0.259 [+0.194, +0.323], P > .999 ** | +0.147 [+0.096, +0.200], P > .999 ** | +0.129 [+0.064, +0.190], P > .999 ** |
| VO | 17 | +0.218 [+0.127, +0.316], P > .999 ** | +0.072 [+0.000, +0.138], P = .980 * | +0.136 [+0.042, +0.231], P = .998 ** |
| LO | 28 | +0.230 [+0.179, +0.283], P > .999 ** | +0.087 [+0.037, +0.142], P = .999 ** | +0.119 [+0.050, +0.187], P = .999 ** |
| TO | 23 | +0.139 [+0.098, +0.182], P > .999 ** | +0.080 [+0.019, +0.140], P = .994 * | +0.094 [+0.038, +0.146], P > .999 ** |

**1b. Location type (Fig. 3d, e), items pooled, % signal (absolute units).**

| Unit | n | HP − opposite | Next − opposite | HP − next |
|---|---|---|---|---|
| Early | 28 | −0.069 [−0.127, −0.011], P = .986 * | −0.030 [−0.070, +0.010], P = .929 | −0.013 [−0.052, +0.024], P = .739 |
| Dorsal | 26 | −0.062 [−0.159, +0.041], P = .895 | −0.017 [−0.084, +0.055], P = .692 | −0.054 [−0.127, +0.022], P = .921 |
| Occipitotemporal | 28 | −0.079 [−0.159, +0.003], P = .968 | −0.014 [−0.070, +0.040], P = .699 | −0.056 [−0.110, −0.001], P = .976 * |
| All areas | 28 | −0.062 [−0.126, −0.003], P = .978 * | −0.021 [−0.064, +0.025], P = .841 | −0.027 [−0.062, +0.008], P = .937 |
| V1 | 28 | −0.067 [−0.137, +0.008], P = .962 | −0.029 [−0.089, +0.034], P = .825 | −0.019 [−0.069, +0.029], P = .784 |
| V2 | 28 | −0.101 [−0.165, −0.039], P = .999 ** | −0.047 [−0.097, +0.001], P = .968 | −0.025 [−0.075, +0.026], P = .836 |
| V3 | 28 | −0.052 [−0.136, +0.026], P = .899 | −0.018 [−0.073, +0.040], P = .739 | −0.024 [−0.085, +0.035], P = .791 |
| V3AB | 26 | −0.055 [−0.154, +0.044], P = .866 | −0.012 [−0.088, +0.067], P = .642 | −0.044 [−0.115, +0.026], P = .894 |
| IPS | 13 | −0.040 [−0.235, +0.170], P = .660 | −0.064 [−0.246, +0.121], P = .757 | +0.035 [−0.157, +0.208], P = .646 |
| hV4 | 28 | −0.080 [−0.193, +0.035], P = .913 | −0.032 [−0.126, +0.058], P = .748 | −0.027 [−0.138, +0.092], P = .685 |
| VO | 17 | −0.078 [−0.236, +0.077], P = .847 | −0.001 [−0.117, +0.115], P = .506 | −0.032 [−0.157, +0.098], P = .695 |
| LO | 28 | −0.059 [−0.176, +0.071], P = .833 | +0.010 [−0.080, +0.099], P = .580 | −0.057 [−0.140, +0.031], P = .907 |
| TO | 23 | −0.116 [−0.195, −0.028], P = .993 * | −0.003 [−0.084, +0.076], P = .531 | −0.084 [−0.148, −0.023], P = .995 * |

**1c. Location type, % of the participant's mean response (relative units, as Fig. 3e).** Pathways and all areas only (per area the mean response is often near 0, so the relative values are unstable).

| Unit | n | HP − opposite | Next − opposite | HP − next |
|---|---|---|---|---|
| Early | 28 | −15.2% [−29.3, −1.6], P = .984 * | −4.7% [−14.0, +4.4], P = .843 | −3.5% [−12.1, +5.7], P = .784 |
| Dorsal | 25 | −10.5% [−23.9, +2.2], P = .944 | −2.2% [−12.0, +8.7], P = .663 | −9.3% [−19.2, +0.6], P = .961 |
| Occipitotemporal | 28 | −9.2% [−19.3, +2.4], P = .948 | −1.7% [−9.5, +5.4], P = .669 | −8.4% [−15.1, −0.6], P = .983 * |
| All areas | 28 | −11.8% [−24.7, +0.0], P = .971 | −3.0% [−11.5, +5.4], P = .767 | −5.7% [−12.5, +0.9], P = .953 |

## Sections 2 to 6: unweighted statistics (checks)

Same data, voxels and window (3.2–6.4 s) as Fig. 3; values in % of each subject's mean response over the four locations and three item types in that pathway (mean ± SEM over subjects where given). Wilcoxon signed-rank tests, uncorrected; repeated-measures ANOVAs. n = 28 (dorsal: n = 26 in the absolute analyses, two subjects lacking dorsal voxels at some item location; n = 25 in the relative analyses, where sub-30 is also left out because its mean dorsal response is not positive). Source: `fig3_item_locations.py` run at n = 28 (`retsupp/visualize/paper/run_excluding.py --exclude 1 2`; the full generated legend is kept as `notes/data/n28/fig3_item_locations_n28_long_legend.md`).

**2. Location type × item type per pathway (Fourier basis).** Repeated-measures ANOVA location type (HP, next to HP, opposite HP) × item type (target, distractor, neutral), with the planned contrasts on the mean over item types.

| Pathway | Location type | Location type × item | HP − next | Next − opposite | HP − opposite | Next left − next right |
|---|---|---|---|---|---|---|
| Early | F(2, 54) = 7.67, p = .001 | F(4, 108) = 0.92, p = .454 | −6.7 ± 3.6%, p = .095 | −14.9 ± 6.0%, p = .022 | −21.6%, p = .005 | +0.3%, p = .582 |
| Dorsal | F(2, 48) = 0.16, p = .857 | F(4, 96) = 0.86, p = .490 | −2.8 ± 8.5%, p = .411 | −3.9 ± 14.7%, p = .634 | −6.7%, p = .241 | +3.1%, p = .312 |
| Occipitotemporal | F(2, 54) = 1.49, p = .235 | F(4, 108) = 1.33, p = .262 | −5.9 ± 6.3%, p = .284 | −3.9 ± 3.8%, p = .210 | −9.8%, p = .090 | −6.1%, p = .327 |
| All areas | F(2, 54) = 6.31, p = .003 | F(4, 108) = 1.63, p = .172 | −6.9 ± 2.7%, p = .018 | −8.6 ± 4.6%, p = .060 | −15.5%, p = .013 | −0.8%, p = .973 |

**3. Per item type: HP − next to HP and next − opposite (Fourier basis).**

| Pathway | Target: HP − next | Target: next − opposite | Distractor: HP − next | Distractor: next − opposite | Neutral: HP − next | Neutral: next − opposite |
|---|---|---|---|---|---|---|
| Early | −2.2%, p = .552 | −12.1%, p = .131 | −6.6%, p = .284 | −23.3%, p = .090 | −11.3%, p = .006 | −9.3%, p = .218 |
| Dorsal | −5.4%, p = .156 | −0.8%, p = .672 | −0.8%, p = .367 | −14.0%, p = .812 | −2.1%, p = .853 | +3.1%, p = .491 |
| Occipitotemporal | −4.4%, p = .537 | −5.1%, p = .086 | −8.6%, p = .479 | −8.7%, p = .144 | −4.6%, p = .295 | +2.1%, p = .814 |
| All areas | −5.5%, p = .274 | −7.8%, p = .202 | −7.2%, p = .109 | −16.8%, p = .077 | −8.0%, p = .014 | −1.4%, p = .614 |

**4. HP against all three LP locations** (deconvolution with two event types per item; ANOVA location (HP, LP) × item type).

| Pathway | Main effect of location | Location × item |
|---|---|---|
| Early | −9.8%, F(1, 27) = 5.72, p = .024 | F(2, 54) = 0.55, p = .579 |
| Dorsal | −21.4%, F(1, 25) = 0.76, p = .393 | F(2, 50) = 0.83, p = .442 |
| Occipitotemporal | −4.2%, F(1, 27) = 0.53, p = .475 | F(2, 54) = 0.67, p = .517 |
| All areas | −8.4%, F(1, 27) = 6.82, p = .015 | F(2, 54) = 0.22, p = .801 |

**5. DCT basis (robustness).**

Fig. 3b, c and HP against LP: target − neutral p < .001 in all three pathways; distractor − neutral p < .001 in all three pathways.

| Pathway | HP − LP | Main effect of location | Location × item |
|---|---|---|---|
| Early | −12%, p = .034 | p = .065 | p = .828 |
| Dorsal | −9%, p = .059 | p = .396 | p = .447 |
| Occipitotemporal | −4%, p = .210 | p = .516 | p = .506 |
| All areas | not computed | p = .026 | p = .921 |

Location types, DCT basis:

| Pathway | Location type | Location type × item | HP − next | Next − opposite | HP − opposite | Next left − next right |
|---|---|---|---|---|---|---|
| Early | F(2, 54) = 6.35, p = .003 | F(4, 108) = 0.35, p = .841 | −5.9 ± 3.8%, p = .164 | −14.0 ± 5.9%, p = .023 | −19.9%, p = .010 | +0.1%, p = .814 |
| Dorsal | F(2, 48) = 0.34, p = .713 | F(4, 96) = 0.47, p = .758 | −3.3 ± 8.0%, p = .252 | −5.7 ± 12.7%, p = .578 | −9.0%, p = .210 | +5.8%, p = .312 |
| Occipitotemporal | F(2, 54) = 1.25, p = .294 | F(4, 108) = 1.03, p = .396 | −6.9 ± 7.0%, p = .210 | −2.6 ± 4.1%, p = .339 | −9.5%, p = .157 | −5.4%, p = .386 |
| All areas | F(2, 54) = 5.40, p = .007 | F(4, 108) = 0.72, p = .579 | −6.7 ± 2.8%, p = .028 | −8.3 ± 4.8%, p = .090 | −15.0%, p = .018 | −0.2%, p = .991 |

## 6. Per area: location type (Fig. 3d, e), n = 28, unweighted

Items pooled; % of each participant's mean response over the twelve item × location cells in that area (window 3.2–6.4 s); participants whose mean response in an area is not positive are left out (n per area). ANOVA: one-way repeated-measures, location type (HP, next to HP, opposite HP), uncorrected F, BH q across the nine areas. Contrasts: mean ± SEM [bootstrapped 95% CI of the mean, 10 000 resamples], participants below 0, Wilcoxon signed-rank p, Benjamini-Hochberg q across the nine areas per contrast. Exploratory per area; the pathway-level tests are in Fig. 3. Data: `notes/data/fig3_location_per_area_n28_2026-11-12.tsv`; code: `retsupp/visualize/paper/fig3_location_per_area.py`.

| Area | n | Location type F (p, q) | HP − opposite (%) | p / q | Next − opposite (%) | p / q | HP − next (%) | p / q |
|---|---|---|---|---|---|---|---|---|
| V1 | 28 | F(2, 54) = 5.79 (.005, .047) | −52.8 ± 16.4 [−86.3, −23.1], 21/28 < 0 | .002 / .015 | −30.3 ± 14.6 [−59.3, −2.9], 18/28 < 0 | .090 / .405 | −22.5 ± 15.6 [−56.0, +3.7], 16/28 < 0 | .150 / .385 |
| V2 | 28 | F(2, 54) = 2.10 (.132, .538) | −20.6 ± 9.4 [−39.1, −3.0], 20/28 < 0 | .010 / .031 | −9.2 ± 12.3 [−30.9, +16.5], 19/28 < 0 | .138 / .413 | −11.4 ± 8.1 [−27.3, +3.8], 19/28 < 0 | .171 / .385 |
| V3 | 28 | F(2, 54) = 0.70 (.503, .754) | −7.0 ± 10.5 [−27.3, +13.3], 19/28 < 0 | .327 / .491 | −9.1 ± 5.3 [−19.7, +0.7], 17/28 < 0 | .210 / .472 | +2.1 ± 7.6 [−11.9, +17.0], 15/28 < 0 | .884 / .884 |
| V3AB | 25 | F(2, 48) = 0.26 (.769, .811) | −9.3 ± 13.5 [−35.6, +15.6], 16/25 < 0 | .275 / .491 | −6.4 ± 16.2 [−37.5, +24.6], 13/25 < 0 | .771 / .868 | −2.9 ± 8.5 [−18.7, +13.9], 14/25 < 0 | .339 / .510 |
| IPS | 13 | F(2, 24) = 1.60 (.222, .538) | −11.7 ± 17.6 [−47.9, +18.9], 8/13 < 0 | .635 / .817 | −35.7 ± 26.7 [−92.4, +4.4], 7/13 < 0 | .273 / .492 | +24.0 ± 14.7 [−1.3, +53.8], 4/13 < 0 | .168 / .385 |
| hV4 | 28 | F(2, 54) = 1.18 (.314, .566) | −12.4 ± 11.2 [−35.9, +7.0], 16/28 < 0 | .264 / .491 | −13.4 ± 7.1 [−27.9, −0.2], 18/28 < 0 | .086 / .405 | +1.0 ± 10.3 [−21.1, +18.2], 12/28 < 0 | .412 / .529 |
| VO | 17 | F(2, 32) = 0.21 (.811, .811) | +16.1 ± 31.5 [−34.8, +83.7], 10/17 < 0 | .747 / .840 | +11.8 ± 14.8 [−13.9, +41.5], 10/17 < 0 | .890 / .890 | +4.3 ± 27.7 [−47.8, +58.2], 10/17 < 0 | .782 / .880 |
| LO | 26 | F(2, 50) = 0.21 (.809, .811) | +1.7 ± 9.3 [−16.1, +19.9], 14/26 < 0 | .842 / .842 | +5.0 ± 6.2 [−6.4, +17.5], 12/26 < 0 | .565 / .727 | −3.3 ± 7.4 [−16.7, +11.7], 17/26 < 0 | .340 / .510 |
| TO | 21 | F(2, 40) = 1.48 (.239, .538) | −111.5 ± 87.4 [−304.2, +17.9], 16/21 < 0 | .010 / .031 | −55.4 ± 63.9 [−197.2, +37.7], 11/21 < 0 | .495 / .727 | −56.1 ± 28.8 [−115.7, −6.2], 15/21 < 0 | .042 / .379 |

- HP − opposite, q < .05: V1, V2, TO.
- Next − opposite, q < .05: none.
- HP − next, q < .05: none.
- Location type ANOVA, q < .05: V1.
