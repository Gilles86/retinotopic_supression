**Supplementary Table S. Responses at the item locations (Fig. 3): statistics per item type, per pathway, against all LP locations, and with a DCT basis.** Same data, voxels and window (3.2–6.4 s) as Fig. 3; values in % of each subject's mean response over the four locations and three item types in that pathway (mean ± SEM over subjects where given). Wilcoxon signed-rank tests, uncorrected; repeated-measures ANOVAs. n = 28 (dorsal: n = 26 in the absolute analyses, two subjects lacking dorsal voxels at some item location; n = 25 in the relative analyses, where sub-30 is also left out because its mean dorsal response is not positive). Source: `fig3_item_locations.py` run at n = 28 (`retsupp/visualize/paper/run_excluding.py --exclude 1 2`; the full generated legend is kept as `notes/data/n28/fig3_item_locations_n28_long_legend.md`).

**1. Location type × item type per pathway (Fourier basis).** Repeated-measures ANOVA location type (HP, next to HP, opposite HP) × item type (target, distractor, neutral), with the planned contrasts on the mean over item types.

| Pathway | Location type | Location type × item | HP − next | Next − opposite | HP − opposite | Next left − next right |
|---|---|---|---|---|---|---|
| Early | F(2, 54) = 7.67, p = .001 | F(4, 108) = 0.92, p = .454 | −6.7 ± 3.6%, p = .095 | −14.9 ± 6.0%, p = .022 | −21.6%, p = .005 | +0.3%, p = .582 |
| Dorsal | F(2, 48) = 0.16, p = .857 | F(4, 96) = 0.86, p = .490 | −2.8 ± 8.5%, p = .411 | −3.9 ± 14.7%, p = .634 | −6.7%, p = .241 | +3.1%, p = .312 |
| Occipitotemporal | F(2, 54) = 1.49, p = .235 | F(4, 108) = 1.33, p = .262 | −5.9 ± 6.3%, p = .284 | −3.9 ± 3.8%, p = .210 | −9.8%, p = .090 | −6.1%, p = .327 |
| All areas | F(2, 54) = 6.31, p = .003 | F(4, 108) = 1.63, p = .172 | −6.9 ± 2.7%, p = .018 | −8.6 ± 4.6%, p = .060 | −15.5%, p = .013 | −0.8%, p = .973 |

**2. Per item type: HP − next to HP and next − opposite (Fourier basis).**

| Pathway | Target: HP − next | Target: next − opposite | Distractor: HP − next | Distractor: next − opposite | Neutral: HP − next | Neutral: next − opposite |
|---|---|---|---|---|---|---|
| Early | −2.2%, p = .552 | −12.1%, p = .131 | −6.6%, p = .284 | −23.3%, p = .090 | −11.3%, p = .006 | −9.3%, p = .218 |
| Dorsal | −5.4%, p = .156 | −0.8%, p = .672 | −0.8%, p = .367 | −14.0%, p = .812 | −2.1%, p = .853 | +3.1%, p = .491 |
| Occipitotemporal | −4.4%, p = .537 | −5.1%, p = .086 | −8.6%, p = .479 | −8.7%, p = .144 | −4.6%, p = .295 | +2.1%, p = .814 |
| All areas | −5.5%, p = .274 | −7.8%, p = .202 | −7.2%, p = .109 | −16.8%, p = .077 | −8.0%, p = .014 | −1.4%, p = .614 |

**3. HP against all three LP locations** (deconvolution with two event types per item; ANOVA location (HP, LP) × item type).

| Pathway | Main effect of location | Location × item |
|---|---|---|
| Early | −9.8%, F(1, 27) = 5.72, p = .024 | F(2, 54) = 0.55, p = .579 |
| Dorsal | −21.4%, F(1, 25) = 0.76, p = .393 | F(2, 50) = 0.83, p = .442 |
| Occipitotemporal | −4.2%, F(1, 27) = 0.53, p = .475 | F(2, 54) = 0.67, p = .517 |
| All areas | −8.4%, F(1, 27) = 6.82, p = .015 | F(2, 54) = 0.22, p = .801 |

**4. DCT basis (robustness).**

Fig. 3b, c and HP against LP: target − neutral p < .001 in all three pathways; distractor − neutral p < .001 in all three pathways.

| Pathway | HP − LP | Main effect of location | Location × item |
|---|---|---|---|
| Early | −12%, p = .034 | p = .065 | p = .828 |
| Dorsal | −9%, p = .059 | p = .396 | p = .447 |
| Occipitotemporal | −4%, p = .210 | p = .516 | p = .506 |
| All areas | — | p = .026 | p = .921 |

Location types, DCT basis:

| Pathway | Location type | Location type × item | HP − next | Next − opposite | HP − opposite | Next left − next right |
|---|---|---|---|---|---|---|
| Early | F(2, 54) = 6.35, p = .003 | F(4, 108) = 0.35, p = .841 | −5.9 ± 3.8%, p = .164 | −14.0 ± 5.9%, p = .023 | −19.9%, p = .010 | +0.1%, p = .814 |
| Dorsal | F(2, 48) = 0.34, p = .713 | F(4, 96) = 0.47, p = .758 | −3.3 ± 8.0%, p = .252 | −5.7 ± 12.7%, p = .578 | −9.0%, p = .210 | +5.8%, p = .312 |
| Occipitotemporal | F(2, 54) = 1.25, p = .294 | F(4, 108) = 1.03, p = .396 | −6.9 ± 7.0%, p = .210 | −2.6 ± 4.1%, p = .339 | −9.5%, p = .157 | −5.4%, p = .386 |
| All areas | F(2, 54) = 5.40, p = .007 | F(4, 108) = 0.72, p = .579 | −6.7 ± 2.8%, p = .028 | −8.3 ± 4.8%, p = .090 | −15.0%, p = .018 | −0.2%, p = .991 |
