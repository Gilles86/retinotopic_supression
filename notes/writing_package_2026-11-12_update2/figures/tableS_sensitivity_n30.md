**Supplementary Table S. Every main result with and without the two pilot participants.** The main analyses leave out sub-01 and sub-02, tested with a pilot version of the paradigm (mapping bar 0.5°/s instead of 0.168°/s, so a sweep took ≈ 14 s instead of ≈ 42 s; the high-probability location at only three of the four positions, in blocks of 2, 3 and 1 runs per session, in the same order in both sessions; the search task itself identical). Per-subject fits do not depend on other subjects; only the group level was recomputed (Wilcoxon tests, Benjamini-Hochberg across areas, bootstrapped CIs, the Fig. 3 ANOVAs, the hierarchy regression, the leave-one-subject-out δ with the pilots also left out of the pool, the Fig. 7 hierarchical model). No result changes sign. Without the pilots the effects at the HP location are stronger (Figs. 3, 5c, 6); the tied distractor gain no longer improves out-of-subject prediction, and the Fig. 7 global-gain RT effect is no longer credible. Sources: `notes/data/sensitivity_excl0102_2026-11.tsv` (`retsupp/visualize/paper/sensitivity_excl_subjects.py`), `notes/data/loso_exact_final2_excl0102_2026-11.tsv`, `notes/data/trialwise_meta_bayes_excl0102_2026-11/`, `notes/sensitivity_excl_sub01_02_2026-11-11.md`. Panel letters as in the 2026-11-11 figures (Fig. 1 c–f; Fig. 6 d = HP vs next, e = eccentricity); in the n = 28 figures Fig. 1e merges the two repeat effects.

Bold = a change at the figure's own threshold.

### Fig. 1 Behaviour (median within-subject difference, ms [95% CI], Wilcoxon)

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| Distractor HP − LP | −25.0 [−32.4, −20.0], 28/28, p = 7.5e-9 | −26.1 [−34.4, −20.9], 30/30, p = 1.9e-9 | no |
| Distractor LP − absent | +42.6, 26/28, p = 1.0e-7 | +43.7, 28/30, p = 2.6e-8 | no |
| Distractor HP − absent | +14.6 [+0.3, +21.7], 19/28, p = .0041 | +15.8 [+2.1, +23.2], 21/30, p = .0012 | no |
| Target HP − LP | +24.3, 24/28, p = 1.5e-6 | +26.8, 26/30, p = 3.9e-7 | no |
| Target repeat − switch | −38.7, 26/28, p = 3.7e-8 | −38.4, 28/30, p = 9e-9 | no |
| Distractor repeat − switch | −15.1, 20/28, p = .0013 | −15.1, 21/30, p = .0035 | no |

### Fig. 2d pRF models (held-out R² gained, median [CI])

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| DoG over Gaussian | +.0033, 28/28 | +.0035, 30/30 | no |
| Joint HRF on top of DoG | +.0106, 27/28 | +.0107, 29/30 | no |
| Both | +.0179, 28/28 | +.0189, 30/30 | no |

### Fig. 3 Item locations

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| Target − neutral (early / dorsal / occ.-temp., % signal) | +0.125 / +0.264 / +0.236, all p ≤ 3e-8 | +0.125 / +0.258 / +0.229, all p ≤ 7e-9 | no |
| Distractor − neutral (LP) | +0.072 / +0.149 / +0.156, p ≤ 1e-4 | +0.072 / +0.147 / +0.150, p ≤ 4e-5 | no |
| Early: location-type ANOVA | F(2,54) = 7.67, p = .0012 | F(2,58) = 6.28, p = .003 | stronger |
| Early: next − opposite | −14.9%, p = .022 | −13.0%, p = .047 | stronger |
| Early: HP − opposite | −21.6%, p = .0052 | −19.0%, p = .011 | stronger |
| Early: HP − next | −6.7%, p = .095 | −6.1%, p = .135 | n.s. both |
| Early: location × item | p = .454 | p = .514 | no |
| All areas: ANOVA | F = 6.31, p = .0035 | F = 4.73, p = .012 | stronger |
| All areas: HP − next / next − opp / HP − opp | −6.9% (.018) / −8.6% (.060) / −15.5% (.013) | −6.4% (.029) / −6.8% (.12) / −13.2% (.028) | stronger, next − opp still n.s. |
| HP vs all LP (early / all areas) | −9.8%, p = .024 / −8.4%, p = .015 | −8.6%, p = .039 / −7.3%, p = .027 | stronger |
| Dorsal, occ.-temp. ANOVA | p = .86, .23 | p = .86, .49 | no |

### Fig. 5 Gain fields (pooled = per-subject median over areas; Wilcoxon; per area BH q < .05)

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| Target gain, joint | +.16, 89%, p = 3e-7; 6 areas (V2, LO drop out) | +.18, 90%, p = 8e-8; 8 areas q < .05 (V2–TO) | **fewer areas** (V2 +.076, LO +.15 were the weakest) |
| Target gain per doubling of pRF size | +.33, q = .0017 | +.32, q = .0006 | no |
| Distractor gain, joint | +.065, 75%, p = .023; V3AB, hV4, TO | +.065, 77%, p = .011; V3AB, hV4, TO | no (weaker p) |
| Distractor on top of target, alone | +.031, p = .019; V3AB, hV4 (TO drops) | +.035, p = .0066; V3AB, hV4, TO | **weaker, one area fewer** |
| Sustained S alone | −.034, 100%, p = 7.5e-9; V2–TO | −.032, 97%, p = 3.7e-9; V2–TO | no |
| **dS alone** | −.014, 79%, p = 1.1e-4; V3 (.046), TO (.035) | −.012, 80%, p = 5.6e-5; V3 (q = .020), TO (.018) | no |
| dS joint (m7) | −.034, 79%, p = .0027; V3, hV4, TO | −.031, 77%, p = .0050; V3, hV4 | stronger (+TO) |
| dD alone | −.010, p = .099; LO | −.013, p = .050; LO | **drops above .05** (was borderline) |
| dT alone | −.034, p = .018 | −.034, p = .016 | no |
| dD − dT alone | +.009, p = .24 | +.002, p = .37 | no |
| dD joint | +.004, p = .29 | −.010, p = .47 | no |
| Split: next − opposite | −.006, p = .73 | +.0001, p = .87 | no |
| Split: HP − next | −.016, p = .070 | −.012, p = .092 | no |
| Split: HP − opposite | −.033, 82%, p = .0048 | −.031, 80%, p = .0087 | stronger |
| Split: target HP − LP | +.047, p = .37 | −.019, p = .46 | no (n.s.) |
| One shared width: dS | −.023, 79%, p = .036 | −.019, 77%, p = .055 | **crosses .05** |

(The hierarchy slope reproduces key_numbers only with `notes/data/fig1_prf_sigma.tsv`, the model-4 size table the
final2 regression used; with the final-fit sizes the n = 30 target slope is +.39. Both n = 28 values use the same table
as their n = 30 counterpart.)

### Supplementary table: LOSO (DoG; ΔR² better / sign-flip direction)

| Term | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| Target gain | 75%, p = .0007 / 96%, p < 1e-4 | 77%, p = .0006 / 97%, p < 1e-4 | no |
| **Distractor gain (tied)** | 61%, p = .16 / 75%, p = .0034 | 77%, p = .031 / 80%, p = .0010 | **better-out-of-subject no longer significant** |
| Distractor on top of target | 61%, p = .33 / 75%, p = .0005 | 63%, p = .11 / 80%, p < 1e-4 | no |
| Sustained S | 93% / 100%, p < 1e-4 | 87% / 100%, p < 1e-4 | no |
| dS | 68%, p = .41 / 75%, p = .0071 | 67%, p = .18 / 77%, p = .0024 | no |
| dD | 43%, p = .25 / 50%, p = .96 | 57%, p = 1.0 / 70%, p = .10 | no (direction gone) |
| dT | 61%, p = .25 / 71%, p = .014 | 63%, p = .24 / 70%, p = .019 | no |

### Fig. 6 Decoding (V1–V3 pooled, free decoder, % of the four-quadrant mean)

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| d: HP / next / opposite | −1.02 / +0.57 / −0.12% | −0.85 / +0.46 / −0.08% | larger |
| d: HP < next (one-sided) | p = .0038 | p = .015 | stronger |
| d: HP < opposite | p = .082 | p = .13 | n.s. both |
| d: Friedman, four quadrants | **p = .013** | p = .062 | **becomes significant** |
| d: next vs opposite (two-sided) | p = .070 | p = .16 | n.s. both |
| e: whole aperture / 1.5–2.5° / 2.5°+ | −1.32% (.010) / −1.64% (.0096) / −1.52% (.020) | −1.10% (.033) / −1.37% (.029) / −1.30% (.040) | stronger |
| e: 0–1.5° | −0.73%, p = .069 | −0.57%, p = .10 | n.s. both |
| Suppl. per area, BH q < .05 | **V2, V3 (q = .038)** | none (V2, V3, V3AB, VO, TO q = .076–.077) | **two areas survive** |
| b: frame-wise r, V1 (median) | .52 | .51 | no |
| Decoded bars, HP half − other half (vertical / horizontal / pooled) | −1.71 (.034) / −1.88 (.002) / −1.80% (.006) | −1.61 (.031) / −1.83 (.001) / −1.72% (.004) | no |
| Decoded bars, next − opposite half, pooled | +1.21%, p = .019 | +1.04%, p = .047 | stronger |
| Gaussian pRFs: HP vs next (two-sided) | p = .0083 | p = .036 | stronger |
| DN pRFs: HP vs next (two-sided, free / outline) | **p = .036** / .0081 | p = .105 / .011 | **free decoder becomes significant** |
| c, insets (outline decoder): HP segment / next-to-HP segment | −1.33 ± 0.49% (p = .016) / +1.29 ± 0.37% (p = .002) | −1.16 ± 0.42% (p = .008) / +1.13 ± 0.33% (p = .002) | similar |

### Fig. 7 Gains and behaviour (hierarchical model; posterior mean [95% HDI], P of the sign)

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| Target gain on target repeat, all areas (% of typical gain) | −19% [−33, −4], P = .997 | −19% [−33, −7], P = .999 | no |
| … early / dorsal / occ.-temp. | −26% (.962) / −22% (.964) / −13% (.945) | −29% (.978) / −19% (.961) / −13% (.956) | **early falls below .975** |
| … absolute units | −0.052, P = .993 | −0.053, P = .998 | no |
| … without SE weighting | P = .968 | P = .978 | slightly below .975 |
| Distractor gain on distractor repeat | −30%, P = .92 | −23%, P = .87 | no (n.s.) |
| Distractor gain per SD log RT | −0.024 [−0.043, −0.005], P = .990 | −0.023 [−0.040, −0.004], P = .993 | no |
| **Global gain per SD log RT** | −0.021 [−0.046, +0.006], P = .943 | −0.030 [−0.060, −0.002], P = .979 | **no longer credible** |
| Target gain per SD log RT | +0.009, P = .80 | +0.009, P = .81 | no |

### Robustness: Gaussian and DN pRFs (pooled, Wilcoxon)

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| dS alone, exact (DoG / Gauss / DN) | p = 9e-5 / 4.2e-4 / .0052 | p = 3e-5 / 2.3e-4 / .0032 | no |
| dS joint (Gauss / DN) | −.088, **p = .027** / −.078, p = .0024 | −.063, p = .078 / −.074, p = .0076 | **Gaussian crosses .05** |
| dS LOSO direction (Gauss / DN) | 79%, p = .0044 / 71%, p = .038 | 70%, p = .011 / 70%, p = .045 | no |
| dS LOSO better (Gauss / DN) | 61%, p = .54 / 64%, p = .24 | 57%, p = .64 / 60%, p = .27 | no |
| Distractor on top of target alone (Gauss / DN) | p = .032 / .12 | p = .011 / .052 | weaker |
| DN distractor gain (tied) LOSO | **18% better, p = .018 (worse out of subject)** | 57% better, p = .56 | **flips to worse** |
| Target gain (all models, joint and alone) | all p ≤ 4e-7 | all p ≤ 1e-7 | no |

### DoG centre/surround tilt κ (supplementary figure)

| Result | n = 28 (main) | n = 30 (with pilots) | Change without the pilots |
| --- | --- | --- | --- |
| κ, full-data fits | −0.036 [−0.082, −0.019] (mean over areas), 82% < 0, p = .011 | median −0.040, p = .0035 | no |
| κ LOSO better / direction | 68%, p = .043 / 93%, p = 2.5e-7 | 70%, p = .055 / 90%, p = 4.7e-7 | no |
| DN baseline field, common b,d weight, LOSO better / direction | 46%, p = .24 / 50%, p = .60 | 43%, p = .18 / 50%, p = .89 | no |
| DN b/d ratio, LOSO better / direction | 57%, p = .27 / 75%, p = .010 | 53%, p = .45 / 73%, p = .025 | no |
| DN baseline fields, held-out runs vs input gain | worse in 28/28 | worse in 30/30 | no |

