# Key numbers (2026-11-10, final fits)

Take every number in the text from here. n = 30 unless stated. Tests: Wilcoxon signed-rank across subjects
(two-sided unless stated); BH = Benjamini-Hochberg across the nine areas. Paths are relative to
`/Users/gdehol/git/retsupp/`.

**The final fits** (every number below unless marked otherwise): DoG pRFs with one HRF per subject × area, fitted
jointly with the pRFs (`derivatives/paper_2026-11/roi_hrf_dog_final2.tsv`), voxels reliable across runs, gain-field
models fitted on all runs (`derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf`). Robustness
rows use the same recipe with Gaussian or DN pRFs, each with its own jointly fitted HRF.

Status: **F** final · **R** robustness (supplement) · **X** exploratory (supplement or text at most, no claim).

## Fig. 1 Behaviour (F; `notes/data/behaviour_rt_2026-11.tsv`, legend `fig1_paradigm_behaviour.md`)

Unchanged from the 2026-11-09 package (behaviour does not depend on any fit). Correct trials, 21,480 in total.

| Effect | Value | Subjects in that direction | p |
|---|---|---|---|
| Distractor at HP vs LP (HP benefit) | 877 vs 905 ms, −27.8 ms (median 26.1) | 30/30 | 1.9 × 10⁻⁹ |
| Distractor at LP vs absent (capture cost) | 905 vs 862 ms, +43.0 ms | 28/30 | 2.6 × 10⁻⁸ |
| Distractor at HP vs absent | 877 vs 862 ms, +15.2 ms | 21/30 | .001 |
| Target at HP vs LP (distractor trials) | 908 vs 880 ms, +28.4 ms | 26/30 | 3.9 × 10⁻⁷ |
| Target repeats its location (repeat − switch) | median −38 ms | 28/30 | < .001 |
| Distractor repeats (repeat − switch, within HP and LP, averaged) | median −15 ms | 21/30 | .003 |

Without the two pilots (sub-01, sub-02): −26.6 ms (p = 7.5 × 10⁻⁹); target at HP +27.3 ms (p = 1.5 × 10⁻⁶).
Design: 8 items on a ring at 4°, target and distractor only at the four diagonal positions; no distractor on 20% of
trials; bar aperture 3.17°. `[TODO: distractor-at-HP probability; printed in Fig. 1a but not in its legend]`.

## Fig. 2 pRF fits (F; `fig2_prf_fits_final2.md`)

- d, held-out R² gained per subject (median over the nine areas; three folds of runs, reliable voxels) over a Gaussian
  pRF with the canonical HRF (`notes/data/prf_cv_roi_summary.tsv` + `notes/data/hrf_comparison_jointhrf_final2_2026-11.tsv`):
  DoG pRF +.0035 (30/30, p = 1.9 × 10⁻⁹); the jointly fitted HRF per subject × area on top of the DoG +.0107 (29/30,
  p = 1.3 × 10⁻⁸); both +.0189 (30/30, p = 1.9 × 10⁻⁹). (Changed from the 09 package, which used a median-of-voxels HRF:
  +.011 and +.017.)
- e, search-display response by distance of the pRF centre from the nearest item: from the final pRFs
  (`notes/data/event_deconv_nearest_final2_2026-11/`); qualitative, no test.
- V1 pRF size flat with eccentricity (≈ .55–.6°), likely a resolution floor of the paradigm; V2 .60 → .75°, V3 .61 → .91°.
- Example subject: sub-23. Contralateral share per area (b): `[TODO: compute from the Fig. 2b table]`.

**R, pRF models with jointly fitted HRF** (`notes/data/prf_jointhrf_summary_final2_2026-11.tsv`; held-out R², reliable
voxels, median over the nine areas per subject, median over subjects): Gaussian .178, DoG .182, DN .191. DN > DoG by
.0063 (27/30, p = 1.2 × 10⁻⁶); DoG > Gaussian by .0033 (28/30, p = 1.9 × 10⁻⁸); joint HRF over canonical HRF (DoG)
+.0114 (26/30, p = 1.6 × 10⁻⁷).

## Fig. 3 Responses at the item locations (F; `fig3_item_locations_final2.md`)

Voxels with a final pRF centre within 2° of an item (`notes/data/event_deconv*_final2_2026-11/`); the bar's predicted
response removed with the final pRFs and HRF; window 3.2–6.4 s.

- Target − neutral (b): early +0.12%, dorsal +0.26%, occipitotemporal +0.23% signal, all p < .001 (BH over pathways).
- Distractor − neutral at LP (c): early +0.07%, dorsal +0.15%, occipitotemporal +0.15% signal, all p < .001.
- Location type, early areas (e, items pooled, % of the subject's mean response): HP − next −6.1 ± 3.6% (p = .135);
  next − opposite −13.0 ± 5.8% (p = .047); HP − opposite −19.0% (p = .011); next left − next right +0.6% (p = .63).
  ANOVA location type F(2, 58) = 6.28, p = .003; location type × item F(4, 116) = 0.82, p = .514.
- All areas: location type F(2, 58) = 4.73, p = .012; HP − next −6.4 ± 2.7% (p = .029); next − opposite −6.8 ± 4.5%
  (p = .119); HP − opposite −13.2% (p = .028); × item F(4, 116) = 1.47, p = .217.
- Dorsal (n = 27; sub-30 left out of the relative analyses because its mean dorsal response is not positive) and
  occipitotemporal: no reliable location effect (F(2, 52) = 0.15, p = .86; F(2, 58) = 0.72, p = .49).
- HP vs all three LP locations: early −8.6%, F(1, 29) = 4.65, p = .039; all areas −7.3%, F(1, 29) = 5.43, p = .027;
  location × item early p = .471.
- DCT basis (robustness), early: location type F(2, 58) = 5.10, p = .009; next − opposite −12.0% (p = .052);
  HP − opposite −17.3% (p = .020); HP − next −5.3% (p = .21); × item p = .86.
- Per item type, early, HP − next / next − opposite: target −0.5% / −12.0%; distractor −7.5% / −19.2%; neutral
  −10.2% (p = .011) / −7.7%; all others p ≥ .10: same direction, item types do not differ.
- Nearly identical to the 09 package (earlier fits): next − opposite −14.0% (p = .029), HP − opposite −19.0% (p = .011).

## Fig. 5 The gain fields (F; `fig5_gain_fields_final2.md`, parameter table `table_parameters_2026-11_final2.md`)

Sources: joint fits `notes/data/gain_field_parameters_final2_2026-11.tsv` (m7_full = all terms fitted together);
estimated alone (each term added alone to its base model, the subject's own estimate, all voxels):
`notes/data/conditional_estimates_dog_final2_2026-11.tsv`. "Pooled" = per-subject median over the nine areas.

**a, joint fit, median per area** (* BH q < .05):
- Target gain: V1 −.008, V2 +.076*, V3 +.27*, V3AB +.34*, IPS +.21*, hV4 +.53*, VO +.50*, LO +.15*, TO +.28*.
  Pooled +.18 [+.13, +.27], 90% of subjects > 0, p = 8 × 10⁻⁸.
- Distractor gain (averaged over its locations): V3AB +.12*, hV4 +.22*, TO +.10*; others n.s. (V1 −.03 … VO +.07).
  Pooled +.065 [+.032, +.15], 77% > 0, p = .011.
- Estimated alone, the distractor gain on top of the target term: V3AB +.049 (q = .007), hV4 +.15 (q = .026),
  TO +.053 (q = .034); pooled +.035, p = .007. Alone on the item model (without the target term) it is ≈ 0
  (pooled −.007, p = .10).
- Target gain rises with pRF size: +.32 per doubling [+.23, +.54], 83% of subjects, q < .001
  (`notes/data/hierarchy_regression_final2_2026-11.tsv`); distractor +.11 [−.07, +.26], q = .20 (n.s.). Within
  subjects the target gain falls with the width of the transient field (median Spearman ρ −.64, p = 3 × 10⁻⁶) and does
  not track the item weight (ρ +.07, p = .34).

**b, field widths (FWHM)**: sustained median 3.3° over all subjects and areas (14% of fits at a σ bound); per area
early ≈ 5.0–5.3°, higher areas 2.5–3.4° (TO 2.5°, VO 2.6°, hV4 2.9°, IPS 3.1°, LO 3.2°, V3AB 3.4°). Neighbouring items
are 5.7° apart. Transient field: often at a bound, poorly constrained. (Changed: the 09 package had 4.8°.)

**Sustained gain shared by all four locations (S, estimated alone)**: negative in every area but V1; pooled −.032
[−.036, −.022], 97% of subjects < 0, p = 3.7 × 10⁻⁹; q < .001 in V2–TO. (A no-control term: it is not HP-specific.)

**c, dS = g_HP − g_LP, sustained gain at HP minus at LP, estimated alone**: pooled −.012 [−.017, −.006], 80% of
subjects < 0, p = 5.6 × 10⁻⁵. Per area: V3 −.032 (q = .020), TO −.019 (q = .018); V1 −.007 (q = .14), hV4 −.015
(q = .14), all others q ≥ .22, all medians negative.
Joint fit (m7_full): pooled −.031 [−.057, −.015], 77% < 0, p = .005; V3 −.077 (q = .034), hV4 −.047 (q = .034); V1
−.066, V2 −.058, TO −.023 (q = .14–.20).
Direction across subjects (leave-one-subject-out sign test, `notes/data/loso_per_step_final2_2026-11.tsv`): the other
29 subjects' dS points the way a subject's data move in 77% of subjects, p = .0024.

**d, transient HP − LP, estimated alone**:
- dD (distractor at HP − at LP): pooled −.013 [−.029, +.004], 63% < 0, p = .050; per area only LO −.027 (q = .015).
- dT (target at HP − at LP): pooled −.034 [−.065, +.005], 67% < 0, p = .016; no area q < .05 (V3 −.073, q = .084).
- dD − dT: pooled +.002, p = .37; no area differs (smallest q = .83).
- Joint dD: pooled −.010, p = .47; no area q < .05. (Changed: the 09 package had dD q < .05 in V3AB, IPS, hV4, VO.)

**Split models** (`notes/data/split_models_final2_2026-11.tsv`, full-data fits):
- Sustained LP gain split into next to HP vs opposite HP (m8_full_adjopp, 252 subject × area fits): next − opposite
  pooled +.0001, p = .87; HP − next −.012 (p = .092; V3 −.097, q = .038); HP − opposite −.031 (80% < 0, p = .0087;
  hV4 −.050, q = .018). The gain is lowest at HP; the neighbours do not differ from the opposite location.
- Target gain split by location (m8_full_dT, 232 fits): target HP − LP pooled −.019, p = .46; distractor HP − LP +.027,
  p = .27; dD − dT p = .72 (fitted jointly, neither is reliable).
- One shared width for sustained and transient fields (m7_full_sigtied, 239 fits): dS pooled −.019, 77% < 0, p = .055;
  FWHM 3.9°.

## Supplementary table: what each term adds to out-of-subject prediction (F; `notes/data/loso_per_step_final2_2026-11.tsv`, `loso_exact_final2_2026-11.tsv`)

Leave-one-subject-out (LOSO): each subject's held-out fit with the term at the median of the other 29 subjects' own
estimates, scored on signal voxels (held-out response beats a model-free null); per subject the median over areas.
"Better" = share of subjects whose ΔR² > 0, Wilcoxon p. "Direction" = sign-flip test: ΔR² at +δ minus at −δ, which
removes the penalty of adding a term and is a true 50% null. Exact (ΔSSE curves) and linear versions agree.

| Term | Better | p | Direction | p | Median δ (others) |
|---|---|---|---|---|---|
| Items (weight on bar-only pRFs, vs items at bar strength) | 100% | < .001 | — | — | −0.77 |
| Target gain (T) | 77% | .0006 | 97% | < .001 | +0.23 |
| Distractor gain, tied (D) | 77% | .031 | 80% | .001 | −0.015 |
| Distractor gain on top of target (D given T) | 63% | .11 | 80% | < .001 | +0.009 |
| Sustained gain, tied (S) | 87% | < .001 | 100% | < .001 | −0.027 |
| dS (sustained HP − LP) | 67% | .18 | 77% | .0024 | −0.007 |
| dD (distractor HP − LP) | 57% | 1.0 | 70% | .10 | −0.012 |
| dT (target HP − LP) | 63% | .24 | 70% | .019 | −0.049 |

Items row, read with care: the step perturbs the item weight of pRFs fitted to the bar alone, whose pRFs never saw the
items; the others' weight is close to "no items" in many areas (own weight at the grid edge in 47% of subject × area
pairs), so it is not a test of items vs none. The fitted item weight with the pRFs refitted (m0/m7) is the relevant
number: item_scale median 2.85 (m0, IQR 1.77–4.57) and 3.08 (m7) relative to the bar. Per area: `figS_loso_per_roi_final2.md`.
(Changed: the 09 package had dS 77%, p = .010 on the earlier fits.)

## DoG centre/surround tilt κ (X; `notes/data/dog_cs_field_full_2026-11.tsv` on the cluster clone, `notes/data/loso_kappa_dnw_2026-11.tsv`)

κ splits the sustained field between the DoG centre and surround (κ < 0: relatively more of the HP effect on the
surround). Full-data fits: κ pooled median −0.040, p = .0035; per area negative, q < .02 in V1 (−0.077), V2, V3, hV4,
LO; dS unchanged by adding κ (p = .63). LOSO: better in 70% of subjects (p = .055); direction 90% (p = 4.7 × 10⁻⁷),
q < .05 in V1–TO; median δ −0.009. Held-out runs (run3, pRFs frozen): worse than without κ in every area (pooled ΔR²
≈ −1 × 10⁻⁴, p = 7 × 10⁻⁹). Consistent in direction, too small to help prediction.

## DN: where does the HP effect act? (X; `figS_dn_gain_locus_complete.md`, `notes/dn_gain_field_variants_2026-11.md`)

- A sustained field on the DN neural baseline (b), surround baseline (d) or both, instead of the input gain:
  held-out runs (run3, pRFs frozen; complete set, `notes/data/dn_basefield_cv_complete_2026-11.tsv`) worse than the
  input-gain model in 30/30 subjects for each (p = 1.9 × 10⁻⁹; mean ΔR² −3.5 × 10⁻⁵ (b), −3.4 × 10⁻⁵ (d), −1.1 × 10⁻⁴ (both)).
- Fitted together, b and d move together (Gauss-Newton correlation 0.999), i.e. an amplitude field that trades off
  with the input gain; the DN-specific b/d direction is not used.
- LOSO: common b,d weight 43% better (p = .18), direction 50% (p = .89); b/d ratio 53% (p = .45), direction 73%
  (p = .025; TO, VO, IPS; ΔR² ≈ 10⁻⁷).
- Recovery: at our effect size a b or d effect is read as input gain in 72% of simulations.
- Conclusion: the data cannot place the HP effect on the DN baselines rather than on the input gain.

## Fig. 6 Decoding (F; `fig6_decoding.md`; pRFs = final DoG joint-HRF fold fits)

Unchanged from the 09 package (it already used the final joint-HRF pRFs).
- Frame-wise r, decoded vs true (median over subjects; shuffled ≈ 0, all p < 10⁻³ one-sided): V1 .51, V2 .47, V3 .40,
  V3AB .29, IPS .12, hV4 .28, VO .16, LO .25, TO .19. n per area: V1–V3 30, V3AB 25, IPS 19, hV4 29, VO 11, LO 29, TO 27.
- Example run: sub-03 ses-1 run-4, V1, r = .62 (rank 1 of 2653); median run r = .30.
- d, HP quadrant − other three, one-sided, BH over nine areas: V2 p = .013, V3 .027, VO .034, TO .031, V3AB .043, all
  q = .076–.077; V1 p = .32; none q < .05.
- e, V1–V3 pooled, % of the four-quadrant mean (mean ± within-subject SEM): HP −0.85 ± 0.38%, next to HP +0.46 ± 0.18%,
  opposite −0.08 ± 0.22%. HP < next one-sided p = .015; HP < opposite p = .13; Friedman p = .06; left vs right p = .22.
  Next vs opposite (two-sided) p = .16.
- f, by eccentricity (V1–V3): 0–1.5° −0.6% (p = .10); 1.5–2.5° −1.4% (p = .029); 2.5°+ −1.3% (p = .040); whole aperture
  −1.1 ± 0.5% (p = .033); bands do not differ (Friedman p = .53).
- c, insets (outline decoder): bar segment in the quadrant facing HP −1.16 ± 0.42% (p = .008); next to HP +1.13 ± 0.33%
  (p = .002).

**Supplements (F).** Canonical HRF (V1–V3): V2, V3 q = .034; HP < next p = .008. Decoded bars, same stimulus: HP half
minus other half, vertical −1.61 ± 0.66% (p = .031), horizontal −1.83 ± 0.51% (p = .001), pooled −1.72% (p = .004);
per area only V3 survives BH (−2.71%, q = .011). Bar-crossing response (V1–V3): HP − LP −2.6%, one-sided p = .017;
with a pre-crossing baseline −2.1%, p = .16 (n.s.).

**Decoder simulation (F; `figS_decode_simulation.md`)**: an HP-only gain field is recovered at about 85% of its true
quadrant pattern; −0.025 gain gives a decoded HP − next of −0.88%, the same order as observed (HP − next −1.31%); an HP-only field puts
the neighbours *below* opposite (−0.24%), unlike the data (next +0.46%, opposite −0.08%, n.s.).

**R, decoding with Gaussian and DN pRFs (joint HRF)** (`notes/data/decoding_prf_models_*_2026-11_final2.tsv`,
`figS_decoding_prf_models_final2.md`), V1–V3 pooled, HP vs next (two-sided): DoG −0.85 vs +0.46% (p = .031);
Gaussian −0.85 vs +0.46% (p = .036); DN −0.65 vs +0.36% (p = .10; outline decoder p = .011). Next vs opposite never
reliable (p = .13–.49). Decoding quality V1 r: DoG .51, Gaussian .49, DN .45.

## Fig. 7 Gains and behaviour (F; `fig7_compact_v2.md`; data `notes/data/trialwise_meta_bayes_2026-11/`)

Per subject × area, the trial-wise gains modulated by trial history and by z(log RT), estimated directly on the BOLD
(final m7 fits, each area's signal voxels), each estimate with a leave-one-run-out jackknife SE; hierarchical
Bayesian model over subjects and areas (group means fixed, areas and subjects random), weighting each estimate by its
SE. Posterior mean [95% HDI], P = posterior probability of the sign. n = 30 (247 subject × area estimates).
- a, target gain on target-repeat trials, % of the area's typical target gain: all nine areas −19% [−33, −7], P = .999;
  early −29% [−57, +0], P = .978; dorsal −19% [−40, +2], P = .961; occipitotemporal −13% [−27, +3], P = .956.
  In absolute gain −0.053 [−0.091, −0.019], P = .998. Without the SE weighting −42%, P = .978.
- a, distractor gain on distractor-repeat trials: −23% [−67, +16], P = .873 (no credible change).
- b, change per SD of log RT (higher = higher gain on slower trials): distractor gain −0.023 [−0.040, −0.004], P = .993;
  global gain −0.030 [−0.060, −0.002], P = .979; target gain +0.009 [−0.012, +0.028], P = .809.
  Without the SE weighting the RT effects vanish (distractor P = .678, global P = .624).
- Convergence: 3 divergences over 27 fits, max R̂ 1.01, min bulk ESS 882; posterior predictive intervals cover 95–99%.
- The October-fit version of Fig. 7 (RT regressed on per-trial gains) is retired. On the final fits that analysis gave
  only isolated effects (`fig7_behaviour_final2.md`).
- Between subjects (text only, earlier fits): dS vs behavioural HP benefit ρ = −.17, p = .37.

## Robustness: the gain fields with Gaussian and DN pRFs (R; joint fits from `gain_field_parameters_final2_2026-11.tsv`)

Joint fit (m7_full), pooled per-subject median over areas:

| Term | DoG | Gaussian | DN |
|---|---|---|---|
| Target gain | +.18, p = 8 × 10⁻⁸ | +.45, p = 1 × 10⁻⁷ | +.51, p = 1.9 × 10⁻⁹ |
| Distractor gain | +.065, p = .011 | +.095, p = .020 | +.15, p = .00095 |
| dS (joint) | −.031, 77% < 0, p = .005 | −.063, 63% < 0, p = .078 | −.074, 67% < 0, p = .0076 |
| dD (joint) | −.010, p = .47 | +.043, p = .39 | +.009, p = .39 |

Per area (joint): target q < .05 beyond V1 under DoG; Gaussian beyond V3 (V1 negative, q = .013); DN in V2–TO.
dS joint q < .05: DoG V3, hV4; Gaussian none (V2 p = .018); DN none (V3, hV4 q = .062).

Estimated alone (exact ΔSSE minimum, areas with ≥ 20 voxels; `notes/data/conditional_exact_final2_allfam_2026-11.tsv`),
pooled per-subject median over areas, and the leave-one-subject-out tests (`notes/data/loso_per_step_allfam_final2_2026-11.tsv`,
`tableS_loso_final2.md`):

| Term | DoG | Gaussian | DN |
|---|---|---|---|
| dS alone | −.012, 80% < 0, p = 5.6 × 10⁻⁵ (V3, TO q < .05) | −.005, 83% < 0, p = 2.3 × 10⁻⁴ (hV4 q = .041) | −.019, 77% < 0, p = .0032 (hV4 q = .029) |
| dS direction (sign-flip, LOSO) | 77%, p = .0024 | 70%, p = .011 | 70%, p = .045 |
| dS better out of subject | 67%, p = .18 | 57%, p = .64 | 60%, p = .27 |
| dD alone | −.013, p = .050 | −.010, p = .44 | −.025, p = .13 |
| dT alone | −.034, p = .016 | −.054, p = .021 | −.040, p = .096 |
| Target gain alone | +.16, p = 1.9 × 10⁻⁹ | +.41, p = 2.6 × 10⁻⁸ | +.35, p = 1.9 × 10⁻⁹ |
| Distractor on top of target alone | +.035, p = .007 (V3AB, hV4, TO) | +.034, p = .011 (V3AB, hV4, TO) | +.041, p = .052 (V3AB, hV4, IPS) |
| Sustained gain S alone | −.032, 97% < 0 | −.042, 100% < 0 | −.020, 97% < 0 |
| Target gain better out of subject | 77%, p = .0006 | 90%, p = 6 × 10⁻⁷ | 87%, p = 1.4 × 10⁻⁵ |

Conclusion: under all three pRF models the sustained gain is lower at HP than at LP when estimated alone (77–83% of
subjects), and the direction generalises across subjects; in no model does dS improve out-of-subject prediction. The
DoG alone values come from the linear sufficient statistics (`conditional_estimates_dog_final2_2026-11.tsv`); the
exact-curve DoG values agree. Figures: `figS_gain_fields_gauss_final2`, `figS_gain_fields_dn_final2`.

Not run with Gaussian or DN pRFs: Fig. 3 (voxel selection by pRF centre) and Fig. 7 (trial-wise gains).
