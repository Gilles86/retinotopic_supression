# Key numbers (2026-11-12, final fits, n = 28)

Take every number in the text from here. **n = 28 (sub-03–sub-30) unless stated**: the two pilot participants (sub-01,
sub-02) are left out of every analysis because their paradigm differed (bar 0.5°/s instead of 0.168°/s, so each sweep
took ≈ 14 s instead of ≈ 42 s; HP at only three of the four locations, in blocks of 2, 3 and 1 runs, in the same order
in both sessions; search task identical). The same results with n = 30 are in `figures/tableS_sensitivity_n30.md`.
Tests: Wilcoxon signed-rank across subjects (two-sided unless stated); BH = Benjamini-Hochberg across the nine areas.
Paths are relative to `/Users/gdehol/git/retsupp/`.

**The final fits** (every number below unless marked otherwise): DoG pRFs with one HRF per subject × area, fitted
jointly with the pRFs (`derivatives/paper_2026-11/roi_hrf_dog_final2.tsv`), voxels reliable across runs, gain-field
models fitted on all runs (`derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf`). Robustness
rows use the same recipe with Gaussian or DN pRFs, each with its own jointly fitted HRF. Per-subject fits do not depend
on other subjects, so leaving the pilots out changes only the group statistics; they were recomputed with
`retsupp/visualize/paper/run_excluding.py --exclude 1 2` (figures), `retsupp/visualize/paper/sensitivity_excl_subjects.py`
(`notes/data/sensitivity_excl0102_2026-11.tsv`, both n), `retsupp/visualize/paper/n28_extra_numbers.py`
(`notes/data/n28/key_numbers_extras_n28_2026-11-12.tsv`), and, for what pools over subjects inside an analysis (the
leave-one-subject-out δ, the Fig. 7 hierarchical model, group-mean decoded images), reruns written to `notes/data/n28/`.

Status: **F** final · **R** robustness (supplement) · **X** exploratory (supplement or text at most, no claim).

## Fig. 1 Behaviour (F; `notes/data/behaviour_rt_2026-11.tsv`, legend `fig1_paradigm_behaviour_v4_n28.md`)

20,040 trials, 17,292 correct (`notes/data/behaviour_trials_2026-11-11.tsv.gz`). (The 2026-11-11 package's "21,480
correct trials" was the total number of trials at n = 30, of which 18,609 correct.)

Effects are the **median** of the within-subject differences (the statistic the Wilcoxon test and the figure use) with a
bootstrapped 95% CI (10 000 resamples over subjects); condition means and the mean difference in brackets.
Fig. 1 has five panels: c (distractor location), d (target location), e (intertrial priming: target and distractor
repeat − switch as within-subject differences, one panel). The raw response times behind e are in the supplementary
`figS_priming_rt_v4_n28`.

| Effect | Median difference [95% CI] | Condition means (mean difference) | Subjects in that direction | p |
|---|---|---|---|---|
| Distractor at HP vs LP (HP benefit; c) | −25.0 ms [−32.4, −20.0] | 874 vs 901 ms (−26.6) | 28/28 | 7.5 × 10⁻⁹ |
| Distractor at LP vs absent (capture cost; c) | +42.6 ms [+27.0, +50.7] | 901 vs 861 ms (+40.0) | 26/28 | 1.0 × 10⁻⁷ |
| Distractor at HP vs absent (c) | +14.6 ms [+0.3, +21.7] | 874 vs 861 ms (+13.4) | 19/28 | .0041 |
| Target at HP vs LP (distractor trials; d) | +24.3 ms [+18.5, +36.4] | 905 vs 877 ms (+27.3) | 24/28 | 1.5 × 10⁻⁶ |
| Target repeats its location (repeat − switch; e) | −38.7 ms [−48.8, −28.4] | (−37.2) | 26/28 | 3.7 × 10⁻⁸ |
| Distractor repeats (repeat − switch, within HP and LP, averaged; e) | −15.1 ms [−29.2, −6.2] | (−19.0) | 20/28 | .0013 |

Source: `notes/data/n28/key_numbers_extras_n28_2026-11-12.tsv` (block "1 behaviour").
Design: 8 items on a ring at 4°, target and distractor only at the four diagonal positions; no distractor on 20% of
trials; bar aperture 3.17°. Distractor at HP on 50% of all trials (62.5% of distractor-present trials), at each LP
location 10%, no distractor 20% (as printed in Fig. 1a; source `retsupp/visualize/paper/plot_fig1.py`).

## Fig. 2 pRF fits (F; `fig2_prf_fits_n28.md`)

- d, held-out R² gained per subject (median over the nine areas; three folds of runs, reliable voxels) over a Gaussian
  pRF with the canonical HRF (`notes/data/prf_cv_roi_summary.tsv` + `notes/data/hrf_comparison_jointhrf_final2_2026-11.tsv`):
  DoG pRF +.0033 [+.0024, +.0048] (28/28, p = 7.5 × 10⁻⁹); the jointly fitted HRF per subject × area on top of the
  DoG +.0106 [+.0064, +.0196] (27/28, p = 5.2 × 10⁻⁸); both +.0179 [+.0141, +.0229] (28/28, p = 7.5 × 10⁻⁹). Median
  over subjects [bootstrapped 95% CI].
- e, search-display response by distance of the pRF centre from the nearest item: from the final pRFs
  (`notes/data/event_deconv_nearest_final2_2026-11/`); qualitative, no test.
- c, pRF size against eccentricity (final pRFs, `notes/data/fig1_prf_sigma_final2.tsv`; mean over subjects of each
  subject's median σ per 0.5° bin, bins 0.25–3.25°): V1 nearly flat (.54–.65°), likely a resolution floor of the
  paradigm; V2 .58 → .79°; V3 .61 → 1.06°.
- Example subject: sub-23 (maps in a: mean fit over all runs, DoG with a flexible HRF per voxel; panel a's maps are not
  from the final fold fits, the legend says so).
- b, contralateral share of pRF centres, from the final pRFs (joint-HRF DoG fold fits, mean over folds; voxels with
  held-out R² ≥ .1 inside the bar aperture; `notes/data/fig1_prf_coverage_final2.tsv`), median [IQR] over subjects (n):
  V1 96.0% [92.5–97.6] (28), V2 94.5% [90.6–97.4] (28), V3 95.6% [93.4–97.4] (28), V3AB 95.9% [91.0–100] (27),
  IPS 89.7% [85.9–97.3] (27), hV4 95.8% [89.9–98.6] (28), VO 100% [100–100] (24), LO 96.9% [93.6–98.6] (28),
  TO 99.6% [96.2–100] (28); chance 50%. n < 28: no pRF centre passing the criterion.

**R, pRF models with jointly fitted HRF** (`notes/data/prf_jointhrf_summary_final2_2026-11.tsv`; held-out R², reliable
voxels, fold mean, median over the nine areas per subject, median over subjects): Gaussian .176, DoG .181, DN .188.
DN > DoG by .0048 [+.0023, +.0094] (25/28, p = 4.8 × 10⁻⁶); DoG > Gaussian by .0029 [+.0025, +.0051] (26/28,
p = 7.5 × 10⁻⁸).

## Fig. 3 Responses at the item locations (F; `fig3_item_locations_n28.md`; per item type, per pathway, HP vs all LP and DCT statistics in `tableS_item_locations_n28.md`)

Voxels with a final pRF centre within 2° of an item (`notes/data/event_deconv*_final2_2026-11/`); the bar's predicted
response removed with the final pRFs and HRF; window 3.2–6.4 s.

- Target − neutral (b), mean over subjects [bootstrapped 95% CI], % signal: early +0.125 [+0.102, +0.150] (n = 28),
  dorsal +0.264 [+0.205, +0.327] (n = 26), occipitotemporal +0.236 [+0.199, +0.277] (n = 28); Wilcoxon signed-rank, Benjamini-Hochberg over the three pathways per panel (b and c separately): all q ≤ 3 × 10⁻⁸ (uncorrected p ≤ 3 × 10⁻⁸).
- Distractor − neutral at LP (c): early +0.072 [+0.040, +0.104], dorsal +0.149 [+0.091, +0.214], occipitotemporal
  +0.156 [+0.110, +0.203]; BH over the three pathways: all q ≤ 1.1 × 10⁻⁴ (early q = 1.1 × 10⁻⁴, dorsal 2.9 × 10⁻⁵, occipitotemporal 2.0 × 10⁻⁶; `notes/data/sensitivity_excl0102_2026-11.tsv`). The Fig. 3 legend states the correction.
- Dorsal n: 26 in b, c (two subjects lack dorsal voxels at some item location); 25 in d, e (sub-30 also left out of
  the relative analyses because its mean dorsal response is not positive).
- Location type, early areas (e, items pooled, % of the subject's mean response): HP − next −6.7 ± 3.6% (p = .095);
  next − opposite −14.9 ± 6.0% (p = .022); HP − opposite −21.6% (p = .005); next left − next right +0.3% (p = .58).
  ANOVA location type F(2, 54) = 7.67, p = .001; location type × item F(4, 108) = 0.92, p = .454.
- All areas: location type F(2, 54) = 6.31, p = .003; HP − next −6.9 ± 2.7% (p = .018); next − opposite −8.6 ± 4.6%
  (p = .060); HP − opposite −15.5% (p = .013); × item F(4, 108) = 1.63, p = .172.
- Dorsal (n = 25) and occipitotemporal: no reliable location effect (F(2, 48) = 0.16, p = .86; F(2, 54) = 1.49,
  p = .24).
- HP vs all three LP locations, two numbers for two deconvolutions (say which one is quoted): in the location-type
  deconvolution of Fig. 3d, e (HP − mean of the three LP locations) early −11.7 ± 4.0%, p = .009, all areas
  −9.7 ± 3.1%, p = .006 (`notes/data/richter_C_D1_location_contrasts_n28_2026-11-11.tsv`; the model comparison in the
  Richter follow-up uses this one); in the earlier HP/LP deconvolution (two location classes per item) early −9.8%,
  F(1, 27) = 5.72, p = .024; all areas −8.4%, F(1, 27) = 6.82, p = .015; location × item early p = .579.
- HP − opposite with 95% CI of the mean: early −21.6 ± 6.8% [−34.9, −8.9]; all areas −15.5 ± 5.3% [−25.4, −5.1].
- DCT basis (robustness), early: location type F(2, 54) = 6.35, p = .003; next − opposite −14.0% (p = .023);
  HP − opposite −19.9% (p = .010); HP − next −5.9% (p = .16); × item p = .84.
- Per item type, early, HP − next / next − opposite: target −2.2% / −12.1%; distractor −6.6% / −23.3%; neutral
  −11.3% (p = .006) / −9.3%; all others p ≥ .09: same direction, item types do not differ.

## Fig. 5 The gain fields (F; `fig5_gain_fields_v6_n28.md`, parameter table `table_parameters_2026-11_n28.md`)

Panel letters (v6, update 4): a = the fitted gain-field maps (new); b = target and distractor gain per area (was a); c = dS (unchanged); d = dD and dT (unchanged); the earlier to-scale panel b is removed.

Sources: joint fits `notes/data/gain_field_parameters_final2_2026-11.tsv` (m7_full = all terms fitted together);
estimated alone (each term added alone to its base model, the subject's own estimate, all voxels):
`notes/data/conditional_estimates_dog_final2_2026-11.tsv`. "Pooled" = per-subject median over the nine areas.
Group values: `notes/data/sensitivity_excl0102_2026-11.tsv` (column `_excl`).

**a, the fitted gain-field maps (V3, hV4, TO; group medians, n = 28)** — per area, the median over participants of each participant's joint-fit parameter (`notes/data/gain_field_parameters_final2_2026-11.tsv`, m7_full; values written to `notes/data/fig5_gain_field_map_parameters_n28_2026-11-12.tsv`, script `retsupp/visualize/paper/fig5_gain_fields_v6.py`). The sustained HP gain is drawn as median g_LP + median joint dS, so HP − LP on the map equals the joint dS below (the median of g_HP itself is lower, because the median of a difference is not the difference of medians). The transient width is set to FWHM 3.2° for display in every area (not interpreted). All values agree with the parameter table.
  - V3: g_LP median −0.004, g_HP median −0.138; drawn at HP: g_LP + joint dS = −0.094 (joint dS median −0.090); sustained width σ 1.94° (FWHM 4.6°); target gain +0.300; distractor gain (averaged over locations) +0.034; transient at HP − LP (joint dD median, used only in the search supplement) +0.114.
  - hV4: g_LP median −0.058, g_HP median −0.160; drawn at HP: g_LP + joint dS = −0.115 (joint dS median −0.057); sustained width σ 1.29° (FWHM 3.0°); target gain +0.558; distractor gain (averaged over locations) +0.216; transient at HP − LP (joint dD median, used only in the search supplement) +0.107.
  - TO: g_LP median −0.354, g_HP median −0.476; drawn at HP: g_LP + joint dS = −0.378 (joint dS median −0.024); sustained width σ 1.07° (FWHM 2.5°); target gain +0.217; distractor gain (averaged over locations) +0.084; transient at HP − LP (joint dD median, used only in the search supplement) +0.000.
  - Other areas (same rule; parameter table): V1 g_LP +0.028, g_HP −0.043, σ 2.21°; V2 g_LP +0.013, g_HP −0.075, σ 2.00°; V3AB g_LP −0.114, g_HP −0.148, σ 1.34°; IPS g_LP −0.331, g_HP −0.402, σ 1.33°; VO g_LP −0.153, g_HP −0.170, σ 1.11°; LO g_LP −0.087, g_HP −0.085, σ 1.37°. The sustained gain is negative at all four locations in 7 of 9 areas (not V1, V2).

**b, joint fit, median per area** (* BH q < .05; was panel a):
- Target gain: V1 −.018, V2 +.048, V3 +.30*, V3AB +.33*, IPS +.21*, hV4 +.56*, VO +.50*, LO +.18 (q = .072),
  TO +.22*. Pooled +.16 [+.13, +.24], 89% of subjects > 0, p = 3.2 × 10⁻⁷. (At n = 30 V2 and LO also reached
  q < .05; they are the two weakest areas.)
- Distractor gain (averaged over its locations): V3AB +.12*, hV4 +.22*, TO +.084*; others n.s. Pooled +.065
  [+.032, +.15], 75% > 0, p = .023.
- Estimated alone, the distractor gain on top of the target term: V3AB +.038 (q = .024), hV4 +.15 (q = .050);
  pooled +.031 [−.002, +.069], 68% > 0, p = .019. Alone on the item model (without the target term) it is ≈ 0
  (pooled −.007, p = .090).
- Target gain rises with pRF size: +.33 per doubling [+.24, +.67], 82% of subjects, q = .0017
  (`notes/data/n28/hierarchy_regression_final2_2026-11.tsv`); distractor +.11 [−.10, +.29], q = .33 (n.s.). Within
  subjects the target gain falls with the width of the transient field (median Spearman ρ −.62, p = 6.5 × 10⁻⁶) and
  does not track the item weight (ρ +.07, p = .35) (`figS_gain_vs_prf_size_n28.md`).

**Field widths (FWHM; supplementary `figS_field_widths_n28`; the maps of Fig. 5a use each area's median sustained width)**: sustained
median 3.2° over all subjects and areas (12% of fits at a σ bound); per area early 4.6–5.2° (V1 5.2°, V2 4.7°,
V3 4.6°), higher areas 2.5–3.2° (TO 2.5°, VO 2.6°, hV4 3.0°, IPS 3.1°, LO 3.2°, V3AB 3.2°). Neighbouring items are
5.7° apart. Transient field: often at a bound, poorly constrained.

**Sustained gain shared by all four locations (S, estimated alone)**: negative in every area but V1; pooled −.034
[−.037, −.024], 100% of subjects < 0, p = 7.5 × 10⁻⁹; q < .001 in V2–TO. (A no-control term: it is not HP-specific.)
In the joint fit (m7_full) the LP gain itself is clearly non-zero: g_LP pooled −.067, negative in 24/28 subjects,
p = 2.3 × 10⁻⁶ (g_HP −.148, 25/28, p = 1.5 × 10⁻⁵): a field at HP alone (g_LP = 0) is rejected by the data. Methods
sentence: "The sustained field has a gain at all four possible target locations, because all four are searched; no
location is therefore a gain-free reference, and the HP effect is defined relative to the other three
(dS = g_HP − g_LP). A field at HP alone is the special case g_LP = 0, which the data reject." 

**c, dS = g_HP − g_LP, sustained gain at HP minus at LP, estimated alone**: pooled −.014 [−.018, −.006], 79% of
subjects < 0, p = 1.1 × 10⁻⁴. Per area: V3 −.030 (q = .046), TO −.019 (q = .035); all other medians negative, q ≥ .13.
Joint fit (m7_full): pooled −.034 [−.066, −.015], 79% < 0, p = .0027; V3 −.090 (q = .032), hV4 −.057 (q = .028),
TO −.024 (q = .044).
Direction across subjects (leave-one-subject-out sign test, `notes/data/n28/loso_per_step_allfam_final2_2026-11.tsv`):
the other 27 subjects' dS points the way a subject's data move in 75% of subjects, p = .0071.
Between subjects (text only): dS estimated alone vs the behavioural HP benefit (LP − HP RT), Spearman ρ = −.11
[−.50, +.30], p = .57; dS joint ρ = −.01, p = .97 (`notes/data/richter_D1_intervals_n28_2026-11-11.tsv`). Individual
differences cannot be resolved.

**d, transient HP − LP, estimated alone**:
- dD (distractor at HP − at LP): pooled −.010 [−.029, +.005], 61% < 0, p = .099; per area only LO −.033 (q = .022).
- dT (target at HP − at LP): pooled −.034 [−.068, +.005], 68% < 0, p = .018; no area q < .05.
- dD − dT: pooled +.009 [−.004, +.052], 18/28 > 0, p = .24; no area differs (`notes/data/update2_numbers_n28_2026-11-12.tsv`).
- Joint dD: pooled +.004 [−.041, +.124], 13/28 < 0, p = .29; no area q < .05 (`notes/data/update2_numbers_n28_2026-11-12.tsv`).
- So the transient reduction at HP is small and **not specific to the distractor** (the Fig. 5d title says "target and
  distractor alike").

**Split models** (full-data fits; `notes/data/gain_field_parameters_final2_2026-11.tsv`):
- Sustained LP gain split into next to HP vs opposite HP (m8_full_adjopp, 234 subject × area fits): next − opposite
  pooled −.006 [−.032, +.020], p = .73; HP − next −.016 [−.069, +.005], p = .070 (no area q < .05); HP − opposite
  −.033 [−.052, −.019], 82% < 0, p = .0048 (hV4 −.057, q = .004). The gain is lowest at HP; the neighbours do not
  differ from the opposite location.
- Target gain split by location (m8_full_dT, 214 fits): target HP − LP pooled +.047, p = .37; distractor HP − LP
  +.046, p = .13; dD − dT p = .48 (fitted jointly, neither is reliable).
- One shared width for sustained and transient fields (m7_full_sigtied, 221 fits): dS pooled −.023 [−.042, −.005],
  79% < 0, p = .036; FWHM 3.7°.

## Supplementary table: what each term adds to out-of-subject prediction (F; `tableS_loso_n28.md`, `notes/data/n28/loso_per_step_allfam_final2_2026-11.tsv`, `notes/data/loso_exact_final2_excl0102_2026-11.tsv`)

Leave-one-subject-out (LOSO): each subject's held-out fit with the term at the median of the other 27 subjects' own
estimates (the pilots are left out of the pool as well as of the test; cluster job 8261937), scored on signal voxels
(held-out response beats a model-free null); per subject the median over areas. "Better" = share of subjects whose
ΔR² > 0, Wilcoxon p. "Direction" = sign-flip test: ΔR² at +δ minus at −δ, which removes the penalty of adding a term
and is a true 50% null.

| Term | Better | p | Direction | p | Median δ (others) |
|---|---|---|---|---|---|
| Items (weight on bar-only pRFs, vs items at bar strength) | 100% | < .001 | — | — | −0.76 |
| Target gain (T) | 75% | .0007 | 96% | < .001 | +0.20 |
| Distractor gain, tied (D) | 61% | .16 | 75% | .0034 | −0.019 |
| Distractor gain on top of target (D given T) | 61% | .33 | 75% | .0005 | +0.006 |
| Sustained gain, tied (S) | 93% | < .001 | 100% | < .001 | −0.031 |
| dS (sustained HP − LP) | 68% | .41 | 75% | .0071 | −0.007 |
| dD (distractor HP − LP) | 43% | .25 | 50% | .96 | −0.010 |
| dT (target HP − LP) | 61% | .25 | 71% | .014 | −0.050 |

Items row, read with care: the step perturbs the item weight of pRFs fitted to the bar alone, whose pRFs never saw the
items; it is not a test of items vs none. The fitted item weight with the pRFs refitted (m0/m7) is the relevant number
(item_scale median ≈ 3 relative to the bar at n = 30: 2.85 m0, 3.08 m7). Per area: `figS_loso_per_roi_n28.md`.
(At n = 30 the tied distractor gain improved out-of-subject prediction, 77%, p = .031; at n = 28 it does not.)

## DoG centre/surround tilt κ (X; `figS_dog_kappa_n28.md`, `notes/data/n28/loso_kappa_2026-11_n28.tsv`)

κ splits the sustained field between the DoG centre and surround (κ < 0: relatively more of the HP effect on the
surround). Full-data fits (`figS_dog_kappa_n28`; pooled = per participant the mean over areas): κ −0.036 [−0.082,
−0.019] (bootstrapped 95% CI), 82% of 28 participants below 0, p = .011; per area q < .05 in V1 (−0.103), V2 (−0.074),
V3 (−0.057), hV4 (−0.047) and LO (−0.023); 5 subject × area fits at the bound |κ| = 2. Held-out runs (run3, pRFs
frozen; 27 participants, no run-3 fits for sub-08): worse than without κ, pooled ΔR² −1.02 × 10⁻⁴ [−1.78, −0.56],
96% below 0, p = 3 × 10⁻⁸, q < .05 in every area. LOSO (n = 28, pilots out of the pool): better in 68% of subjects
(p = .043); direction 93% (p = 2.5 × 10⁻⁷); median δ −0.009. (n = 30, earlier summary: pooled median −0.040,
p = .0035; LOSO 70%, p = .055; direction 90%.) Consistent in direction, too small to help prediction. Adding κ does not change dS (n = 28): per participant median over areas of dS(with κ) − dS(m7) −0.0001 [−0.0011, +0.0002], 12/28 larger, p = .33; dS with κ −.025 [−.053, −.010], p = .0027 (`notes/data/dog_cs_field_full_2026-11.tsv`, rows m8_full_dogk, dS − dS_m7; `notes/data/update2_numbers_n28_2026-11-12.tsv`).
κ and the neighbour pattern (`notes/data/n28/kappa_neighbour_2026-11_n28.tsv`): run through the Fig. 3 pipeline on
noise-free predictions, the fitted tilt changes next − opposite (early areas) by +0.27% (61% of subjects > 0, p = .45),
opposite in sign and ~50× smaller than the measured −14.9%; the joint fit (m7_full) predicts −1.3% (p = .16).

## DN: where does the HP effect act? (X; `figS_dn_gain_locus_n28.md`, `notes/dn_gain_field_variants_2026-11.md`)

- A sustained field on the DN neural baseline (b), surround baseline (d) or both, instead of the input gain:
  held-out runs (run3, pRFs frozen; `notes/data/dn_basefield_cv_complete_2026-11.tsv`) worse than the input-gain model in
  28/28 subjects for each (p = 7.5 × 10⁻⁹; mean ΔR² −3.3 × 10⁻⁵ (b), −3.3 × 10⁻⁵ (d), −1.1 × 10⁻⁴ (both)).
- Fitted together, b and d move together (Gauss-Newton correlation of w_b and w_d, median over 252 subject × area fits 0.9992, 5th–95th percentile 0.991–1.000, n = 28; `notes/data/dn_basefield_full_2026-11.tsv`, model m8_full_dnbd), i.e. an amplitude field that trades
  off with the input gain.
- LOSO (n = 28, `notes/data/n28/loso_dnw_2026-11_n28.tsv`): common b,d weight 46% better (p = .24), direction 50%
  (p = .60); b/d ratio 57% (p = .27), direction 75% (p = .010; ΔR² ≈ 10⁻⁷).
- Recovery (simulation): at our effect size a b or d effect is read as input gain in 72% of simulations.
- Conclusion: the data cannot place the HP effect on the DN baselines rather than on the input gain.

## Fig. 6 Decoding (F; `fig6_decoding_n28.md`; pRFs = final DoG joint-HRF fold fits)

- Frame-wise r, decoded vs true (median over subjects; shuffled ≈ 0, all p < 10⁻³ one-sided): V1 .52, V2 .48, V3 .41,
  V3AB .29, IPS .11, hV4 .27, VO .16, LO .24, TO .19. n per area: V1–V3 28, V3AB 23, IPS 17, hV4 27, VO 10, LO 27,
  TO 25.
- Example run: sub-03 ses-1 run-4, V1, r = .62 (rank 1 of 2458 subject × run × area decodes); median run r = .30.
  Group-mean frames (a, bottom row): mean of 28 subjects (V1) (`notes/data/n28/decoding_frames_jointhrf_2026-11.npz`,
  cluster job 8262318).
- Supplementary `figS_decoding_hp_per_area_n28`, HP quadrant − other three, one-sided, BH over nine areas:
  **V2 (p = .007) and V3 (p = .008) q = .038**; hV4 p = .058, VO .065, TO .086, V3AB .10 (q = .15); V1 p = .22.
- d, V1–V3 pooled, % of the four-quadrant mean (mean ± within-subject SEM): HP −1.02 ± 0.38%, next to HP
  +0.57 ± 0.17%, opposite −0.12 ± 0.24%. HP < next one-sided p = .004; HP < opposite p = .082; Friedman across the four
  quadrants p = .013; left vs right p = .24. Next vs opposite (two-sided) p = .070. Differences (mean ± SEM [bootstrapped 95% CI of the mean], n = 28, `notes/data/update2_numbers_n28_2026-11-12.tsv`): next − opposite +0.68 ± 0.32% [+0.08, +1.30], p = .070; HP − opposite −0.90 ± 0.53% [−1.97, +0.12], p = .16 (two-sided; one-sided p = .082 as above); HP − next −1.58 ± 0.53% [−2.66, −0.58].
- e, by eccentricity (V1–V3): 0–1.5° −0.7% (p = .069); 1.5–2.5° −1.6% (p = .010); 2.5°+ −1.5% (p = .020); whole
  aperture −1.3 ± 0.5% (p = .010); bands do not differ (Friedman p = .51).
- c, insets (outline decoder): bar segment in the quadrant facing HP −1.33 ± 0.49% (p = .016); next to HP
  +1.29 ± 0.37% (p = .002).

**Supplements (F).** Canonical HRF (`figS_decoding_canonical_n28`, V1–V3): V2, V3 q = .013; HP < next p = .002.
Decoded bars, same stimulus (`figS_decoded_bars_rotated_n28`; mean ± SEM [bootstrapped 95% CI of the mean], `notes/data/update2_numbers_n28_2026-11-12.tsv`: HP half − other half pooled −1.80 ± 0.56% [−2.87, −0.73], vertical −1.71 ± 0.70% [−3.05, −0.38], horizontal −1.88 ± 0.54% [−2.89, −0.82]; next − opposite half pooled +1.21 ± 0.45% [+0.35, +2.09], vertical +1.29 ± 0.45% [+0.40, +2.15], horizontal +1.12 ± 0.59% [−0.01, +2.26], p = .019, .010, .10): HP half minus other half, vertical −1.71% (p = .034),
horizontal −1.88% (p = .002), pooled −1.80% (p = .006); next − opposite half pooled +1.21% (p = .019); per area
(`figS_decoded_bars_per_roi_n28`) only V3 survives BH (−2.82%, q = .013). Bar-crossing response
(`figS_bar_crossing_response_n28`, V1–V3): HP − LP −3.1% of the LP response, one-sided p = .010; with a pre-crossing
baseline −2.7%, p = .10 (n.s.).

**Decoder simulation (F; `figS_decode_simulation_n28.md`, `notes/data/decode_sim_summary_2026-11_n28.tsv`)**: an
HP-only gain field is recovered at about 85% of its true quadrant pattern; −0.025 gain gives a decoded HP − next of
−0.88% (true −1.03%), the same order as observed (HP − next −1.59%); an HP-only field puts the neighbours *below*
opposite (−0.24%), unlike the data (next +0.57%, opposite −0.12%, n.s.).

**R, decoding with Gaussian and DN pRFs (joint HRF)** (`notes/data/decoding_prf_models_quadrants_2026-11_n28.tsv`,
`figS_decoding_prf_models_n28.md`), V1–V3 pooled, HP vs next (two-sided), free decoder: DoG −1.02 vs +0.57%
(p = .008); Gaussian −1.02 vs +0.57% (p = .008); DN −0.85 vs +0.46% (p = .036; outline decoder p = .008). Next vs
opposite reliable only with the Gaussian pRFs and the outline decoder (p = .048; otherwise p = .053–.45). Decoding quality V1 r: DoG .52, Gaussian .50, DN .45.

## Fig. 7 Gains and behaviour (F; `fig7_compact_n28.md`; data `notes/data/n28/trialwise_meta_bayes_2026-11/`)

Per subject × area, the trial-wise gains modulated by trial history and by z(log RT), estimated directly on the BOLD
(final m7 fits, each area's signal voxels), each estimate with a leave-one-run-out jackknife SE; hierarchical Bayesian
model over subjects and areas (group means fixed, areas and subjects random), weighting each estimate by its SE, refitted
on the 28 participants. Posterior mean [95% HDI], P = posterior probability of the sign. n = 28 (230 subject × area
estimates).
- a, target gain on target-repeat trials, % of the area's typical target gain: all nine areas −19% [−33, −4],
  P = .997; early −26% [−57, +3], P = .962; dorsal −22% [−47, +2], P = .964; occipitotemporal −13% [−30, +5],
  P = .945. In absolute gain −0.052 [−0.093, −0.012], P = .993. Student-t likelihood: P = .994. Without the SE
  weighting −44%, P = .968 (same direction, just below .975).
- a, distractor gain on distractor-repeat trials: −30% [−75, +13], P = .920 (no credible change).
- b, change per SD of log RT (higher = higher gain on slower trials): **distractor gain −0.024 [−0.043, −0.005],
  P = .990**; global gain −0.021 [−0.046, +0.006], P = .943 (**no longer credible**; it was P = .979 at n = 30);
  target gain +0.009 [−0.013, +0.031], P = .80. Without the SE weighting the RT effects vanish (distractor P = .69,
  global P = .51). Panel b's title is now "Lower distractor gain on slower trials".
- Convergence: 1 divergence over 27 fits, max R̂ 1.01, min bulk ESS 880.
- The October-fit version of Fig. 7 (RT regressed on per-trial gains) is retired.

## Richter et al. follow-up (2026-11-11; n = 28; `richter_followup_2026-11-11.md` in this package, code `retsupp/visualize/paper/richter_followup.py`, data `notes/data/richter_*_n28_2026-11-11.tsv`)

Medians of within-subject differences with bootstrapped 95% CIs and Wilcoxon p unless "mean ± SEM".

**B. Behaviour by location relative to HP** (correct trials; next = the two adjacent diagonal locations pooled;
target cells on distractor-present trials; `richter_B_behaviour_{cells,contrasts}_n28_2026-11-11.tsv`). Cell means:
distractor at HP 874 ms, next 912, opposite 880, absent 861; target at HP 905, next 881, opposite 870 ms.
- Distractor: next − opposite +37.8 ms [+24.2, +45.5], p < .001; HP − next −37.8 [−41.9, −30.6], p < .001;
  HP − opposite −3.8 [−15.9, +3.7], p = .34. The benefit against all LP locations comes from the neighbours being slow;
  opposite HP is about as fast as HP.
- Target: HP − next +20.6 [+15.7, +29.7], HP − opposite +30.2 [+22.8, +48.3], next − opposite +10.4 [+0.2, +18.4],
  all p ≤ .005; accuracy at HP −4.9% vs opposite (p < .001). A graded target cost HP > next > opposite.
- Richter et al. report the benefit only at HP and no target cost; ours differs on both.

**D5. Target vs distractor at the item locations** (both at LP; window 3.2–6.4 s, % signal; BH over pathways):
early +.038 [+.004, +.105], q = .015; dorsal +.126 [+.049, +.184], q < .001 (n = 26); occipitotemporal +.097
[+.055, +.146], q < .001. So target > distractor > neutral in early visual cortex, as in Richter et al.

**D6. Location effect per early area** (exploratory, uncorrected): HP − opposite V1 −29.8% [−61.2, −12.9], p = .002;
V2 −20.0% [−41.9, −3.9], p = .010; V3 −11.4% [−18.5, +0.2], p = .33; next − opposite n.s. in each (p = .09–.21).
Unlike Richter et al. (V2 > V1), the reduction is clearest in V1 (V1 is noisy: single-area percentages have small
denominators).

**A1. Fig. 3e contrasts per session and block** (descriptive; the inference is the session and block difference,
which is not significant; early areas, items pooled, % of the subject's full-data mean response; mean ± SEM; `richter_A1_carry_over_common_denominator_n28_2026-11-11.tsv`):

| Split | HP − opposite | Next − opposite | HP − next |
|---|---|---|---|
| All runs (Fig. 3e) | −21.6 ± 6.8, p = .005 | −14.9 ± 6.0, p = .022 | −6.7 ± 3.6, p = .095 |
| Session 1 | −2.2 ± 13.8, p = .26 | −3.5 ± 12.6, p = .66 | +1.4 ± 12.3, p = .75 |
| Session 2 | −34.6 ± 13.2, p = .013 | −20.2 ± 11.8, p = .11 | −14.3 ± 11.5, p = .14 |
| Session 2 − 1 | −32.4 ± 23.0, p = .39 | −16.7 ± 21.0, p = .58 | −15.7 ± 22.7, p = .37 |
| Block 1 (HP new in the session) | −34.4 ± 15.1, p = .023 | −15.1 ± 10.0, p = .095 | −18.1 ± 7.7, p = .032 |
| Block 2 (opposite = HP of block 1) | +3.1 ± 15.8, p = .81 | −2.0 ± 11.1, p = .99 | +5.2 ± 9.8, p = .88 |

All areas: session 1 HP − opposite −7.8 ± 9.7, p = .16; session 2 −20.2 ± 8.5, p = .019. Session and block
differences are not significant (p ≥ .26): the effects do not differ reliably between sessions or blocks. Each split
holds a quarter to half of the data, so per-split tests are underpowered; do not contrast a significant with a
non-significant split.

**A2. Locations recoded by HP history** (early, % of the split's mean, n = 27): current HP 83.2 ± 14.3; HP in the
previous block 99.5 ± 7.9; HP in the previous session 113.7 ± 14.1; never HP 101.0 ± 5.9. ANOVA on the LP locations,
location × was-HP-before: was HP before F(1, 26) = 0.39, p = .54 (all areas p = .71). No reduction at former HP
locations.

**A3. Sustained gain per session** (estimated alone with the all-run fits fixed; `richter_A3_session_gains_n28_2026-11-11.tsv`):
dS session 1 −.006 [−.021, +.021], p = .96; session 2 −.035 [−.062, −.005], 68% < 0, p = .011; session 2 − 1 −.022,
p = .095 (the inference: no reliable session difference). Next − opposite ≈ 0 in both sessions (+.005, p = .36;
−.002, p = .61).

**A4. Behaviour by HP history** (`richter_A4_behaviour_history_n28_2026-11-11.tsv`): raw, faster at a former-HP opposite
location (distractor −31.0 ms, p = .012; target −34.2 ms, p = .007), but after the practice change is subtracted
(difference in differences) no carry-over: previous block distractor −9.3 ms [−31.1, +9.2], p = .35; previous session
+17.5 ms [−27.7, +38.7], p = .37; target p ≥ .48.

**C. How much of the measured HP reduction the joint fit reproduces** (noise-free m7_full predictions through the Fig. 3
pipeline; mean ± SEM; `richter_C_D1_location_contrasts_n28_2026-11-11.tsv`, `richter_C_term_contrasts_n28_2026-11-11.tsv`):

| Contrast | Measured, early | m7 predicted, early | Measured, all areas | m7 predicted, all areas |
|---|---|---|---|---|
| HP − LP (mean of three) | −11.7 ± 4.0, p = .009 | −5.4 ± 3.7, p = .036 | −9.7 ± 3.1, p = .006 | −4.1 ± 3.0, p = .015 |
| HP − opposite | −21.6 ± 6.8, p = .005 | −6.3 ± 3.8, p = .023 | −15.5 ± 5.3, p = .013 | −4.7 ± 3.0, p = .010 |
| HP − next | −6.7 ± 3.6, p = .095 | −4.9 ± 3.7, p = .045 | −6.9 ± 2.7, p = .018 | −3.8 ± 3.0, p = .020 |
| Next − opposite | −14.9 ± 6.0, p = .022 | −1.3 ± 0.8, p = .16 | −8.6 ± 4.6, p = .060 | −0.9 ± 0.5, p = .17 |

About half of HP − LP, about 30% of HP − opposite, most of HP − next, none of next − opposite. By term (early, n = 27):
the sustained HP − LP term carries the predicted reduction (sustained only −7.3 ± 4.8%, p = .025; its own contribution
−4.7 ± 4.6%); the distractor transient dD pulls it back slightly (+2.2 ± 2.1%); the target transient dT adds nothing
(+0.8 ± 1.9%). No variant predicts the neighbour effect (+0.7 to +1.9%).

## Discussion checks (2026-11-12; n = 28; `discussion_checks_2026-11-12.md` in this package, code `retsupp/visualize/paper/discussion_checks.py`, data `notes/data/discussion_*_n28_2026-11-12.tsv`)

- Behaviour by HP history (medians of within-subject differences, bootstrapped 95% CI, Wilcoxon). All blocks: distractor
  next − opposite +37.8 ms [+24.2, +45.5], p < .001; distractor HP − opposite −3.8 [−15.9, +3.7], p = .34; target
  HP − next +20.6, p < .001; target HP − opposite +30.2, p < .001. Session 1 block 1 only (no location ever HP; ~77 HP,
  30 next, 16 opposite, 33 absent correct trials per subject): +34.4 [+5.9, +53.2], p = .025; −4.3 [−43.6, +15.2],
  p = .57; +19.5, p = .048; +31.9, p = .004. s1b1 − other blocks: −6.5 (p = .58), +8.7 (.83), −10.4 (.51), +1.5 (.92).
  Relation × session-block rm-ANOVA: distractor F(6, 162) = 0.27, p = .95; target F = 0.52, p = .79.
- Model (m7) vs measured item responses, early areas, % of the subject's mean (mean ± SEM; gap = measured − predicted,
  paired Wilcoxon): HP − LP −11.7 ± 4.0 vs −5.4 ± 3.7, gap −6.3 ± 4.7, p = .095; HP − next −6.7 ± 3.7 vs −4.9 ± 3.7,
  gap −1.8 ± 4.5, p = .70; HP − opposite −21.6 ± 6.8 vs −6.3 ± 3.8, gap −15.4 ± 7.1, p = .026; next − opposite
  −14.9 ± 6.0 vs −1.3 ± 0.8, gap −13.6 ± 6.2, p = .040. Between-subject ρ(measured, predicted): .20 (p = .31), .16,
  .20, −.14 (all n.s.). All areas: HP − LP gap −5.7 ± 3.6 (p = .081), HP − opposite gap p = .040, next − opposite gap
  p = .081.

## Robustness: the gain fields with Gaussian and DN pRFs (R; joint fits from `gain_field_parameters_final2_2026-11.tsv`)

Joint fit (m7_full), pooled per-subject median over areas:

| Term | DoG | Gaussian | DN |
|---|---|---|---|
| Target gain | +.16, p = 3.2 × 10⁻⁷ | +.45, p = 4.1 × 10⁻⁷ | +.53, p = 7.5 × 10⁻⁹ |
| Distractor gain | +.065, p = .023 | +.095, p = .032 | +.15, p = .0017 |
| dS (joint) | −.034 [−.066, −.015], 22/28 < 0, p = .0027 | −.088 [−.111, −.001], 19/28 (68%) < 0, p = .027 | −.078 [−.102, −.027], 20/28 < 0, p = .0024 |
| dD (joint) | +.004, p = .29 | +.055, p = .14 | +.024, p = .22 |

Per area (joint): target q < .05 in V3–TO except LO under DoG; under Gaussian in V1 (negative), V3AB, IPS, hV4, VO, LO,
TO; under DN in V2–TO. dS joint q < .05: DoG V3, hV4, TO; Gaussian none (smallest q = .13); DN none (hV4 q = .059).

Estimated alone (exact ΔSSE minimum, areas with ≥ 20 voxels; `notes/data/conditional_exact_final2_allfam_2026-11.tsv`;
DoG from the linear statistics, as in Fig. 5), pooled per-subject median over areas, and the leave-one-subject-out tests
(`notes/data/n28/loso_per_step_allfam_final2_2026-11.tsv`, `tableS_loso_n28.md`):

| Term | DoG | Gaussian | DN |
|---|---|---|---|
| dS alone | −.014 [−.018, −.006], 22/28 < 0, p = 1.1 × 10⁻⁴ (V3, TO q < .05) | −.008 [−.026, −.001], 23/28 < 0, p = 4.2 × 10⁻⁴ (hV4 q = .048) | −.028 [−.041, −.007], 21/28 < 0, p = .0052 (hV4 q = .020) |
| dS direction (sign-flip, LOSO) | 75%, p = .0071 | 79%, p = .0044 | 71%, p = .038 |
| dS better out of subject | 68%, p = .41 | 61%, p = .54 | 64%, p = .24 |
| dD alone | −.010, p = .099 | −.004, p = .69 | −.018, p = .24 |
| dT alone | −.034, p = .018 | −.067, p = .022 | −.066, p = .081 |
| Target gain alone | +.16, p = 7.5 × 10⁻⁹ | +.44, p = 1.0 × 10⁻⁷ | +.35, p = 7.5 × 10⁻⁹ |
| Distractor on top of target alone | +.031, p = .019 (V3AB, hV4) | +.028, p = .032 (V3AB, hV4, TO) | +.033, p = .12 (V3AB, IPS, hV4) |
| Sustained gain S alone | −.034, 100% < 0 | −.043, 100% < 0 | −.021, 96% < 0 |
| Target gain better out of subject | 75%, p = .0007 | 89%, p < .001 | 86%, p = .0001 |
| Distractor gain (tied) better out of subject | 61%, p = .16 | 64%, p = .030 | 18%, p = .018 (worse) |

Conclusion: under all three pRF models the sustained gain is lower at HP than at LP when estimated alone (75–82% of
subjects) and jointly (p ≤ .027), and the direction generalises across subjects; in no model does dS improve
out-of-subject prediction. Figures: `figS_gain_fields_gauss_n28`, `figS_gain_fields_dn_n28`.

Not run with Gaussian or DN pRFs: Fig. 3 (voxel selection by pRF centre) and Fig. 7 (trial-wise gains).

## Methods facts (from `writing_followup_2026-11-10.md`, part A; sources as given there)

**Participants.** 30 participants completed both MRI sessions (sub-01–sub-30). The first two (sub-01, sub-02) were
tested with a pilot version of the paradigm (bar 0.5°/s instead of 0.168°/s, ≈ 14 s instead of ≈ 42 s per sweep; HP at
only three of the four locations, in blocks of 2, 3 and 1 runs, the same order in both sessions; search task identical)
and are left out of all analyses, on paradigm grounds; **n = 28** (sub-03–sub-30). The n = 30 results are in
Supplementary Table `tableS_sensitivity_n30.md`: no result changes sign; the effects at the HP location get stronger
without the pilots, the weakest distractor-gain and global-gain statements weaker. Of the 28: age known for 27 (one
missing), mean 23.4 years, SD 3.6, range 18–33; 20 female, 8 male (self-reported sex). 26 right-handed, 2 left-handed (self-reported; source: the lab's participant sheet, BIDS `participants.tsv`). Before
the MRI sessions every participant did a behavioural pretest session; 15 people did the pretest without continuing to
the MRI sessions (reasons not recorded). Six further people started MRI participation but are not in the dataset
(6 rows of the lab sheet with an MRI code outside sub-01–30, 6 distinct participants): two scans were stopped after two
blocks because of discomfort, one was cancelled, one was awaiting medical (GP) clearance, and two have no reason
recorded [authors: confirm]. Subject × area pairs without enough voxels drop out per figure, as listed
above. Source: the lab's subject sheet, aggregated (`notes/data/n28/key_numbers_extras_n28_2026-11-12.tsv`, block
"demographics"); per-subject values in the BIDS `participants.tsv` (not part of this package). **Ethics approval (body,
number) and payment: to be supplied by the authors (VU team)** [the sheet shows only varying amounts and course credit].

**Eye tracking (E12, checked 2026-11-12; `experiment/main.py`, `session.py`, `trial.py`, `settings/default.yml`)**:
the behavioural pretest session ran with an EyeLink tracker (exptools2 `PylinkEyetrackerSession`, calibration before
each block, left eye): during the search display a beep sounded when gaze left a 1° window around fixation
(`gaze_threshold_deg: 1`, `eyemovements_alert: True`), as feedback. In the two MRI sessions the tracker was off
(`main.py`: `eyetracker_on = False` for sessions > 1), so fixation in the scanner was not monitored. No eye-tracking
files are in the dataset, no fixation criterion for continuing to the MRI sessions is in the code, and the lab sheet
records only the date of the eye-tracked pretest (for 26 of the 28 participants) and no exclusion for fixation: how many
of the 15 who did not continue were excluded for fixation is not recoverable from the files.

**Task** (`experiment/session.py`, `trial.py`, `stimuli.py`; per-run `*_expsettings.yml`; A4):
- Response: report whether the target (the item with the odd orientation) carries a small grey dot (radius 0.1°);
  keys `b` = dot, `y` = no dot (scanner button box; which fingers is not in the code). First press only, during the
  1.5 s search display; RT from search onset. On each trial two of the four diagonal and two of the four cardinal items
  carry a dot. (`experiment/instructions.yml` and `experiment/README.md` name other keys, from the behavioural-lab
  version; use `b`/`y`.)
- Timing: fixation 1.0 s, search 1.5 s, feedback 1.0 s (a red "x" after an error or no response; nothing after a
  correct response), interval 1.3, 2.9 or 4.5 s (20 each per run, shuffled); 60 trials per run; 14.3 s blank at the end.
- Trial list per run (fixed counts, shuffled): distractor at HP 30 (50%), at each LP location 6 (10%), absent 12 (20%);
  target 15 times at each diagonal location, never at the distractor's location. (The settings field
  `likely_distractor_probability: 0.67` is unused; do not cite it.)
- HP blocks (sub-03–30): four blocks of three runs, each diagonal location HP once; blocks 1–2 in session 1, 3–4 in
  session 2; within a session HP moves to the opposite location, between sessions to the other diagonal; eight orders
  in rotation. sub-20 session 1 and sub-24 session 2 have five runs (one block of two). (Pilots sub-01, sub-02, left out:
  2, 3 and 1 runs at three locations per session.)
- Display: mid-grey background; eight bars 1.5° × 0.375° on a ring at 4°; one item has the odd colour (distractor), one
  the odd orientation (target, always the common colour); colours orange and green, assigned at random per trial, as is
  the target orientation; display shifted 1.1° down. Fixation: white disc 0.25° with a black cross and a grey centre
  dot. Screen 70.9 cm at 196 cm, 1920 × 1080 (≈ 20.5° wide).
- Bar: checkerboard in a circular aperture of radius 3.17°, width 0.79°, contrast reversing at 8 Hz, 0.168°/s (42.4 s
  per sweep); rightward, leftward, downward, upward with 4.8 s rests, twice per run; independent of the trials.

**Acquisition and preprocessing** (A5):
- Philips Ingenia 7T, SENSE-Head-7T coil; gradient-echo EPI, TR 1.6 s, TE 24 ms, flip angle 66°, 1.68 × 1.68 × 1.7 mm,
  multiband 3, in-plane SENSE 2.4, partial Fourier 0.80, 258 volumes per run; magnitude and phase saved. Field maps:
  one phase-difference map per session half (TE 4.5/5.5 ms). Anatomy: MP2RAGE 0.70 × 0.70 × 0.85 mm; T1w for fMRIPrep
  = UNI × INV2/max(INV2), N4-corrected.
- NORDIC (`NIFTI_NORDIC`, magnitude + phase, temporal_phase 1, phase_filter_width 10).
- fMRIPrep 25.2.3, output spaces T1w and fsnative, 4 dummy scans; analyses volumetric in T1w space; fsnative only for
  the neuropythy atlas.
- Cleaning: motion-outlier frames interpolated, then nilearn `clean_img(standardize='psc')` with DVARS, FD, 10 aCompCor,
  6 motion parameters + derivatives, non-steady-state and cosine regressors; percent signal change of each voxel's run
  mean.
- Distortion correction (checked 2026-11-12, fMRIPrep reports `derivatives/fmriprep/sub-*.html`): fieldmap-based, from
  the phase-difference maps ("FMB (fieldmap-based) – phase-difference map") in every run of every subject (356 runs; no
  run used SyN); boilerplate in `references/fmriprep_CITATION.md`.
- 2D or 3D EPI: **not recoverable from the files here.** The BOLD sidecars are a hand-written template
  (`retsupp/prepare/convert_raw_mri_data.py`, `default_json`) that says `MRAcquisitionType: 3D` together with
  `MultibandAccelerationFactor: 3` and a 1.7 mm slice thickness; multiband (simultaneous multi-slice) acceleration
  implies a 2D multi-slice acquisition, so "3D" is most likely a template error. The raw PAR/REC headers of the
  functional runs are not on the cluster (sourcedata holds only behaviour); check the exam card or a PAR header.

**Decoder and software** (A6):
- Decoder (`retsupp/decode/decode_final.py`): Bayesian inversion of the fitted pRF encoding model in braincoder; noise
  model `ResidualFitter` on training-run residuals (multivariate t, covariance ρ·ττᵀ + (1 − ρ)·diag(τ²) + σ²·WWᵀ; van
  Bergen et al. 2015); `StimulusFitter` finds the MAP stimulus image, every pixel of every frame free, Gaussian (L2)
  prior, L2 = 1.0; bar-only paradigm; pRFs from the fold that held the decoded run out; at most 2000 voxels per area by
  cross-run reliability (`decode_final.py --max-voxels`, default 2000; the 2026-11-11 package's "200" was wrong), BOLD SD ≤ 20%. Cite braincoder: de Hollander, Renkert, Ruff & Knapen (2024), Zenodo,
  doi:10.5281/zenodo.10778413.
- Versions: TensorFlow 2.14.0, Keras 2.14.0, nilearn 0.13.0 (fits; the cleaning env no longer exists, its nilearn
  version is not recoverable), nibabel 5.2.1, NumPy 1.26.4, SciPy 1.15.2, pandas 2.3.3, statsmodels 0.14.6 (Python
  3.10); nideconv commit 46146f7 (Fourier basis, 13 regressors, −4 to 20 s); neuropythy 0.12.16; PyMC 5.28.3, nutpie
  0.16.8, bambi 0.17.2, ArviZ 0.23.4; fMRIPrep 25.2.3; matplotlib 3.10.6, seaborn 0.13.2.
- braincoder (checked 2026-11-12): the cluster env (`retsupp_cuda`) uses the editable checkout
  `~/git/retsupp/libs/braincoder`; its HEAD has been f82f316 ("gamma_pdf in log space", 2026-11-07) since 2026-11-07
  10:10 CET (git reflog; no later checkout; working tree clean now). All final fits were written after that: joint-HRF
  DoG pRF fits from 2026-11-07 10:48, the gain-field ladder 2026-11-09, Gaussian and DN fits 2026-11-09/10, decodes
  2026-11-07 10:15 to 2026-11-10. So the final fits ran on braincoder f82f316 (= the repo's `libs/braincoder` submodule
  pointer); whether the working tree was clean at each run is not recorded.

**Global gain (Fig. 7; A8).** Exists only in the trial-wise model (`retsupp/modeling/fit_trialwise_af_gains.py`,
regressor G), not in the m7 fits (baseline value 0): a per-trial multiplier on the whole, spatially unweighted stimulus
drive (bar plus all eight items, through each voxel's pRF, HRF and amplitude) during the 1.5 s search window. It absorbs
non-spatial per-trial amplitude changes, so the target and distractor gains index the spatially specific part.


## Counts (k of n) for every pooled share (2026-11-12)

See `counts_k_of_n_n28.md` in this package: k / n for every share above, from the per-participant tables (`notes/data/update2_numbers_n28_2026-11-12.tsv`). One correction: the Gaussian joint dS is 19/28 = 68% < 0 (was written 67%).
