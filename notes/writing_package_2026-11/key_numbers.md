# Key numbers (updated 2026-11-09)

Take every number in the text from here. n = 30 unless stated. Tests: Wilcoxon signed-rank across subjects
(two-sided unless stated); BH = Benjamini-Hochberg. Paths are relative to `/Users/gdehol/git/retsupp/`.

Status: **F** final · **P** provisional, **will change after the final DoG rerun** (job 8196318, one HRF per subject
× ROI) · **Proto** October 2026 fits, will be redone · **X** exploratory. "Computed" = not printed in a legend;
computed on 2026-11-09 from the named file with the same test the figure uses.

## Fig. 1 Behaviour (F; `notes/data/behaviour_rt_2026-11.tsv`, `notes/results_overview_2026-11.md` §3)

Correct trials, 21,480 trials in total.

| Effect | Value | Subjects in that direction | p |
|---|---|---|---|
| Distractor at HP vs LP (HP benefit) | 877 vs 905 ms, −27.8 ms (median 26.1) | 30/30 | 1.9 × 10⁻⁹ |
| Distractor at LP vs absent (capture cost) | 905 vs 862 ms, +43.0 ms | 28/30 (93%) | 2.6 × 10⁻⁸ |
| Distractor at HP vs absent | 877 vs 862 ms, +15.2 ms | 21/30 (70%) | .001 |
| Target at HP vs LP (distractor trials) | 908 vs 880 ms, +28.4 ms | 26/30 | 3.9 × 10⁻⁷ |
| Target repeats its location (repeat − switch) | median −38 ms | 28/30 | < .001 |
| Distractor repeats (repeat − switch, within HP and LP, averaged) | median −15 ms | 21/30 | .003 |

Without the two pilots (sub-01, sub-02): −26.6 ms (p = 7.5 × 10⁻⁹); target at HP +27.3 ms (p = 1.5 × 10⁻⁶).
Design (Fig. 1a, b): 8 items on a ring at 4°, target and distractor only at the four diagonal positions; no distractor
on 20% of trials; bar aperture 3.17°; four sweep directions, 42 s each with 4.8 s rest, twice per run.
`[TODO: distractor-at-HP probability; printed in Fig. 1a but not in its legend]`.

## Fig. 2 pRF fits (F, except d)

- Held-out R² gained per subject (median over the nine ROIs; three folds of runs, reliable voxels) over a Gaussian
  pRF with the canonical HRF (computed from `notes/data/prf_cv_roi_summary.tsv` + `notes/data/hrf_comparison_2026-11.tsv`,
  as `fig2_prf_fits.py gain_r2`): DoG pRF +.0035 (30/30 positive, p < .001); one HRF per subject × ROI on top of the DoG
  +.011 (29/30, p < .001); both +.017 (29/30, p < .001). **d may change** when the jointly fitted HRF replaces the
  median-of-voxels HRF (legend says so). This also meets the pre-specified HRF rule (> 15/30 positive, p < .05).
- Held-out R² by pRF model (reliable voxels, median over subjects; `prf_cv_roi_summary.tsv`, per-voxel HRF variants):
  Gaussian .201, DoG .210, DN .216; with a fitted HRF DoG .238, DN .245. DN > DoG in 30/30 subjects (by ≈ .007 with
  fitted HRF).
- V1 pRF size is flat with eccentricity (≈ .55–.6°), most likely a resolution floor of the paradigm
  (`notes/open_issues_2026-11.md` c). V2 rises .60 → .75°, V3 .61 → .91°.
- Contralateral share per ROI (Fig. 2b): `[TODO: not in the legend; compute from the Fig. 2b table]`.
- Example subject: sub-23.

## Fig. 3 Responses at the item locations (F; legend of `fig3_item_locations`; `notes/data/event_deconv_relative{,_dct}_2026-11/`)

Voxels with pRF centre within 2° of an item; window 3.2–6.4 s; % signal or % of the subject's mean response.

- Target − neutral (b): early +0.12%, dorsal +0.26%, occipitotemporal +0.22% signal, all p < .001 (BH over pathways).
- Distractor − neutral at LP (c): early +0.07%, dorsal +0.17%, occipitotemporal +0.16% signal, all p < .001.
- Location type, early areas (e, item types pooled): HP − next −5.0 ± 3.4% (p = .158); next − opposite −14.0 ± 5.9%
  (p = .029); HP − opposite −19.0% (p = .011); next left − next right +0.4% (p = .78). ANOVA location type
  F(2, 58) = 6.61, p = .003; location type × item F(4, 116) = 1.01, p = .406.
- All areas: location type F(2, 58) = 4.80, p = .012; HP − next −5.0% (p = .045); next − opposite −7.2% (p = .119);
  HP − opposite −12.2% (p = .029); × item F(4, 116) = 1.83, p = .128.
- Dorsal and occipitotemporal: no reliable location effect (F = 0.87, p = .42; F = 0.58, p = .57).
- HP vs all three LP (two event types per item): early −8.1%, F(1, 29) = 4.52, p = .042; all areas −6.3%, p = .032.
- DCT basis (robustness), early: location type F(2, 58) = 4.17, p = .020; next − opposite −13.4% (p = .015);
  HP − opposite −17.6% (p = .033); HP − next −4.3% (p = .22); × item p = .94. Early HP vs LP −9% (p = .067).
- Per item type, early, HP − next / next − opposite: target −0.5% / −11.8%; distractor −4.9% / −22.7%;
  neutral −9.6% / −7.7% (all p ≥ .055): same direction, none reliable alone.

## Table 1 and LOSO (P; `notes/data/loso_per_step_2026-11.tsv`, `table1_ladder.md`, `figS_loso_per_roi.md`)

Share of 30 subjects predicted better when the term is added with the size from the other 29 (median over ROIs).

| Term | DoG | p | DN | p |
|---|---|---|---|---|
| Items (vs items at bar strength; **see caveat**) | 100% | < .001 | 100% | < .001 |
| Items vs no items (rescored, `open_issues_2026-11.md` a) | 80% | .0013 | 3% | < .001 (worse) |
| Target gain (T) | 90% | < .001 | 97% | < .001 |
| Distractor gain, tied (D) | 50% | .81 | 53% | .84 |
| Sustained gain, tied (S) | 90% | < .001 | 90% | < .001 |
| dS | 77% | .010 | 63% | .23 |
| dD | 63% | .09 | 67% | .08 |
| D given T | pending | | pending | |

Per ROI (figS_loso_per_roi, earlier canonical-HRF fits): dS 77–94% with q < .05 in V2, V3, V3AB, IPS, hV4, VO, TO;
dD q < .05 in V3AB, IPS, hV4, VO (78–83%); D on top of T q < .05 only in hV4 (79%).

## Fig. 5 Gain fields (P; all numbers change after the rerun)

Sources: `notes/data/conditional_estimates_dog_2026-11.tsv` (estimated alone), `conditional_dT_dog_2026-11.tsv`,
`notes/figures/paper_2026-11/table_parameters_2026-11.md` (joint m7_full), Fig. 5 legend.

**a, joint fit (DoG), median per ROI** (* BH q < .05):
- Target: V1 +.005, V2 +.12*, V3 +.30*, V3AB +.30*, IPS +.39*, hV4 +.34*, VO +.35*, LO +.26*, TO +.25*.
- Distractor (averaged over locations): V1 +.03, V2 +.05, V3 +.03, V3AB +.14*, IPS +.18*, hV4 +.21*, VO +.10*, LO +.09, TO +.09.
- Pooled (subject median over ROIs): T +.25 [+.21, +.32], 3% < 0; D +.11 [+.04, +.12], 20% < 0, both p < .001.

**b, field widths (FWHM)**: sustained median 4.8° over all subjects and ROIs, 12% of fits at a σ bound; per ROI 3.7°
(TO) to 6.6° (IPS). Transient: often at a bound (17–73% per ROI; IPS 73%, mostly the upper bound), poorly
constrained. Neighbouring items are 5.7° apart.

**c, dS = g_HP − g_LP, estimated alone** (computed; Wilcoxon, BH over nine ROIs):
V1 −.004 (q = .10), V2 −.004 (q = .019), V3 −.015 (q = .002), V3AB −.011 (q < .001), IPS −.009 (q < .001),
hV4 −.022 (q = .001), VO −.013 (q = .019), LO −.007 (q = .044), TO −.001 (q = .09). Seven of nine ROIs q < .05.
Pooled: median −.009 [−.014, −.006], 80% of subjects < 0, p < .001.
Joint (m7_full): V1 −.042, V2 −.046, V3 −.052 (q = .07), others −.001 to −.023, no ROI q < .05; pooled −.018
[−.032, −.004], 73% < 0, p = .006.

**d, transient HP − LP, estimated alone** (Fig. 5 legend; computed for medians):
- dD (distractor): V3AB −.032 (q = .003), IPS −.037 (q = .039), hV4 −.052 (q < .001), VO −.038 (q = .008); V1–V3,
  LO, TO n.s. Pooled −.031 [−.038, −.013], 83% < 0, p < .001. Joint: −.07 to −.10 in V3AB, IPS, hV4, VO, q ≈ .09–.15;
  pooled −.018 [−.059, +.024], p = .44.
- dT (target): no ROI q < .05 (smallest q = .07, V3; IPS positive trend q = .10); joint q ≥ .36. Pooled (computed)
  median −.013, p = .088.
- dD − dT: no ROI reliable (V3 +.053, q = .094; IPS −.097, q = .094; others q = .855); pooled median −.005, p = .89.

**Along the hierarchy** (`notes/data/hierarchy_regression_2026-11.tsv`, figS_gain_vs_prf_size): target +.24 per
doubling of pRF size [+.15, +.27], 90% of subjects, q < .001; distractor +.06 [+.03, +.11], 77%, q = .01. Target gain
correlates with item weight (ρ = .25, p = .011) and negatively with field size (ρ = −.22, p = .049) within subjects.

## DN robustness of Fig. 5 (P; `figS_gain_fields_dn.md`; canonical-HRF DN fits, final DN fits pending)

- Target (joint DN) larger than DoG everywhere: V1 +.21 to TO +.78, all q < .01.
- dS estimated alone: negative in all nine ROIs (V1 −.007 … VO −.029); q < .05 in V3, V3AB, IPS, hV4, TO; pooled
  median −.014, p = .001. Joint DN pooled −.027 [−.038, −.017], 80% < 0, p = .004.
- dD estimated alone: V3AB −.051, IPS −.058, VO −.071 (q < .01); V1, V3 positive and n.s.
- LOSO dS under DN: 63%, p = .23 (DN constrains dS about four times less; `open_issues_2026-11.md` b).

## Fig. 6 Decoding (F; legend; `notes/data/decoding_*_jointhrf_2026-11.*`, `decoding_prf_models_*_2026-11.tsv`)

- Frame-wise r, decoded vs true (median over subjects; shuffled ≈ 0, all p < 10⁻³ one-sided): V1 .51, V2 .47, V3 .40,
  V3AB .29, IPS .12, hV4 .28, VO .16, LO .25, TO .19. n per ROI: V1–V3 30, V3AB 25, IPS 19, hV4 29, VO 11, LO 29, TO 27.
- Example run: sub-03 ses-1 run-4, V1, r = .62 (rank 1 of 2653); median run r = .30.
- d, HP quadrant − other three quadrants, one-sided, BH over nine ROIs (unconstrained decoder): V2 p = .013,
  V3 .027, VO .034, TO .031, V3AB .043, all q = .076–.077; V1 p = .32 (q = .41); none q < .05. Outline decoder: V2, V3,
  V3AB, TO q = .090.
- **Pre-specified test** (`primary_test_2026-11.md`: V1–V3, one-sided, Holm): computed from the same p values:
  V2 p_Holm = .039, V3 .054, V1 .32. Not reported in the legend (see caveats).
- e, per quadrant, V1–V3 pooled, % of the four-quadrant mean (mean ± within-subject SEM): HP −0.85 ± 0.38%, next to
  HP +0.46 ± 0.18%, opposite −0.08 ± 0.22%. HP < next one-sided p = .015; HP < opposite p = .13; Friedman p = .06;
  left vs right neighbour p = .22. Next vs opposite (two-sided, `decoding_prf_models_quadrants_2026-11.tsv`): p = .16
  (unconstrained), .13 (outline).
- f, by eccentricity (V1–V3): 0–1.5° −0.6% (p = .10); 1.5–2.5° −1.4% (p = .029); 2.5°+ −1.3% (p = .040); whole
  aperture −1.1 ± 0.5% (p = .033); bands do not differ (Friedman p = .53); fovea (< 0.75°) +0.2%.
- c, insets (outline decoder, same stimulus): bar segment in the quadrant facing HP −1.16 ± 0.42% (p = .008); in the
  quadrant next to HP +1.13 ± 0.33% (p = .002).

**Supplements to Fig. 6 (F).**
- Canonical HRF (V1–V3, BH over three): V2 q = .034, V3 q = .034 (outline .029); HP < next p = .008; r .51/.47/.41.
- DN pRFs: r .46/.44/.41 in V1–V3; HP < next p = .057; no ROI q < .05 unconstrained; outline V3 q = .021.
- Next vs opposite across pRF models (`decoding_prf_models_quadrants_2026-11.tsv`): p = .13–.43, never reliable;
  sign: next > opposite in every model.
- Decoded bars, same stimulus: HP-quadrant half minus other half, vertical −1.61 ± 0.66% (p = .031), horizontal
  −1.83 ± 0.51% (p = .001), pooled −1.72% (p = .004); next-to-HP half minus opposite half, pooled +1.04% (p = .047).
  Per ROI only V3 survives BH (−2.71%, q = .011).
- Bar-crossing response (V1–V3, pRF within 2° of an item): HP − LP −0.060% signal (−2.6%), 67% < 0, one-sided
  p = .017; with a pre-crossing baseline −2.1%, p = .16 (n.s.).

## Fig. 7 Gains and behaviour (Proto; legend; `notes/data/trialwise_rt_bayes_summary_davg_2026-11.tsv`)

- a, target gain repeat − switch: V3AB −.40 (q = .020), IPS −.39 (q = .025), hV4 −.40 (q = .025), VO −.34 (q = .009);
  others q ≥ .13. Distractor gain repeat − switch: no ROI q < .05 (IPS −.32, p = .008, q = .073).
- b, Bayesian hierarchical model of log RT (z within subject; slope per SD of gain; 95% HDI; P = posterior
  probability of the sign; not corrected across ROIs):
  - Target gain → slower: V3AB +.030 (P = .995), hV4 +.031 (P = .996), LO +.022 (P = .988).
  - Distractor gain on HP trials → faster: V1 −.036 (P = .994), V2 −.037 (P = .998); on LP trials V2 −.034 (P = .983).
    HP − LP difference reliable only in IPS (+.046, P = .98; uncorrected).
  - Global gain → faster in V2 −.027 (P = .979), V3 −.036 (P = .998); → slower in IPS +.070 (P = .998).
  - Covariates: target repeat −.21 SD, distractor repeat −.06 SD of log RT in every ROI model.
- Between subjects (text only): dS vs behavioural HP benefit ρ = −.17, p = .37; no ROI survives correction.
  Run-level learning: none (sustained HP − LP slope within block p = .86, across blocks p = .17).

## Sensitivity (P)

- Within-subject CV detects dS only near −.05, about three times our estimate (figS_cv_power).
- Per-subject dS noise SD ≈ .03 (DoG; .066 under DN) against an effect of ≈ .01–.02.

## Exploratory (X)

- DN parameters per HP condition (`notes/data/dn_by_hp_stats_2026-11.tsv`, n = 28 in V1–V3): HP-quadrant vs
  opposite-quadrant pRFs larger σ, lower peak amplitude and higher surround amplitude in V1–V3 (BH over ROIs
  q ≈ .01–.10, V2 the strongest); neural and surround baselines n.s. except LO (q = .08). Friedman tests across the
  four quadrants mostly n.s. Do not claim.
- DN baseline-field variants (canary, sub-03): b- or d-field alone does nothing; together they move along the
  amplitude direction and trade off against the input gain. Full results pending.
