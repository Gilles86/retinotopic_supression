# Item 2: does the sustained field alone explain the item responses at HP? (n = 28)

**Question.** Fig. 3 shows, in early areas, less response at the HP location than at the LP locations, and about equally
for the target, the distractor and a neutral item. Is that what the sustained HP − LP gain alone predicts? Or do the
transient HP − LP terms (distractor, dD; target, dT) need to be added?

**Short answer.** The sustained field alone (dS) is enough. In early areas it predicts about two thirds of the measured
HP − LP reduction (−0.029 vs −0.045 % signal, items pooled). It predicts the same reduction for every item type, as
the data show. What it leaves unexplained is not significant (−0.027, p = .14).

The joint fit as fitted also contains a distractor-transient HP − LP term (dD), and in early areas that term is
positive. So the joint fit predicts no HP reduction for the distractor (−0.002), and a target − distractor interaction
(−0.025, p = .043) that the data do not show (−0.001, p = .54). For the distractor, the joint fit fits worse than dS
alone (measured − model −0.075, p = .025 vs −0.028, p = .099).

Without HP terms the predicted HP − LP is 0 (|median| ≤ 0.007 % signal), so the voxel selection and the design do not
create the effect.

The location × item-type interaction is not significant in any pathway or early area, in the data or after removing
either model.

## Data and method

**Models.** All three come from the joint gain-field fit m7_full (DoG pRF; item weight, target, distractor and
sustained gains; pRFs refitted together with the gains). Its root is
`derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`. The joint polish is **not fully
converged**; a convergence run is in progress elsewhere. m7_full has dS and dD but no dT, so variant (i) is the joint
fit with a term set equal, not the ladder model without the terms:
- **(i) sustained HP − LP only (S):** m7_full with dD set to 0. g_HP_dyn and g_LP_dyn are both set to their mean,
  weighted by that participant's distractor pulses at HP vs LP, so the average distractor gain is unchanged.
- **(ii) joint fit (SD):** m7_full as fitted, with dS and dD.
- **(iii) no HP terms (nohp):** as S, and the sustained gain set to the mean over the four locations,
  (g_HP + 3·g_LP)/4.
- **Supplement (SDT, SDT0):** the joint fit plus the target-transient HP − LP term, dT (m8_full_dT). This model exists
  only with the pRFs held fixed (`..._reliable_roihrf`) and only for some participant × area fits. SDT0 is the same
  parameters with dT equalised.

**Pipeline.** The noise-free predicted BOLD goes through the same steps as the measured refit data of Fig. 3 (the
refit selection, `event_deconv_relative_polished_2026-11-12`, which matches the Fig. 3 numbers to within 1–2%):
- voxels reliable across runs, with a refitted pRF centre within 2° of an item location;
- minus the prediction for the bar alone (the same pRFs with every gain at 0);
- averaged per location, then nideconv with twelve event types (item × HP / next left / next right / opposite);
- the window 3.2–6.4 s, pre-onset subtracted.

The bar-alone prediction and the voxel counts reproduce the measured export exactly (sub-03 check: max |Δbar| =
5 × 10⁻⁵, voxel counts identical).

**Contrasts.** HP − LP per item type, with LP the mean of next left, next right and opposite.
- **Absolute (% signal):** main tables, because model and data are on the same scale.
- **Relative (% of the participant's mean over the twelve cells, Fig. 3e units):** at the end. A model's own mean
  response is the denominator, so a small predicted mean inflates the relative numbers.

**Statistics.** As in key_numbers:
- measured: median [bootstrapped 95% CI], k of n, Wilcoxon p;
- per area (V1, V2, V3): BH q across the three areas;
- model columns: median, k of n and p of the noise-free prediction. The p only shows whether the prediction is
  consistent across participants; it is not a test of the data;
- "measured − X": per participant, measured minus predicted;
- interaction: repeated-measures ANOVA location (HP, LP) × item type, on the measured cells and on measured − model.

Dorsal n = 26 (two participants lack dorsal voxels at some item location).

## Table 2a. HP − LP per item type, measured vs predicted (% signal)

| Group | Item | Measured | S predicted (dS only) | SD predicted (joint fit) | No HP terms | Measured − S | Measured − SD |
|---|---|---|---|---|---|---|---|
| Early | target | −0.055 [−0.106, −0.004], 20/28 < 0, p = .16 | −0.029, 20/28, p = .0032 | −0.033, 22/28, p = .0034 | +0.000 | −0.050, 17/28, p = .55 | −0.051, 17/28, p = .55 |
| Early | distractor | −0.041 [−0.116, −0.015], 21/28 < 0, p = .0071 | −0.028, 21/28, p = .0034 | **−0.002**, 14/28, p = .68 | +0.001 | −0.028, 16/28, p = .099 | **−0.075, 18/28, p = .025** |
| Early | neutral | −0.054 [−0.082, −0.017], 22/28 < 0, p = .0032 | −0.026, 20/28, p = .0071 | −0.029, 20/28, p = .0041 | +0.000 | −0.026, 20/28, p = .090 | −0.029, 20/28, p = .077 |
| Early | pooled | **−0.045 [−0.088, −0.012], 20/28 < 0, p = .0095** | **−0.029**, 20/28, p = .0041 | −0.018, 19/28, p = .017 | +0.000 | **−0.027, 16/28, p = .14** | −0.029, 18/28, p = .066 |
| Dorsal | target | −0.077 [−0.205, +0.030], 17/26, p = .075 | −0.020 | −0.026 | +0.000 | −0.067, p = .17 | −0.067, p = .16 |
| Dorsal | distractor | −0.083 [−0.160, +0.025], 17/26, p = .063 | −0.024 | −0.015 | +0.001 | −0.041, p = .16 | −0.092, p = .080 |
| Dorsal | neutral | −0.067 [−0.107, +0.030], 15/26, p = .24 | −0.021 | −0.023 | +0.007 | −0.021, p = .53 | −0.020, p = .53 |
| Dorsal | pooled | −0.040 [−0.141, −0.004], 18/26 < 0, p = .038 | −0.023 | −0.027 | +0.002 | −0.030, p = .23 | −0.051, p = .12 |
| Occ.-temporal | target | +0.003 [−0.182, +0.057], 15/28, p = .58 | −0.029 | −0.028 | +0.002 | +0.026, p = .94 | +0.019, p = .96 |
| Occ.-temporal | distractor | −0.075 [−0.181, +0.009], 18/28, p = .095 | −0.034 | −0.017 | +0.001 | −0.044, p = .33 | −0.038, p = .13 |
| Occ.-temporal | neutral | −0.019 [−0.103, +0.062], 17/28, p = .85 | −0.022 | −0.026 | +0.004 | −0.002, p = .78 | +0.002, p = .76 |
| Occ.-temporal | pooled | −0.033 [−0.149, +0.029], 16/28, p = .34 | −0.030 | −0.020 | +0.001 | +0.008, p = .76 | −0.027, p = .54 |
| V1 | target | −0.033 [−0.083, +0.009], 19/28, p = .20, q = .60 | −0.010 | −0.017 | −0.001 | −0.033, p = .35 | −0.036, p = .44 |
| V1 | distractor | −0.078 [−0.108, −0.035], 22/28, p = .032, q = .048 | −0.011 | +0.006 | +0.001 | −0.059, p = .030, q = .089 | −0.067, p = .030, q = .045 |
| V1 | neutral | −0.081 [−0.124, −0.015], 20/28, p = .034, q = .051 | −0.010 | −0.011 | +0.001 | −0.085, p = .040, q = .12 | −0.084, p = .045, q = .14 |
| V1 | pooled | −0.057 [−0.092, −0.016], 22/28, p = .022, q = .065 | −0.009 | −0.003 | −0.000 | −0.061, p = .048, q = .14 | −0.059, p = .045, q = .14 |
| V2 | target | −0.052 [−0.124, +0.064], 17/28, p = .55, q = .63 | −0.029 | −0.031 | +0.001 | −0.042, p = .60 | −0.043, p = .58 |
| V2 | distractor | −0.043 [−0.118, −0.008], 20/28, p = .020, q = .048 | −0.025 | −0.017 | −0.000 | −0.028, p = .21 | −0.047, p = .022, q = .045 |
| V2 | neutral | −0.045 [−0.075, −0.003], 21/28, p = .012, q = .036 | −0.023 | −0.024 | +0.002 | −0.029, p = .18 | −0.029, p = .18 |
| V2 | pooled | −0.039 [−0.089, +0.004], 19/28, p = .053, q = .080 | −0.027 | −0.015 | +0.001 | −0.028, p = .32 | −0.038, p = .19 |
| V3 | target | −0.069 [−0.135, +0.091], 15/28, p = .63, q = .63 | −0.054 | −0.054 | +0.001 | +0.032, p = .44 | +0.030, p = .41 |
| V3 | distractor | −0.035 [−0.168, +0.045], 17/28, p = .25, q = .25 | −0.057 | −0.008 | −0.001 | −0.014, p = .63 | −0.055, p = .40 |
| V3 | neutral | −0.022 [−0.072, +0.057], 15/28, p = .88, q = .88 | −0.052 | −0.052 | +0.001 | +0.023, p = .17 | +0.023, p = .15 |
| V3 | pooled | −0.042 [−0.130, +0.057], 16/28, p = .51, q = .51 | −0.054 | −0.052 | −0.000 | +0.013, p = .52 | +0.003, p = .65 |

Measured − model q values are BH across V1–V3 and are given only where p < .05.

**In plain words:** in early areas the sustained HP − LP gain alone predicts a uniform HP reduction of about two thirds
of the measured one, and the rest is not significant. The joint fit's positive distractor-transient term cancels the
predicted reduction for the distractor, and the data do not show that. In V1 the measured reduction (−0.057) is larger
than either model predicts (−0.009 / −0.003); the shortfall is p ≈ .05 and not significant after BH (q = .14).

## Table 2b. Location × item type: the interaction contrasts (differences of HP − LP, % signal)

| Group | Contrast | Measured | S predicted | SD predicted | Measured − S | Measured − SD |
|---|---|---|---|---|---|---|
| Early | target − neutral | −0.018 [−0.057, +0.086], p = .55 | −0.002 | −0.002 | −0.008, p = .52 | −0.004, p = .54 |
| Early | distractor − neutral | −0.008 [−0.056, +0.043], p = .57 | −0.002 | +0.024, 20/28 > 0, p = .060 | −0.006, p = .69 | −0.019, p = .15 |
| Early | target − distractor | −0.001 [−0.036, +0.050], p = .54 | +0.000 | **−0.025, 22/28 < 0, p = .043** | −0.004, p = .45 | **+0.032, p = .048** |
| Dorsal | target − neutral | −0.006 [−0.114, +0.043], p = .53 | −0.003 | −0.004 | +0.005, p = .71 | +0.004, p = .67 |
| Dorsal | distractor − neutral | −0.010 [−0.131, +0.034], p = .35 | −0.001 | +0.019 | +0.008, p = .50 | −0.029, p = .30 |
| Dorsal | target − distractor | −0.010 [−0.051, +0.022], p = .78 | +0.001 | −0.018 | −0.014, p = .98 | +0.009, p = .84 |
| Occ.-temporal | target − neutral | −0.037 [−0.169, +0.044], p = .20 | −0.004 | −0.003 | −0.041, p = .23 | −0.042, p = .22 |
| Occ.-temporal | distractor − neutral | −0.029 [−0.191, +0.033], p = .077 | −0.003 | −0.002 | −0.021, p = .10 | −0.059, p = .015 |
| Occ.-temporal | target − distractor | +0.027 [−0.013, +0.060], p = .25 | −0.000 | +0.009 | +0.025, p = .24 | +0.025, p = .37 |
| V1 | target − neutral | +0.071 [−0.005, +0.119], p = .086, q = .26 | +0.001 | +0.000 | +0.065, q = .26 | +0.061, q = .26 |
| V1 | distractor − neutral | −0.010 [−0.057, +0.051], p = .87, q = .92 | +0.000 | +0.014 | −0.011, q = .99 | −0.043, q = .35 |
| V1 | target − distractor | +0.058 [+0.002, +0.119], p = .030, q = .089 | +0.002 | −0.013 | +0.073, p = .025, q = .074 | +0.104, q = .20 |
| V2 | target − neutral | +0.013 [−0.097, +0.156], p = .34 | +0.000 | −0.001 | +0.010, q = .26 | +0.011, q = .31 |
| V2 | distractor − neutral | −0.030 [−0.061, +0.066], p = .92 | −0.004 | +0.010 | −0.025, q = .99 | −0.035, q = .35 |
| V2 | target − distractor | −0.031 [−0.058, +0.063], p = .73 | +0.002 | −0.007 | −0.034, q = .68 | −0.019, q = .20 |
| V3 | target − neutral | −0.033 [−0.095, +0.008], p = .18 | −0.005 | −0.005 | −0.028, q = .26 | −0.027, q = .26 |
| V3 | distractor − neutral | −0.079 [−0.161, +0.060], p = .12 | −0.003 | +0.020 | −0.079, q = .36 | −0.109, p = .019, q = .057 |
| V3 | target − distractor | +0.020 [−0.081, +0.073], p = .63 | −0.002 | −0.028 | +0.019, q = .68 | +0.072, q = .20 |

Repeated-measures ANOVA, location (HP, LP) × item type, interaction term (% signal):

| Group | Measured | S predicted | SD predicted | Measured − S | Measured − SD |
|---|---|---|---|---|---|
| Early | F(2,54) = 1.47, p = .24 | F = 0.18, p = .84 | F = 1.95, p = .15 | F = 1.57, p = .22 | F = 2.98, p = .059 |
| Dorsal | F(2,50) = 0.94, p = .40 | F = 4.05, p = .023 | F = 0.37, p = .70 | F = 0.70, p = .50 | F = 1.01, p = .37 |
| Occ.-temporal | F(2,54) = 2.67, p = .079 | F = 2.84, p = .067 | F = 0.40, p = .67 | F = 2.38, p = .10 | F = 2.97, p = .060 |
| V1 | F(2,54) = 2.82, p = .068 | F = 0.16, p = .85 | F = 0.43, p = .65 | F = 2.90, p = .064 | F = 2.45, p = .096 |
| V2 | F(2,54) = 1.17, p = .32 | F = 0.11, p = .89 | F = 0.96, p = .39 | F = 1.24, p = .30 | F = 2.10, p = .13 |
| V3 | F(2,54) = 2.33, p = .11 | F = 4.31, p = .018 | F = 1.67, p = .20 | F = 2.22, p = .12 | F = 4.55, p = .015 |

In relative units (Fig. 3e) the measured interaction is p ≥ .18 everywhere. Early: F(2,54) = 0.66, p = .52, as in the
Fig. 3 legend.

The S-predicted interaction is a few thousandths of a % signal (Table 2b). Its F is "significant" in dorsal and V3 only
because a noise-free prediction has very little spread across participants.

**In plain words:** the data show no location × item-type interaction in any pathway or early area. The sustained-only
model predicts none. The joint fit predicts a small one in early areas (less HP reduction for the distractor than for
the target), and the data contradict it (measured − joint fit, target − distractor +0.032, p = .048). One uncorrected
exception is V1 target − distractor (+0.058, p = .030, q = .089). There the target shows less HP reduction than the
distractor, which is the opposite of the joint fit's direction, and it does not survive BH.

## Table 2c. Relative units (Fig. 3e), early areas and pathways, items pooled and per item

| Group | Item | Measured | S predicted | SD predicted | Measured − S | Measured − SD |
|---|---|---|---|---|---|---|
| Early | pooled | −10.9 [−23.0, −2.6], 20/28 < 0, p = .0089 | −15.0 | −7.6 | −1.2, p = .58 | −4.8, p = .15 |
| Early | target | −15.2 [−23.5, −0.8], p = .34 | −14.7 | −13.0 | −2.7, p = .99 | −3.8, p = .97 |
| Early | distractor | −13.0 [−25.6, −4.5], p = .016 | −12.6 | −0.8 | −5.1, p = .35 | −12.5, p = .074 |
| Early | neutral | −10.9 [−24.6, −4.4], p = .0017 | −14.2 | −14.0 | −7.4, p = .24 | −9.9, p = .22 |
| Dorsal | pooled | −10.6 [−20.0, −0.8], 18/26, p = .053 | −2.4 | −5.3 | −3.5, p = .26 | −6.7, p = .19 |
| Occ.-temporal | pooled | −5.8 [−15.7, +3.6], 16/28, p = .34 | −5.7 | −4.5 | +2.2, p = .96 | −3.7, p = .40 |

**In plain words:** in Fig. 3e's units, the sustained-only model predicts the early HP reduction in full (−15% vs
−11% measured, residual p = .58), including the distractor (−12.6% vs −13.0%). The joint fit predicts almost none for
the distractor (−0.8%).

## Table 2d (supplement). Adding the target-transient HP − LP term, dT (pRFs held fixed; % signal)

The m8_full_dT fits exist only for some participant × area pairs, so every column uses only those pairs: early n = 27,
dorsal 23, occipitotemporal 25.

| Group | Item | Measured (same pairs) | Joint fit (SD) | Joint + dT (SDT) | SDT with dT equalised (SDT0) | Measured − SDT |
|---|---|---|---|---|---|---|
| Early | target | −0.045 [−0.093, −0.001], p = .29 | −0.032 | −0.007 | −0.026 | −0.047, p = .63 |
| Early | distractor | −0.037 [−0.101, −0.013], p = .014 | +0.005 | +0.005 | +0.007 | −0.063, p = .046 |
| Early | neutral | −0.056 [−0.077, −0.010], p = .0082 | −0.019 | −0.015 | −0.017 | −0.042, p = .13 |
| Dorsal | target | −0.060 [−0.205, +0.039], p = .20 | −0.028 | −0.036 | −0.017 | −0.067, p = .26 |
| Occ.-temporal | target | +0.031 [−0.228, +0.119], p = .58 | −0.032 | −0.044 | −0.004 | +0.049, p = 1.0 |

**In plain words:** adding dT does not bring the model closer to the data. In early areas the fitted dT removes most of
the predicted HP reduction for the target (−0.007 vs −0.026 without it), while the measured target reduction stays
(−0.045).

## Sources

- Code:
  - `retsupp/visualize/paper/predict_location_timeseries_polished.py` (predictions, the variants, voxel counts);
  - `retsupp/visualize/paper/slurm_jobs/predict_location_timeseries_polished.sh` (cluster, one task per participant,
    then both deconvolutions);
  - `retsupp/visualize/paper/items_model_vs_measured.py --part hp` (statistics, local).
- Tables:
  - `notes/data/items_hp_model_vs_measured_n28_2026-11-13.tsv` (every group × contrast × source × unit; also the
    rows for the dT pairs, `measured@SDT`, `SD@SDT`);
  - `notes/data/items_hp_model_vs_measured_anova_n28_2026-11-13.tsv` (ANOVAs; main effect of location too).
- Deconvolved predictions:
  - `notes/data/event_deconv_polished_pred_2026-11-13_relative_{nohp,S,SD,SDT,SDT0}/`;
  - cluster: `derivatives/paper_2026-11/location_timeseries_polished_pred_2026-11-13_*`.
- Measured: `notes/data/event_deconv_relative_polished_2026-11-12/`.
- Fitted terms per participant × area: `notes/data/polished_pred_overlap_2026-11-13/sub-XX_terms.tsv`. Joint-fit
  medians (fractions): dS V1 −0.079, V2 −0.077, V3 −0.095; dD V1 +0.103, V2 +0.029, V3 +0.116, hV4 +0.107.
