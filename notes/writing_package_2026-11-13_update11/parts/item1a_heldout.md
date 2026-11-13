# Item 1a: do the transient HP − LP terms improve held-out prediction?

**Bottom line.** Held-out prediction does not favour any HP − LP term, including the sustained dS that the paper
reports. Every added term makes the held-out run slightly worse:
- dS vs none: 3/28 participants better, median ΔR² −0.29 × 10⁻⁴;
- adding dD to dS: 0/28 better, −0.89 × 10⁻⁴;
- adding dT to dS + dD: 1/28 better, −0.81 × 10⁻⁴.

24 of 28 participants predict best with no HP terms at all. In sample these terms explain only about 0.2–0.3 × 10⁻⁴
of the variance each, less than the cost of estimating them from the other runs. So leave-one-run-out
cross-validation cannot decide whether the transient terms belong in the model. It says the same about dS, which is
significant across participants. This agrees with the earlier simulation of the held-out test
(`notes/data/cv_power_simulation_2026-11.tsv`: within-participant CV leaving out one HP condition detects a planted dS only at about three times
its estimate). What the transient terms add has to be judged from the group-level estimates and their power (item 1c),
not from held-out R².

Models: all four share the base terms (item weight, target gain, distractor gain, sustained gain at all four
locations).
- **none:** base terms only, no HP terms (the m4_TDS term set);
- **dS:** + the sustained HP − LP term;
- **dS + dD:** + the distractor transient HP − LP term (the HP − LP terms of the joint fit, m7_full);
- **dS + dD + dT:** + the target transient HP − LP term (m8_full_dT).

Leave-one-run-out cross-validation (12 folds; 11 for sub-20 and sub-24): in each fold the base terms (item weight, target gain, distractor gain, sustained gain at all four locations) and the HP − LP terms of each model were fitted on the other runs and the held-out run was predicted. pRFs and field widths: the all-runs fit of the model with item weight, target, distractor and sustained gain (m4_TDS) in the reference root (pRFs refitted together with the gain fields, update 8, `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`; that polish was not run to full convergence), identical in every model, so their leak into the held-out run is shared. The HP − LP terms enter linearly (least squares around that fit). Held-out R² per participant × area = 1 − ΣSSE / ΣSST over the held-out runs and voxels; ΔR² in units of 10⁻⁴ (0.01 percentage points of variance). Per area: median over participants [bootstrapped 95% CI], participants with ΔR² > 0, Wilcoxon p, BH q across nine areas; pooled: per participant the median over the nine areas; pathway: median over its areas, BH across three.

Held-out R² of the model without HP terms (base): median over participants × areas 0.250.

## Sustained HP − LP (dS) vs no HP terms (ΔR² = R²[dS] − R²[none], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.50 [-1.14, -0.22] | 4/28 | .0013 | .012 |
| V2 | -0.33 [-0.67, -0.11] | 7/28 | .060 | .089 |
| V3 | -0.15 [-0.67, +0.20] | 12/28 | .36 | .41 |
| V3AB | -0.16 [-1.35, +0.08] | 9/27 | .055 | .089 |
| IPS | -0.61 [-1.64, -0.28] | 4/25 | .0061 | .018 |
| hV4 | -0.07 [-0.43, +0.04] | 10/28 | .19 | .25 |
| VO | -0.46 [-0.83, +1.18] | 8/21 | .81 | .81 |
| LO | -0.19 [-0.37, -0.10] | 7/28 | .010 | .023 |
| TO | -0.56 [-1.16, -0.09] | 7/28 | .0041 | .018 |
| **Pooled** | -0.29 [-0.58, -0.21] | 3/28 | 6.7 × 10⁻⁶ | n/a |
| Early (pathway) | -0.28 [-0.51, -0.11] | 5/28 | 3.4 × 10⁻⁴ | .001 |
| Dorsal (pathway) | -0.30 [-1.43, -0.08] | 8/27 | .020 | .020 |
| Occipitotemporal (pathway) | -0.28 [-0.49, -0.18] | 5/28 | .0022 | .0034 |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 3 of 28 participants (median ΔR² -0.29 × 10⁻⁴, p = 6.7 × 10⁻⁶); areas with q < .05: V1, IPS, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## Adding the distractor transient HP − LP (dD) to dS (ΔR² = R²[dS + dD] − R²[dS], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.27 [-0.78, +0.13] | 10/28 | .27 | .27 |
| V2 | -0.52 [-0.90, -0.25] | 4/28 | 3.7 × 10⁻⁵ | 6.6 × 10⁻⁵ |
| V3 | -0.43 [-0.90, -0.22] | 5/28 | .002 | .0023 |
| V3AB | -1.11 [-1.37, -0.38] | 5/27 | 3.5 × 10⁻⁵ | 6.6 × 10⁻⁵ |
| IPS | -2.43 [-4.16, -1.93] | 3/25 | 2.7 × 10⁻⁵ | 6.6 × 10⁻⁵ |
| hV4 | -0.53 [-0.91, -0.38] | 2/28 | 2.5 × 10⁻⁷ | 1.1 × 10⁻⁶ |
| VO | -2.47 [-3.71, -1.76] | 3/21 | 3.5 × 10⁻⁴ | 4.5 × 10⁻⁴ |
| LO | -1.09 [-2.10, -0.30] | 4/28 | 9.3 × 10⁻⁵ | 1.4 × 10⁻⁴ |
| TO | -0.69 [-1.25, -0.36] | 1/28 | 1.9 × 10⁻⁷ | 1.1 × 10⁻⁶ |
| **Pooled** | -0.89 [-1.32, -0.47] | 0/28 | 7.5 × 10⁻⁹ | n/a |
| Early (pathway) | -0.48 [-1.14, -0.18] | 5/28 | .0017 | .0017 |
| Dorsal (pathway) | -1.58 [-2.94, -1.22] | 4/27 | 1.3 × 10⁻⁶ | 2.0 × 10⁻⁶ |
| Occipitotemporal (pathway) | -1.02 [-1.59, -0.73] | 1/28 | 1.5 × 10⁻⁸ | 4.5 × 10⁻⁸ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 0 of 28 participants (median ΔR² -0.89 × 10⁻⁴, p = 7.5 × 10⁻⁹); areas with q < .05: V2, V3, V3AB, IPS, hV4, VO, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## Adding the target transient HP − LP (dT) to dS + dD (ΔR² = R²[dS + dD + dT] − R²[dS + dD], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.59 [-0.88, -0.40] | 5/28 | .0044 | .0057 |
| V2 | -0.52 [-0.61, -0.20] | 6/28 | 5.3 × 10⁻⁴ | .0011 |
| V3 | -0.45 [-0.76, -0.16] | 4/28 | 7.2 × 10⁻⁵ | 3.2 × 10⁻⁴ |
| V3AB | -0.85 [-1.21, -0.56] | 3/27 | 6.0 × 10⁻⁴ | .0011 |
| IPS | -2.17 [-3.08, -1.44] | 4/25 | .0015 | .0022 |
| hV4 | -0.36 [-0.99, -0.07] | 7/28 | .077 | .087 |
| VO | -1.56 [-2.45, -0.86] | 5/21 | .12 | .12 |
| LO | -1.17 [-1.96, -0.66] | 0/28 | 7.5 × 10⁻⁹ | 6.7 × 10⁻⁸ |
| TO | -0.78 [-1.18, -0.49] | 6/28 | 1.2 × 10⁻⁴ | 3.6 × 10⁻⁴ |
| **Pooled** | -0.81 [-1.02, -0.56] | 1/28 | 1.5 × 10⁻⁸ | n/a |
| Early (pathway) | -0.53 [-0.61, -0.35] | 5/28 | 1.5 × 10⁻⁴ | 1.5 × 10⁻⁴ |
| Dorsal (pathway) | -1.36 [-1.68, -0.98] | 4/27 | 8.4 × 10⁻⁵ | 1.3 × 10⁻⁴ |
| Occipitotemporal (pathway) | -0.85 [-1.34, -0.48] | 2/28 | 1.9 × 10⁻⁷ | 5.6 × 10⁻⁷ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 1 of 28 participants (median ΔR² -0.81 × 10⁻⁴, p = 1.5 × 10⁻⁸); areas with q < .05: V1, V2, V3, V3AB, IPS, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## Adding both transient terms (dD + dT) to dS (ΔR² = R²[dS + dD + dT] − R²[dS], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.79 [-1.27, -0.30] | 6/28 | .022 | .022 |
| V2 | -1.13 [-1.62, -0.46] | 2/28 | 3.3 × 10⁻⁶ | 1.2 × 10⁻⁵ |
| V3 | -0.56 [-1.56, -0.29] | 4/28 | 1.2 × 10⁻⁴ | 1.5 × 10⁻⁴ |
| V3AB | -1.61 [-2.21, -0.82] | 2/27 | 5.5 × 10⁻⁶ | 1.2 × 10⁻⁵ |
| IPS | -4.68 [-8.11, -2.67] | 3/25 | 1.0 × 10⁻⁴ | 1.5 × 10⁻⁴ |
| hV4 | -0.92 [-2.16, -0.55] | 3/28 | 8.0 × 10⁻⁶ | 1.4 × 10⁻⁵ |
| VO | -3.77 [-5.09, -1.18] | 5/21 | .0016 | .0018 |
| LO | -2.35 [-3.89, -1.71] | 2/28 | 3.7 × 10⁻⁸ | 3.4 × 10⁻⁷ |
| TO | -1.36 [-2.15, -1.08] | 3/28 | 4.8 × 10⁻⁶ | 1.2 × 10⁻⁵ |
| **Pooled** | -1.64 [-1.97, -1.01] | 1/28 | 2.2 × 10⁻⁸ | n/a |
| Early (pathway) | -0.75 [-1.58, -0.36] | 3/28 | 6.3 × 10⁻⁵ | 6.3 × 10⁻⁵ |
| Dorsal (pathway) | -2.95 [-4.23, -2.33] | 2/27 | 3.1 × 10⁻⁶ | 4.6 × 10⁻⁶ |
| Occipitotemporal (pathway) | -1.96 [-3.13, -1.15] | 1/28 | 2.2 × 10⁻⁸ | 6.7 × 10⁻⁸ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 1 of 28 participants (median ΔR² -1.64 × 10⁻⁴, p = 2.2 × 10⁻⁸); areas with q < .05: V1, V2, V3, V3AB, IPS, hV4, VO, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## All three HP − LP terms vs no HP terms (ΔR² = R²[dS + dD + dT] − R²[none], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -1.11 [-1.93, -0.65] | 5/28 | .0071 | .008 |
| V2 | -0.77 [-1.86, -0.57] | 3/28 | 1.3 × 10⁻⁵ | 3.9 × 10⁻⁵ |
| V3 | -0.73 [-2.04, -0.16] | 8/28 | .0071 | .008 |
| V3AB | -2.09 [-3.56, -0.81] | 4/27 | 5.5 × 10⁻⁵ | 1.2 × 10⁻⁴ |
| IPS | -5.51 [-7.24, -3.65] | 4/25 | 3.8 × 10⁻⁴ | 5.6 × 10⁻⁴ |
| hV4 | -1.34 [-2.44, -0.53] | 5/28 | 1.5 × 10⁻⁴ | 2.7 × 10⁻⁴ |
| VO | -3.12 [-5.17, -2.03] | 4/21 | .024 | .024 |
| LO | -2.81 [-4.18, -1.89] | 2/28 | 4.1 × 10⁻⁷ | 3.7 × 10⁻⁶ |
| TO | -1.67 [-3.81, -1.15] | 3/28 | 1.5 × 10⁻⁶ | 6.9 × 10⁻⁶ |
| **Pooled** | -1.91 [-2.96, -1.33] | 2/28 | 3.7 × 10⁻⁸ | n/a |
| Early (pathway) | -0.90 [-1.94, -0.62] | 4/28 | 1.1 × 10⁻⁵ | 1.1 × 10⁻⁵ |
| Dorsal (pathway) | -3.61 [-6.39, -2.09] | 2/27 | 9.5 × 10⁻⁶ | 1.1 × 10⁻⁵ |
| Occipitotemporal (pathway) | -2.10 [-3.51, -1.70] | 2/28 | 3.7 × 10⁻⁸ | 1.1 × 10⁻⁷ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 2 of 28 participants (median ΔR² -1.91 × 10⁻⁴, p = 3.7 × 10⁻⁸); areas with q < .05: V1, V2, V3, V3AB, IPS, hV4, VO, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## Model with the best held-out prediction per participant (median over areas of R² − R²[base])

| Model | Participants |
|---|---|
| No HP terms (item weight, target, distractor, sustained gain) | 24 |
| Sustained HP − LP only (dS) | 2 |
| dS + distractor transient HP − LP (dD) (the HP terms of the joint fit) | 0 |
| dS + dD + target transient HP − LP (dT) | 2 |

*Summary:* how the participants split over the four models; a model counts for a participant when its median held-out gain over the nine areas is the largest.

## For scale: in-sample R² gain of the same terms (all runs; ≥ 0 by construction)

| Contrast | Median over participants of the pooled in-sample ΔR² (× 10⁻⁴) |
|---|---|
| Sustained HP − LP (dS) vs no HP terms | +0.29 |
| Adding the distractor transient HP − LP (dD) to dS | +0.23 |
| Adding the target transient HP − LP (dT) to dS + dD | +0.24 |
| Adding both transient terms (dD + dT) to dS | +0.82 |
| All three HP − LP terms vs no HP terms | +1.40 |

*Summary:* in sample each added term explains only about 10⁻⁴ of the variance or less; held out, that gain is smaller than the cost of estimating the term, so held-out prediction cannot separate these models.

Sources: `notes/data/hp_terms_loro_2026-11-13.tsv` (per participant × area × held-out run × model: held-out SSE and SST, fold coefficients), `notes/data/hp_terms_heldout_r2_2026-11-13.tsv` (held-out R² per participant × area × model), `notes/data/hp_terms_heldout_stats_2026-11-13.tsv` (every number in the tables), `notes/data/hp_terms_insample_dr2_2026-11-13.tsv` (in-sample ΔR²). Code: `retsupp/modeling/hp_terms_heldout_power.py` (cluster, `retsupp/modeling/slurm_jobs/hp_terms_heldout_power.sh`, output `derivatives/hp_terms_heldout_power_v2_2026-11-13`), `retsupp/visualize/paper/hp_terms_heldout_power_stats.py` (tables).
