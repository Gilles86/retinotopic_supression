# Item 1a: do the transient HP − LP terms improve held-out prediction?

Leave-one-run-out cross-validation (12 folds; 11 for sub-20 and sub-24): in each fold the base terms (item weight, target gain, distractor gain; the zero-sum model has no sustained level shared by the four locations) and the HP − LP terms of each model (dS in the zero-sum direction, g_HP + ¾·dS, g_LP − ¼·dS) were fitted on the other runs and the held-out run was predicted. pRFs and field widths: the all-runs fit of the zero-sum main model (m5_TDS_dS_zs, pRFs and gain fields fitted jointly by LM, `..._zs_lm_ref` m7_full), identical in every model, so their leak into the held-out run is shared. The HP − LP terms enter linearly (least squares around that fit). Held-out R² per participant × area = 1 − ΣSSE / ΣSST over the held-out runs and voxels; ΔR² in units of 10⁻⁴ (0.01 percentage points of variance). Per area: median over participants [bootstrapped 95% CI], participants with ΔR² > 0, Wilcoxon p, BH q across nine areas; pooled: per participant the median over the nine areas; pathway: median over its areas, BH across three.

Held-out R² of the model without HP terms (base): median over participants × areas 0.250.

## Sustained HP − LP (dS) vs no HP terms (ΔR² = R²[dS] − R²[none], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.31 [-0.96, +0.43] | 12/28 | .75 | .90 |
| V2 | -0.03 [-0.48, +0.31] | 14/28 | .85 | .90 |
| V3 | +0.12 [-0.54, +0.92] | 15/28 | .33 | .81 |
| V3AB | -0.14 [-0.74, +0.51] | 13/27 | .80 | .90 |
| IPS | -0.40 [-0.92, +0.31] | 8/25 | .35 | .81 |
| hV4 | -0.01 [-0.16, +0.15] | 14/28 | .90 | .90 |
| VO | -0.19 [-0.81, +1.91] | 9/21 | .45 | .81 |
| LO | -0.06 [-0.38, +0.17] | 13/28 | .45 | .81 |
| TO | -0.25 [-0.93, +0.08] | 11/28 | .24 | .81 |
| **Pooled** | -0.11 [-0.24, +0.00] | 10/28 | .099 | n/a |
| Early (pathway) | -0.03 [-0.31, +0.35] | 14/28 | .78 | .95 |
| Dorsal (pathway) | +0.17 [-1.16, +0.45] | 16/27 | .95 | .95 |
| Occipitotemporal (pathway) | -0.06 [-0.23, +0.08] | 14/28 | .51 | .95 |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 10 of 28 participants (median ΔR² -0.11 × 10⁻⁴, p = .099); areas with q < .05: none; pathways with q < .05: none.

## Adding the distractor transient HP − LP (dD) to dS (ΔR² = R²[dS + dD] − R²[dS], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.17 [-0.53, +0.12] | 10/28 | .28 | .28 |
| V2 | -0.53 [-1.11, -0.30] | 4/28 | 5.5 × 10⁻⁵ | 1.2 × 10⁻⁴ |
| V3 | -0.49 [-0.78, -0.24] | 5/28 | .0017 | .0022 |
| V3AB | -1.10 [-1.74, -0.31] | 4/27 | 8.4 × 10⁻⁵ | 1.3 × 10⁻⁴ |
| IPS | -2.94 [-4.01, -2.21] | 3/25 | 6.6 × 10⁻⁶ | 3.0 × 10⁻⁵ |
| hV4 | -0.69 [-1.01, -0.36] | 3/28 | 2.8 × 10⁻⁵ | 8.3 × 10⁻⁵ |
| VO | -2.72 [-3.44, -1.46] | 5/21 | .0055 | .0062 |
| LO | -1.35 [-1.99, -0.68] | 3/28 | 7.2 × 10⁻⁵ | 1.3 × 10⁻⁴ |
| TO | -0.68 [-1.48, -0.41] | 1/28 | 1.5 × 10⁻⁸ | 1.3 × 10⁻⁷ |
| **Pooled** | -0.80 [-1.44, -0.49] | 0/28 | 7.5 × 10⁻⁹ | n/a |
| Early (pathway) | -0.39 [-0.93, -0.15] | 5/28 | 4.7 × 10⁻⁴ | 4.7 × 10⁻⁴ |
| Dorsal (pathway) | -2.00 [-2.62, -1.50] | 3/27 | 3.7 × 10⁻⁷ | 5.6 × 10⁻⁷ |
| Occipitotemporal (pathway) | -1.27 [-1.88, -0.78] | 1/28 | 2.2 × 10⁻⁸ | 6.7 × 10⁻⁸ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 0 of 28 participants (median ΔR² -0.80 × 10⁻⁴, p = 7.5 × 10⁻⁹); areas with q < .05: V2, V3, V3AB, IPS, hV4, VO, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## Adding the target transient HP − LP (dT) to dS + dD (ΔR² = R²[dS + dD + dT] − R²[dS + dD], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.55 [-0.85, -0.34] | 5/28 | .0061 | .0091 |
| V2 | -0.44 [-0.64, -0.11] | 6/28 | .0027 | .0048 |
| V3 | -0.39 [-0.81, -0.17] | 4/28 | 1.5 × 10⁻⁵ | 6.8 × 10⁻⁵ |
| V3AB | -0.84 [-1.65, -0.59] | 3/27 | 4.3 × 10⁻⁴ | 9.6 × 10⁻⁴ |
| IPS | -2.06 [-2.91, -1.19] | 6/25 | .011 | .014 |
| hV4 | -0.42 [-1.00, -0.03] | 9/28 | .10 | .10 |
| VO | -1.07 [-1.91, -0.39] | 3/21 | .046 | .052 |
| LO | -1.26 [-2.69, -0.54] | 1/28 | 7.5 × 10⁻⁸ | 6.7 × 10⁻⁷ |
| TO | -0.96 [-1.42, -0.49] | 5/28 | 1.1 × 10⁻⁴ | 3.2 × 10⁻⁴ |
| **Pooled** | -0.64 [-1.11, -0.57] | 1/28 | 1.5 × 10⁻⁸ | n/a |
| Early (pathway) | -0.55 [-0.78, -0.34] | 4/28 | 1.5 × 10⁻⁴ | 2.3 × 10⁻⁴ |
| Dorsal (pathway) | -1.50 [-1.76, -0.80] | 5/27 | .0025 | .0025 |
| Occipitotemporal (pathway) | -1.09 [-1.32, -0.64] | 3/28 | 1.3 × 10⁻⁶ | 3.8 × 10⁻⁶ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 1 of 28 participants (median ΔR² -0.64 × 10⁻⁴, p = 1.5 × 10⁻⁸); areas with q < .05: V1, V2, V3, V3AB, IPS, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## Adding both transient terms (dD + dT) to dS (ΔR² = R²[dS + dD + dT] − R²[dS], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.83 [-1.21, -0.29] | 7/28 | .017 | .017 |
| V2 | -0.95 [-1.33, -0.49] | 3/28 | 1.2 × 10⁻⁴ | 2.3 × 10⁻⁴ |
| V3 | -0.79 [-1.35, -0.30] | 3/28 | 1.5 × 10⁻⁴ | 2.3 × 10⁻⁴ |
| V3AB | -1.96 [-2.93, -0.77] | 3/27 | 2.6 × 10⁻⁵ | 7.8 × 10⁻⁵ |
| IPS | -4.88 [-6.35, -3.16] | 5/25 | 1.4 × 10⁻⁴ | 2.3 × 10⁻⁴ |
| hV4 | -0.64 [-2.03, -0.41] | 5/28 | .0017 | .0022 |
| VO | -2.83 [-4.90, -0.56] | 5/21 | .014 | .016 |
| LO | -2.44 [-3.40, -1.84] | 1/28 | 2.2 × 10⁻⁸ | 2.0 × 10⁻⁷ |
| TO | -1.40 [-2.32, -1.17] | 2/28 | 2.5 × 10⁻⁷ | 1.1 × 10⁻⁶ |
| **Pooled** | -1.39 [-1.76, -1.01] | 0/28 | 7.5 × 10⁻⁹ | n/a |
| Early (pathway) | -1.04 [-1.26, -0.58] | 4/28 | 7.2 × 10⁻⁵ | 7.2 × 10⁻⁵ |
| Dorsal (pathway) | -3.44 [-4.26, -2.05] | 2/27 | 1.0 × 10⁻⁶ | 1.6 × 10⁻⁶ |
| Occipitotemporal (pathway) | -2.13 [-3.14, -1.06] | 1/28 | 2.2 × 10⁻⁸ | 6.7 × 10⁻⁸ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 0 of 28 participants (median ΔR² -1.39 × 10⁻⁴, p = 7.5 × 10⁻⁹); areas with q < .05: V1, V2, V3, V3AB, IPS, hV4, VO, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## All three HP − LP terms vs no HP terms (ΔR² = R²[dS + dD + dT] − R²[none], × 10⁻⁴)

| Area | Median ΔR² [95% CI] | k > 0 / n | p | q |
|---|---|---|---|---|
| V1 | -0.59 [-1.82, +0.21] | 11/28 | .49 | .49 |
| V2 | -0.94 [-1.63, -0.19] | 8/28 | .077 | .12 |
| V3 | -0.51 [-1.84, +0.24] | 12/28 | .33 | .37 |
| V3AB | -1.64 [-3.23, -0.42] | 6/27 | .0014 | .0042 |
| IPS | -4.65 [-10.34, -2.94] | 5/25 | .0023 | .0051 |
| hV4 | -0.62 [-2.03, -0.22] | 7/28 | .077 | .12 |
| VO | -1.90 [-3.43, +0.41] | 7/21 | .16 | .20 |
| LO | -2.38 [-4.08, -1.34] | 5/28 | 1.1 × 10⁻⁵ | 5.0 × 10⁻⁵ |
| TO | -1.91 [-3.69, -1.42] | 3/28 | 1.5 × 10⁻⁶ | 1.4 × 10⁻⁵ |
| **Pooled** | -1.51 [-2.40, -1.35] | 4/28 | 3.2 × 10⁻⁷ | n/a |
| Early (pathway) | -1.02 [-1.61, -0.19] | 8/28 | .020 | .020 |
| Dorsal (pathway) | -3.97 [-5.39, -1.88] | 4/27 | 2.4 × 10⁻⁴ | 3.5 × 10⁻⁴ |
| Occipitotemporal (pathway) | -1.58 [-3.13, -1.33] | 4/28 | 1.5 × 10⁻⁶ | 4.6 × 10⁻⁶ |

*Summary:* Pooled over areas, the model with the added term(s) predicts the held-out run better in 4 of 28 participants (median ΔR² -1.51 × 10⁻⁴, p = 3.2 × 10⁻⁷); areas with q < .05: V3AB, IPS, LO, TO; pathways with q < .05: Early, Dorsal, Occipitotemporal.

## Model with the best held-out prediction per participant (median over areas of R² − R²[base])

| Model | Participants |
|---|---|
| No HP terms (item weight, target, distractor, sustained gain) | 17 |
| Sustained HP − LP only (dS) | 8 |
| dS + distractor transient HP − LP (dD) (the HP terms of the joint fit) | 2 |
| dS + dD + target transient HP − LP (dT) | 1 |

*Summary:* how the participants split over the four models; a model counts for a participant when its median held-out gain over the nine areas is the largest.

## For scale: in-sample R² gain of the same terms (all runs; ≥ 0 by construction)

| Contrast | Median over participants of the pooled in-sample ΔR² (× 10⁻⁴) |
|---|---|
| Sustained HP − LP (dS) vs no HP terms | +0.81 |
| Adding the distractor transient HP − LP (dD) to dS | +0.35 |
| Adding the target transient HP − LP (dT) to dS + dD | +0.19 |
| Adding both transient terms (dD + dT) to dS | +0.81 |
| All three HP − LP terms vs no HP terms | +2.28 |

*Summary:* in sample each added term explains only about 10⁻⁴ of the variance or less; held out, that gain is smaller than the cost of estimating the term, so held-out prediction cannot separate these models.

Sources: `notes/data/hp_terms_loro_zs_lm_2026-11-16.tsv` (per participant × area × held-out run × model: held-out SSE and SST, fold coefficients), `notes/data/hp_terms_heldout_r2_zs_lm_2026-11-16.tsv` (held-out R² per participant × area × model), `notes/data/hp_terms_heldout_stats_zs_lm_2026-11-16.tsv` (every number in the tables), `notes/data/hp_terms_insample_dr2_zs_lm_2026-11-16.tsv` (in-sample ΔR²). Code: `retsupp/modeling/hp_terms_heldout_power.py` (cluster, `retsupp/modeling/slurm_jobs/hp_terms_heldout_power.sh`, output `derivatives/hp_terms_heldout_power_v2_2026-11-13`), `retsupp/visualize/paper/hp_terms_heldout_power_stats.py` (tables).
