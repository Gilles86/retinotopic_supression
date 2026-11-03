# DN parameters across the visual hierarchy vs Aqil et al. 2021 (2026-11-03)

Aqil, Knapen & Dumoulin 2021 (PNAS) report that b, d and the size ratio φ = σ2/σ1 all decrease along V1, V2, V3, V3AB, hV4, LO, TO, VO, IPS. Read from the PMC text, which cut off partway: the legends of Fig. 3D, 3G and 4D are verbatim; their equation and voxel selection (R² > .3, eccentricity to 4.5°) come from an unverified summary. No values or per-ROI statistics were available.

Here: m5 and m6 from `prf_cv_roi`, fitted on the training runs of each fold. Voxels with training reliability > 0, positive amplitude, eccentricity ≤ 3.17°, SD ≤ 20%. Median per subject × fold × ROI with at least 20 voxels, averaged over folds. Trend: per-subject Spearman ρ across the ROIs in Aqil's order; Wilcoxon across 29 subjects. Eccentricity control: log parameter, residual after regressing on eccentricity and its square within subject. Stats: `notes/data/dn_hierarchy_stats.tsv`. Figure: `notes/figures/dn_hierarchy_aqil.pdf`. Script: `retsupp/visualize/paper/plot_dn_hierarchy.py`.

| Parameter | Reported | m6 ρ | m6 p | m6 ρ, ecc control | m5 ρ | Across-fold r |
|---|---|---:|---:|---:|---:|---:|
| σ centre | up (standard) | +.71 | < .001 | +.86 | +.75 | .91 |
| φ = σ2/σ1 | down | +.28 | .018 | −.02 (p=.93) | +.10 (p=.28) | .74 |
| b | down | −.70 | < .001 | −.67 | −.77 | .69 |
| d | down | +.80 | < .001 | +.68 | +.79 | .67 |
| b/d | not reported | −.90 | < .001 | −.75 | −.89 | .74 |
| d/c (peak units) | not reported | +.82 | < .001 | +.55 | +.83 | .69 |
| b/a (peak units) | not reported | +.07 (p=.49) | | | +.35 | |

Result: σ works as a positive control; b goes the reported way; φ does not decrease; d goes the opposite way.

Limits:
- a, b and d share one free scale (multiplying all three leaves the prediction unchanged, checked numerically). b and d start at 1 in `_adapt_m5_from_m2`, so raw values depend on the fit's path. b/a shows no trend in m6. b/d and φ do not have this problem. Whether Aqil's fits pin the scale was not checked.
- One bar stimulus at one contrast in a 3.17° aperture; Aqil used several stimulus conditions. 23% of voxels have a surround σ above 5°, beyond the stimulus grid, so φ and d are probably poorly constrained here.
- With training R² > .3 (closer to Aqil), only 13–16 subjects keep at least 5 ROIs and few voxels remain outside V1–V3 and LO: b weakens (m6 ρ −.30, p=.22), φ rises (+.60, p=.002; ecc control +.12, p=.36), d +.86, b/d −.90.
- VO (median 8 reliable voxels per fold) and IPS (31) enter for few subjects.
