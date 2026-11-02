# Fitting lessons, retsupp tables (2026-11-02)

Named version of the tables in the braincoder skill (`gilles-claude-skills/skills/braincoder/references/convergence.md`, `r2_and_cv.md`, `performance.md`), which use generic labels because that repo is public.

Terms: items = m0_prf vs m00_bar; T = m1_T vs m0_prf; D = m2_D vs m0_prf; S = m3_S vs m0_prf; dS = m5_TDS_dS vs m4_TDS; dD = m6_TDS_dD vs m4_TDS.

Sources: `derivatives/af_prf_cv_ladder_nested/lessons_20261102/{fold_table,train_monotonic}.tsv.gz`, `within_cv_20261102/`, `stability_20261101/shared_params.tsv`; `af_fixed_group_mechanisms/`. Figures: `notes/figures/fixed_group_mechanisms_loso_illustrated.pdf`, `fixed_group_mechanisms_within_subject_cv_supp.pdf`.

## Outlier-variance voxels (found 2026-11-02)

`cleaned` BOLD is psc per run, so low-mean voxels (mask edge, dropout) can have huge variance. sub-03 fold 0, all 9,894 ROI voxels: the top voxel holds 51% of test SST, the top 10 hold 90% (SD 418% vs median 1.1%). Pooled test R² without them is m1 .102, m2 .104, m3 .107, m4 .109, against .0070, −.0024, .0068, −.0027 with them. The negative m2 and m4 values are this, not runaway fits. Raw mean falls with SD (median 947 for SD ≤ 3, 170 for SD > 10; mask median 777).

The AF voxels (at most 500 per ROI) are affected too: top voxel median 5.7% of SST, top 10 median 26%, worst cases sub-07 IPS 67%, sub-21 SPL1 62%, sub-07 V2 61% (top 10: 95%).

Offline robustness of the LOSO test (signal voxels, δ from the other subjects, same voxel limit applied to δ and to the score). % of subjects improved, p:

| Limit on voxel SD | Signal voxels per subject | items | T | D | S | dS | dD |
|---|---:|---:|---:|---:|---:|---:|---:|
| None (reported) | 1795 | 100 | 90 | 50, p=.94 | 87 | 80, p=.006 | 73, p=.011 |
| ≤ 10% | 1786 | 100 | 93 | 37, p=.44 | 90 | 73, p=.004 | 67, p=.036 |
| ≤ 5% | 1732 | 100 | 87 | 53, p=.79 | 90 | 67, p=.016 | 67, p=.077 |
| ≤ 3% | 1560 | 100 | 100 | 53, p=.64 | 93 | 67, p=.029 | 73, p=.061 |

dS survives every limit. dD keeps its direction but is no longer below .05 at ≤ 5% and ≤ 3%. Limitation: the base fits still contain these voxels, and σ_AF, σ_dyn and the shared gains are fitted to summed squared error across the ROI, so high-variance voxels can dominate them. Only a refit without them removes that. Script: session scratchpad `robust_loso.py`, to move into the repo.

## % of subjects whose prediction improves (signal voxels; variance-weighted ΔR²)

| Term | Leave one HP location out (n=30) | Leave one run per HP location out (n=26, preliminary) | Leave one subject out, zero parameters (n=30) |
|---|---:|---:|---:|
| items | 100 | 96 | 100 |
| T | 73 (p=.0016) | 65 (p=.15) | 90 (p<.0001) |
| S | 83 (p<.0001) | 77 (p=.0003) | 87 (p<.0001) |
| D | 23 (p=.0019) | 27 (p=.067) | 50 (p=.94) |
| dS | 20 (p<.0001) | 35 (p=.27) | 80 (p=.006) |
| dD | 7 (p<.0001) | 12 (p<.0001) | 73 (p=.011) |

Percentages are of the per-subject median over the 9 regions V1–IPS. p is a Wilcoxon test across subjects.

## Per-voxel-mean vs pooled: dS with δ from the other subjects' m5 fits

| ROI | Per-voxel mean ×10⁻³ (subjects > 0) | Pooled ×10⁻⁵ (subjects > 0) | Pooled, conditional δ ×10⁻⁵ |
|---|---:|---:|---:|
| V1 | −0.02 (47%) | −0.36 (7%) | +0.00 |
| V2 | +0.29 (80%) | −12.6 (27%) | +0.08 |
| V3 | +0.05 (50%) | −1.95 (37%) | +0.48 |
| hV4 | +0.04 (83%) | +0.83 (60%) | +0.70 |
| All | +0.04 (83%) | −0.15 (40%) | +0.09 |

All voxels, not only signal voxels.

## dS: conditional optimum vs joint fit, group medians

| ROI | Conditional b/a on m4 | m5 fit |
|---|---:|---:|
| V2 | −0.005 | −0.042 |
| V3 | −0.013 | −0.051 |
| hV4 | −0.021 | −0.030 |
| VO | −0.013 | −0.030 |
| V3AB | −0.011 | −0.019 |
| IPS | −0.009 | −0.016 |

So the m5 values overstate the effect; the size to report is closer to the conditional one.

## ΔcvR² by voxel reliability, unbounded nested, ×10⁻³

| null cvR² | ≤0 | 0–.05 | .05–.1 | .1–.2 | .2–.3 | >.3 |
|---|---:|---:|---:|---:|---:|---:|
| Share of voxels | 52% | 10% | 7% | 10% | 7% | 14% |
| items | +15.41 | +11.28 | +8.33 | +5.03 | +2.65 | +1.41 |
| T | −0.40 | +0.01 | +0.01 | −0.04 | −0.26 | −0.20 |
| D | −0.40 | −0.15 | −0.12 | −0.09 | −0.10 | −0.09 |
| S | −0.03 | +0.05 | +0.03 | +0.03 | +0.02 | −0.02 |
| dS | −3.02 | −0.24 | −0.27 | −0.43 | −0.60 | −0.92 |
| dD | −1.19 | −0.15 | −0.20 | −0.23 | −0.33 | −0.38 |

Signal voxels (null > 0): V1 83%, V2 80%, V3 75%, LO 59%, hV4 37%, TO 35%, V3AB 29%, VO 13%, IPS 12%, SPL1 1.7%, FEF 0.4%.

## Train-fit monotonicity (share of folds worse than the parent)

| Child | Old ladder, mean R² | Old ladder, weighted | Nested bounded, mean R² | Nested bounded, weighted |
|---|---:|---:|---:|---:|
| m1_T | 19% | 3% | 16% | 0% |
| m2_D | 32% | 5% | 25% | 0% |
| m3_S | 21% | 16% | 20% | 0% |
| m3b_TD | 28% | 11% | 26% | 0% |
| m4_TDS | 16% | 14% | 23% | 0% |
| m5_TDS_dS | 46% | 23% | 33% | 0% |
| m6_TDS_dD | 29% | 14% | 26% | 0% |
| m7_full | 32% | 13% | 29% | 0% |

This corrects the 2026-10-29 note: most of the "richer models fit worse" pattern was the metric.

## Ladder runtime per fold (8 CPUs, loco bounded, median 395 voxels, about 2300 training TRs)

| Model | Optimizer | Median s | 90th pct s | Median L-BFGS iterations over 5 starts |
|---|---|---:|---:|---:|
| m00_bar | Adam | 1634 | 3516 | – |
| m0_prf | Adam | 1722 | 3359 | – |
| m1_T | L-BFGS | 90 | 177 | 75 |
| m2_D | L-BFGS | 92 | 184 | 75 |
| m3_S | L-BFGS | 92 | 190 | 64 |
| m3b_TD | L-BFGS | 93 | 186 | 84 |
| m4_TDS | L-BFGS | 120 | 231 | 159 |
| m5_TDS_dS | L-BFGS | 127 | 244 | 194 |
| m6_TDS_dD | L-BFGS | 123 | 230 | 176 |
| m7_full | L-BFGS | 129 | 245 | 219 |

Add about 4 min per subject to load the full-brain BOLD (19 s per run, 12 runs, about 270k voxels).
