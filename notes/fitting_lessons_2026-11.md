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

| ≤ 50% (drops 13 voxels, 32% of variance) | 1795 | 100 | 90 | 60, p=.28 | 87 | 77, p=.025 | 73, p=.009 |
| ≤ 20% (drops 53 voxels, 36% of variance) | 1794 | 100 | 93 | 50, p=.87 | 90 | 80, p=.002 | 70, p=.014 |

The ≤ 5% and ≤ 3% rows are too strict and should not be read as a weakening of dD: voxels with SD 3–20% have the best signal of all (table below). Only above 20% does reliability collapse.

| Voxel SD (%) | Voxels | Share of variance | Median R² (m4_TDS) | Median cross-run null R² | % with null > 0 |
|---|---:|---:|---:|---:|---:|
| ≤ 1 | 17,801 | 1.6% | .055 | −.050 | 27 |
| 1–2 | 65,703 | 16.6% | .092 | −.011 | 47 |
| 2–3 | 17,243 | 12.1% | .187 | +.109 | 65 |
| 3–5 | 7,659 | 13.2% | .246 | +.172 | 67 |
| 5–10 | 2,412 | 12.9% | .281 | +.193 | 67 |
| 10–20 | 357 | 7.8% | .269 | +.177 | 65 |
| 20–50 | 40 | 3.8% | .124 | −.093 | 43 |
| > 50 | 13 | 31.9% | .069 | −5.2 | 0 |

These voxels mostly already fail the signal selection used for scoring (null > 0), so they barely touch the score. They reach the result through δ and through the shared parameters of the base fits, where they enter the summed squared error. Refit canary with `MAX_BOLD_SD=20` (job 7967396: sub-03 V3AB, sub-07 V2, sub-07 IPS, sub-20 SPL1), to `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20`. Per-run diagnostic: `retsupp/modeling/diagnose_outlier_voxels.py` (job 7967395). Offline script: session scratchpad `robust_loso2.py`, to move into the repo.

## Refit canaries: shared parameters under three voxel treatments

Likelihood fits are dropped (decision 2026-11-02): no full `--loss nll` ladder. The columns below and the noise-weighted section further down remain as a record only. Outlier removal (`maxsd20`) continues.

Full-data nested bounded fits. `sse` is the original fit; `maxsd20` drops voxels above 20% SD and refits m00 and m0 as well (job 7967396); `nll` minimizes Σ_v (T/2) log SSE_v for the shared terms but reuses the m0 of `sse` (job 7970095, 17–33 min per subject × ROI). Table: `notes/data/shared_fit_comparison_canary_3versions_20261102.tsv`. The contaminated cases were picked because they were the worst, so they do not estimate a typical change.

g_HP − g_LP in m5_TDS_dS and g_HP_dyn − g_LP_dyn in m6_TDS_dD:

| Subject, ROI | Voxels dropped | dS sse | dS maxsd20 | dS nll | dD sse | dD maxsd20 | dD nll |
|---|---:|---:|---:|---:|---:|---:|---:|
| sub-03 V3AB | 0 | +.003 | +.003 | −.007 | .000 | .000 | −.076 |
| sub-07 V2 | 13 | −10.32 | +.122 | +.037 | +1.762 | −.132 | −.087 |
| sub-07 IPS | 1 | −.155 | +.002 | .000 | −.378 | −.063 | −.038 |
| sub-20 SPL1 | 4 | +.564 | +.039 | +.130 | +.921 | +.605 | +.611 |
| sub-08 V3 | not run | −.018 | | −.037 | −.194 | | −.203 |
| sub-12 V3AB | not run | −.018 | | −.019 | −.132 | | −.095 |
| sub-15 IPS | not run | −.017 | | −.010 | −.131 | | −.119 |
| sub-23 hV4 | not run | −.052 | | −.045 | −.152 | | −.125 |

In sub-07 V2 the m5 sse fit puts g_HP at its bound of −10. Without those 13 voxels the dS sign flips. The ordinary cases barely move between sse and nll. Item_scale in m3_S also depends on the outliers: sub-20 SPL1 1.32 (sse), 1.59 (nll), 24.33 (maxsd20); sub-07 V2 5.03, 5.00, 2.49. nll stays near sse there because it starts from the sse m0, so it does not remove the outliers' effect on the base fit.

Above 20% SD: 53 voxels in 23 of 330 subject × ROI combinations, 12 subjects, 8 of the 23 in sub-07. Refits of the other 20: job 7975291, same root. One further case to watch has no voxel above 20%: sub-21 SPL1, top voxel 62% of SST.

## LOSO with noise weighting (offline, base fits still SSE)

Each subject's δ is taken as Σ(b/σ²_v) / Σ(a/σ²_v), with σ²_v the voxel's residual variance in the base fit, as a per-voxel Gaussian likelihood would weight it (`noise_model='gaussian'` in braincoder 0.6+). Scored as ΔlogL = −(T/2)·log(1 + ΔSSE/SSE0), or as pooled ΔR². % of subjects improved, p (signal voxels):

| δ from | Score | dS | dD |
|---|---|---:|---:|
| Unweighted (reported) | Pooled ΔR² | 80, p=.006 | 73, p=.011 |
| Unweighted | ΔlogL | 73, p=.035 | 73, p=.007 |
| Noise-weighted | Pooled ΔR² | 77, p=.012 | 60, p=.28 |
| Noise-weighted | ΔlogL | 67, p=.088 | 63, p=.28 |

dD depends on how voxels are weighted when its strength is estimated, so treat it as less secure than dS. Items, T and S hold in all rows; D stays at chance. Script: session scratchpad `ml_loso.py`, to move into the repo.

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
