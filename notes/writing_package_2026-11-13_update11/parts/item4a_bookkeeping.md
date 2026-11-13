# Item 4a: where every gain-field number in the Results comes from

## The fits used

**Fits.** One reference fit (update 8) is used throughout, n = 28 (sub-01 and sub-02 left out).
- **pRF model:** DoG pRFs.
- **HRF:** one per participant × area, fixed.
- **Voxel selection, the "reliable" selection** (the same for every row unless stated):
  - per area, voxels with p_signal > 0.5;
  - then the top 500 by R²;
  - then leave-one-run-out reliability > 0;
  - BOLD SD ≤ 20%.
  
  It gives 241 participant × area fits. The looser r > 0.2 / r > 0.1 selections are robustness checks only (update 9).

**Roots** (under `/shares/zne.uzh/gdehol/ds-retsupp/derivatives/`):
- **REF**, `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`: pRFs refitted together with the
  gain fields (alternating polish). It contains only two models:
  - m4_TDS: item weight, target, distractor and sustained gain;
  - m7_full: all terms, i.e. m4_TDS plus HP − LP sustained (dS) and HP − LP distractor (dD).
- **BASE**, `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf`: the same ladder with the pRFs held at the
  fit without gain fields. Every model except m4_TDS and m7_full comes from here.
- **ALONE**, `af_fixed_group_mechanisms_polished_2026-11-12`: terms estimated alone. Each term is added alone to its
  base model by a linearised one-parameter fit. Only rows with `valid` = True (linearity check) are used.
  - dS, dD, dT: base m4_TDS from REF.
  - S, T, D, items, DonT: base models from BASE.
- **WIDTHS**, `field_widths_reliable_polished_2026-11-12`: per-run sufficient statistics on REF's m7_full
  (`fit_width_profiles.py`). The five gains are re-solved linearly with everything else at the joint fit. This gives
  the jackknife SEs, the reliable-gain subsets and the width profiles.

**Tables** (`notes/data/`):
- **G:** `gain_field_parameters_polished_2026-11-12.tsv` (rows `family` = dog; `model` = m7_full or m4_TDS from REF).
- **C:** `conditional_estimates_polished_2026-11-12.tsv` (column `mech`, rows `valid`).
- **W:** `field_widths_reliable_polished_2026-11-12_n28_*.tsv`.
- **H10:** `hierarchy_gain_terms_polished_2026-11-12_n28_*.tsv` (update 10).
- **HR:** `hierarchy_regression_polished_2026-11-12_n28.tsv`.
- **Generated key numbers:** `notes/data/reference_numbers_polished_2026-11-12_n28.{md,tsv}`
  (`retsupp/visualize/paper/reference_numbers.py`).

## Number → fit

| Results number (Fig.) | Value as reported | Model and fit | Estimated alone or joint | pRFs | Root | File and column |
|---|---|---|---|---|---|---|
| Target gain, per area and pooled (5b) | pooled +17% [+13, +24] | m7_full | joint | refitted | REF | G, dog/m7_full, `g_T_dyn` |
| Distractor gain, averaged over its locations (5b) | pooled +8.0% [+4.4, +13] | m7_full | joint | refitted | REF | G, dog/m7_full, (`g_HP_dyn` + 3·`g_LP_dyn`)/4 |
| Distractor gain on top of the target term | pooled +3.1% | m1_T + distractor (DonT) | alone | held fixed | ALONE on BASE m1_T | C, `mech` = DonT |
| **dS estimated alone**, per area, pooled, per pathway (5c, lead statistic) | pooled −1.0% [−1.9, −0.4], p = 2.7 × 10⁻⁴ | m4_TDS + dS | alone | refitted | ALONE on REF m4_TDS | C, `mech` = dS; per pathway `dS_pooled_by_pathway_n28_2026-11-12_v11.tsv` |
| dS joint (5c legend, text) | pooled −3.5% [−6.7, −1.3], p = .0044 | m7_full | joint | refitted | REF | G, dog/m7_full, `g_HP` − `g_LP` |
| dD estimated alone (5d) | pooled −0.8%, p = .14; LO −3.3%* | m4_TDS + dD | alone | refitted | ALONE on REF m4_TDS | C, `mech` = dD |
| dT estimated alone (5d) | pooled −3.6%, p = .032 | m4_TDS + target at HP | alone | refitted | ALONE on REF m4_TDS | C, `mech` = dT |
| dD − dT (5d) | pooled +1.0%, p = .17 | difference of the two rows above, per participant × area | alone | refitted | ALONE on REF m4_TDS | C, dD − dT |
| dD joint (5d legend) | pooled +0.1%, p = .35 | m7_full | joint | refitted | REF | G, dog/m7_full, `g_HP_dyn` − `g_LP_dyn` |
| dT joint | not available | the joint fit ties the target gain across locations | — | — | — | — |
| Sustained gain at LP and at HP (5a, under the maps; sign only) | LP pooled −11%, HP −21% | m7_full | joint | refitted | REF | G, dog/m7_full, `g_LP`, `g_HP` |
| Sustained gain shared by all four locations (S) | pooled −3.4% | m0_prf + sustained (location-blind) | alone | held fixed | ALONE on BASE m0_prf | C, `mech` = S |
| Item weight (ratio to the bar) | pooled 3.39 | m7_full | joint | refitted | REF | G, dog/m7_full, `item_scale` |
| Fig. 5a map: sustained HP − LP drawn | V3 −2.7, hV4 −2.4, TO −1.9% | = dS estimated alone (area median) | alone | refitted | ALONE on REF m4_TDS | `fig5_gain_field_map_parameters_n28_2026-11-12_v11.tsv`, `dS_map` |
| Fig. 5a map: target and distractor values | V3 +33 / +6%, … | m7_full | joint | refitted | REF | same file, `T`, `D` |
| Sustained field width, FWHM (5a maps, field-width supplement) | 2.8° [2.7, 2.9] among fits with a reliable sustained gain | m7_full fitted `sigma_AF`; subset with jackknife \|g_HP or g_LP\| / SE > 2 | joint | refitted | WIDTHS on REF m7_full | W `_widths_thr2.tsv`, `_fits.tsv` (`fwhm_sus`, `sus_z`) |
| Narrower than 5.7° (28/28); early vs higher areas (3.4° vs 2.7°, p = .013; p = .58 at > 3 SE) | — | as above | joint | refitted | WIDTHS | W `_claims_thr2.tsv`, `_early_vs_higher_thr{2,3}.tsv` |
| Transient width at a bound (59%) | — | m7_full `sigma_dyn`, fits with a reliable target gain | joint | refitted | WIDTHS | W `_fits.tsv` (`bound_dyn`, `g_T_z`) |
| Field volume g·2πσ² | sustained HP −2.95 deg², … | m7_full | joint | refitted | WIDTHS | W `_volume.tsv` |
| Gain–width trade-off ρ | — | m7_full | joint | refitted | WIDTHS | W `_tradeoff.tsv` |
| Width profile intervals | 17.7° … 1.5° | m7_full, σ re-solved on a grid | joint | refitted | WIDTHS | W `_fits.tsv` (`sus_prof_*`) |
| Target gain vs pRF size (+27% per doubling) | — | m7_full target against the area's pRF σ (`fig1_prf_sigma.tsv`) | joint | refitted | REF | HR |
| Hierarchy slopes of every term (update 10: target +27.4 pp, dS alone −0.7 pp, dS / S ratio, within-area terciles, width slope) | — | target, distractor, LP: m7_full; dS, dD, dT: ALONE; S: ALONE on BASE; within-area bins: m7_full per pRF-size tercile | as listed | refitted, except S | REF, ALONE, BASE; bins `derivatives/gain_bins_polished_2026-11-12` | H10 `_regression`, `_dissociation`, `_proportion_*`, `_bin_slopes`, `_width_slope` |
| LP gain trade-off with item weight (ρ = −.73) and pRF amplitude (−.17) | — | m7_full, refitted − held fixed | joint | both | REF vs BASE | `prf_refit_2026-11-12_shared.tsv` |
| **New (update 11):** 1b Bayes factors and TOST | — | dD, dT, dD − dT alone; dD joint | both | refitted | ALONE, REF | `update11_gain_terms_polished_2026-11-12_n28_null_evidence.tsv` |
| **New (update 11):** 3b sizes and dS / target ratio | −0.20 [−0.45, −0.07] | m7_full (T, D, dS) | joint | refitted | REF (+ WIDTHS for target reliability) | `..._sizes.tsv`, `..._ratio.tsv` |
| **New (update 11):** 3c gain–dS correlations | within participants ρ −0.17 (target) | m7_full; dS also alone; split halves from WIDTHS | both | refitted | REF, ALONE, WIDTHS | `..._correlations.tsv`, `..._correlations_split_half.tsv` |

## Not on the reference fit (pRFs held at the fit without gain fields)

Say so in the Methods. These numbers come from the BASE root or from earlier tables:
- Gaussian and DN robustness: G rows `family` = gauss / dn; Figs. S10, S11; the DN columns of Table S3.
- Leave-one-subject-out tests: `loso_per_roi_final2_2026-11.tsv`; Fig. S8, Table S4.
- κ and the split models (next vs opposite HP; target split).
- Per-session dS and dS vs behaviour.
- Baseline vs gain.
- The terms estimated alone on BASE models: S, T, D, items, DonT.

Fig. 7 (trial-wise gains) is on the refitted fits since update 9 (`trialwise_signal_se_polished_2026-11-12`).

**In plain language.** Every target, distractor, item-weight, sustained-LP/HP and width number is from the joint fit
with refitted pRFs. Every dS, dD and dT quoted as "estimated alone" is from the refitted model with item weight,
target, distractor and sustained gain, with that one term added. Only S and DonT among the gain-field numbers still
use base models with the pRFs held fixed. The robustness, leave-one-subject-out and split-model numbers are all on the
held-fixed fits.

## Rules the text should follow

- **"Joint fit" vs "estimated alone".** Name the fit whenever dS is quoted. dS joint (−3.5%) and dS estimated alone
  (−1.0%) differ by a factor of 3.5. The lead statistic is the one estimated alone.
- **Sizes compared side by side** (target vs dS, item 3b) should come from the same fit (joint), or the mix should be
  said.
- **The LP / shared sustained gain:** quote its sign and where it is significant, not its size.
