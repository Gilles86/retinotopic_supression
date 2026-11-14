# Zero-sum sustained field, fitted by joint LM: the new reference (2026-11-13/14)

Inputs of writing-package update 13. Main model: **m5_TDS_dS_zs**: item weight, target gain, distractor gain and a
zero-sum sustained field (dS), with no transient HP − LP terms. Supplementary model: **m7_full_zs**, which adds the
transient HP − LP distractor term (dD). All gain-field numbers below come from 28 participants (sub-03 … sub-30) and
the current voxel selection (reliable, max BOLD SD 20).

## Models and fitting (`retsupp/modeling/fit_cv_ladder_nested.py`, ZERO_SUM_MODELS)

The sustained field has no condition-invariant part. With g_HP + 3·g_LP = 0 it averages to 1 over the four HP
conditions, and dS = g_HP − g_LP = 4/3·g_HP. The reason is in `notes/joint_optimisation_2026-11-13.md`: with the pRFs
free, the common level is a pRF-shape term. All models are fitted with every parameter free (per-voxel DoG pRFs plus
shared terms) by the joint Levenberg–Marquardt of `joint_lm`, with 400 iterations per start. Each model has two
starts; the lower training SSE is kept, and both are recorded in `optimizer_info[0]['candidates']`.

| model | terms | starts | start kept (of 241) |
|---|---|---|---|
| m4_TDS_zs | item weight, target, distractor (HP = LP); no sustained field | clamped m3b_TD; polished m4_TDS with sustained gains at 0 | 96 cold / 145 warm |
| **m5_TDS_dS_zs** (main) | + zero-sum sustained field (dS) | parent m4_TDS_zs; clamped m5_TDS_dS projected (g_HP = ¾ dS, g_LP = −¼ dS) | 222 parent / 19 cold |
| m7_full_zs (supplementary) | + transient HP − LP (dD) | parent m4_TDS_zs; clamped m7_full projected | 211 parent / 30 cold |

Checks in the code:

- Each start's prediction equals its parent's (asserted; max |Δpred| was 0 in every canary).
- The zero-sum constraint holds (asserted).
- Any term that ends on a bound is recorded (`info['on_bound']`; table `notes/data/zero_sum_fits_2026-11-13.tsv`,
  one row per cell × model × start).

The two starts end 0–11 000 SSE apart (median 47), so both are needed: a single start can stick, for example with
σ_dyn on its 0.5° bound.

Bounds (zero-sum fits only, `zs_bounds`):

- The sustained field is non-negative: g_HP ∈ [−1, 3], g_LP ≥ −1.
- Transient gains ∈ [−1, 5]. The lower bound keeps the drive non-negative while the distractor or target is on.
- Item scale ≥ 0.05.
- Widths as before, 0.5–8°.

Runs:

- Root `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm`, 241 cells × 3 models.
- Jobs:
  - 8341754: m4_TDS_zs + m7_full_zs.
  - 8342980–8343220: m5_TDS_dS_zs, one job per cell.
  - Re-runs: two cells (sub-13 V2 and V3) timed out at 8 h on a slow node and were resumed from their per-start
    checkpoints (8371705, 8371713). sub-30 IPS m5 was resubmitted (8371707). Exclude u24-chaihm0-634 from now on as well.
- Up to 3 h 13 min per cell for two models at 500 voxels. Peak RSS 25.6 GB, from data loading.

View root `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm_ref` (symlinks, with a
README):

- `m7_full → m5_TDS_dS_zs`. Every script that reads "m7_full" therefore uses the main model: the joint numbers,
  Fig. 5, the hierarchy gain bins, the width profiles and the Fig. 7 trial-wise gains.
- `m4_TDS → m4_TDS_zs`. This is the base of the estimated-alone terms.
- m7_full_zs, m5_TDS_dS_zs and m4_TDS_zs are also linked under their own names.

Estimated-alone (`af_fixed_group_mechanisms_zs_lm_2026-11-13`, `fit_fixed_group_mechanisms --zero-sum`):

- dS alone moves g_HP by +¾δ and g_LP by −¼δ on top of m4_TDS_zs, at the other participants' median σ_AF of the main
  model.
- dD and dT are as before, on m4_TDS_zs.
- Invalid linearised fits (lin_err > 1e-3) are excluded, as before (`valid`).

Everything was regenerated with `retsupp/visualize/paper/zero_sum_reference.sh` (cluster-submit, cluster-collect,
local). That script wraps `regenerate_reference.sh`, `regenerate_hierarchy_gain_terms.sh` and `rt_links_rerun.sh` with
TAG `zs_lm_2026-11-13` and SUFFIX `_zs_n28`. Outputs are new files only:

- Tables: `notes/data/*zs_lm_2026-11-13*`, and `reference_numbers_zs_lm_2026-11-13_n28.{md,tsv}` for the key numbers.
- Figures in `notes/figures/paper_2026-11/`:
  - `fig5_gain_fields_v11_zs_n28`
  - `fig7_compact_zs_lm_2026-11-13_n28`
  - `figS_field_widths_v3_zs_n28`
  - `figS_gain_field_map_search_v3_zs_n28`
  - `figS_gain_vs_prf_size_zs_lm_2026-11-13_zs_n28`
  - `figS_pooled_dS_dD_zs_lm_2026-11-13_zs_n28`
  - `figS_hierarchy_gain_terms_zs_lm_2026-11-13_zs_n28`
  - `table_parameters_2026-11_zs_lm_2026-11-13_zs_n28`
  - `table_hierarchy_gain_terms_zs_lm_2026-11-13_zs_n28`

The main model has no transient HP − LP term, so dD (joint) is 0 by construction and its Wilcoxon p is undefined. The
Benjamini–Hochberg steps of `hierarchy_regression`, `hierarchy_gain_terms`, `fig5_gain_fields` and
`fig5_gain_fields_v6` now use `fig45_common.fdr`, which returns NaN for a NaN p and corrects over the rest. The
outputs for the earlier references are unchanged.

## TO and other fits on bounds

TO (sub-03, sub-09, sub-26 in the canaries) is degenerate under every bound set tried: the item scale goes to its floor
and the transient gains go to their bounds. This also happens with the old bounds (item scale ≥ 0.001, transient gains
≤ 10); the tighter bounds cost 150–330 SSE.

- **Cause:** the transient terms scale the item drive, so once the item scale goes to 0 they lose their meaning. The
  model also has no location-blind onset transient for the search array.
- **Not done:** reparametrising the transient terms would change the model class that every downstream script uses.
  A clean fix would need a new term.
- **Decision:** keep the bounds above, record on-bound terms, and report with and without the on-bound fits.

Main model, 241 cells: 42 have the item scale at its floor, 10 have g_HP at a bound, and 18 have a transient gain at a
bound (σ bounds are common and harmless). The comparison table below drops the on-bound fits term by term.

## Comparison: refitted-polished (alternating) vs zero-sum LM

Statistics as in the paper:

- **Per area:** median over participants in %, bootstrapped 95% CI, participants below 0, Wilcoxon q (BH across the
  nine areas).
- **Pooled:** per participant the median over areas, then Wilcoxon p.

Full table, with V3AB, IPS, VO and LO as well: `notes/data/compare_zero_sum_2026-11-13.{md,tsv}`. Excerpt below
(pooled and the areas that carry the claims; "excl." = main model without fits that ended on that term's bound).

| quantity | refitted-polished | zero-sum LM, main model (m5_TDS_dS_zs) | main, excl. on-bound | + transient HP − LP (m7_full_zs) |
|---|---|---|---|---|
| **dS alone**, pooled | −1.0% [−1.9, −0.4], 22/28, p = 2.7 × 10⁻⁴ | −1.5% [−2.4, −0.9], 21/28, p = .0071 | — | — |
| dS alone, V3 / hV4 / TO | −2.7%* / −2.4%* / −1.9%* (q = .036 each) | −4.8% (q = .11) / −2.5% (q = .32) / −1.2% (q = .32) | — | — |
| **dS joint**, pooled | −3.5% [−6.7, −1.3], 22/28, p = .0044 | −5.5% [−7.6, −2.1], 21/28, p = .012 | −5.1% [−6.0, −2.8], p = .0061 (10 excl.) | −5.6% [−8.4, −2.2], 22/28, p = .0089 |
| dS joint, hV4 | −5.6%* (q = .024) | −8.2%* (q = .022) | −7.6% (q = .055) | −8.2% (q = .061) |
| **Target gain**, pooled | +17% [+13, +24], 3/28 < 0, p = 3.2 × 10⁻⁷ | +23% [+14, +35], 3/28 < 0, p = 4.1 × 10⁻⁷ | +21%, p = 8.2 × 10⁻⁷ (48 excl.) | +19%, p = 8.2 × 10⁻⁷ |
| target, areas with q < .05 | V3, V3AB, IPS, hV4, VO, TO | V3, V3AB, IPS, hV4, VO, LO, TO | V3, V3AB, hV4, VO, LO, TO | V3, V3AB, IPS, hV4, VO, LO, TO |
| **Distractor gain**, pooled | +8.0% [+4.4, +13], 7/28 < 0, p = .023 | +10% [+1.4, +17], 8/28 < 0, p = .0065 | +10%, p = .0082 (44 excl.) | +7.0% [−1.6, +15], p = .050 |
| distractor, areas with q ≤ .05 | V3AB, hV4, TO | V3AB, hV4 (q = .050) | V3, V3AB, hV4, VO, TO | V3AB |
| **Item weight**, pooled | 3.4 [2.0, 4.9] | 2.8 [1.5, 4.2] | 3.0 (42 excl.) | 2.8 |
| dD alone, pooled | −0.8%, p = .14 | −1.2%, p = .31 | — | — |
| dT alone, pooled | −3.6%, p = .032 | −2.6%, p = .019 | — | — |

Reading:

- **dS joint:** the same direction and a somewhat larger magnitude. Pooled it goes from −3.5% to −5.5% (main model);
  the supplementary model gives the same value. hV4 stays significant in the main model.
- **dS alone:** pooled it remains clearly below 0 (p = .0071), but no single area survives BH any more (V3, hV4 and TO
  had q = .036). It is now estimated on a base without a sustained field, at a common σ_AF.
- **Target gain:** larger and more widespread (7 of 9 areas).
- **Distractor gain:** stronger pooled in the main model (p = .0065). It weakens when dD is added (m7_full_zs,
  p = .050), consistent with the finding that the transient HP − LP terms are not supported.
- **Item weight:** about the same.

Not meaningful in this model:

- "Sustained gain at LP (joint)" is −dS/4 by construction.
- "dD joint" is 0 in the main model.
- The key-numbers file still lists both; drop them from the text.

RT links (`notes/data/rt_links_between_subjects_zs_lm_2026-11-13.tsv`, `rt_links_trialwise_zs_lm_2026-11-13.tsv`,
Fig. 7 `fig7_compact_zs_lm_2026-11-13_n28`):

- dS vs the behavioural HP benefit: ρ = +0.09 (alone) and +0.12 (joint), both non-significant, as before.
- Trial-wise:
  - Target gain on target repeats: −19% [−29, −9].
  - Distractor gain per SD log RT: −2.7 [−4.7, −0.7].
  - Global gain per SD log RT: −2.3 [−5.1, +0.5].
  - These are essentially unchanged from the earlier reference.

Caveat, field widths: `fit_width_profiles` re-solves g_HP and g_LP freely at each σ_AF. The sustained width figure
(`figS_field_widths_v3_zs_n28`) and the width rows of the key numbers therefore do not respect the zero-sum constraint.
A separate zero-sum width analysis exists (`retsupp/visualize/paper/width_profiles_zero_sum.py`,
`notes/data/width_profiles_zero_sum_2026-11-14*.tsv`, commit 55138c5); use that for the sustained width.
