# Zero-sum sustained field, fitted by joint LM: the new reference (2026-11-13)

STATUS: fits running / done on the cluster; downstream regeneration and the comparison table not yet run (cluster
unreachable from the laptop since ~19:10, VPN). Sections marked TODO are filled once that is done.

## Models (`retsupp/modeling/fit_cv_ladder_nested.py`, ZERO_SUM_MODELS)

The sustained field has no condition-invariant part: g_HP + 3·g_LP = 0, so it averages to 1 over the four HP
conditions and dS = g_HP − g_LP = 4/3·g_HP (why: `notes/joint_optimisation_2026-11-13.md`). All three models are fitted
with every parameter free (per-voxel DoG pRFs + shared terms) by the joint Levenberg–Marquardt of `joint_lm`
(400 iterations per start), from two starts each; the lower training SSE is kept and both are recorded
(`optimizer_info[0]['candidates']`):

| model | terms | starts |
|---|---|---|
| m4_TDS_zs | item weight, target, distractor (HP = LP); no sustained field | clamped m3b_TD (cold); polished m4_TDS with sustained gains 0 (warm) |
| **m5_TDS_dS_zs** (main model) | + zero-sum sustained field (dS) | parent m4_TDS_zs; clamped m5_TDS_dS projected onto zero-sum (g_HP = ¾ dS, g_LP = −¼ dS) |
| m7_full_zs (supplementary) | + transient HP − LP (dD) | parent m4_TDS_zs; clamped m7_full projected onto zero-sum |

Checks in the code: each start's prediction equals its parent's (max |Δpred| printed and asserted; 0 in every canary),
the zero-sum constraint holds (asserted), and terms that end on a bound are listed (`info['on_bound']`).

Bounds (zero-sum fits only, `zs_bounds`): sustained field non-negative (g_HP ∈ [−1, 3], g_LP ≥ −1); transient gains
∈ [−1, 5] (≥ −1: drive cannot become negative while the distractor / target is on); item scale ≥ 0.05; widths as
before (0.5–8°).

## Canaries (sub-03, 09, 17, 23 × V1, V3, hV4, TO, IPS; TO also sub-26)

Table: `notes/data/zero_sum_canary_2026-11-13.tsv` (cluster checkout; `summarize_zero_sum`).

- The two starts matter: they end 0.2–1 477 SSE apart (median 109); the cold start sometimes sticks with σ_dyn on its
  0.5° bound (sub-09 IPS: 340 428 vs 340 034 from the warm start).
- Without the condition-invariant level the LM moves m4_TDS_zs little from the clamped fits (tens to hundreds of SSE):
  the large drops in the identifiability study came from g_LP reshaping the pRFs.
- dS joint (m7_full_zs) tracks the refitted-polished reference closely in most cells (e.g. sub-03 V1 −0.126 vs −0.129,
  V3 0.042 vs 0.040; sub-09 V1 0.156 vs 0.157; sub-17 V3 −0.224 vs −0.225; sub-23 V3 −0.506 vs −0.502), with
  exceptions where the reference sat on bounds (sub-17 IPS 0.65 vs 1.80; sub-26 TO).
- Runtime: up to 3 h 13 min per cell for 2 models (500 voxels); peak RSS 25.6 GB (data loading).

## TO decision

TO (sub-03, 09, 26) is degenerate under every bound set tried: the item scale goes to its floor and the transient
gains to their bounds (also with the old bounds item ≥ 0.001, transient ≤ 10; the tighter bounds cost 150–330 SSE).
Cause: the transient terms scale the item drive, so once the item scale goes to 0 they lose their meaning; the model
has no location-blind onset transient for the search array, which TO seems to want. A reparametrisation of the
transient terms would change the model class every downstream script uses, so it was not done. Decision: keep the
bounds above, record on-bound terms per cell, and report the paper's statistics (medians, Wilcoxon, BH) with and
without on-bound fits (`compare_zero_sum` third column). A clean fix would need a new term (a location-blind onset
transient).

## Runs

- m4_TDS_zs + m7_full_zs: job 8341754 (241 cells, one task each), root
  `derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm`.
- m5_TDS_dS_zs: jobs 8342980–8343220 (one per cell, `afterok` on its 8341754 task), same root.

## TODO once the cluster is reachable

1. Check all three models have 241 fits (`ls <root>/<model>/sub-*/*cv-fits.pkl | wc -l`), resubmit failures.
2. `summarize_zero_sum --zs-root ..._zs_lm --out notes/data/zero_sum_fits_2026-11-13.tsv`.
3. View root `..._zs_lm_ref`: `m4_TDS -> ../..._zs_lm/m4_TDS_zs`, `m7_full -> ../..._zs_lm/m5_TDS_dS_zs` (main model:
   every 'joint' number, hierarchy bins, width profiles, Fig. 7), `m7_full_zs`, `m5_TDS_dS_zs` symlinked under their
   own names.
4. `ZERO_SUM=1 SUBMIT_ALONE=1 REF_ROOT=..._zs_lm_ref ALONE_FOLDER=af_fixed_group_mechanisms_zs_lm_2026-11-13
   TAG=zs_lm_2026-11-13 SUFFIX=_zs_n28 bash retsupp/visualize/paper/regenerate_reference.sh cluster-submit`, then
   `cluster-collect`, then `local`; the same variables for `regenerate_hierarchy_gain_terms.sh` and `rt_links_rerun.sh`.
   (The scripts' sbatch lines exclude only u24-chaihm0-633; add u24-chaiam0-613.)
5. m7_full_zs numbers alongside: `collect_gain_parameters --root dog=..._zs_lm` (rows m7_full_zs, m5_TDS_dS_zs).
6. `compare_zero_sum --ref notes/data/reference_numbers_polished_2026-11-12_n28.tsv --zs
   notes/data/reference_numbers_zs_lm_2026-11-13_n28.tsv --zs-gain-table ... --zs-summary ... --out
   notes/data/compare_zero_sum_2026-11-13` → the comparison table below.

Caveat for step 4: the sustained-width profiles (`fit_width_profiles`) re-solve g_HP and g_LP freely at each width, not
along the zero-sum direction, and the "sustained (mean of 4 locations)" width summary is exactly the part zero-sum
removes; the sustained-width figure needs re-thinking before it is used with this reference.

## Comparison: refitted-polished (alternating) vs zero-sum LM

TODO (dS alone, dS joint, target, distractor, item weight; per area and pooled; main model m5_TDS_dS_zs, m7_full_zs
alongside).
