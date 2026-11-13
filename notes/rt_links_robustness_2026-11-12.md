# Links between gain-field estimates and response times: refitted pRFs and voxel-selection robustness (2026-11-12/13)

Question: do the paper's links between the gain-field estimates and response times (RT) change (1) when the pRFs are
refitted together with the gain fields (the update-8 reference fit) and (2) when the voxel selection is loosened
(warm-started joint fits on looser model-free cross-run reliability selections, r > 0.2 and r > 0.1)?

**Decision rule:** the voxel-selection check kept the current selection (`notes/voxel_selection_robustness_2026-11-12.md`),
so the r > 0.2 and r > 0.1 results are supplementary. They go to the main text only if the main fit changes; it does not.

**Version of every number below:** "refitted" = the alternating polish root
`..._reliable_roihrf_polished` (update 8). A polish run to convergence (`..._polished_conv`) is coming; the RT links are
rerun on it with one command (bottom of this note), and the numbers here are then superseded for the main text.

n = 28 (sub-03 … sub-30) throughout.

## Fits compared

| Label | What it is | Root (`/shares/zne.uzh/gdehol/ds-retsupp/derivatives/`) | Estimated-alone dS |
|---|---|---|---|
| Earlier | Joint gain-field fits with the pRFs held fixed (clamped), reliable voxels | `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf` | `conditional_estimates_dog_final2_2026-11.tsv` |
| Refitted | pRFs refitted together with the gain fields (alternating polish), same voxels | `..._reliable_roihrf_polished` | `af_fixed_group_mechanisms_polished_2026-11-12` |
| r > 0.2 | Refitted, looser voxel selection (cross-run reliability r > 0.2 vs a shift null), warm-started | `..._relnull_r0.2_roihrf_polished` (249 subject × area cells) | `af_fixed_group_mechanisms_dS_relnull_r0.2_polished_2026-11-13` (new, dS only) |
| r > 0.1 | Same, r > 0.1 | `..._relnull_r0.1_roihrf_polished` (251 cells) | `af_fixed_group_mechanisms_dS_relnull_r0.1_polished_2026-11-13` (new, dS only) |

As for every paper number, estimated-alone fits that fail the linearity check (`valid` = False) are left out
(refitted 228 of 241 area cells kept at n = 28; r > 0.2 92%, r > 0.1 91% valid).

## A. Between subjects: dS vs the behavioural HP benefit

dS (g_HP − g_LP), per subject the median over the nine areas, vs the HP benefit (LP − HP RT, correct trials; median
25 ms). Spearman ρ, percentile bootstrap 95% CI over subjects (10 000 resamples). Table
`notes/data/rt_links_between_subjects_2026-11-13.tsv`; code `retsupp/visualize/paper/rt_links_between_subjects.py` (same
computation as `richter_followup.d1_intervals`; it reproduces the earlier ρ = −.11 and −.01 exactly).

| Fit | dS estimated alone: ρ [95% CI], p | dS joint (m7): ρ [95% CI], p |
|---|---|---|
| Earlier (paper) | −.11 [−.50, +.30], p = .57 | −.01 [−.39, +.38], p = .97 |
| Refitted | −.13 [−.46, +.25], p = .51 | +.06 [−.34, +.44], p = .77 |
| r > 0.2 | −.07 [−.45, +.33], p = .72 | −.08 [−.44, +.31], p = .67 |
| r > 0.1 | −.14 [−.52, +.26], p = .48 | −.04 [−.42, +.37], p = .85 |

No link in any fit; the paper sentence ("individual differences cannot be resolved") holds. Only the numbers change:
for the refitted reference, "Spearman ρ = −.13 [−.46, +.25], p = .51; dS joint ρ = +.06, p = .77".

## B. Fig. 7: trial-wise gains, history and RT

Trial-wise gains (`fit_trialwise_af_gains.py`, parametric model, leave-one-run-out jackknife SEs) on top of each fit's
m7_full, hierarchical Bayesian meta-analysis over subjects and areas (`trialwise_meta_bayes.py`), mean over all nine
areas, posterior mean [95% HDI], P = posterior probability of the sign shown. a: % of the area's typical gain;
b: gain per SD of log RT × 100. Table `notes/data/rt_links_trialwise_comparison_2026-11-13.tsv`
(`rt_links_compare.py`; primary, Student-t and no-SE-weighting variants).

Voxels: "signal" = each area's signal voxels, as the paper's Fig. 7 (voxels outside the m7 fit take the joint-HRF fold
pRFs); "own" = the fit's own voxels, areas with ≥ 10 voxels (new `--min-voxels 10`). The robustness roots can only be
judged with their own voxels (the signal-voxel set would undo the looser selection), so the refitted fit was also run
with its own voxels as the bridge.

| Effect | Earlier (paper, signal) | Refitted (signal) — new Fig. 7 | Refitted (own) | r > 0.2 (own) | r > 0.1 (own) |
|---|---|---|---|---|---|
| Target gain, target repeat | −19% [−33, −4], P = .997 ** | **−18% [−31, −4], P = .995 \*\*** | −16% [−29, −4], P = .994 * | −15% [−25, −4], P = .996 ** | −18% [−28, −9], P > .999 *** |
| Distractor gain, distractor repeat | −30% [−75, +13], P = .920 | −28% [−66, +11], P = .933 | −7% [−39, +23], P = .666 | −21% [−60, +19], P = .866 | −25% [−60, +9], P = .940 |
| Distractor gain per SD log RT | −2.4 [−4.3, −0.5], P = .990 * | **−2.0 [−3.7, −0.2], P = .990 \*** | −2.2 [−3.9, −0.5], P = .992 * | −1.5 [−3.0, −0.0], P = .975 | −1.3 [−2.9, +0.1], P = .960 |
| Global gain per SD log RT | −2.1 [−4.6, +0.6], P = .942 | −1.9 [−4.7, +0.6], P = .928 | −2.4 [−5.1, +0.4], P = .955 | −1.8 [−4.3, +0.6], P = .933 | −0.8 [−3.4, +1.9], P = .716 |
| Target gain per SD log RT | +0.9 [−1.3, +3.1], P = .804 | +0.9 [−0.9, +2.9], P = .833 | +0.8 [−1.1, +2.6], P = .823 | +1.3 [−0.1, +2.6], P = .968 | +1.2 [−0.0, +2.4], P = .970 |
| Subject × area estimates | 230 | 230 | 222 | 238 | 250 |

Stars: P > .975 *, > .995 **, > .9995 ***.

Sensitivity (refitted, signal): Student-t likelihood gives the same conclusions (target repeat P = .995, distractor RT
P = .988). Without the SE weighting the target-repeat effect keeps its direction but is no longer near the threshold
(−24%, P = .891; earlier −44%, P = .968), and the RT effects vanish as before (distractor P = .603, global P = .557).
In absolute gain units the target-repeat effect is −0.049 [−0.085, −0.014], P = .994 (earlier −0.052, P = .993).

Convergence (27 fits per version: 5 effects × 6 variants, global gain without relative variants; nutpie, 4 × 2000
draws, target_accept .99):

| Version | Divergences (all 27 fits) | Max R̂ | Min bulk ESS | PPC: 95% intervals cover | Area means inside PPC band |
|---|---|---|---|---|---|
| Refitted (signal) | 1 (target repeat, relative) | 1.01 | 700 | 95–99% | all |
| Refitted (own) | 1 (target repeat, absolute) | 1.01 | 851 | 94–99% | all |
| r > 0.2 (own) | 8 (distractor repeat, Student-t variant only) | 1.01 | 760 | 94–99% | not in every fit |
| r > 0.1 (own) | 3 (target repeat abs/rel, target RT relative; 1 each) | 1.01 | 608 | 93–99% | not in every fit |

**Which links survive.**
- Target gain lower on target-repeat trials: survives the refit (−18%, P = .995) and both looser selections
  (P ≥ .994). Robust.
- Distractor gain lower on slower trials: survives the refit (−2.0 per SD log RT, P = .990, same as before) and the
  refit with the fit's own voxels (P = .992), but weakens with looser selections (r > 0.2 P = .975, just at the
  threshold; r > 0.1 P = .960, not credible; per group only early areas at r > 0.1). It stays an SE-weighted effect
  that vanishes without the weighting. State it as in the paper, with the robustness rows in the supplement.
- Global gain on slower trials: not credible in any version (P = .72–.96), as in the paper.
- Target gain on slower trials: not credible (P = .80–.97); with looser selections it drifts toward higher target gain
  on slower trials (P = .968/.970), below threshold.
- Distractor repeat: no credible change in any version.
- Between subjects (A): no link in any version.

So the main fit does not change: the paper keeps its RT statements, with updated numbers from the refitted version, and
the r > 0.2 / r > 0.1 rows go to the supplement.

## Figures

- `notes/figures/paper_2026-11/fig7_compact_polished_n28.{pdf,png,md}` — new Fig. 7 on the refitted fits (same layout
  as `fig7_compact_n28`; legend regenerated from the posterior; per-group and per-area tables in the .md).
- `notes/figures/paper_2026-11/figS_fig7_relnull_r02_n28.*`, `figS_fig7_relnull_r01_n28.*` — supplementary versions on
  the looser selections (fit's own voxels). At r > 0.1 panel b's title becomes neutral ("Gains on slower trials") because
  no RT effect over all nine areas is credible.
- `fig7_compact.py` changes (default output unchanged, checked: the n = 28 legend re-renders identically): `--fit-label`;
  panel-b title and RT sentence follow the posterior when the distractor RT effect is not credible; the no-SE phrase says
  "not credible" below P = .95 instead of "just short of the threshold".

## Data and jobs

Trial-wise outputs (cluster `derivatives/`, local copies of the parametric/check/bins tables under `/data/ds-retsupp/derivatives/`):

| Version | Folder | SLURM |
|---|---|---|
| Refitted, signal voxels | `af_trialwise_gains_signal_se_polished_2026-11-12` | 8316523 (4–30), 8315809 (3), 8318739 (5, rerun) |
| Refitted, own voxels (≥ 10) | `af_trialwise_gains_ownvox10_polished_2026-11-12` | 8318717 (14 subjects with an area < 10 voxels or a failed task); the other 14 copied from `af_trialwise_gains_ownvox_polished_2026-11-12` (8315468, 8316268; identical settings, all areas ≥ 10 voxels) |
| r > 0.2, own voxels (≥ 10) | `af_trialwise_gains_ownvox10_relnull_r0.2_polished_2026-11-13` | 8331458 |
| r > 0.1, own voxels (≥ 10) | `af_trialwise_gains_ownvox10_relnull_r0.1_polished_2026-11-13` | 8331459 |
| dS estimated alone, r > 0.2 / r > 0.1 | `af_fixed_group_mechanisms_dS_relnull_r0.{2,1}_polished_2026-11-13` (MECHANISMS=dS, base m4_TDS of the root) | 8331523/8331711, 8331524/8331712 |

Failed tasks, all rerun or superseded: node u24-chaiam0-613 gave `PermissionError` on files under `/shares` (canary
8315467 sub-03, 8316523_5, 8316268_29; now excluded); 8316268_4 and _22 hit a singular matrix on 1-voxel IPS fits with
the fit's own voxels, which led to `--min-voxels`.

Local tables: `notes/data/trialwise_{signal_se,ownvox10}_polished_2026-11-12/`,
`notes/data/trialwise_ownvox10_relnull_r0.{2,1}_polished_2026-11-13/`, meta-analyses
`notes/data/trialwise_meta_bayes_<same tag>/`, gain parameters `notes/data/gain_field_parameters_relnull_r0.{2,1}_polished_2026-11-13.tsv`,
estimated-alone `notes/data/conditional_estimates_dS_relnull_r0.{2,1}_polished_2026-11-13.tsv`.

Resources (sacct): trial-wise tasks 8–17 min, ~1 CPU used of 2, MaxRSS up to 27 GB (signal voxels) and 26.5 GB
(r > 0.1); dS estimated alone 4 min, 21 GB.

## Rerun on another reference (e.g. the converged polish) — one command per stage

`retsupp/visualize/paper/rt_links_rerun.sh` (variables `REF_ROOT`, `ALONE_FOLDER`, `TAG`, `VOXELS`):

```bash
# cluster, ~/retsupp_plan after git pull
REF_ROOT=af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_conv ALONE_FOLDER=<its estimated-alone folder> \
    TAG=<tag> bash retsupp/visualize/paper/rt_links_rerun.sh cluster-submit      # then, when done: ... cluster-collect
# laptop, repo root
REF_ROOT=... ALONE_FOLDER=... TAG=<tag> bash retsupp/visualize/paper/rt_links_rerun.sh local
```

`local` writes `fig7_compact_<tag>_n28`, `rt_links_between_subjects_<tag>.tsv` and `rt_links_trialwise_<tag>.tsv`
(each next to the earlier version).
