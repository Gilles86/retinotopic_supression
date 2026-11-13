# Cleanup proposal: derivatives, notes/data and working notes (2026-11-13)

**Read-only inventory. Nothing has been moved, deleted or changed.** Every action below needs the author's OK, item by
item. Several jobs and agents were still running when this was made (see "Running jobs").

Inputs: cluster `/shares/zne.uzh/gdehol/ds-retsupp/derivatives/` (207 folders; sizes and newest-file dates from one
`find` traversal per folder in an `srun -A hare.econ.uzh -p standard` job, 8332688, cross-checked against the CephFS
recursive stats); local `/data/ds-retsupp/derivatives/` (83 entries, local `du`); repo `notes/data/` (415 entries) and
`notes/*.md` (24 notes). "Referenced" means the exact folder name appears in `PAPER_REPRODUCTION.md`, the writing
packages `notes/writing_package_2026-11-12_*`, the newest notes `notes/*2026-11-1[23]*.md`, the defaults or documented
roots of tracked code outside `archive/`, or names a source table of the key numbers (update 4 and update 8).

Sizes are in GiB (written GB). Full per-item tables (status, size, date, reason, proposed action, reference counts):

- `/Users/gdehol/git/retsupp/notes/data/cleanup_2026-11-13/cluster_derivatives.tsv`
- `/Users/gdehol/git/retsupp/notes/data/cleanup_2026-11-13/local_derivatives.tsv`
- `/Users/gdehol/git/retsupp/notes/data/cleanup_2026-11-13/notes_data.tsv`

## Summary

| Where | KEEP | USED | RUNNING/RECENT | UNKNOWN | SUPERSEDED | Total |
|---|---|---|---|---|---|---|
| Cluster derivatives | 916.6 GB (7) | 131.0 GB (49) | 5.7 GB (18) | 5.6 GB (8) | **412.7 GB (125)** | 1471.7 GB (207) |
| Local derivatives | 171.5 GB (6) | 0.5 GB (14) | — | 0.3 GB (7) | **85.8 GB (56)** | 258.1 GB (83) |

Biggest wins (cluster):

1. `prf_regressed_out`, **201 GB**: input of the eccentric GLM only, which is not in the paper or in
   `PAPER_REPRODUCTION.md`; reproducible from `cleaned` + the m4 fit with `retsupp/eccentric_glm/clean_data.py`.
   Proposed: delete (no backup needed).
2. `glmsingle`, **181 GB**: GLMsingle, archived from the paper on 2026-11-12. Recomputing costs hours per subject.
   Proposed: delete, or back up to g_econ first if GLMsingle may come back (the share is append-only, so that copy is
   permanent).
3. `eccentric_glm` + `stimulus_rois` (7.4 GB), May `decoded` (6.0 GB), `prf_conditionfit.bak` (5.2 GB) and 13
   superseded ladder generations `af_prf_cv_ladder*` (8.4 GB). About 27 GB together.
4. Locally: `glmsingle` (46 GB), `prf_regressed_out` (19 GB), `eccentric_glm` (7 GB), `decoded` (6 GB). The working
   copies of `fmriprep` / `cleaned` / `mean_signal` (162 GB) also exist on the cluster and could go if not needed
   locally.

Deleting everything marked SUPERSEDED would take the cluster from 1.47 TB to about 1.06 TB. About 917 GB of that is
KEEP (fmriprep, cleaned, NORDIC, mean signal, neuropythy, whole-brain pRFs, `paper_2026-11`) and 107 GB is
`cleaned_bold_cache` (USED by `decode_final.py`, but a regenerable cache).

### Risks and things to settle first

- **Code defaults still point at superseded roots.** `retsupp/visualize/paper/fig3_collect_loso.py` defaults to
  `af_fixed_group_exact_{dog,dn}`; `collect_conditional_estimates.py` defaults to `af_fixed_group_mechanisms_maxsd20`;
  `fit_trialwise_af_gains.sh` writes to `af_trialwise_gains` (or `_hp`) by default. The paper tables come from the
  `*_final2_2026-11` / `_polished` roots. Point the defaults at those before deleting, or a bare rerun will fail.
- **Two near-duplicate pairs** (estimated-alone dS, voxel-selection robustness):
  `af_fixed_group_mechanisms_relnull_r0.{1,2}_polished_dS` and `af_fixed_group_mechanisms_dS_relnull_r0.{1,2}_polished_2026-11-13`
  have the same file lists but different contents. The notes name the dated pair. The `regenerate_reference.sh` example
  names the undated pair, and both have tables in `notes/data/`. The author should say which pair is canonical.
- **Local-only data.** Local `prf_conditionfit` (2.2 GB vs 0.1 GB on the cluster) and `prf_roi_cache` (0.11 vs
  0.01 GB) hold more than the cluster copies. `prf_runfit`, `prf_summaries{,.conditionwise,.runwise}`,
  `attention_model`, `glmsingle_summaries` and several `af_prf_joint_*` folders exist only locally. Copy them before
  deleting anything locally (see `~/.claude/reference/data-archival.md`: about 23k local derivative files are on
  neither the cluster nor g_econ).
- **`cleaned` (385 GB) is on neither g_econ nor anywhere else.** `backup_to_gecon.py` covers raw, sourcedata and
  fmriprep only. It is reproducible with `clean.py`, but it is the input to everything. KEEP.
- `af_trialwise_gains_ownvox_polished_2026-11-12`: per the RT-links note, 14 subjects of `..._ownvox10_polished` were
  copied from it. Check the copy before deleting it.
- Untracked `notes/data` tables were being written while this was made (one vanished mid-scan), so re-list them
  before acting.
- The `.git` folder is 2.8 GB. Tracked `notes/data/trialwise_meta_bayes_*polished*` folders hold about 720 MB of
  posteriors. Untracking them is not proposed here; the history already holds them.

### Running jobs (squeue, 2026-11-13 ~09:00)

483 tasks `cvl_nested_full_bounded_polish_maxsd20_sse_dog_hprot0_reliable_roihrf_warmpolish`, 1 task of the r > 0.1
warm polish, `side_hpc_prf_lateral_occipital_true`, `side_hpc_prf_precentral_scramble`, `side_prf_atlas_group`,
`side_signedprf_sub-30` (the last two write outside `derivatives/`), and the `heal_env_holds` scrontab. Output roots
written in the last 24 h are listed as RUNNING/RECENT below. Re-run the classification after these finish.

## 1. Cluster derivatives

### Totals (cluster)

| Status | Folders | GB |
|---|---|---|
| KEEP | 7 | 916.6 |
| RUNNING/RECENT | 18 | 5.7 |
| SUPERSEDED | 125 | 412.7 |
| UNKNOWN | 8 | 5.6 |
| USED | 49 | 131.0 |
| **All** | 207 | 1471.7 |

### KEEP (irreproducible or expensive)

| Folder | GB | Last modified | Why |
|---|---|---|---|
| `fmriprep` | 459.6 | 2026-11-06 | fMRIPrep 25.2.3 outputs (incl. FreeSurfer); pipeline stage in PAPER_REPRODUCTION.md; days of compute |
| `cleaned` | 384.9 | 2026-01-06 | confound-cleaned BOLD, input to every fit; not on g_econ |
| `mean_signal` | 33.8 | 2026-01-06 | mean time series (pipeline stage, PAPER_REPRODUCTION.md) |
| `nordic` | 26.0 | 2025-12-16 | NORDIC-denoised BOLD (pipeline stage); input to fMRIPrep |
| `prf` | 9.6 | 2026-05-19 | whole-brain pRF fits m1-m6 (GPU sweep; Fig. 2 inputs, CLAUDE.md QC); expensive |
| `paper_2026-11` | 1.7 | 2026-11-13 | PAPER_REPRODUCTION.md, rt_links_rerun.sh / regenerate_reference.sh (roi_hrf_dog_final2.tsv), writing packages |
| `neuropythy` | 1.0 | 2026-05-19 | visual-area atlas (PAPER_REPRODUCTION.md; all ROIs) |

### USED

| Folder | GB | Last modified | Referenced by |
|---|---|---|---|
| `cleaned_bold_cache` | 106.8 | 2026-05-12 | decode_final.py, simulate_decode.py, fit_prf.py (regenerable cache of cleaned BOLD) |
| `decoded_final_2026-11` | 16.7 | 2026-11-12 | Fig. 6 (decode_final.py, fig6_collect.py, figS_decoded_bars.py, group_decoded_frames.py) |
| `decoded_sim_fitted_2026-11` | 1.5 | 2026-11-12 | Fig. 6 model prediction (simulate_decode.py --mode fitted) |
| `af_prf_cv_ladder_nested_run3_frozenprf_bounded_maxsd20_dn_basefield_2026-11` | 0.6 | 2026-11-09 | figS_dn_gain_locus held-out panel (dn_basefield_cv*_2026-11.tsv) |
| `af_trialwise_gains_signal_se_2026-11` | 0.6 | 2026-11-10 | Fig. 7 at update 4 (trialwise_signal_se_2026-11/, "Earlier (paper, signal)" in rt_links_robustness_2026-11-12.md) |
| `af_trialwise_gains_ownvox10_relnull_r0.1_polished_2026-11-13` | 0.6 | 2026-11-13 | rt_links_robustness_2026-11-12.md (figS_fig7_relnull_r01) |
| `af_trialwise_gains_ownvox10_relnull_r0.2_polished_2026-11-13` | 0.5 | 2026-11-13 | rt_links_robustness_2026-11-12.md (figS_fig7_relnull_r02) |
| `prf_cv_roi_jointhrf` | 0.5 | 2026-11-10 | paper pRFs with joint HRF (PAPER_REPRODUCTION.md stage; decode_final.py) |
| `af_trialwise_gains_signal_se_polished_2026-11-12` | 0.5 | 2026-11-12 | new Fig. 7 (rt_links_robustness_2026-11-12.md, rt_links_rerun.sh) |
| `af_trialwise_gains_ownvox10_polished_2026-11-12` | 0.5 | 2026-11-12 | rt_links_robustness_2026-11-12.md |
| `prf_cv_roi` | 0.3 | 2026-11-02 | pRF CV fits per ROI (decode_final.py, voxel_reliability_null.py, fit_prf_cv_roi_hrf_joint.py) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf` | 0.2 | 2026-11-09 | PAPER_REPRODUCTION.md (BASE_ROOT, final fits of update 4); key_numbers update4/8 |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_reliable_roihrf` | 0.2 | 2026-11-10 | DN robustness fits (gain_field_parameters_final2_2026-11.tsv; key_numbers update4 "Robustness") |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_gauss_reliable_roihrf` | 0.2 | 2026-11-10 | Gaussian robustness fits (gain_field_parameters_final2_2026-11.tsv) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_basefield_2026-11` | 0.2 | 2026-11-09 | figS_dn_gain_locus (dn_basefield_full_2026-11.tsv), dn_gain_field_variants_2026-11.md |
| `decoded_sim_2026-11` | 0.1 | 2026-11-09 | figS_decode_simulation (simulate_decode.py, collect_simulate_decode.py) |
| `af_prf_cv_ladder_nested_run3_frozenprf_bounded_maxsd20_reliable_roihrf_dogcs_2026-11` | 0.1 | 2026-11-09 | figS_dog_kappa panel c (run3 held-out dR2) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.1_roihrf_polished` | 0.1 | 2026-11-12 | voxel_selection_robustness_2026-11-12.md (gain_field_parameters_relnull_r0.1_polished_2026-11-13.tsv) |
| `prf_cv_roi_sharedhrf` | 0.1 | 2026-11-09 | build_roi_hrf_table.py, decode_final.py |
| `af_fixed_group_mechanisms_final2_2026-11` | 0.08 | 2026-11-10 | conditional_estimates_dog_final2_2026-11.tsv (key_numbers update4: dS, S estimated alone) |
| `af_fixed_group_mechanisms_polished_2026-11-12` | 0.08 | 2026-11-12 | PAPER_REPRODUCTION.md ALONE_FOLDER (update 8) |
| `af_fixed_group_mechanisms_clampedcheck_2026-11-12` | 0.08 | 2026-11-12 | prf_refit_with_gains_2026-11-12.md (check of the polished fit) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.2_roihrf_polished` | 0.07 | 2026-11-12 | regenerate_reference.sh example, voxel_selection_robustness, WPKG update8 |
| `af_fixed_group_exact_dog_final2_2026-11` | 0.04 | 2026-11-10 | loso_exact_final2*.tsv (tableS_loso, key_numbers update4) |
| `af_fixed_group_exact_dn_final2_2026-11` | 0.04 | 2026-11-10 | loso_exact_robust_final2*.tsv (LOSO with DN pRFs) |
| `af_fixed_group_exact_gauss_final2_2026-11` | 0.04 | 2026-11-10 | loso_exact_robust_final2*.tsv (LOSO with Gaussian pRFs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished` | 0.03 | 2026-11-12 | PAPER_REPRODUCTION.md reference fit (update 8); regenerate_reference.sh / rt_links_rerun.sh REF_ROOT |
| `prf_cv_roi_reliability_null_folds` | 0.03 | 2026-11-12 | decode_final.py --selection relnull; figS_decoding_voxel_selection |
| `prf_dn_by_hp_2026-11` | 0.03 | 2026-11-09 | figS_dn_params_by_hp (fit_dn_by_hp.py, collect_dn_by_hp.py) |
| `af_fixed_group_mechanisms_dS_relnull_r0.1_polished_2026-11-13` | 0.02 | 2026-11-13 | rt_links_robustness_2026-11-12.md, voxel_selection_robustness_2026-11-12.md |
| `af_fixed_group_exact_dnw_2026-11` | 0.02 | 2026-11-10 | loso_kappa_dnw / n28/loso_dnw_2026-11_n28.tsv |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_dogcs_2026-11` | 0.02 | 2026-11-09 | figS_dog_kappa (dog_cs_field_full_2026-11.tsv), collect_dog_cs_field.py |
| `prf_cv_roi_reliability` | 0.02 | 2026-11-03 | reliable voxels (PAPER_REPRODUCTION.md stage) |
| `prf_cv_roi_reliability_null` | 0.02 | 2026-11-12 | voxel_reliability_null.py; decoding_voxel_selection_robustness_2026-11-12.md |
| `af_fixed_group_mechanisms_dS_relnull_r0.2_polished_2026-11-13` | 0.01 | 2026-11-13 | rt_links_robustness_2026-11-12.md, voxel_selection_robustness_2026-11-12.md |
| `af_fixed_group_linear_session2_2026-11-11` | 0.01 | 2026-11-11 | richter_followup.py A3 (per-session dS) |
| `af_fixed_group_linear_session1_2026-11-11` | 0.01 | 2026-11-11 | richter_followup.py A3 (per-session dS), notes/richter_followup_2026-11-11.md |
| `gain_bins_2026-11-12` | 0.01 | 2026-11-12 | hierarchy_gain_terms.py, fit_gain_bins.sh |
| `field_widths_reliable_2026-11-12` | 0.01 | 2026-11-12 | field_widths_reliable_2026-11-12.md, WPKG update4 (widths, fixed pRFs) |
| `field_widths_reliable_polished_2026-11-12` | 0.01 | 2026-11-12 | regenerate_reference.sh (field_widths_reliable_${TAG}), PAPER_REPRODUCTION.md widths |
| `af_fixed_group_exact_kappa_2026-11` | 0.01 | 2026-11-10 | loso_kappa_*.tsv (figS_dog_kappa LOSO, n28) |
| `baseline_vs_gain_2026-11-12` | 0.01 | 2026-11-12 | baseline_vs_gain_2026-11-12.md, WPKG, figS_baseline_vs_gain.py |
| `decoded_model_pred_2026-11` | 0.01 | 2026-11-11 | decode_model_prediction.py (code default) |
| `prf_v1_resolution_2026-11-12` | 0.01 | 2026-11-12 | v1_prf_size_floor_2026-11-12.md |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.1_roihrf` | 0.00 | 2026-11-12 | base (unpolished) root of the r>0.1 polished fit; a warm-polish job on r0.1 is running |
| `sim_prf_size_floor_2026-11-12` | 0.00 | 2026-11-12 | v1_prf_size_floor_2026-11-12.md |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.2_roihrf` | 0.00 | 2026-11-12 | base (unpolished) root of the r>0.2 polished fit |
| `dn_baseline_field_recovery_2026-11` | 0.00 | 2026-11-09 | figS_dn_gain_locus recovery (collect_dn_baseline_field.py) |
| `dog_cs_field_recovery_2026-11` | 0.00 | 2026-11-09 | dog_cs_field recovery (recover_dog_cs_field.sh) |

### RUNNING/RECENT (written in the last 24 h or by a running job; re-check after the jobs finish)

| Folder | GB | Last modified | Note | Proposed afterwards |
|---|---|---|---|---|
| `side_hpc_region_ts` | 5.0 | 2026-11-12 | written in the last 24 h (running job output) | re-check after current jobs |
| `af_trialwise_gains_ownvox_polished_2026-11-12` | 0.5 | 2026-11-12 | written in the last 24 h; af_trialwise_gains_ownvox10_polished_2026-11-12 (14 subjects copied from here) | delete after the author confirms the copy (after the running jobs) |
| `side_hpc_prf_cv` | 0.1 | 2026-11-12 | written in the last 24 h (running job output) | re-check after current jobs |
| `af_fixed_group_mechanisms_relnull_r0.1_polished_dS` | 0.02 | 2026-11-13 | written in the last 24 h; same file list as af_fixed_group_mechanisms_dS_relnull_r0.1_polished_2026-11-13 but contents differ; conditional_estimates_relnull_r0.1_polished_dS.tsv; regenerate_reference.sh example names this folder | author to decide after the running jobs |
| `af_fixed_group_mechanisms_relnull_r0.2_polished_dS` | 0.01 | 2026-11-13 | written in the last 24 h; same file list as ..._dS_relnull_r0.2_polished_2026-11-13 but contents differ; regenerate_reference.sh example names this folder | author to decide after the running jobs |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_conv` | 0.01 | 2026-11-13 | written in the last 24 h (running job output) | re-check after current jobs |
| `decoded_sim_fitted_2026-11.canary_concat_bak_202611121422` | 0.01 | 2026-11-12 | written in the last 24 h; backup copy of a superseded or rerun folder | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.1_roihrf_coldtest` | 0.00 | 2026-11-12 | written in the last 24 h; canary / test run | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.2_roihrf_coldtest` | 0.00 | 2026-11-12 | written in the last 24 h; canary / test run | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.1_roihrf_coldtest_polished` | 0.00 | 2026-11-12 | written in the last 24 h; canary / test run | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_lbfgscanary_bak` | 0.00 | 2026-11-12 | written in the last 24 h; backup copy of a superseded or rerun folder | delete (after the running jobs) |
| `af_fixed_group_mechanisms_relnull_r0.2_polished_dS_fallbackbase_bak` | 0.00 | 2026-11-13 | written in the last 24 h; backup copy of a superseded or rerun folder | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_4rounds_partial_bak` | 0.00 | 2026-11-12 | written in the last 24 h; backup copy of a superseded or rerun folder | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_alttest` | 0.00 | 2026-11-12 | written in the last 24 h; canary / test run | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_conv_warmr02` | 0.00 | 2026-11-13 | written in the last 24 h (running job output) | re-check after current jobs |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_cputest` | 0.00 | 2026-11-12 | written in the last 24 h; canary / test run | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished_gputest` | 0.00 | 2026-11-12 | written in the last 24 h; canary / test run | delete (after the running jobs) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.2_roihrf_coldtest_polished` | 0.00 | 2026-11-12 | written in the last 24 h; canary / test run | delete (after the running jobs) |

### UNKNOWN

| Folder | GB | Last modified | What is known |
|---|---|---|---|
| `cleaned_roi_cache` | 3.6 | 2026-05-11 | build_cleaned_roi_cache.sh; regenerable cache |
| `prf.hrffix` | 1.3 | 2026-11-04 | read by retsupp/visualize/pycortex/render_fig1_prf.py; check whether any Fig. 2 render still uses it |
| `af_trialwise_gains_hp` | 0.6 | 2026-11-06 | default output of fit_trialwise_af_gains.sh HP_NEUTRAL=1; notes/data/trialwise_rt_bayes_hp_2026-11/ (not in a package) |
| `af_prf_cv_ladder_nested_full_bounded_maxsd20_dT` | 0.04 | 2026-11-06 | dT variant ladder; joint_dT_dog_2026-11.tsv? not cited in packages |
| `af_fixed_group_mechanisms_maxsd20_dT` | 0.02 | 2026-11-06 | joint_dT_dog_2026-11.tsv? not cited in packages |
| `prf_roi_cache` | 0.01 | 2026-05-15 | Subject.get_prf_roi cache (utils/data.py); regenerable |
| `prf_diagnostics` | 0.01 | 2026-05-19 | R2-mixture p_signal maps (compute_r2_mixture.py); user no longer uses p_signal for selection |
| `outlier_voxels` | 0.00 | 2026-11-03 | diagnose_outlier_voxels.py (project_outlier_variance_voxels memory) |

### SUPERSEDED, grouped (every folder with its size is in `notes/data/cleanup_2026-11-13/cluster_derivatives.tsv`)

| Group | Superseded by / reason | Folders | GB | Proposed action |
|---|---|---|---|---|
| `af_fixed_group_exact_dn` (0.07), `af_fixed_group_exact_dog` (0.07) | *_final2_2026-11 versions (note: fig3_collect_loso.py default still names these) | 2 | 0.1 | delete |
| `af_prf_cv_ladder`, `af_prf_cv_ladder_nested`, `af_prf_cv_ladder_run3`, `af_prf_cv_ladder_nested_bounded`, `af_prf_cv_ladder_nested_run3_bounded`, `af_prf_cv_ladder_nested_full_bounded_maxsd20_dn`, … (13 folders) | ..._maxsd20_{,dn_,gauss_}reliable_roihrf (final2) and its _polished refit; earlier October-November ladder generations (CV-ladder verdict era) | 13 | 8.4 | delete |
| `glmsingle` (180.5) | GLMsingle (archived 2026-11-12: not in the paper) | 1 | 180.5 | backup to g_econ (permanent) or delete; recompute is hours/subject |
| `af_fixed_group_mechanisms`, `af_fixed_group_mechanisms_dn`, `af_fixed_group_mechanisms_maxsd20`, `af_fixed_group_mechanisms_maxsd20_even`, `af_fixed_group_mechanisms_maxsd20_hprot1`, `af_fixed_group_mechanisms_maxsd20_hprot2`, … (10 folders) | af_fixed_group_mechanisms_final2_2026-11 (+ linear_session*, split models in gain_field_parameters_final2; HP-rotation null archived). collect_conditional_estimates.py default still names _maxsd20 | 10 | 0.7 | delete |
| `af_trialwise_gains_signal_2026-11` (0.6), `af_trialwise_gains` (0.5), `af_trialwise_gains_final2_2026-11` (0.4) | af_trialwise_gains_signal_se_2026-11 (and the _polished versions) | 3 | 1.5 | delete |
| `af_prf_cv_shiftvsgain`, `af_prf_cv_v2`, `af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_repeat_pSig0.5`, `af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_targetRepeat_pSig0.5`, `af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_pSig0.5_base-m6`, `af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_allSharedSigma_pSig0.5`, … (38 folders) | archived AF / Klein / shift analyses (conditionwise_shifts README; May 2026 AF "sharedSigma" era) | 38 | 0.9 | delete (code archived; small) |
| `prf_conditionfit` (0.1) | archived conditionwise pRF shifts (archive/conditionwise_shifts/README) | 1 | 0.1 | backup to g_econ, then delete |
| `af_fixed_group_gain` (0.04) | archived fixed-group gain (archive/paper_cleanup_2026-11-12) | 1 | 0.0 | delete |
| `af_learning_gains_nested` (0.00) | archived learning-gain models (paper_cleanup README) | 1 | 0.0 | delete |
| `prf_conditionfit.bak` (5.2), `af_prf_joint_dynamic_v3_dog_with_target_sharedSigma_allSharedSigma_pSig0.5.bak.2026-05-15` (0.02), `af_prf_cv_ladder_nested_full_bounded_maxsd20_gauss_reliable_roihrf_partialhrf_bak_2026-11-10` (0.01), `af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_reliable_roihrf_badtable_bak_2026-11-09` (0.00), `af_prf_cv_shiftvsgain_smoke.bak.20260531_155537` (0.00) | backup copy of a superseded or rerun folder | 5 | 5.2 | delete |
| `prf.snakemake_demo`, `_m2`, `prf.bench.gpu10x`, `prf_cv_roi_bench_cpu16`, `prf_cv_roi_bench_cpu32`, `prf_cv_roi_bench_gpu`, … (17 folders) | benchmark / debug / smoke output | 17 | 0.1 | delete |
| `af_trialwise_gains_ladder_canary`, `af_trialwise_gains_refactorcheck_2026-11`, `af_prf_cv_ladder_nested_fasttest_2026-11`, `af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_basefield_canary`, `af_prf_cv_ladder_nested_full_bounded_maxsd20_dn.canary1_nofloor`, `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_dogcs_canary`, … (20 folders) | canary / test run | 20 | 0.1 | delete |
| `decoded` (6.0), `v1_decode_handoffs` (0.05), `decoded_paradigm_sweep` (0.01), `decode` (0.00), `decoded_paradigm` (0.00) | decoded_final_2026-11 (decode_final.py); May 2026 VSS decoding | 5 | 6.1 | delete (VSS-era; reproducible with decode.py) |
| `eccentric_glm` (7.2), `stimulus_rois` (0.2), `eccentric_glm_summary` (0.01) | eccentric GLM: not in PAPER_REPRODUCTION.md / the paper | 3 | 7.4 | backup the small summary, delete the rest |
| `af_fixed_group_exact_dog_DonT` (0.01) | exploratory variant; LOSO tables come from *_final2_2026-11 | 1 | 0.0 | delete |
| `prf_regressed_out` (201.4) | input of the eccentric GLM only (not in the paper); reproducible from cleaned + m4 via eccentric_glm/clean_data.py | 1 | 201.4 | delete |
| `neuropythy_backup_20260509` (0.1) | neuropythy (current atlas) | 1 | 0.1 | delete |
| `figures` (0.00), `voxel_traces_cache` (0.00) | stray / empty | 2 | 0.0 | delete |


**Notes on the SUPERSEDED groups.** The `af_prf_cv_ladder*` generations (`af_prf_cv_ladder`, `_run3`, `_nested`,
`_nested_bounded`, `_nested_bounded_polished`, `_nested_full_bounded`, `_full_bounded_maxsd20`, `_maxsd20_dn`,
`_maxsd20_hprot1`, `_maxsd20_dn_hprot1`, `_maxsd20_reliable`, `_nll`, `_nested_run3_bounded`) predate the
reliable-voxel, per-area-HRF fits (`..._maxsd20_{,dn_,gauss_}reliable_roihrf`, the final2 roots of update 4) and their
polished refit. The run-3 held-out panels of the supplements use the `run3_frozenprf_*` roots, which stay. The
`af_fixed_group_mechanisms` family before `_final2_2026-11` (`_dn`, `_maxsd20`, its even/odd/ses/hprot splits) is
superseded by `_final2_2026-11`, `af_fixed_group_linear_session{1,2}_2026-11-11` (per-session dS) and the
HP-rotation null, which has been archived. The archived AF / Klein / shift roots (`af_prf_joint_*`, `af_prf_klein_*`,
`af_prf_cv_shiftvsgain`, `af_prf_cv_v2`, `af_three_models`, `sim_refit`, `gam_clusters`, `prf_distractors`) are the
May-2026 "sharedSigma" era and the conditionwise-shift work (`archive/conditionwise_shifts/README.md`). They total
under 1 GB, so deleting them gains little; keep them if in doubt. `prf_conditionfit` is the archived conditionwise
fit. The README says the data were left in place, and the local copy is larger (see Risks), so back it up before
deleting.

Recommendation per SUPERSEDED folder: **delete** for canaries, tests, smokes, `_bak` copies, benchmarks and the
superseded generations above (all reproducible from tracked code). **Back up to g_econ first** only for
`glmsingle` (if it may return) and `prf_conditionfit` (merge the larger local copy first). Nothing in SUPERSEDED is
raw or expensive except `glmsingle`.

## 2. Local `/data/ds-retsupp/derivatives`

Statuses follow the cluster folder of the same name. For KEEP/USED the cluster copy is canonical, so the local copy is
a convenience (local disk is at 96%, 81 GB free).

### Totals (local /data/ds-retsupp/derivatives)

| Status | Folders | GB |
|---|---|---|
| KEEP | 6 | 171.5 |
| SUPERSEDED | 56 | 85.8 |
| UNKNOWN | 7 | 0.3 |
| USED | 14 | 0.5 |
| **All** | 83 | 258.0 |

| Folder | Status | GB | On cluster | Proposed action |
|---|---|---|---|---|
| `fmriprep` | KEEP | 75.4 | yes | keep (cluster copy is canonical; the local copy can go when not needed for local work) |
| `cleaned` | KEEP | 57.1 | yes | keep (cluster copy is canonical; the local copy can go when not needed for local work) |
| `glmsingle` | SUPERSEDED | 45.9 | yes | backup to g_econ (permanent) or delete; recompute is hours/subject |
| `mean_signal` | KEEP | 29.1 | yes | keep (cluster copy is canonical; the local copy can go when not needed for local work) |
| `prf_regressed_out` | SUPERSEDED | 19.5 | yes | delete |
| `prf` | KEEP | 9.8 | yes | keep (cluster copy is canonical; the local copy can go when not needed for local work) |
| `eccentric_glm` | SUPERSEDED | 7.2 | yes | backup the small summary, delete the rest |
| `decoded` | SUPERSEDED | 5.7 | yes | delete (VSS-era; reproducible with decode.py) |
| `prf_conditionfit` | SUPERSEDED | 2.2 | yes | backup to g_econ, then delete |
| `prf_conditionfit.bak` | SUPERSEDED | 0.9 | yes | delete |
| `af_prf_cv_ladder_nested_bounded` | SUPERSEDED | 0.7 | yes | delete |
| `prf_runfit` | SUPERSEDED | 0.5 | no | copy to cluster or g_econ first, then delete locally |
| `af_prf_cv_ladder_nested_run3_bounded` | SUPERSEDED | 0.5 | yes | delete |
| `prf_summaries.conditionwise` | SUPERSEDED | 0.5 | no | backup to g_econ, then delete locally |
| `attention_model` | SUPERSEDED | 0.4 | no | backup to g_econ, then delete locally |
| `prf_cv_roi` | USED | 0.3 | yes | keep |
| `prf_summaries.conditionwise.bak` | SUPERSEDED | 0.2 | no | delete |
| `stimulus_rois` | SUPERSEDED | 0.2 | yes | backup the small summary, delete the rest |
| `prf_summaries.runwise` | SUPERSEDED | 0.2 | no | backup or delete |
| `af_prf_cv_ladder_nested` | SUPERSEDED | 0.2 | yes | delete |
| `af_fixed_group_mechanisms_dn` | SUPERSEDED | 0.1 | yes | delete |
| `af_fixed_group_mechanisms_maxsd20` | SUPERSEDED | 0.1 | yes | delete |
| `af_fixed_group_mechanisms` | SUPERSEDED | 0.1 | yes | delete |
| `prf.hrffix` | UNKNOWN | 0.1 | yes | author to decide |
| `prf_roi_cache` | UNKNOWN | 0.1 | yes | author to decide |
| `glmsingle_summaries` | SUPERSEDED | 0.09 | no | keep small copy or backup |
| `prf_summaries` | UNKNOWN | 0.07 | no | copy to cluster first |
| `af_fixed_group_exact_dn` | SUPERSEDED | 0.07 | yes | delete |
| `af_fixed_group_exact_dog` | SUPERSEDED | 0.07 | yes | delete |

## 3. Repo `notes/data/`

415 entries (2.8 GB on disk; 955 tracked files, 190 untracked entries). An entry counts as referenced when its name
(or a code template that produces it) appears in tracked code outside `archive/`, `PAPER_REPRODUCTION.md`, the
writing packages or the paper figure `.md` files. Shell-templated outputs of `rt_links_rerun.sh` and the Richter
follow-up inputs were checked by hand. Template matching is approximate, so "keep" is generous and "keep (check)" marks
2026-11 tables that are probably written by a kept script under a templated name.

| Proposal | Tracked | Entries | MB |
|---|---|---|---|
| delete | no | 63 | 152.2 |
| delete (or archive) | no | 1 | 198.5 |
| git mv to archive/paper_cleanup_2026-11-12/notes/data/ | yes | 19 | 52.2 |
| keep | no | 50 | 412.9 |
| keep | yes | 230 | 2071.4 |
| keep (archive with its note) | no | 1 | 12.9 |
| keep (archive with its note) | yes | 15 | 1.0 |
| keep (check) | yes | 30 | 10.6 |

**delete (or archive)** (1 entries, 198.5 MB): `rt_bayes_distonly/` (198.5 MB).

**delete** (63 entries, 152.2 MB): `prf_validity_summary_m6.tsv` (36.9 MB), `sim_refit_cache/` (30.2 MB), `prf_validity_summary_m1.tsv` (19.8 MB), `prf_validity_summary_m3.tsv` (19.5 MB), `prf_validity_summary_m5.tsv` (17.5 MB), `prf_validity_summary_m2.tsv` (14.5 MB), `sim_refit_8subs.tsv` (5.4 MB), `trialwise_af_bins.tsv` (3.7 MB), `trialwise_af_runfit.tsv` (1.6 MB), `af_fits_all_latest.tsv` (0.9 MB), `af_fits_all.tsv` (0.4 MB), `trialwise_af_cv.tsv` (0.3 MB), `cond_frozen_hrf_test.tsv` (0.3 MB), `af_dog_v3_target_allSharedSigma_parameters.tsv`, `af_results_REPEAT.tsv`, `af_shared_params_by_subject.tsv`, `repeat_af_fits_shared_params.tsv`, `trialwise_af_check.tsv`, `af_gauss_v3_target_sharedSigma_parameters.tsv`, `af_results_DOG_m4_NEW_all.tsv`, `sim_refit_vs_analytic_sub-03_V1.tsv`, `af_results_NEUTRAL.tsv`, `af_results_GAUSS_partial.tsv`, `af_dog_v3_target_sharedSigma_pSig0.95_parameters.tsv`, `af_results_DOG_m4_NEW.tsv`, `af_dog_v3_target_sharedSigma_pSig0.95_apt0.5_parameters.tsv`, `sim_refit_sub-03_V1.tsv`, `matched_voxel_threshold_sweep.tsv`, `trialwise_af_multcomp.tsv`, `model_signal_audit.tsv`, `model_signal_audit_v2.tsv`, `prf_surface_coverage.tsv`, `repeat_rmanova_per_roi.tsv`, and 30 untracked `fig3_example_voxel_scan_sub-*_2026-11.tsv` scratch files.

**git mv to archive/paper_cleanup_2026-11-12/notes/data/** (19 entries, 52.2 MB): `event_deconv_relative_dct_final2_2026-11/` (19.9 MB), `trialwise_signal_2026-11/` (13.5 MB), `event_deconv_dct_final2_2026-11/` (8.8 MB), `trialwise_final2_2026-11/` (5.5 MB), `event_deconv_nearest_dct_2026-11/` (2.7 MB), `loso_exact_robust_final2_2026-11.tsv` (0.6 MB), `loso_exact_final2_vslow_2026-11.tsv` (0.3 MB), `loso_exact_final2_2026-11.tsv` (0.3 MB), `conditional_dS_rotation_null_2026-11.tsv` (0.1 MB), `hrf_comparison_2026-11.tsv` (0.1 MB), `voxel_selection_candidates_2026-11.tsv` (0.1 MB), `fig3_voxel_scan/` (0.1 MB), `fig3_example_voxel_scan_sub-15_V2_2026-11.tsv`, `fig3_example_voxel_scan_sub-21_hV4_2026-11.tsv`, `fig3_example_voxel_scan_sub-21_V3_2026-11.tsv`, `fig3_example_voxel_scan_sub-21_V3AB_2026-11.tsv`, `fig3_voxel_scan_2026-11.tsv`, `trialwise_rt_bayes_summary_davg_final2_2026-11.tsv`, `trialwise_rt_bayes_summary_davg_signal_2026-11.tsv`.

**keep (check)** (30 entries, 10.6 MB): `bar_items_joint_2026-11/` (9.1 MB), `loso_linear_final2_2026-11.tsv` (0.3 MB), `fig3_voxel_selection_n28_2026-11-12_cells.tsv` (0.3 MB), `field_widths_reliable_polished_2026-11-12_n28_fits.tsv` (0.2 MB), `loso_per_roi_final2_2026-11.tsv` (0.1 MB), `gain_field_parameters_relnull_r0.1_polished_2026-11-13.tsv`, `gain_field_parameters_relnull_r0.2_polished_2026-11-13.tsv`, `conditional_dS_oddeven_2026-11.tsv`, `conditional_dS_sessions_2026-11.tsv`, `baseline_vs_gain_step1_n28_2026-11-12_per_location.tsv`, `hierarchy_slopes_final2_2026-11.tsv`, `hierarchy_slopes_polished_2026-11-12_n28.tsv`, `hrf_by_roi_2026-11.tsv`, `trialwise_rt_bayes_hp_2026-11/`, `fig3_voxel_selection_n28_2026-11-12_voxels.tsv`, `baseline_vs_gain_step1_n28_2026-11-12_per_roi.tsv`, `conditional_estimates_dS_relnull_r0.1_polished_2026-11-13.tsv`, `conditional_estimates_dS_relnull_r0.2_polished_2026-11-13.tsv`, `joint_dT_dog_2026-11.tsv`, `kappa_neighbour_2026-11_fits.tsv`, `dS_by_sensitivity_2026-11.tsv`, `dn_by_hp_stats_2026-11_radial.tsv`, `dog_cs_field_2026-11_tests.tsv`, `dog_cs_field_full_2026-11_tests.tsv`, `field_widths_reliable_n28_2026-11-12_volume.tsv`, `field_widths_reliable_polished_2026-11-12_n28_volume.tsv`, `fig3_voxel_selection_n28_2026-11-12_stats.tsv`, `fig5_gain_field_map_parameters_n28_2026-11-12_v11.tsv`, `loso_per_step_final2_2026-11.tsv`, `split_models_final2_2026-11.tsv`.

**keep (archive with its note)** (16 entries, 13.9 MB): `decode_sweep/` (12.9 MB), `loso_exact_robust_final2_excl0102_2026-11.tsv` (0.6 MB), `gain_field_parameters_voxsel_relnull_2026-11-13.tsv` (0.3 MB), `shared_fit_comparison_canary_3versions_20261102.tsv`, `rt_links_trialwise_comparison_2026-11-13.tsv`, `voxel_selection_robustness_2026-11-13.tsv`, `conditional_estimates_relnull_r0.1_polished_dS.tsv`, `conditional_estimates_relnull_r0.2_polished_dS.tsv`, `discussion_model_vs_measured_corr_n28_2026-11-12.tsv`, `discussion_model_vs_measured_n28_2026-11-12.tsv`, `followup_A1_behaviour_ci_2026-11-10.tsv`, `followup_A1_fig2_fig3_ci_2026-11-10.tsv`, `followup_A3_contralateral_2026-11-10.tsv`, `loso_per_step_excl0102_2026-11.tsv`, `m4_phantom_cross_subject.tsv`, `rt_links_between_subjects_2026-11-13.tsv`.


Untracked "delete" items are the May–October AF/sharedSigma tables (`af_*`, `af_results_*`, `repeat_*`), `sim_refit*`,
`prf_validity_summary_m1–m6` (May pRF validity; 108 MB), the October trial-wise AF tables, `cond_frozen_hrf_test.tsv`
and the Fig. 3 example-voxel scan scratch. None is read by kept code. `rt_bayes_distonly/` (198 MB, untracked) is read
only by `plot_trialwise_rt_bayes.py` (the October Fig. 7, retired). `notes/data/sig_voxel_counts_m4_m6.tsv` is cited in
`CLAUDE.md` and stays.

## 4. Working notes `notes/*.md`

Kept (current, or cited by `PAPER_REPRODUCTION.md`, a package, code, `CLAUDE.md` or memory): every
`*2026-11-1[123]*.md`, `dn_gain_field_variants_2026-11.md`, `fitting_lessons_2026-11.md`, `prf_models.md`,
`rsync_commands.md`, `writing_followup_2026-11-10.md` (part A is the source of the Methods facts in update 4's
`key_numbers.md`). `palette_proposal_2026-11-13.md` stays until the palette is decided.

Proposed to move to `archive/paper_cleanup_2026-11-12/notes/` with `git mv` (history kept):

| Note | Superseded by | References to update |
|---|---|---|
| `primary_test_2026-11.md` — "retsupp: pre-specified tests (2026-11-06)" | The framing is retired (no pre-specification); the HP-rotation null was archived; the dS test is reported from estimates and LOSO in update 4/8 `key_numbers.md` | docstrings in `retsupp/decode/decode_final.py`, `retsupp/visualize/paper/fig6_decoding.py`; `decoding_prep_2026-11.md`; memory `project_key_question_dS` / `feedback_no_prespecification` |
| `decoding_prep_2026-11.md` — "Decoding (step 7): preparation, 2026-11-06" | `decode_final.py`, `PAPER_REPRODUCTION.md` Fig. 6 rows, `decoding_voxel_selection_robustness_2026-11-12.md` | none |
| `robustness_plan_2026-11.md` — "Robustness analyses across pRF models (plan, 2026-11-09)" | Plan carried out: update 4 `key_numbers.md` "Robustness", `sensitivity_excl_sub01_02_2026-11-11.md` | memory `project_robustness_prf_models` points at it |
| `writing_followup_2026-11-10_A.md` — "part A: information the text still needs" | Same text is section A of `writing_followup_2026-11-10.md` (kept) | none |
| `writing_followup_2026-11-10_B.md` — "part B: figure changes" | Figure changes went into the v3/v4 figures and the packages (update 4 onward) | none |
| `m4_phantom_diagnosis.md` — "M4 phantom-perfect voxel diagnosis" (May) | Fixed by `sd_min = 0.2` in the models; the paper uses the CV pRFs | one comment in `retsupp/decode/decoder.py` |

Writing packages: update 4_final → 5 → 6 → 7 → 8 build on each other (each applies on top of the previous one), and
`PAPER_REPRODUCTION.md` cites update 4_final and update 8, so all stay. The `.zip` copies duplicate the folders (about
4 MB); keep them as the handed-over artefacts.
