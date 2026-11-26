# Update 6 for the writing agent (26 Nov 2026): final data pass

Builds on updates 4 and 5. Where they disagree, this file wins. Figures (PNG, PDF, legend .md) and tables are in `figures/` of this package.

Rules (unchanged):
- Joint zero-sum fits only (DoG main model).
- Gain terms: Student-t hierarchical model, posterior mean [95% HDI], P = posterior probability of the sign; credible = P > .975.
- Single-area claims only from each area fitted on its own.
- n = 28.

## Status

| Item | Status | One line |
|---|---|---|
| A1 Scan and design facts | done | Run = 258 vol × 1.6 s = 412.8 s (6.9 min); "about 6.6 minutes" is first pulse to last trial, so correct it. 60 trials per run; 720 per participant (660 for sub-20, sub-24 in fMRI); 20,040 fMRI trials |
| A2 Fig. 3 voxel selection | done | The gain-field fit's voxels with pRF centre within 2° of an item location; no other criterion. pRF centres from the earlier (non-zero-sum) joint DoG fit; on the main model's voxels the early HP − opposite is the same (−0.094 vs −0.092% signal) |
| A3 Gain-field voxel selection | done, one correction | No "R² > 0.05" step is applied. 1 to 500 voxels per fit (median 131); "16 to 395" is the range of per-area medians |
| A4 Measured vs predicted, precision-weighted | **not done today** | Needs a leave-one-run-out deconvolution of measured and predicted responses on the same voxels; running locally, will follow. Until then keep the unweighted numbers marked pending |
| A5 Field value at the neighbour | done | 1.8% of peak (95% range 0.35 to 5.9%), early and all areas |
| A6 Joint dD | done, confirmed | +1.3 [−1.6, +4.0]; early +4.7 [+0.8, +8.7]; held-out prediction worse in 28 of 28, Wilcoxon p = 7.5 × 10⁻⁹ |
| A7 Fits on a bound (Table S4) | done, one correction | 58 of 241 fits have the item weight or a gain on a bound (not 22 to 27); no conclusion changes; dorsal dS P = .965 to .970 without them |
| A8 Supplementary figures | done | S1, S2, S5, S11 as is; S6, S7, S14, S15 re-rendered. S15 now against the main model (it was the dD variant). S7's looser-selection arms are older fits (legend says so) |
| A9 Table S1 in full | done | Precision-weighted own-data columns plus the Wilcoxon and ANOVA sections |
| A10 Supplementary Note 1 | done | 248 words, from the code |
| B1 S10, joint dD | done | No cluster job needed |
| B2 S12, S13 decoding | done | Zero-sum simulation existed; eccentricity figure with model band; per-area figure data only |
| B3 Table S3, n = 30 | **needs cluster, running** | sub-01, sub-02 had no zero-sum fits; SLURM 8902950 (fits) and 8902951 (SEs), expected about 19:00; table follows tonight |
| B4 Fig. 6c, d hierarchical | done, **one conclusion changes** | Only V3 credible (−1.40 [−2.81, −0.06], P = .984); all areas P = .90, early P = .968 |
| C Consistency check | done | All listed numbers confirmed. Target repeat −17.1% is the model without the neutral-at-previous-target term; Fig. 7c shows the model with it (−21.7%): cite one model consistently |


## A1 to A3: design facts and voxel selection

## A1. Scan and design facts (done)

| Quantity | Value | Source |
|---|---|---|
| TR | 1.6 s | BOLD json `RepetitionTime` (cluster, e.g. `sub-05_ses-1_task-search_run-1_rec-NORDIC_bold.json`) |
| Volumes per run | 258 acquired and 258 analysed. The NIfTIs have 258 volumes, and 258 scanner pulses are logged per run. The 4 "dummy scans" are only flagged by fMRIprep (`--dummy-scans 4`); no volumes are removed. | NIfTI shapes (128 × 128 × 81 × 258); events `pulse` rows |
| Run duration | 258 × 1.6 = 412.8 s = 6.9 min (6 min 53 s) | |
| Run structure | 14.3 s fixation at the start; 60 search trials from 14.4 s to 397 s after the first pulse (382.5 s, 6.4 min); 14.3 s fixation at the end | events.tsv: `intro_dummy_scan`, `trial_start`, `iti`, `outro_dummy_scan` |
| Trial | Fixation 0.5 s, pre-target 0.5 s, search display 1.5 s, feedback 1.0 s, ITI 0.5 to 4.5 s (mean 2.7 s). Trial onsets are 6.4 s apart on average (4.7 to 8.1 s). | events.tsv durations |
| Trials per run | 60 in every one of the 336 logged runs. 48 have a distractor (80%): 30 at HP (62.5% of distractor trials) and 6 at each of the three LP locations (12.5% each). 12 have no distractor. | events.tsv `feedback` rows, `distractor_location` (10 = none) |
| Runs per participant | 12 (2 sessions × 6). With fMRI, sub-20 (session 1) and sub-24 (session 2) have 5 runs in one session, so 11. Their behavioural logs have all 12 runs. | `Subject.get_runs` (retsupp/utils/data.py:324); logs |
| Trials per participant | 720 (12 × 60). With fMRI, sub-20 and sub-24 have 660. | |
| Trials in the dataset (n = 28) | fMRI: 26 × 720 + 2 × 660 = 20,040 trials in 334 runs. Behaviour (all logs): 28 × 720 = 20,160 trials in 336 runs. | |

**"About 6.6 minutes" is not the run duration.** It is the time from the first scanner pulse to the end of the last trial (397 s = 6.6 min). A run lasted 412.8 s (6.9 min), including 14.3 s of fixation at the start and at the end. The trials themselves took 6.4 min.

**Further checks:**
- Which behavioural numbers the paper uses for sub-20 and sub-24 (12 or 11 runs) was not checked here. The behavioural analyses should say which.
- One log (sub-08, session 2, run 5) recorded only 111 pulses, but its NIfTI has 258 volumes. This is a logging gap, not a short run.

**Suggested sentence (Methods):**
> "Each run lasted 412.8 s (258 volumes, TR = 1.6 s): 14.3 s of fixation, 60 search trials (one every 6.4 s on average; search display 1.5 s, feedback 1 s, inter-trial interval 0.5 to 4.5 s) and another 14.3 s of fixation. Each participant completed 12 runs in two sessions (720 trials; 11 runs and 660 trials in two participants with one five-run session), 20,040 trials with fMRI in total."

## A2. Fig. 3 voxel selection (done)

**Rule.** The rule is in `retsupp/visualize/paper/export_location_timeseries.py`, lines 87 to 130. Its output is `derivatives/paper_2026-11/location_timeseries_final2`, written 10 Nov at commit 7417667 with `--ladder-root` set to the final DoG fits. The jackknife deconvolution reads that output (`slurm_jobs/deconv_jackknife.sh`). A voxel is used if all of the following hold:

1. **It is one of the voxels of the gain-field fits of its area.** That is the main model's selection (A3): p_signal > 0.5, at most 500 by R², BOLD SD ≤ 20%, cross-run reliability > 0. The voxel pool and the bar prediction come from the fits of the ladder root.
2. **Cross-run reliability > 0.** This is the same measure as in A3 (`prf_cv_roi_reliability`, mean over the three folds), so the criterion is redundant here.
3. **Its pRF centre lies within 2° of one of the four possible target and distractor locations.** The centre is the voxel's x, y in the joint DoG gain-field fit (m7_full of that root, all runs). The voxel is assigned to the nearest location (`dist.min(1) < 2`; the locations are 5.66° apart, so no voxel can be within 2° of two).

There is no further criterion: no additional R², p_signal or SD threshold, and no minimum number of voxels. The signal is each voxel's BOLD minus its predicted response to the mapping bar alone, averaged over the location's voxels. Pathways and all areas pool voxels, weighted by voxel count (`fig3_meta.windows`, `fig45_common.item_timecourses`).

**Voxel counts** per participant and area, summed over the four locations. The `n_voxels` column of `event_deconv_final2_jk_plain_2026-11-25` (all runs), sub-03 to sub-30:

| Area | Participants with ≥ 1 voxel | Median | Range |
|---|---|---|---|
| V1 | 28 | 102 | 11 to 139 |
| V2 | 28 | 101 | 7 to 151 |
| V3 | 28 | 60.5 | 11 to 134 |
| V3AB | 26 | 27.5 | 3 to 151 |
| IPS | 13 | 6 | 1 to 54 |
| hV4 | 28 | 28 | 1 to 89 |
| VO | 17 | 6 | 1 to 48 |
| LO | 28 | 12 | 1 to 103 |
| TO | 23 | 8 | 1 to 66 |

- **Coverage:** 219 participant × area cells, 10,126 voxels in total. That is 22% of the 45,405 gain-field voxels; a median of 26% in V1 and 8 to 9% in LO and TO.
- **Per location** (participant × area × location), the median is 25 voxels in V1 and V2, 14 in V3, and 3 to 8 elsewhere. The minimum is 1.
- **Missing cells:** 22 of the 241 gain-field cells have no voxel within 2° of any location.

**Suggested sentence (Methods):**
> "For the responses at the item locations, we used the voxels of the gain-field fits (Gain-field model, voxel selection) whose pRF centre, in the joint fit, lay within 2° of one of the four possible target and distractor locations, each assigned to the nearest location (median per participant: V1 102, V2 101, V3 61, V3AB 28, hV4 28, LO 12, TO 8, VO 6, IPS 6 voxels; IPS in 13 and VO in 17 participants). From each voxel's time series we subtracted its predicted response to the mapping bar and averaged over the voxels of each location."

## A3. Gain-field voxel selection, main zero-sum DoG fits (done)

**Rule as run.** The code is `retsupp/modeling/fit_cv_ladder_nested.py`, lines 815 to 840, with `choose_voxel_mask` (`retsupp/modeling/fit_af_prf_cv_v2.py:329`) and `select_roi_voxels_psignal` (`retsupp/modeling/fit_dog_dynamic_af_braincoder.py:400`). The root is `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf(_zs_lm)`. Its meta.json gives selector "psignal", p_signal_thr 0.5 and max_voxels 500; the root name encodes maxsd20 and reliable. The zero-sum fits use exactly the voxels of the non-zero-sum reference: the fitter raises an error if the voxel sets differ (`fit_cv_ladder_nested.py:845`). The steps:

1. **The area:** the voxels of the area's neuropythy atlas labels, combined where needed (V3AB = V3A + V3B, LO = LO1 + LO2, TO = TO1 + TO2, VO = VO1 + VO2).
2. **p_signal > 0.5:** the voxel's posterior probability of signal in the R² mixture model of the whole-brain DoG pRF fit (model 4, flexible HRF, mean over runs). Voxels with NaN are excluded.
3. **At most 500 voxels per area,** the top 500 by the R² of that pRF fit.
4. **BOLD SD ≤ 20%,** applied after the ranking.
5. **Cross-run reliability > 0.** The reliability is model-free: the leave-one-run-out R² of the mean time series of the other runs, computed within each of three folds' training runs and averaged over folds (`retsupp/modeling/cv/prf_cv_reliability.py`).

**One correction: there is no R² > 0.05 criterion.** `r2_thr=0.05` is passed, but in the p_signal branch it is not used (`fit_af_prf_cv_v2.py:346-356`). Delete "R² > 0.05" from the Methods. Everything else in the Methods description is correct.

**Counts** (`n_vox`, `gain_field_parameters_zs_lm_2026-11-13.tsv`; family dog, model m7_full, which is the m5_TDS_dS_zs fit; m5_TDS_dS_zs rows are identical):
- **Fits:** 241 participant × area fits (n = 28; V3AB in 27, IPS in 25 and VO in 21 participants), 45,405 voxels.
- **Per fit:** median 131 voxels, range 1 to 500.
- **The Methods' "16 to 395 per area (median 131)"** is the range of the per-area medians (VO 16, V1 394.5) together with the median over all fits. Say so explicitly, because single fits range from 1 to 500.
- **Small fits:** 38 fits have fewer than 20 voxels, mostly in IPS and VO; three have 1 voxel (sub-04, sub-22 and sub-29, all IPS). They are all in the hierarchical model, where their large SEs give them little weight.

| Area | Participants | Min | Median | Max |
|---|---|---|---|---|
| V1 | 28 | 44 | 394.5 | 498 |
| V2 | 28 | 26 | 379 | 500 |
| V3 | 28 | 61 | 363.5 | 500 |
| V3AB | 27 | 3 | 90 | 267 |
| IPS | 25 | 1 | 29 | 208 |
| hV4 | 28 | 12 | 115 | 364 |
| VO | 21 | 2 | 16 | 87 |
| LO | 28 | 9 | 216 | 494 |
| TO | 28 | 4 | 80 | 355 |

**Suggested sentence (Methods):**
> "Per area, we selected voxels whose posterior probability of carrying signal, from a two-component mixture model of the R² of the whole-brain DoG pRF fit, exceeded 0.5; kept at most the 500 with the highest R²; excluded voxels with a BOLD standard deviation above 20%; and kept those with positive cross-run reliability (the R² with which the mean of the other runs predicts each run). This left a median of 131 voxels per participant and area (range 1 to 500; per-area medians from 16 in VO to 395 in V1), 45,405 voxels in 241 participant × area fits."

## A4. Measured vs predicted, precision-weighted

**Status: not done today.** The leave-one-run-out deconvolution of the measured and the predicted item-location responses (early areas, the main model's voxels) is running locally. Until it is in, mark the measured vs predicted sentences (−9.2 vs −7.2%, p = .94; −17.1 vs −1.5%, p = .023) as pending.

## A6. Joint dD variant on the zero-sum fits

**Status: done (confirmed, no change).**

**Numbers.** These come from `notes/data/dD_meta_zs_2026-11-25/bayes.tsv` (Student-t hierarchical model). dD is the distractor gain at HP minus at LP, in percentage points:
- **All areas:** +1.3 [−1.6, +4.0], P(> 0) = .82. Confirmed.
- **Early:** +4.7 [+0.8, +8.7], P(> 0) = .990. Confirmed.
- **Dorsal:** −0.4 [−6.6, +5.7].
- **Occipitotemporal:** −0.5 [−4.5, +3.0].

**Held-out claim.** Source: `notes/data/hp_terms_heldout_stats_zs_lm_2026-11-16.tsv` (update 13b, item 1a), code `retsupp/modeling/hp_terms_heldout_power.py --zero-sum`.
- **Method:** leave-one-run-out prediction with 12 folds (11 for sub-20 and sub-24). In each fold, the gains of each model were refitted on the other runs, and the held-out run was predicted. The pRFs and field widths were held at the all-runs fit; they are the same in every model, so any leak from them is shared by all models.
- **Comparison:** the zero-sum model with dS, with versus without dD. Each participant's value is the median over the nine areas of the change in held-out R².
- **Result:**
  - Adding dD made held-out prediction worse in **28 of 28 participants**: median ΔR² −0.80 × 10⁻⁴ [−1.44, −0.49], two-sided Wilcoxon signed-rank p = 7.5 × 10⁻⁹. This is the smallest p possible with n = 28, since all 28 participants have the same sign.
  - Per pathway, it was worse in 23 of 28 participants (early, p = .00047), 24 of 27 (dorsal, p = 3.7 × 10⁻⁷) and 27 of 28 (occipitotemporal, p = 2.2 × 10⁻⁸).
  - Per area, it was worse in every area except V1 (V1: 18 of 28, p = .28).
- **Caveat:** this is the closed-form linear refit of the gains around the joint fit. It is not a full refit with the pRFs left free in each fold.

**Suggested sentence:**

> "Letting the distractor gain differ between the HP and the LP locations did not reveal a distractor-specific effect (HP − LP, +1.3 percentage points over all areas, 95% HDI [−1.6, +4.0]), and adding this term made the prediction of held-out runs worse in all 28 participants (Wilcoxon signed-rank test, p = 7.5 × 10⁻⁹)."


## A5. Width of the sustained field: value at the neighbouring location

**Status: done.**

**How it was computed.**
- The bootstrap was recomputed from `notes/data/width_profiles_zero_sum_2026-11-14_profiles.tsv`, with 500 draws, participants resampled and seed 0.
- Width profiles are summed over participants, and over areas within the unit.
- The best width is the grid point with the lowest summed error. The grid has 17 widths, σ 0.5 to 8°.
- The field's value at the neighbouring location is exp(−d² / 2σ²), with d = 5.66°.
- The original bootstrap code was not saved. This recomputation reproduces the published width range and P values (4.0 to 5.6°; P(FWHM < 5.66°) = .99 for all areas, .98 for early areas).

| Unit | Group FWHM | 95% bootstrap range of FWHM | Value at the neighbouring location, % of peak | 95% bootstrap range | Largest in any draw | P(FWHM < 5.66°) |
|---|---|---|---|---|---|---|
| All areas | 4.7° | 4.0 to 5.6° | 1.8% | 0.35 to 5.9% | 24% | .99 |
| Early (V1 to V3) | 4.7° | 4.0 to 5.6° | 1.8% | 0.35 to 5.9% | 14% | .98 |

**Note.** The widths fall on grid points (FWHM 4.0, 4.7, 5.6, 6.7, 7.9°), so the bootstrap range is coarse. For all areas, 87 of 500 draws gave 4.0°, 355 gave 4.7°, 51 gave 5.6° and 7 gave a wider field. For early areas the counts were 144, 311, 36 and 9.

**Suggested sentence.**
> "At the group-level width, the sustained field had fallen to 1.8% of its peak at the neighbouring item locations (95% bootstrap range 0.4 to 5.9%), in early visual cortex as over all areas."

## A7. Fits on a bound (Supplementary Table S4)

**Status: done.**

**Table.** `notes/figures/paper_2026-11/tableS4_bound_fits_n28.md`. Data: `notes/data/bound_fits_zs_dog_2026-11-26/`. Script: `scratchpad/u6/bounds.py`, run locally.

**What counts as on a bound.**
- **A term at its bound.** The fit's gain differs from the unbounded jackknife solution (the `at_bound` flag of `gain_meta_zs`). This applies to 22 fits for dS, 27 for the target gain and 27 for the distractor gain. These are the "22 to 27 of 241" in the Methods.
- **The item weight at a bound.** In 42 fits the item weight is at 0.05 (26) or at 50 (16).
- **Strict exclusion.** Any gain or the item weight on a bound: 58 of 241 fits, mostly in TO (16), IPS (14), LO (7) and VO (6).

**Correction for the Methods.** The text says "Fits whose item weight or a gain ended on a bound (22 to 27 of 241 ...)". The 22 to 27 counts only the gains. With the item weight included, it is 58 of 241.

**Result.** No conclusion changes.
- **dS:** without the bound fits it is −3.2 to −3.5 over all areas and −5.4 to −6.0 in early areas. It stays credible over all areas, in early areas and in occipitotemporal areas.
- **Dorsal dS:** already borderline (P = .977). Without the bound fits it is −1.8 to −1.9 with P = .965 to .970, just below the .975 threshold. Do not describe the dorsal reduction as robust.
- **Target and distractor gains:** they become larger when the bound fits are left out. The early target gain then becomes credible (+10.2, P = .981; strict exclusion +13.2, P = .993). This does not change "no conclusion about the early target gain", which rests on the pRF-model dependence.

**Suggested sentence (Methods).**
> "In 58 of the 241 participant × area fits, a gain or the item weight ended on a bound of the fit (mostly in TO and IPS). These fits were kept; without them, the HP − LP difference was −3.5% over all areas (95% HDI [−4.9, −1.8]) and −6.0% in early visual cortex, and the target and distractor gains were, if anything, larger (Supplementary Table S4)."

## Part C. Consistency check

| Number in the text | Status | Current value and source |
|---|---|---|
| dS −3.2 [−4.8, −1.6] | Confirmed | −3.2 [−4.8, −1.6], P > .999 (`notes/data/gain_meta_zs_dog_2026-11-25/bayes.tsv`, variant t) |
| dS early −5.5, dorsal −1.9, occipitotemporal −2.2 | Confirmed | Early −5.5 [−8.0, −3.0], dorsal −1.9 [−3.7, −0.0] (P = .977), occipitotemporal −2.2 [−3.9, −0.6] (P = .996) |
| Target +23.5 [+15.3, +31.9] | Confirmed | Same file |
| Target early +9.4, dorsal +30.6, occipitotemporal +30.5 | Confirmed | Early +9.4 [−0.3, +19.2] (P = .974, not credible), dorsal +30.6 [+19.1, +43.1], occipitotemporal +30.5 [+20.4, +40.6] |
| Distractor +4.7 [+0.9, +8.9] | Confirmed | P = .988 |
| Gaussian dS −5.2 [−9.9, −0.2] | Confirmed | P = .982 (`gain_meta_zs_gauss_2026-11-26/bayes.tsv`) |
| DN dS −5.7 [−8.5, −3.2] | Confirmed | P = .999 (`gain_meta_zs_dn_2026-11-26/bayes.tsv`) |
| Bar slope at HP −2.9% [−4.2, −1.1], 21 of 28, p = .020 | Confirmed | All areas, HP vs LP: per participant the median over areas, median −2.94% [−4.18, −1.11], 21 of 28 below 0, Wilcoxon p = .020 (`notes/data/baseline_vs_gain_step1_zs_n28_2026-11-25.tsv`). This is a Wilcoxon statistic, not the hierarchical model, because these terms have no per-cell SEs. |
| Decoding −1.14 ± 0.39% | Confirmed | Recomputed from `notes/data/decoding_readout_jointhrf_2026-11.tsv` (free decoder, n = 28). Per participant and area: HP quadrant minus the other three, as % of the four-quadrant mean. Mean over the nine areas: −1.14 ± 0.39% (mean ± SEM), one-sided Wilcoxon p = .0033. |
| Target repeat −17% [−26.4, −6.9] | Confirmed | −17.1% [−26.4, −6.9], P > .999 (`notes/data/trialwise_meta_bayes_signal_se_zs_lm_2026-11-25_pathavg/summary.tsv`, T_target_repeat, rel_t, all areas). This comes from the model without the neutral-at-previous-target term. In the model with that term (Fig. 7c), the value is −21.7% (update 4, item 8a). |
| Distractor gain per SD log RT −2.7 [−4.7, −0.8] | Confirmed | −2.70 [−4.75, −0.80] percentage points, P = .996 (same file, D_rt, abs_t, all areas) |
| FWHM 4.7° (4.0 to 5.6) | Confirmed | A5 above |

No number in the list needs correcting.

## A8 and A9 (fork 4): supplementary figures without refit, and Table S1 in full

All files are in `/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/`. Each figure has a PNG, a PDF and a legend `.md`, and every PNG was checked by eye. Nothing is committed.

## A8. Supplementary figures

| No. | Status | File | Data source | Notes |
|---|---|---|---|---|
| S1 | done (as is) | `figS_rt_by_location_n28_palA` | Behaviour | No model, so it is compliant. Rendered 13 Nov; unchanged. |
| S2 | done (as is) | `figS_priming_rt_v4_n28_palA` | Behaviour | Compliant; unchanged. |
| S5 | done (as is) | `figS_decoding_prf_models_n28_palA` | Decoder with DoG, Gaussian and DN pRFs; no gain fields | Compliant. The statistics are Wilcoxon (two-sided, next vs opposite). The colour of the "Opposite" key label is a shade off (minor). |
| S6 | done (re-rendered) | `figS_item_locations_per_area_abs_weighted_n28` (new) | Fig. 3 deconvolution (`fig3_robustness_*_n28_2026-11-12.tsv`) plus `notes/data/fig3_meta_2026-11-25/own_data.tsv` | Stars and legend now come from the precision-weighted own-data fits (update 4, item 13). HP − opposite is credible in V2 (−0.101% [−0.165, −0.039], P = .999) and TO (−0.116%, P = .993); V1 is P = .962. The Wilcoxon q values stay as a check sentence. The legend colours are corrected (purple shades, not red and grey). The old `figS_item_locations_per_area_abs_n28` is kept. |
| S7 | done (re-rendered) | `figS_item_locations_voxel_selection_zs_n28` (new) | Main-model arm: `notes/data/event_deconv_relative_zs_lm_2026-11-16` (main model's voxels and zero-sum pRFs, bar-only prediction from them). Looser arms: `event_deconv_relative_relnull_r0.{2,1}_2026-11-12` | **Caveat:** the r > 0.2 and r > 0.1 arms come from the earlier m7_full fits (not zero-sum), because no zero-sum fits exist for those selections. The legend says so. The pRF blur of panel b uses the polished root's pRF centres. Main model, HP − opposite: early −0.094 ± 0.033% signal (p = .004), all areas −0.085 ± 0.032 (p = .006). Voxels per item location in V1, V2 and V3 are 24.5, 24.75 and 13.25, nearly the same as Fig. 3's selection. Stats: `notes/data/fig3_voxel_selection_zs_n28_2026-11-26_*.tsv`. The legend's selection description (p_signal > 0.5, R² > 0.05, top 500, SD ≤ 20%, reliability > 0) should be aligned with fork A3's answer. |
| S11 | done (as is) | `figS_decoding_voxel_selection_n28_palA` | Decoder only | Compliant, Wilcoxon statistics. Fork B4 covers the hierarchical decoding statistics. |
| S14 | done (new) | `figS_fig7_relnull_zs_n28` (one figure: top r > 0.2, bottom r > 0.1) and the single renders `figS_fig7_relnull_r02_zs_n28`, `figS_fig7_relnull_r01_zs_n28` | `notes/data/trialwise_meta_bayes_relnull_r0.{1,2}_zs_lm_2026-11-25_pathavg` (zero-sum main model, pathway = mean of areas, Student-t primary) | See the S14 notes below the table. |
| S15 | done (new figure) | `figS_items_type_model_vs_measured_zs_n28` | `notes/data/items_type_model_vs_measured_n28_zs_lm_main_2026-11-26.tsv` (new) | **Fix:** the existing `..._zs_lm_2026-11-16.tsv` compared the measured contrasts with the SD prediction. SD is the zero-sum model plus dD (m7_full_zs), not the main model. The table was recomputed against the main model as fitted (variant S, whose dD = 0) with a new `--items-variant` option. See the S15 notes below the table. |

**S14 notes (looser voxel selections for Fig. 7):**
- **What holds:** the target-repeat reduction (−17% at r > 0.2, −20% at r > 0.1, P > .999) and the lower distractor gain on slower trials (P = .996 at r > 0.2, P = .978 at r > 0.1).
- **At r > 0.1 only:** the distractor-repeat effect becomes credible (−50% [−93, −8], P = .994), carried by the dorsal pathway. The global RT gain is credibly positive in the dorsal pathway.
- **Code changes:**
  - New `--rt-ylim` option in `fig7_compact`; the default is unchanged.
  - The distractor-repeat legend sentence now follows the posterior. It used to say "does not change ... credibly" even at P = .994.
- **Stacking:** the scratchpad script `stack_fig7.py` combines the two renders (vector PDF via pypdf).

**S15 notes (item types, measured vs predicted):**
- **The conclusions do not change.** The measured − predicted difference is credible for target − neutral and distractor − neutral in every pathway and in V1 to V3 (q ≤ .009). It is not credible for target − distractor anywhere (q ≥ .11).
- **Early values:** target − neutral is measured +0.127 against a predicted +0.033; distractor − neutral is measured +0.061 against a predicted +0.016.
- **Statistics:** Wilcoxon with BH, not precision-weighted, which matches the table.

## A9. Supplementary Table S1

**Done:** `tableS1_item_locations_weighted_n28.md`, built by `retsupp/visualize/paper/tableS1_item_locations_weighted.py`.

**Structure:**
- **Section 1 (new, primary):** precision-weighted own-data fits, posterior mean [95% HDI] and P, for Early, Dorsal, Occipitotemporal, All areas and all nine areas.
  - 1a: the item contrasts, in % signal.
  - 1b: HP − opposite, next − opposite and HP − next, in % signal.
  - 1c: the same three contrasts in relative units, for pathways and all areas only.
- **Sections 2 to 6:** the earlier table, unchanged in content (location × item ANOVAs, per-item-type contrasts, HP vs all LP, DCT basis, the per-area Wilcoxon/ANOVA table), relabelled as unweighted checks. The one em-dash is replaced by "not computed".

**Key values:**
- Early: HP − opposite −0.069% [−0.127, −0.011], P = .986; next − opposite −0.030, P = .929.
- All areas: HP − opposite −0.062, P = .978.
- Occipitotemporal: HP − next −0.056, P = .976.
- Areas: V2 and TO are credible for HP − opposite.
- SE calibration (all areas): 0.79 for HP − opposite and 0.70 for target − neutral (0.67 expected).

**Provenance:** Fig. 3, S6 and Table S1 use the Fig. 3 deconvolution on the "final2" voxels and pRF centres (the earlier m7_full fits), not on the zero-sum refit. The data are model-free, so this only changes which voxels sit at an item location. The main-model voxel arm in S7 gives nearly identical values (early HP − opposite −0.094 against −0.092% signal).

## Code changes (uncommitted)

- **`fig3_voxel_selection.py`:**
  - New arm `zs`.
  - New options `--sels` and `--name`.
  - The legend describes the zero-sum selection and that the looser arms are not zero-sum.
- **`fig7_compact.py`:**
  - New `--rt-ylim` option.
  - The distractor-repeat legend sentence is conditional on the posterior.
- **`items_model_vs_measured.py`:** new `--items-variant` option (default SD, unchanged).
- **`fig3_robustness.py`:** `figure_abs(..., own=...)` takes its stars and legend from the precision-weighted own-data fits.
- **New scripts:** `figS_items_type_model_vs_measured.py`, `tableS1_item_locations_weighted.py`.
- **New data:**
  - `notes/data/fig3_voxel_selection_zs_n28_2026-11-26_*.tsv`
  - `notes/data/items_type_model_vs_measured_n28_zs_lm_main_2026-11-26.tsv`
  - `notes/data/items_voxel_overlap_n28_zs_lm_main_2026-11-26.tsv`

## Update 6, items A10 and B1 to B4

## Status

| Item | Status |
|---|---|
| A10 Supplementary Note 1 (joint latent RT model) | done |
| B1 S10, joint dD fit | done (figure rendered from existing outputs, no cluster job needed) |
| B2 S12, S13 decoding by eccentricity and per area | done: the zero-sum simulation already existed (update 4, item 3; SLURM 8837804), so both figures are re-rendered with it, no new job |
| B3 Table S3, n = 30 | **needs cluster, running**: sub-01 and sub-02 had no zero-sum fits. Jobs 8902950 (18 tasks) and 8902951 (SEs) started; table pending |
| B4 Fig. 6c, d in the hierarchical model | done; **one conclusion changes** (see B4) |

## A10. Supplementary Note 1 (248 words)

> Supplementary Note 1. Joint latent model of single-trial response times. For every correct trial, we estimated a target gain, a distractor gain (distractor trials only) and a global gain by weighted least squares on top of each participant's gain-field fit, in each of the nine areas. Each trial's 3 × 3 sampling covariance came from 200 circular-shift placebo estimates (model residuals shifted within runs by at least 20 volumes). Per participant, the areas' deviations from their fitted gains, and their covariances, were averaged with weights 1/median SE², and scaled to their standard deviation over trials. Each trial's latent gains equal a condition mean (participant intercept plus group and participant effects of distractor condition, target at HP, target repeat, distractor repeat and session) plus a deviation with participant-specific standard deviations and an LKJ(2) correlation; the observed gains are normal around the latent gains with the trial's placebo covariance. Log response time is normal, with fixed effects of distractor condition, target at HP, both repeats, trial number (linear and quadratic), run and session, a participant intercept, participant-specific residual standard deviations, and participant slopes on the three latent deviations drawn from a multivariate normal (group means Normal(0, 0.05), HalfNormal(0.03) standard deviations, LKJ(2)). Hierarchical terms were non-centred, and the latent gains were integrated out analytically. We sampled with NUTS (PyMC, NumPyro), four chains of 1,000 draws after 1,000 tuning steps, target acceptance 0.9: maximum R̂ 1.01, minimum bulk ESS 318, one divergence. Posterior predictive checks reproduced the response-time effects of Fig. 1.

Source: `retsupp/visualize/paper/rt_joint_latent_model.py` (priors and structure read from the code), run on the zero-sum per-trial gains (`derivatives/af_trialwise_trialse_zs_lm_2026-11-16`); diagnostics from `notes/data/rt_joint_latent_zs_lm_2026-11-16/main_results.tsv` (max R̂ 1.01, min bulk ESS 318, min tail ESS 746, 1 divergence of 4,000 draws) and `notes/update13b_reruns_zero_sum_2026-11-16.md` (PPC: Fig. 1 contrasts inside their 95% bands, 97.6% of participant × condition cells inside). One caveat for the text: the PPC misses the right skew of RT (median and 90th percentile inside in 1 of 5 cells) and the distractor-gain moments (inside in 2 of 5 cells). The note does not need to say so, but do not write "the model reproduced the data" without that qualification.

## B1. S10: the joint dD fit

Files: `notes/figures/paper_2026-11/figS_dD_joint_zs_n28_palA.{png,pdf,md}`; code `retsupp/visualize/paper/figS_dD_joint.py` (new). Inputs: `notes/data/dD_meta_zs_2026-11-25/` (jackknife SEs, calibration 0.73) and `notes/data/hp_terms_heldout_r2_zs_lm_2026-11-16.tsv`.

**a, dD (distractor gain at HP − at LP), percentage points, Student-t hierarchical model:**
- All areas +1.3 [−1.6, +4.0], P(> 0) = .82.
- Early +4.7 [+0.8, +8.7], P = .990; dorsal −0.4 [−6.6, +5.7]; occipitotemporal −0.5 [−4.5, +3.0].
- No single area credible (V3 +5.3, P = .974; V1 +4.4, V2 +4.3, P = .96).

**b, held out (leave one run out), adding dD to the main model, ΔR² × 10⁻⁴:**
- All areas (per participant, median over areas): −0.80 [−1.44, −0.49], **0 of 28 participants better, Wilcoxon signed-rank p = 7.5 × 10⁻⁹**.
- Early −0.39 (5/28 better), dorsal −2.00 (3/27), occipitotemporal −1.27 (1/28); every area except V1 (−0.17, 10/28, q = .28) worse at q < .01 (BH over nine).

Suggested sentence:
> "Letting the distractor gain differ between the HP and the LP locations did not reveal a distractor-specific reduction (all areas +1.3 percentage points [−1.6, +4.0]; early visual cortex +4.7 [+0.8, +8.7]), and it made the prediction of held-out runs worse in all 28 participants (Wilcoxon p = 7.5 × 10⁻⁹; Supplementary Fig. S10)."

## B2. S12 and S13: decoding by eccentricity and per area

The zero-sum simulation already existed (`notes/data/decode_simfit_zs_*`, update 4 item 3), so no cluster job was started.

- **S12, by eccentricity:** `figS_decoding_eccentricity_n28_palA_zs.{png,pdf,md}`. Data unchanged; model line and band from the zero-sum simulation (whole aperture predicted −1.3 ± 0.5%, was −1.2 ± 0.5% with the old fit). Legend names the zero-sum model. Per band (data): 0 to 1.5° −0.7 ± 0.6% (p = .069), 1.5 to 2.5° −1.6 ± 0.7% (p = .010), 2.5°+ −1.5 ± 0.7% (p = .020); bands do not differ (Friedman p = .51).
- **S13, per area with both decoders:** `figS_decoding_hp_per_area_v10_n28_palA_zs.{png,pdf,md}`. This figure has no model content (data only: unconstrained decoder and decoder with the bar outline given), so it is the same as the current version; re-rendered under the _zs name only for a consistent package.

Suggested sentence (unchanged in substance):
> "The decoded difference did not grow toward the aperture edge, next to the items (Supplementary Fig. S12), as predicted by the gain-field model."

## B3. Table S3, n = 30: running on the cluster

- **sub-01 and sub-02 had no zero-sum DoG fits** (the view root `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm_ref` starts at sub-03). Their earlier non-zero-sum ladder fits exist.
- **Started** (account hare.econ.uzh as in the job script, partition standard, code at 87dd93e in a new cluster worktree `~/retsupp_u6`, because `~/retsupp_plan` could not pull over untracked files):
  - **8902950**, `zs_dog_sub01-02`, 18 tasks (2 participants × 9 areas): the same settings as the main DoG zero-sum fits (CV_SCHEME=full, BOUNDED, MAX_BOLD_SD=20, RELIABLE_ONLY, roi_hrf_dog_final2 HRFs, MODELS m4_TDS_zs and m5_TDS_dS_zs, LM 400 iterations, ZS_SUFFIX=_zs_lm). All 18 running since 11:47 CET. The Gaussian zero-sum cells took 4 min to 7.3 h, so expect them done by about 19:00 CET.
  - **8902951**, `wprof_zs_sub01-02`: the jackknife SE step (width profiles into `field_widths_zs_lm_2026-11-14`), after the fits (about 20 min).
- **One difference from the main fits:** the polished fits that served as the second LM start do not exist for sub-01 and sub-02, so their zero-sum fits start only from the clamped parent model (one start instead of two). Say so in the table legend.
- **Then (local, about 30 min):** collect the estimates with sub-01 and sub-02 included, rerun `gain_meta_zs` with n = 30, write Table S3. **Not done today.** Manuscript sentence status: "Table S3 (n = 30): pending".

## B4. Fig. 6c and d in the hierarchical model

Inputs: `notes/data/decode_meta_2026-11-25/` (per participant × area jackknife SEs, calibration median 0.62, 0.67 expected; Student-t hierarchical model, 0 divergences, R̂ 1.00, min bulk ESS 1,929) and, new, each area and each pathway fitted on its own data (`notes/data/decode_meta_own_2026-11-26/own_data.tsv`; pathway = per participant the mean of its areas, SEs combined assuming independent area errors).

Decoded intensity, quadrant facing HP minus the other three, % of the mean of the four:

| Unit | Fig. 6 now (mean ± SEM, one-sided Wilcoxon) | Hierarchical model (posterior mean [95% HDI], P(< 0)) | Own data, no pooling |
|---|---|---|---|
| All nine areas | −1.14 ± 0.39%, p = .003 | −0.71 [−1.76, +0.39], P = .90 | −0.50 [−1.71, +0.87], P = .79 |
| Early (V1 to V3) | −1.35 ± 0.51%, p = .010 | −0.97 [−1.99, +0.05], P = .968 | −1.01 [−2.12, +0.24], P = .955 |
| Occipitotemporal | −1.06 ± 0.41%, p = .006 | −0.82 [−2.54, +0.87], P = .83 | −0.40, P = .64 |
| Dorsal | −0.12 ± 0.75%, p = .56 | −0.11, P = .53 | −0.39, P = .63 |
| V2 | −1.68 ± 0.73%, q = .033 | −0.75 [−2.00, +0.44], P = .88 | −0.66, P = .83 |
| V3 | −1.62 ± 0.59%, q = .033 | **−1.40 [−2.81, −0.06], P = .984** | **−1.99 [−3.51, −0.25], P = .991** |
| Other areas | none q < .05 | none credible | none credible |

**Does a conclusion change? Yes.**
- With the Wilcoxon statistics, the decoded reduction holds over all areas, in early and in occipitotemporal areas, and in V2 and V3.
- With the hierarchical model, only V3 is credible, in the pooled model and on its own data. The early pathway is a trend (P = .968 pooled, .955 on its own data). All areas and occipitotemporal are not credible.
- The reason is the weighting: the jackknife SEs of the decoded effect differ a lot between participants, and the participants who carry the mean effect have noisy estimates (the same pattern as Fig. 3, update 4 item 13). The direction is the same everywhere except IPS.

**Recommendation.** If the paper uses one framework for all gain and decoding results, rewrite the decoding claim as "lower in the quadrant facing HP in V3, with the same direction in early visual cortex as a whole", and keep the Wilcoxon results as the supplementary check. Suggested sentence:
> "Through the same hierarchical model as the gain-field terms, the decoded intensity in the quadrant facing HP was credibly lower in V3 (−1.4% [−2.8, −0.1], P = .984; on V3's own data −2.0% [−3.5, −0.2]), and pointed the same way in early visual cortex as a whole (−1.0% [−2.0, +0.0], P = .968), but not over all nine areas (P = .90); with equal weights per participant (Wilcoxon), the reduction was present over all areas (−1.14 ± 0.39%, p = .003)."

## Files (not committed)

- New code: `retsupp/visualize/paper/figS_dD_joint.py`.
- Figures: `notes/figures/paper_2026-11/figS_dD_joint_zs_n28_palA.{png,pdf,md}`, `figS_decoding_eccentricity_n28_palA_zs.{png,pdf,md}`, `figS_decoding_hp_per_area_v10_n28_palA_zs.{png,pdf,md}`.
- Data: `notes/data/decode_meta_own_2026-11-26/own_data.tsv`.
- Git: I pushed `paper-plan-2026-11` to origin (it carried the three earlier local commits up to 87dd93e; no new commit) so the cluster worktree has the code.
