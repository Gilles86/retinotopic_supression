# Update 2 to writing package 2026-11-12 (n = 28)

The writing agent has update 1 but not the base package, so the manuscript is at n = 28 while its figures were the
n = 30 versions. This update is therefore **complete**: it contains the whole package (all text files and the full
`figures/` folder at n = 28), not only the differences. Replace everything you have with it.

Every number comes from an existing file (named below); no model was refitted. New numbers were computed by
`retsupp/visualize/paper/update2_numbers.py` (→ `notes/data/update2_numbers_n28_2026-11-12.tsv`) and
`retsupp/visualize/paper/update2_tables.py`.

## What is in it

| File | Status | Content |
|---|---|---|
| `figures/` | complete | PDF + PNG + legend `.md` of the seven main figures (`fig1_paradigm_behaviour_v4_n28`, `fig2_prf_fits_n28`, `fig3_item_locations_n28`, `fig4_model_hp_blocks`, `fig5_gain_fields_n28`, `fig6_decoding_n28`, `fig7_compact_n28`), every supplementary figure and table, and the new items below |
| `all_figures_2026-11-12.pdf` | rebuilt | Main figures first, then the Fig. 5b prototype, then supplementary figures in the proposed S order (S20 follows in update 3), then the tables as text in `figures/` |
| `key_numbers.md` | updated | New CIs, k/n, Fig. 3b, c correction, κ vs dS, DN b–d correlation, Methods facts (eye tracking, the six further MRI participants, distortion correction, 2D/3D, braincoder) |
| `counts_k_of_n_n28.md` | new | k / n for every pooled share in `key_numbers.md` |
| `claims_and_caveats.md` | updated | Four caveats added (fixation not monitored in the scanner; fieldmap-based SDC; 2D/3D unresolved; κ does not change dS) |
| `README.md` | updated | Figure list with the new items; CHANGES "update 2" |
| `figures/tableS_sessions_blocks_n28.md` | new | Fig. 3e contrasts and dS per session and block (C8) |
| `figures/figS_rt_by_location_n28.*` | new | RT by location relative to HP, all blocks and session 1 block 1 (C9) |
| `figures/fig5b_gain_field_map_prototype_n28.*` | new, prototype | The fitted gain field as a map (D10); not yet decided |
| `richter_followup_2026-11-11.md`, `prompt_writing_agent_2026-11-12.md`, `discussion_checks_2026-11-12.md` | included | As in the base package |
| `references/fmriprep_CITATION.md`, `references/fmriprep_CITATION.bib` | new | fMRIPrep's own Methods boilerplate and its 31 references |

## What to do in the manuscript

1. Point every figure reference at the n = 28 files in `figures/` and renumber the supplements as in A3 below.
2. Replace percentages of participants by k of n where the text gives counts (`counts_k_of_n_n28.md`); the Gaussian
   joint dS is 19/28 (68%), not 67%.
3. Add the new CIs (B5) and state that Fig. 3b, c are BH-corrected over the three pathways (B6).
4. Methods: eye tracking only in the pretest (E12); the six further MRI participants (E13); fieldmap-based distortion
   correction (E14); braincoder f82f316 (E15); leave the 2D/3D question open until checked (E14).
5. Discussion: the session/block table (C8) and the RT-by-location figure (C9) are the supplementary support for the
   carry-over and neighbour sentences already in update 1.

## Per item

**A1 Missing files.** All present in `figures/` (every README-listed file exists; `tableS_loso_n28.md` and
`table_parameters_2026-11_n28.md` included). The parameter table's title said "n = 30" although its values are n = 28;
fixed (title and generator `retsupp/visualize/paper/fig5_gain_fields.py`).

**A2 Text files.** `richter_followup_2026-11-11.md` and `prompt_writing_agent_2026-11-12.md` included.

**A3 Proposed numbering** (figures and tables numbered separately, in the order the Results cite them; Methods-only
items last):

| No. | File | Supports |
|---|---|---|
| Fig. S1 | `figS_priming_rt_v4_n28` | Raw RTs behind the priming panel (Fig. 1e) |
| Fig. S2 | `figS_rt_by_location_n28` | RT by location relative to HP; pattern present before any location was HP (Fig. 1, Discussion) |
| Fig. S3 | `figS_field_widths_n28` | Field widths (FWHM) per area (Fig. 5b) |
| Fig. S4 | `figS_gain_vs_prf_size_n28` | Target gain rises with pRF size (Fig. 5a) |
| Fig. S5 | `figS_pooled_dS_dD_n28` | dS and dD per participant, alone and joint (Fig. 5c, d) |
| Fig. S6 | `figS_loso_per_roi_n28` | Out-of-subject prediction per area and term (Methods/Fig. 5) |
| Fig. S7 | `figS_dog_kappa_n28` | DoG centre/surround tilt κ: consistent, too small to help prediction |
| Fig. S8 | `figS_gain_fields_gauss_n28` | Fig. 5 with Gaussian pRFs (robustness) |
| Fig. S9 | `figS_gain_fields_dn_n28` | Fig. 5 with DN pRFs (robustness) |
| Fig. S10 | `figS_dn_gain_locus_n28` | HP effect cannot be placed on the DN baselines rather than the input gain |
| Fig. S11 | `figS_dn_params_by_hp_n28` | DN pRF parameters per HP condition (exploratory) |
| Fig. S12 | `figS_decoding_all_rois_n28` | The bar decoded from each area (Fig. 6a) |
| Fig. S13 | `figS_decoding_all_rois_examples_n28` | Example participants per area (Fig. 6a) |
| Fig. S14 | `figS_decoding_hp_per_area_n28` | HP quadrant − other three, per area (Fig. 6d) |
| Fig. S15 | `figS_decoded_bars_rotated_n28` | Same bar, weaker in the HP half (Fig. 6) |
| Fig. S16 | `figS_decoded_bars_per_roi_n28` | Same, per area |
| Fig. S17 | `figS_decoding_canonical_n28` | Fig. 6 with canonical-HRF pRFs (robustness) |
| Fig. S18 | `figS_decoding_prf_models_n28` | Fig. 6 quadrant contrasts with Gaussian and DN pRFs (robustness) |
| Fig. S19 | `figS_decode_simulation_n28` | The decoder recovers an HP-only gain field of the fitted size |
| Fig. S20 | `figS_decoded_map_model_vs_measured_n28` (follows in update 3) | Decoded map, measured vs the joint fit's noise-free prediction (D11) |
| Fig. S21 | `figS_bar_crossing_response_n28` | Measured response when the bar crosses a pRF near an item (cite as consistent only) |
| Table S1 | `tableS_item_locations_n28.md` | Fig. 3 statistics per item type, pathway, HP vs all LP, DCT basis |
| Table S2 | `tableS_sessions_blocks_n28.md` | Fig. 3e and dS per session and block; differences n.s. (C8) |
| Table S3 | `table_parameters_2026-11_n28.md` | All gain terms per area, DoG and DN, joint and alone |
| Table S4 | `tableS_loso_n28.md` | What each term adds to out-of-subject prediction (Methods) |
| Table S5 | `tableS_sensitivity_n30.md` | Every main result with the two pilots included (Methods, participants) |

If Fig. 5b becomes the gain-field map (D10), Fig. S3 (widths) stays as the width supplement.

**B4 Exact counts.** `counts_k_of_n_n28.md`: k / n for every pooled share (Fig. 5 terms joint and alone, gain vs pRF
size, dS/dD/dT, split models, Gaussian and DN, κ, all LOSO "better" and "direction" columns for DoG, Gaussian, DN, κ
and the DN baseline fields). A separate table rather than inline edits, so the text can take k/n from one place. Only
mismatch: Gaussian joint dS 19/28 = 68% (written 67%; corrected in `key_numbers.md`). Source:
`notes/data/update2_numbers_n28_2026-11-12.tsv`.

**B5 CIs** (bootstrapped 95%, 10 000 resamples over participants; `notes/data/update2_numbers_n28_2026-11-12.tsv`):
- dS alone: Gaussian −.008 [−.026, −.001], DN −.028 [−.041, −.007]; dS joint: Gaussian −.088 [−.111, −.001],
  DN −.078 [−.102, −.027] (medians).
- dD − dT alone +.009 [−.004, +.052]; joint dD +.004 [−.041, +.124] (medians).
- Decoded half-bars (mean ± SEM [CI of the mean]): HP half − other half pooled −1.80 ± 0.56% [−2.87, −0.73],
  vertical −1.71 ± 0.70% [−3.05, −0.38], horizontal −1.88 ± 0.54% [−2.89, −0.82]; next − opposite half pooled
  +1.21 ± 0.45% [+0.35, +2.09], vertical +1.29 ± 0.45% [+0.40, +2.15], horizontal +1.12 ± 0.59% [−0.01, +2.26].
- Fig. 6d quadrants (mean ± SEM [CI]): next − opposite +0.68 ± 0.32% [+0.08, +1.30], p = .070; HP − opposite
  −0.90 ± 0.53% [−1.97, +0.12], p = .16 two-sided (.082 one-sided).

**B6 Fig. 3b, c correction.** Benjamini-Hochberg over the three pathways, per panel (`fig3_item_locations.py`,
`stats.false_discovery_control`); the legend already said so; `key_numbers.md` now states it with the q values (all
q ≤ 1.1 × 10⁻⁴).

**B7 κ and the DN baselines at n = 28.** Adding κ does not change dS: median per-participant change −0.0001
[−0.0011, +0.0002], 12/28 larger, p = .33 (`notes/data/dog_cs_field_full_2026-11.tsv`, dS − dS_m7 in the κ rows).
DN: Gauss-Newton correlation of w_b and w_d, median over 252 subject × area fits 0.9992 (5th–95th percentile
0.991–1.000; `notes/data/dn_basefield_full_2026-11.tsv`, both-baselines model).

**C8** `figures/tableS_sessions_blocks_n28.md` (from `notes/data/richter_A1_carry_over_common_denominator_n28_2026-11-11.tsv`
and `richter_A3_session_gains_n28_2026-11-11.tsv`): no session or block difference is significant (early: session 2 − 1
p = .37–.58, block 2 − 1 p = .24–.56; dS session 2 − 1 p = .095).

**C9** `figures/figS_rt_by_location_n28` (made by a parallel agent; legend next to it).

**D10** `figures/fig5b_gain_field_map_prototype_n28` (prototype; made by a parallel agent; legend says it is the fitted
parametric field, not data).

**D11** Not in this update: the decoded map from the joint fit's noise-free predictions next to the measured map (Fig. 6c) follows in update 3.

**E12 Eye tracking.** Pretest session only: EyeLink via exptools2 (`PylinkEyetrackerSession`), calibrated before each
block, left eye; during search a beep when gaze left a 1° window (`settings/default.yml`: `gaze_threshold_deg: 1`,
`eyemovements_alert: True`). Off in both MRI sessions (`experiment/main.py`: `eyetracker_on = False` for sessions > 1).
No eye-tracking files in the dataset; no fixation criterion for continuing is in the code; the lab sheet records only
the date of the eye-tracked pretest (26 of the 28) and no fixation-based exclusion. How many of the 15 who did not
continue were excluded for fixation: not recoverable.

**E13 Six further MRI participants.** The lab sheet has 6 rows (6 distinct participants) with an MRI code outside
sub-01–30: two scans stopped after two blocks because of discomfort, one cancelled, one awaiting medical (GP) clearance,
two without a recorded reason. Aggregated from non-identifying columns only.

**E14 Acquisition.** Distortion correction: fieldmap-based from the phase-difference maps in all 356 runs of all 30
subjects; no run used SyN (fMRIPrep reports `derivatives/fmriprep/sub-*.html`: "FMB (fieldmap-based) –
phase-difference map"). 2D vs 3D: not recoverable from the files here — the sidecar is a hand-written template
(`retsupp/prepare/convert_raw_mri_data.py`) whose "3D" conflicts with multiband 3 (which implies 2D multi-slice); the
raw functional PAR headers are not on the cluster. Check the exam card.

**E15 braincoder.** f82f316 (2026-11-07, "gamma_pdf in log space"): the cluster env's editable checkout has been at
this commit since 2026-11-07 10:10 CET with no later checkout (git reflog), and every final fit was written after that
(DoG joint-HRF fits from 2026-11-07 10:48; ladder 2026-11-09; Gaussian/DN 2026-11-09/10; decodes 2026-11-07 10:15 to
2026-11-10). Equals the repo's `libs/braincoder` submodule pointer. Whether the working tree was clean at each run is not
recorded.

**E16 References.** No `.bib` or reference list for the paper exists in the repo (incl. archive). Included instead:
fMRIPrep's citation boilerplate and its BibTeX (`references/`). The repo also holds PDFs of Klein et al. 2014,
Richter et al. 2025, Theeuwes 2025 and Tuncok et al. 2025 (`notes/papers/`, not included); braincoder's
`libs/braincoder/docs/references.bib` covers only the library's own docs.
