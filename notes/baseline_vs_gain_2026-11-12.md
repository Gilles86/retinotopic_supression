# Baseline shift vs response gain at the HP location (2026-11-12, n = 28; phase 1)

Question from the Introduction: is the effect at the HP location a gain on the stimulus drive (response gain, e.g.
Reynolds & Heeger, 2009) or a shift of the baseline (e.g. Kastner et al., 1999)? Final DoG joint-HRF fits, sub-03–30,
gains in percent. Phase 1: identifiability, a local slope test, and the cluster design (not submitted).

## Step 0. Identifiability of a sustained additive baseline at HP — not identifiable

- Cleaning (`retsupp/preprocess/clean.py`): per run, nilearn `image.clean_img(..., confounds, standardize='psc')`
  with fMRIPrep's cosine (high-pass) regressors, motion, aCompCor in the confounds; `psc` expresses each voxel as %
  of its own run mean and removes that mean. Checked on the Fig. 3 location series (final fits, sub-03–30, 8,162
  ROI × location × run series): the largest |run mean| of the cleaned BOLD is 3.3 × 10⁻⁵ % signal change, against a
  median within-run SD of 1.23%.
- The gain-field model has one baseline per voxel, shared by all runs (`baseline` in `fit_cv_ladder_nested.py`
  PER_VOXEL_COLS); with zero run means it simply sits near 0.
- HP is constant within a run (it moves every 3 runs), and runs are cleaned separately.
- So a sustained additive offset at HP (a constant per voxel per block) is removed by the run-mean normalisation
  before any model sees the data: there is nothing left to estimate. (Even without the normalisation, block-level
  offsets would be confounded with run-to-run drifts in signal level.) In the model code this is shown directly:
  the share of the additive sustained regressor that survives removing each run's mean is written out per subject ×
  ROI (`add_sustained_kept`, expected 0 to float precision).
- Methods sentence: "Because each run's time series was expressed as percent signal change of its own mean and the
  HP location was constant within a run, a sustained additive offset at the HP location is not identifiable in these
  data; we therefore tested the two forms of a baseline shift that remain testable: a change in the slope of the
  response to the mapping bar, and a trial-locked additive response."
- What remains testable: (i) the slope of the response to the bar, which varies within a run (a gain scales it, an
  offset does not) — step 1; (ii) an additive baseline that is not constant within a run, i.e. trial-locked to
  fixation or search onsets (Richter et al.'s proactive, omission-type response) — step 3. The intercept of step 1 is
  not informative either: with zero run means it is determined by the slope.

## Step 1. Slope of the bar response by location (run locally; done)

`retsupp/visualize/paper/baseline_vs_gain.py step1`, on the Fig. 3 location series
(`/data/ds-retsupp/derivatives/paper_2026-11/location_timeseries_final2/`: per subject × ROI × ring location, voxels
with cross-run reliability > 0 and a pRF centre within 2° of the location; `raw` = their mean BOLD, `bar` = the
bar-only prediction of the final pRFs and per-ROI HRF). Per location one OLS over all runs: free intercept per run, one
bar slope per relation of the location to the run's HP location (HP, next, opposite), and a 10-bin FIR (0–16 s) per item
class (target/distractor/neutral × at HP/at LP) so the item responses do not load on the bar. Slopes averaged over the
four locations per subject × ROI (each location is HP, next and opposite in different runs); relative bar gain at HP =
100 × (β_HP / mean(β_next, β_opp) − 1) %. Subject × ROI pairs without a positive bar slope at LP are dropped. Per
participant the median over the areas of a group; median over participants, bootstrapped 95% CI (10,000), Wilcoxon.
Data: `notes/data/baseline_vs_gain_step1_n28_2026-11-12.tsv` (+ `_per_roi`, `_per_location`).

| Group | HP vs LP (%) | k < 0 | p | HP vs next (%) | p | next vs opposite (%) | p |
|---|---|---|---|---|---|---|---|
| All areas (n = 28) | −2.1 [−4.6, −0.7] | 20/28 | .012 | −3.5 [−5.2, −0.9] (21/28) | .0011 | −0.5 [−3.0, +3.0] | .78 |
| Early V1–V3 (28) | −3.4 [−4.9, −0.8] | 20/28 | .10 | −3.0 [−5.9, −0.9] | .034 | +1.3 [−1.3, +4.6] | .19 |
| Dorsal V3AB, IPS (26) | −2.2 [−5.8, +3.9] | 18/26 | .55 | −0.7 | .84 | −0.4 | .65 |
| Occipitotemporal hV4, VO, LO, TO (28) | −2.5 [−6.0, −0.7] | 21/28 | .063 | −2.3 | .48 | −1.3 | .92 |

Reading: the response to the bar itself is about 2–3% smaller when it crosses the HP location than when it crosses
another location — a change in slope, i.e. a gain on the drive, of the same size as the fitted sustained HP − LP gain
(dS estimated alone −1.4%). This is a model-light check (only the bar-only pRF prediction is used as the regressor).
Not yet done: the variant with all aperture voxels weighted by their pRF's overlap with the sustained field (needs the
voxel-level BOLD; can be added to the cluster job if wanted).

## Steps 2–4. Cluster design (code committed, not submitted)

`retsupp/modeling/fit_baseline_terms.py` (+ `retsupp/modeling/slurm_jobs/fit_baseline_terms.sh`), in the fixed-group
framework of `fit_fixed_group_mechanisms` on each subject's final joint fit (m7_full, root
`af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf`, HRF table `paper_2026-11/roi_hrf_dog_final2.tsv`):
all other parameters fixed, each added term linear, its amplitude a least-squares coefficient on the joint fit's
residual, one amplitude per subject × ROI shared by the voxels; estimated on all runs but run 3 and scored on run 3
(held-out ΔR²), plus all-run estimates.

- Step 2 (additive vs multiplicative sustained field): per step 0 the additive sustained term is not identifiable; the
  job only writes `add_sustained_kept` (the regressor's sum of squares left after run-mean removal) as the
  demonstration. The multiplicative dS is included for the joint designs.
- Step 3 (trial-locked baseline): stick at every search onset (`target` events) or fixation onset (`pre-target`),
  convolved with the ROI's HRF, amplitude per voxel w_v = overlap of the pRF (centre Gaussian) with the sustained field
  (the fit's σ_AF) at HP minus the mean at the three LP locations (closed form); a non-specific trial response (mean
  overlap over the four locations) as nuisance. Designs: trial (search), trial (fixation), trial + dD + dT, dD + dT,
  dS, dS + trial — so we see whether dS, dD and dT change when the trial-locked term is included.
- Step 4 (recovery): from the joint fit with the sustained HP gain tied to LP, simulated data = signal + phase-
  randomised residual (one phase spectrum per run shared across voxels): (i) multiplicative, dS = −1.4% (group value);
  (ii) trial-locked additive with the amplitude matched to (i)'s mean change at HP voxels (w_v > 0.5) in the 0–8 s
  after search onsets; (iii) sustained additive followed by run-mean removal (expected: no signal at all). Each
  simulated dataset is fitted with dS, trial and dS + trial; the confusion matrix (which term the group test detects,
  per generator) is computed offline from the per-subject outputs. 10 simulated datasets per subject × ROI.
- Checked locally (synthetic arrays): the solver recovers known amplitudes, held-out ΔR² behaves, the phase
  randomisation keeps each run's power spectrum, the overlap formula is right at the centre.

### Jobs for OK (account hare.econ.uzh, `--exclude=u24-chaihm0-633`, from `~/retsupp_plan` after `git pull`)

| Job | Script | Array | Resources | Est. wall time | Outputs | Step |
|---|---|---|---|---|---|---|
| Canary, sub-03, all 9 ROIs, 10 sims | `retsupp/modeling/slurm_jobs/fit_baseline_terms.sh` | 3 | 2 CPU, 24 GB, 45 min | ~10–20 min (fixed-group dS + dD on 9 ROIs took 4 min; here ~8 predictions per ROI plus the simulation algebra) | `derivatives/baseline_vs_gain_2026-11-12/sub-03/sub-03_roi-*_{terms,sim}.tsv` | 2, 3, 4 |
| Main, after the canary | same | 4–30 (27 tasks; sub-01/02 not needed) | same | ~10–20 min each, ~30 min total with parallel dispatch | same, per subject | 2, 3, 4 |

Then locally: collect the per-subject tables, pooled tests (per participant median over areas; Wilcoxon, bootstrapped
CI, k of n; per area BH over nine), held-out ΔR² per design against the joint fit, the recovery confusion matrix, the
supplementary figure (a schematic, b step-1 slopes, c held-out ΔR², d confusion matrix), key_numbers additions and
wording, added to notes/writing_package_2026-11-12_update5/ (shared with the per-area Fig. 3 update: add files, append sections to its key_numbers.md and UPDATE_NOTES.md, rezip notes/writing_package_2026-11-12_update5.zip).
