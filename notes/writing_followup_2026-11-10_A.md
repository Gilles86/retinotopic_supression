# Writing follow-up, part A: information the text still needs (2026-11-10)

No refits and no changes to analyses. Paths are relative to `/Users/gdehol/git/retsupp/` unless absolute. The
contradictions with `notes/writing_package_2026-11-10/key_numbers.md` (and other flags for the writer) are listed at
the end.

## A1. 95% CIs (bootstrap over subjects, 10 000 resamples, percentile)

**Behaviour (Fig. 1 table).** Per-subject condition means from `notes/data/behaviour_rt_2026-11.tsv` (c, d) and from
`repeats_raw()` in `retsupp/visualize/paper/fig1_paradigm_behaviour.py` on
`notes/data/trialwise_af_trial_table.tsv.gz` (e, f; behaviour only, no fit enters). n = 30, Wilcoxon signed-rank.
Table: `notes/data/followup_A1_behaviour_ci_2026-11-10.tsv`.

| Effect | Mean [95% CI] (ms) | Median [95% CI] (ms) | Subjects in direction | p |
|---|---|---|---|---|
| Distractor HP − LP | −27.8 [−33.7, −21.9] | −26.1 [−34.4, −20.9] | 30/30 | 1.9 × 10⁻⁹ |
| Distractor LP − absent | +43.0 [+33.1, +52.5] | +43.7 [+33.7, +54.2] | 28/30 | 2.6 × 10⁻⁸ |
| Distractor HP − absent | +15.2 [+7.0, +22.9] | +15.8 [+2.1, +23.2] | 21/30 | .0012 |
| Target HP − LP | +28.4 [+20.5, +36.5] | +26.8 [+19.9, +37.5] | 26/30 | 3.9 × 10⁻⁷ |
| Target repeat − switch | −36.6 [−43.8, −29.2] | −38.4 [−46.9, −28.4] | 28/30 | < .001 |
| Distractor repeat − switch (within HP and LP, averaged) | −17.2 [−28.1, −6.7] | −15.1 [−27.4, −6.2] | 21/30 | .0035 |

(Part B's difference strips in Fig. 1 v3 use `fig45_common.median_ci` with 5 000 resamples; their CIs can differ from
these in the first decimal.)

**pRF-model gains (Fig. 2d).** `gain_r2()` in `retsupp/visualize/paper/fig2_prf_fits.py` on
`notes/data/prf_cv_roi_summary.tsv` + `notes/data/hrf_comparison_jointhrf_final2_2026-11.tsv`; per subject the median
over the nine areas, then the median over subjects (as key_numbers). Table:
`notes/data/followup_A1_fig2_fig3_ci_2026-11-10.tsv`.

| Held-out R² gained over Gaussian pRF + canonical HRF | Median [95% CI] | n > 0 | p |
|---|---|---|---|
| DoG pRF | +.0035 [+.0024, +.0049] | 30/30 | 1.9 × 10⁻⁹ |
| Joint HRF on top of DoG | +.0107 [+.0075, +.0212] | 29/30 | 1.3 × 10⁻⁸ |
| Both | +.0189 [+.0150, +.0244] | 30/30 | 1.9 × 10⁻⁹ |

**Item minus neutral per pathway (Fig. 3b, c).** `item_timecourses` + `diff_tests` of
`retsupp/visualize/paper/fig3_item_locations.py` on `notes/data/event_deconv_final2_2026-11/`; per subject the mean
over 3.2–6.4 s, then the **mean** over subjects (the statistic key_numbers reports), % signal. Same table file.

| Contrast | Early | Dorsal | Occipitotemporal |
|---|---|---|---|
| Target − neutral | +0.125 [+0.102, +0.148], n = 30 | +0.258 [+0.203, +0.317], n = 28 | +0.229 [+0.193, +0.269], n = 30 |
| Distractor − neutral (LP) | +0.072 [+0.042, +0.102], n = 30 | +0.147 [+0.093, +0.207], n = 28 | +0.150 [+0.107, +0.195], n = 30 |

All p ≤ 3.9 × 10⁻⁵ (BH over pathways q ≤ 3.9 × 10⁻⁵). Dorsal n = 28 here (two subjects lack dorsal voxels at some
item location); the relative analyses of Fig. 3d, e use n = 27 (sub-30 also left out, as key_numbers says).

## A2. Fig. 1e, f annotations: mean or median

- The figure annotations (−37 ms, −17 ms) were the **mean** of the within-subject differences
  (`diff.mean()` in `rt_panel`, `retsupp/visualize/paper/fig1_paradigm_behaviour.py`). Exact: −36.6 and −17.2 ms.
- key_numbers' −38 and −15 ms are the **medians** (−38.4, −15.1 ms).
- The annotation code now prints the **median** (`diff.median()` in `rt_panel`). This edit is already in the working
  copy of `fig1_paradigm_behaviour.py`, made together with part B's strip changes; I did not edit the file further and
  did not render Fig. 1. With medians, the c–f brackets read: HP − LP −26, LP − absent +44, target HP − LP +27,
  target repeat −38, distractor repeat −15 ms.
- For part B: the legend text used without `--strips` (`LEGEND`) still says only "mean ± within-subject SEM"; the
  `--strips` legend already says the brackets give the median difference. If Fig. 1 v3 is rendered with `--strips`,
  nothing else is needed.
- key_numbers needs one change for consistency (see Contradictions 1).

## A3. Contralateral share per area (Fig. 2b)

`contralateral()` (`retsupp/visualize/paper/plot_fig1.py`) on `notes/data/fig1_prf_coverage.tsv` (pRF centres with
R² ≥ .1 inside the 3.17° aperture). Table: `notes/data/followup_A3_contralateral_2026-11-10.tsv`.

| Area | n | Median (%) | IQR (%) | Range (%) |
|---|---|---|---|---|
| V1 | 30 | 95.9 | 90.6–97.7 | 86.2–100 |
| V2 | 30 | 94.3 | 88.7–97.2 | 81.3–99.3 |
| V3 | 30 | 95.1 | 92.7–97.6 | 86.0–100 |
| V3AB | 30 | 99.4 | 92.8–100 | 81.0–100 |
| IPS | 23 | 96.4 | 91.0–100 | 50.0–100 |
| hV4 | 30 | 94.6 | 92.3–95.8 | 80.2–100 |
| VO | 24 | 100 | 100–100 | 84.6–100 |
| LO | 30 | 96.6 | 94.0–97.8 | 81.9–100 |
| TO | 30 | 100 | 94.9–100 | 81.2–100 |

n < 30: subjects with no pRF centre passing the criterion in that area. Note: this table comes from the model-4 mean
fits (DoG, flexible HRF, all runs; `collect()` in `plot_fig1.py`), not from the final joint-HRF fold fits (see
Contradictions 3).

## A4. Task, from the task code and the run logs

Sources: `experiment/session.py`, `experiment/trial.py`, `experiment/stimuli.py`, the per-run settings
(`/data/ds-retsupp/sourcedata/behavior/logs/sub-XX/ses-Y/run-Z/*_expsettings.yml`; 362 files, 4 variants that differ
only outside the task parameters below), a raw run log
(`.../sub-10/ses-3/run-4/sub-10_ses-3_task-ret_sup_run-4_events.tsv`), and `Subject.get_hpd_locations()`.

**Response rule.** Participants reported whether the target (the item with the odd orientation) carried a small grey
dot. Keys `y` = no dot, `b` = dot (settings `keys: [y, b]`; the logged responses are `y` and `b`; `correct` =
`keys.index(response) == dot on the target`). Responses count only during the 1.5 s search display, the first key
press only; RT is measured from search onset. Which buttons or fingers `y` and `b` were on the scanner button box is
not in the code. Note: `experiment/instructions.yml` still says "left"/"up" buttons and the README says F/J; both are
from the behavioural-lab version, not the scanner sessions.

**Dots.** On every trial exactly two of the four diagonal items (where target and distractor can be) and two of the
four cardinal items carry a dot (`dot_presence` in `SingletonTrial`), so the target has a dot on half of the trials
on average. Dot: grey disc, radius = item length / 15 = 0.1°.

**Trial timing (confirmed).** Fixation 1.0 s (two phases of 0.5 s, `trial_start` and `pre-target`), search display
1.5 s, feedback 1.0 s, interval 1.3, 2.9 or 4.5 s (each 20 times per run, shuffled). Feedback: a red "x"
(0.5° high) at fixation after an error or no response; after a correct response the fixation point stays
(`show_correct_feedback: false`). 60 trials per run. A 14.3 s blank follows the last trial (`blank`), and one precedes
the first trial after the scanner trigger (`DummyWaiterTrial`).

**Trial list per run (fixed counts, shuffled).** Distractor at HP 30 trials (50%), at each LP location 6 (10% each),
no distractor 12 (20%); the target 15 times at each of the four diagonal locations, never at the distractor's
location (so on distractor trials the target is at HP on 12 of 48). The settings field
`likely_distractor_probability: 0.67` is not used by the trial generator (`create_trials` builds the list above);
do not cite 0.67.

**HP blocks.** Subjects 3–30: four blocks of three runs, each diagonal location HP in exactly one block; blocks 1–2
in session 1, blocks 3–4 in session 2. Within a session HP moves to the diagonally opposite location; between sessions
it moves to the other diagonal (e.g. sub-03: upper left, lower right | lower left, upper right). Eight orders,
assigned in rotation (sub-03 = sub-11 = sub-19 = sub-27, and so on). Fig. 1b's "every 3 runs" is correct for them.
Exceptions: sub-20 session 1 and sub-24 session 2 have five runs (one HP block of two runs). Pilots sub-01 and sub-02:
per session, 2 runs at one location, 3 at a second and 1 at a third (sub-01: LR LR UR UR UR LL in both sessions;
sub-02: UL UL UR UR UR LL in both).

**Display.** Background mid-grey (window colour 0 in PsychoPy's −1…1 RGB). Eight items on a ring at 4°: bars
1.5° × 0.375°. Each trial one item has the odd colour (the distractor) and one has the odd orientation (the target,
always in the common colour); colours orange (PsychoPy RGB .78, .14, −1) and green (−1, .33, −1), which one is the
distractor colour is drawn at random per trial, and so is the target orientation (horizontal or vertical). The whole
display is shifted 1.1° down on the screen (`stimulus_shift`), fixation included. Fixation: white disc 0.25° with a
black cross 0.25° long and a grey centre dot 0.125°. A dark-grey circle outlines the 3.17° bar aperture. Screen 70.9 cm
wide at 196 cm, 1920 × 1080 (≈ 20.5° wide).

**Bar.** Checkerboard bar (PsychoPy `sqrXsqr`) inside a circular aperture of radius 3.17° (= 4 − 1.5/1.8), bar width
= aperture diameter / 8 = 0.79°, contrast reversing every 1/16 s (8 Hz flicker), moving at 0.168°/s, so one sweep
takes 42.4 s. Sequence per cycle: rest, rightward, rest, leftward, rest, downward, rest, upward, rest, break
(rest 4.8 s, break 4.8 s); two cycles per run. The bar runs independently of the trials.

## A5. Acquisition and preprocessing

**Scanner and sequence** (BOLD sidecars, `/shares/zne.uzh/gdehol/ds-retsupp/sub-*/ses-*/func/*_part-mag_bold.json`,
identical in all 359 runs; written by the template in `retsupp/prepare/convert_raw_mri_data.py`): Philips Ingenia 7T,
receive coil `SENSE-Head-7T-Ant` + `SENSE-Head-7T-Post`; gradient-echo EPI, TR 1.6 s, TE 24 ms, flip angle 66°,
voxels 1.68 × 1.68 × 1.7 mm, FOV 215 × 215 × 137.7 mm (matrix 128 × 128 × 81 in the NIfTI), multiband factor 3,
in-plane SENSE 2.4, partial Fourier 0.80, phase encoding j, fat suppression SPIR, total readout time 27.6 ms,
258 volumes (7:05.6 min) per run. Magnitude and phase saved (for NORDIC). Field maps: one phase-difference map per
session half (echo times 4.5 and 5.5 ms; each `IntendedFor` three runs). The sidecar says `MRAcquisitionType: 3D`
together with multiband 3, which is contradictory (see Flags).

**Anatomy.** MP2RAGE (INV1, INV2, UNI), NIfTI voxels 0.70 × 0.70 × 0.85 mm (352 × 352 × 216). The T1w for fMRIPrep
is UNI × INV2/max(INV2) (background suppression), then N4 bias correction (`create_preprocessed_t1w`,
`retsupp/prepare/utils.py`).

**NORDIC.** `NIFTI_NORDIC` on magnitude + phase, `ARG.temporal_phase = 1`, `ARG.phase_filter_width = 10`, other
options default (`retsupp/nordic/run_nordic.m`). The NORDIC MATLAB version is not recorded in the repo.

**fMRIPrep 25.2.3** (container `/shares/zne.uzh/containers/fmriprep-25.2.3`; `retsupp/cluster_preproc/fmriprep.sh`):
`--output-spaces T1w fsnative`, `--dummy-scans 4`, `--force syn-sdc`, NORDIC BOLD only (bids filter). Analyses are
**volumetric** in each subject's T1w space (`space-T1w_desc-preproc_bold`); the surface (fsnative) output is used only
for neuropythy's retinotopic atlas, which is then resampled to BOLD space for the ROIs.

**Cleaning and % signal change** (`retsupp/preprocess/clean.py`, `Subject.get_confounds`): per run, motion-outlier
frames linearly interpolated, then `nilearn.image.clean_img(..., confounds, standardize='psc')`: regression of DVARS,
framewise displacement, 10 aCompCor components, 6 motion parameters and their derivatives, non-steady-state and
cosine drift regressors, and scaling to percent signal change of each voxel's run mean. Runs cropped to 258 volumes
where cleaned BOLD has 259. The env it ran in (`neural_priors2`) no longer exists, so the nilearn version used is
not recoverable (current retsupp envs: nilearn 0.13.0 cluster, 0.12.0 local).

## A6. Software and the decoder

- **Decoder** (`retsupp/decode/decode_final.py`, built on `retsupp/decode/decode.py`): Bayesian inversion of the
  fitted pRF encoding model in braincoder. Noise model: `ResidualFitter` on the training-run residuals (a
  multivariate t with covariance ρ·ττᵀ + (1 − ρ)·diag(τ²) + σ²·WWᵀ, the van Bergen et al. (2015, Nat Neurosci)
  form with the pRF-overlap term); then `StimulusFitter` finds the MAP stimulus image, every pixel of every frame a
  free parameter, with an L2 (Gaussian) prior, L2 = 1.0, learning rate 0.5; bar-only paradigm; pRFs from the fold that
  held the decoded run out; up to 200 voxels per area by cross-run reliability, BOLD SD ≤ 20%. Cite braincoder:
  de Hollander, Renkert, Ruff & Knapen (2024), Zenodo, doi:10.5281/zenodo.10778413 (`libs/braincoder/CITATION.cff`).
- **braincoder:** editable install of `libs/braincoder` (`~/git/retsupp/libs/braincoder` on the cluster), now at
  commit f82f316 (2026-11-07, `git describe` v0.4-73-gf82f316); the package reports version 0.1.0, which is not
  informative. Fits ran on whatever commit was checked out at the time (see Flags).
- **Cluster env `retsupp_cuda`** (all pRF, gain-field, trial-wise and decoding fits): Python 3.10.19, TensorFlow
  2.14.0, Keras 2.14.0, nilearn 0.13.0, nibabel 5.2.1, NumPy 1.26.4, SciPy 1.15.2, pandas 2.3.3, statsmodels 0.14.6.
- **nideconv** (Fig. 3 deconvolution): 0.1rc5.dev59+g46146f7ab, git commit 46146f7 of
  github.com/VU-Cog-Sci/nideconv (installed in `~/pylibs/nideconv`); Fourier basis, 13 regressors, −4 to 20 s.
- **neuropythy** 0.12.16 (env `retsupp_neuropythy`, Python 3.10.20; `register_retinotopy`, Benson & Winawer 2018).
- **Bayesian models:** env `bauer` (cluster and local), Python 3.11.15: PyMC 5.28.3, nutpie 0.16.8, bambi 0.17.2,
  ArviZ 0.23.4, PyTensor 2.38.2. Used for the trial-by-trial RT models (bambi + nutpie) and the Fig. 7 hierarchical
  model (`trialwise_meta_bayes.py`, PyMC + nutpie).
- **fMRIPrep** 25.2.3; **figures:** matplotlib 3.10.6, seaborn 0.13.2 (local env `retsupp`).

## A7. Demographics, ethics, payment

Not found in the repo or the dataset: there is no `participants.tsv` in the BIDS root, `dataset_description.json` is
the unfilled template, and no note in the repo mentions age, sex, handedness, ethics approval or payment. The only
raw scanner header locally (`/data/ds-retsupp/sourcedata/mri/sub-24_ses1/*.PAR`) is an anonymised test scan. Known
from the data: 30 participants scanned, two of them pilots (sub-01, sub-02, with a different run order); no
participant excluded from any analysis (some subject × area pairs drop out for lack of voxels, listed per figure in
key_numbers). The rest has to come from the lab's records (Spinoza Centre / VU Amsterdam ethics).

## A8. The "global gain" of Fig. 7

Defined only in the trial-wise model (`retsupp/modeling/fit_trialwise_af_gains.py`, regressor `G`); it does **not**
exist in the m7 fits (its prior / canonical value is 0). For each trial it multiplies the whole, spatially unweighted
stimulus drive of the voxel during that trial's pulse (search onset to feedback, 1.5 s): `D_glob = paradigm @ rfᵀ`,
the bar plus all eight items as painted in the model's paradigm (items scaled by the fitted item weight), through
each voxel's own pRF, HRF and amplitude. Its spatial extent is therefore the entire display, every location weighted
equally; it absorbs non-spatial per-trial changes in response amplitude so that the target and distractor gains index
the spatially specific part. In the Fig. 7 parametric model the global gain also gets the RT and history modulators.

## Contradictions with key_numbers.md and other flags

1. **Fig. 1 table mixes statistics.** key_numbers gives means for rows c, d (−27.8, +43.0, +15.2, +28.4 ms; one
   median in brackets) and medians for rows e, f (−38, −15 ms). The tests are Wilcoxon, and the figure annotation now
   prints medians. Suggest medians throughout: −26.1, +43.7, +15.8, +26.8, −38.4, −15.1 ms, with the CIs above
   (the condition means 877/905/862 ms etc. can stay as means).
2. **Fig. 3 dorsal n.** key_numbers says dorsal n = 27 for the relative analyses; the b, c differences use n = 28.
   Not a conflict, but the legend/text should give 28 for b, c and 27 for d, e.
3. **Fig. 2b is not on the final fits.** The contralateral share (and `fig1_prf_coverage.tsv`) comes from the
   model-4 mean fits (DoG, flexible HRF, all runs; table written 2026-11-05 by `collect()`), while the package marks Fig. 2 FINAL on
   the final joint-HRF fits. As a sanity check this is harmless, but the legend should name the fits it uses.
4. **HP probability in the settings.** `likely_distractor_probability: 0.67` in every expsettings file is unused; the
   true design is 50% / 10% × 3 / 20% (as key_numbers and Fig. 1a say).
5. **Sidecar acquisition type.** The BOLD sidecars come from a hard-coded template and say `MRAcquisitionType: 3D`
   with `MultibandAccelerationFactor: 3`; a multiband EPI is normally 2D. Check the exam card before writing "3D".
6. **Distortion correction.** fMRIPrep ran with `--force syn-sdc` although phase-difference field maps exist; check
   in the fMRIPrep reports which correction was applied before describing it.
7. **braincoder version.** The fits used an editable checkout whose commit changed over the project (e.g. f82f316 of
   2026-11-07 fixed a float32 overflow in the HRF gamma pdf for narrow HRFs). Cite the Zenodo DOI and the commit the
   final fits ran on; the latter is not recorded per fit.
8. **Response instructions.** `experiment/instructions.yml` (left/up) and `experiment/README.md` (F/J) do not match
   the scanner keys (y/b); use the scanner keys.
9. Out of scope, for the record: the docstring of `retsupp/decode/decode_final.py` still calls the decoding a
   "pre-specified test"; code only, not in the package.
