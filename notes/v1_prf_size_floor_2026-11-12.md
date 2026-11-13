# Is the flat V1 pRF size a floor set by the fitting procedure? (2026-11-12)

Question: the manuscript says "In V1, pRF size stayed between 0.54 and 0.65° across eccentricities, which most likely
reflects a floor set by the spatial resolution of the paradigm rather than a property of V1." Would a
higher-resolution paradigm in the fit fix this?

**Answer: no.** The fitting procedure (50 px over ±5°, i.e. 0.2°/px; grid σ from 1°; `sd_min = 0.2`; DoG; one HRF per
subject × area) recovers simulated pRFs down to σ ≈ 0.3–0.4° without bias, at every fitting resolution tested. Its
only floor is `sd_min = 0.2°`, far below 0.54°. Refitting real V1 at 100 px with lower bounds leaves the bulk of V1 at
the same σ. The voxels that do shrink lose held-out R², so they are overfits, not hidden small pRFs. The flat
0.55–0.65° is the size of the voxel-level pRF that the data support. Most likely it is true V1 size broadened by
factors that do not depend on eccentricity and that no paradigm resolution can remove, mainly eye-position variability
and the scatter of receptive fields within a voxel.

## Where the number comes from

`notes/data/fig1_prf_sigma_final2.tsv` (`retsupp/visualize/paper/plot_fig1.py collect_final`): the final pRFs (DoG,
joint HRF per subject × area, `prf_cv_roi_jointhrf/*_dog_final2_stage-3_pars.pkl.gz`, parameters averaged over the
three folds). Selection: held-out R² ≥ .1 (mean over folds), ≥ 50% of the pRF inside the 3.17° aperture,
σ ∈ [0.3, 4]. Statistic: per subject the median σ in each 0.5° bin, then the mean over subjects. **Size is σ of the
DoG centre, not FWHM** (FWHM = 2.355 σ ≈ 1.3–1.5°). Spread over subjects (n = 28) is narrow: 10th–90th percentile
0.48–0.64° at 0.25–1.75°, 0.53–0.74° at 2.25–3.25°.

Fit settings (`fit_prf_cv_roi.py` → `fit_prf_cv_roi_hrf_joint.py`): paradigm `get_stimulus_with_distractors`
(bar + item rectangles), 50 × 50 px over ±5° (0.204°/px); m1 grid x, y ∈ linspace(−3, 3, 12), σ ∈ linspace(1, 4, 8)
(`fit_prf.grid_fit` raises the grid floor to 1°); `sd_min = 0.2` in the model (σ = 0.2 + softplus); DoG surround
from m1 (srf_size 4, srf_amplitude 0.05); then the shared HRF (delay ≥ 2 s, dispersion ≥ 0.1).

Stimulus: bar 0.79° wide, 0.168°/s (0.27° per TR, so the TR does not undersample the sweep), 3.17° aperture, TR 1.6 s.
Voxels 1.68 × 1.68 × 1.7 mm.

## 1. Parameter recovery by simulation

`retsupp/modeling/cv/sim_prf_size_floor.py` (+ `slurm_jobs/sim_prf_size_floor.sh`), outputs
`derivatives/sim_prf_size_floor_2026-11-12/`, summary `notes/data/v1_prf_size_floor_2026-11-12/sim_recovery_summary.tsv`.
Real stimulus of sub-23 (training runs of fold 0, T = 2064). The truth is rendered on a 200 px grid (0.05°/px) with the
canonical HRF and AR(1) noise (φ = .3) at R² .35 (median of real V1 voxels with held-out R² > .1) or .15. True σ is
0.1–1.4°, at eccentricities 0.75–3°, with 24 voxels per σ. The full paper chain is run (m1 grid → GD, m2 DoG, joint
HRF stages 2 and 3 with 3000 iterations).

Median recovered σ (final stage, DoG truth, R² .35):

| true σ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| paper settings (50 px, grid ≥ 1°, sd_min .2) | 0.20 | 0.20 | 0.24 | 0.36 | 0.47 | 0.56 | 0.76 | 1.00 |
| 100 px, same bounds | 0.20 | 0.21 | 0.26 | 0.37 | 0.47 | 0.57 | 0.75 | 0.99 |
| 150 px, same bounds | 0.20 | 0.20 | 0.25 | 0.38 | 0.48 | 0.57 | 0.77 | 0.99 |
| 100 px, grid ≥ 0.15°, sd_min .05 | 0.09 | 0.16 | 0.25 | 0.39 | 0.48 | 0.58 | 0.76 | 1.00 |
| 150 px, grid ≥ 0.15°, sd_min .05 | 0.09 | 0.15 | 0.25 | 0.39 | 0.48 | 0.58 | 0.77 | 0.99 |

- No floor near 0.55° under any setting. Under the paper's settings, recovery is accurate from 0.4° up. Below 0.3° the
  estimate stops at `sd_min = 0.2°`.
- The fitting resolution makes no difference (50, 100 and 150 px agree within 0.02°). The grid floor of 1° does not
  matter either: GD travels down from it. Only `sd_min` sets the floor.
- The same holds for a Gaussian truth and at R² .15 (summary table). The HRF is recovered (delay 4.44 s, dispersion
  0.89 against 4.5, 0.75). A truth HRF that is narrower (4 s, 0.4) or broader (5.5 s, 1.5) is absorbed by the
  joint-HRF stage and does not change σ by more than 0.05°.
- **Eye position.** A random eye position per TR (iid normal, SD s), which the model cannot know about, adds roughly
  in quadrature. True 0.2 / 0.4 / 0.6 is recovered as 0.23 / 0.45 / 0.63 at s = 0.3°, and as 0.35 / 0.56 / 0.72 at
  s = 0.5°. A broadening of about 0.4–0.5° that does not depend on eccentricity, added to true V1 σ of about 0.25–0.45°
  rising with eccentricity, reproduces the measured 0.54 → 0.65°.

Time courses of σ 0.2 and 0.6 at 2° eccentricity correlate at r = .94 on this stimulus, so 1 − r² ≈ 11% of the signal
variance separates them. The stimulus does not hide small pRFs.

## 2. Real V1 refit at higher resolution

`retsupp/modeling/cv/refit_v1_prf_resolution.py` (+ `slurm_jobs/refit_v1_prf_resolution.sh`), V1 only, sub-03,
sub-08, sub-17 and sub-23, three folds, the same chain as the paper. Outputs are in
`derivatives/prf_v1_resolution_2026-11-12/`, with tables in `notes/data/v1_prf_size_floor_2026-11-12/real_v1_*.tsv`.
'paper50' reproduces the published final fits: per-voxel σ r = .999 (sub-03) and .986 (sub-23), median σ identical.
'free100' uses 100 px, a grid from 0.15° and `sd_min = 0.05`.

Voxels with held-out R² ≥ .1 inside the aperture (4327 voxel × fold), mean over subjects of the median σ:

| eccentricity | 0–1° | 1–2° | 2–3.17° |
|---|---|---|---|
| paper50 | 0.56 | 0.60 | 0.67 |
| free100 | 0.52 | 0.63 | 0.68 |

- 91% of voxels stay where they were: σ 0.5–0.8° (median shift of the centre 0.02°). These voxels gain +.003 in
  held-out R² (median, all four subjects positive).
- 9% (401, mostly sub-23 and sub-17 near the fovea) drop to a separate cluster at σ 0.1–0.25°, with a gap at
  0.3–0.4°. These voxels **lose** held-out R² (median −.026 against their paper50 fit; paper50 σ for the same voxels
  was 0.48). They are needle solutions found from the low grid, not small pRFs that the coarse fit missed (rule 6 of
  the braincoder skill).
- In the published fits no V1 voxel with held-out R² > .1 sits near the 0.2° bound (sub-03, sub-23: none < 0.25°).
  The simulation shows that true σ of 0.2–0.3° would be recovered there. The data do not support smaller V1 pRFs.

## Cause, by the four candidates

- (a) Fitting grid / paradigm resolution: **no** (simulation, resolution 50–150 px identical; real refit unchanged for
  the bulk of V1).
- (b) Parameter bounds: `sd_min = 0.2` is a floor, but at 0.2°, not 0.55°. Neither the grid floor (1°) nor the DoG
  surround constraints raise σ (DoG and Gaussian truths both recovered).
- (c) The stimulus itself (bar 0.79°, TR 1.6 s, 0.27°/TR, 3.17° aperture) does **not** cause it: the model includes
  bar width and HRF, and recovery from this exact stimulus is unbiased from 0.4°. Voxel size and BOLD point spread
  (1.7 mm voxels; about 0.1° at 1° eccentricity, about 0.25° at 3°, assuming V1 magnification of roughly 10 and
  4.5 mm/°) broaden σ, but they would make it grow with eccentricity, not flatten it.
- (d) **Mostly real at the voxel level**: true V1 size plus broadening that does not depend on eccentricity. The likely
  sources are eye-position variability across TRs (about 0.4–0.5° SD needed; plausible in a task whose items sit at
  4°; there are no eye-tracking data to check) and position scatter within a voxel. Measured voxel-level V1 σ in other
  studies is in a similar range; check the specific numbers before citing.

## Cost of a refit at higher resolution, and what it would touch

At 100 px the V1-only refit took 2.8× as long (50 against 18 min per subject × fold, 16 CPUs; 11–25 GB). Refitting
everything at 100 px means:

- `fit_prf_cv_roi` (m1–m6) and `fit_prf_cv_roi_hrf_joint` for DoG, Gaussian and DN, 28 subjects × 3 folds. DoG
  final2 alone took 0.5–2 h per task at 50 px, so about 1.5–6 h at 100 px.
- Downstream at `resolution = 50`: reliable voxels; the gain-field ladder and the polished refits
  (`fit_cv_ladder_nested.py`); terms estimated alone and LOSO; field widths; the Fig. 3 deconvolutions;
  decoding (`decode_final.py`) and its simulations.
- Figs. 2–6 and every number in `key_numbers.md` would change.

Order of magnitude: well over 10,000 CPU-hours and several days of queue, for no expected gain. The simulation shows
the same σ, and the real refit gives +.003 held-out R² in V1 for the bulk of voxels. That gain is not separated from
the lower bounds, and the needle voxels lose more.

## Recommendation

Do not refit. Change the sentence: the floor is not a resolution floor.

Suggested wording:

> In V1, pRF size (σ) stayed between 0.54 and 0.65° across eccentricities. This is not a limit of the fitting
> procedure: in simulations with the same stimulus, noise level and fitting steps, pRFs were recovered without bias
> down to σ ≈ 0.4°, independently of the spatial resolution used in the fit, and refitting V1 at twice the resolution
> did not change it. The flat profile more likely reflects broadening that does not depend on eccentricity, such as
> small variations in eye position and the scatter of receptive fields within a voxel, on top of the small true
> pRFs of central V1.

A shorter version for Results: "In V1, pRF size stayed between 0.54 and 0.65° across eccentricities; parameter
recovery showed that the fits resolve σ down to about 0.4°, so this flat profile is not a resolution limit of the
fit (Supplementary Note)."
