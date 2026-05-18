# Conditionwise PRF shifts: data, models, and how to compare them

Context for the post-VSS figure that resolves the "shift models predict
shifts across the whole visual field" confusion. This document explains
the pipeline so the next person (often me, a week from now) doesn't
re-invent the existing infrastructure.

## The data — m4 conditionwise PRF fits

For each subject, m4 (DoG + flexible HRF) is fit separately on the
runs of each HP-distractor condition. So per voxel you get **four
PRF locations** — one per condition. These live at:

```
derivatives/prf_conditionfit/model4/sub-XX/condition-<cond>/sub-XX_desc-{x,y,sd,...}.nii.gz
```

with `<cond>` ∈ {upper_left, upper_right, lower_left, lower_right}.

Access via `Subject.get_prf_parameters_volume(model=4, type='conditionwise')`
— returns a DataFrame indexed by condition × parameter, values are
NIfTI images (per-voxel).

The **observed shift** for voxel v in condition C is
`obs_xy[v, C] - base_xy[v]`, where `base_xy[v]` is the voxel's
baseline PRF location (mean across conditions, or m4 mean fit — the
existing pipeline uses the AF-fit's `x, y` as base, which is close to
m4 mean because the AF fit warm-starts from there).

## The models — two competing parameterisations

Both are **4-AF models** (one AF Gaussian at each of the 4 ring
positions, ring positions on a circle at 4° eccentricity, 45° offsets).
The HP/LP distinction sits in different parameters in each model:

| Model | HP vs LP via | Free params (shared per ROI) |
|---|---|---|
| Our AF+1 (`DoGDynamicAttentionField...`) | gain (g_HP, g_LP); shared σ_AF | σ_AF, g_HP, g_LP |
| Klein 4-AF (Klein/Womelsdorf style) | size (σ_HP, σ_LP); unit gain, no g | σ_HP, σ_LP |

For both, the modulation field at grid location g under condition C is:

```
M_C(g) = 1 + sign * Σ_ℓ w_ℓ(C) * A_ℓ(g)
```

- **Our AF**: `w_ℓ(C) = g_HP` if ℓ is the HP-ring for C else `g_LP`,
  all `A_ℓ` have the same σ_AF.
- **Klein**: `w_ℓ(C) = 1` (unit), `A_ℓ` has σ_HP if HP-ring for C else σ_LP.

`sign = -1` for the retsupp data (suppression — RFs shift AWAY from HP).

The **predicted observed PRF center** under each model is the
**numerical center-of-mass** of `S_v(g) · M_C(g)` on a fine grid
(`S_v` = the voxel's baseline Gaussian PRF). No closed form for either.

## Pipeline files (existing — don't reinvent)

### Compute predicted shifts from our AF+1 fits
`retsupp/visualize/paper/predict_shifts_from_af.py`

- Inputs: AF-fit pickles (per sub, ROI), conditionfit-model index.
- Loads fit pickle → shared (σ_AF, g_HP, g_LP) + per-voxel (x, y, sd).
- Computes numerical CoM under the AF modulation per (voxel, condition).
- Loads observed conditionwise PRF centers via `Subject`.
- Writes long-format TSV: one row per (subject, ROI, voxel, condition)
  with columns:
  - `base_x, base_y, base_sd`  — baseline PRF
  - `pred_x, pred_y`           — our AF model prediction
  - `obs_x, obs_y`             — m4 conditionwise PRF (data)
  - `obs_r2`                   — conditionwise fit R²
  - `proj_pred, proj_obs`      — projection on AWAY-FROM-HP axis
  - `sigma_AF, g_HP, g_LP`     — fitted AF shared params

### Fit Klein 4-AF on the observed shifts
`retsupp/visualize/paper/fit_klein_static_shift.py`

- Inputs: TSV from `predict_shifts_from_af.py`.
- For each (sub, ROI): minimise Σ ||klein_predicted_shift - obs_shift||²
  over (σ_HP, σ_LP) via `scipy.optimize.minimize` (L-BFGS-B, 2 params).
- Writes merged TSV with `pred_klein_x, pred_klein_y` per row + a
  separate `*.fits.tsv` with the fitted σs per (sub, ROI).
- Runs **locally** in seconds. NOT a cluster job.

### Plot vector fields in the rotated frame
`retsupp/visualize/paper/plot_rotated_shift_fields.py`

- Reads the merged TSV.
- Rotates per (sub × condition) so the HP location ends at canonical
  (0, +4°). Stacks all 4 conditions × all subjects in the same frame.
- Per ROI: 3-panel page (PRED | OBS | OBS−PRED) of binned 2D shift
  vector fields. **To add Klein**: add a 4th panel column reading the
  `pred_klein_*` fields.

### Sister scripts (related, useful)
- `plot_rotated_shift_heatmaps.py` — scalar heatmaps (projection on
  AWAY-FROM-HP axis), per ROI. 3 rows (proj / Δx / Δy) × 3 cols
  (PRED/OBS/RESID).
- `compare_amplitude_vs_position_shift.py` — compares the AMPLITUDE
  formulation (our AF) vs the POSITION-SHIFT analytical formulation
  (Sumiya AF+ closed form) on the SAME fitted params. Diagnostic for
  the analytical approximation.
- `demo_foveal_af_influence.py` — shows AF reach inside the aperture
  (Gaussian tails from the ring positions extend into the centre).

## When to extend, when to leave alone

- **Adding a new AF variant**: if it's a per-ring σ Klein-style model
  (no gain), fit it post-hoc on shifts via `fit_klein_static_shift.py`
  (cheap, local). BOLD-level fits (joint with PRF) are the
  publication-grade alternative but add hundreds of GPU-hours; usually
  not necessary for the comparison.
- **Adding a new viz**: extend `plot_rotated_shift_fields.py` or
  `plot_rotated_shift_heatmaps.py` (add a column), don't write a
  parallel plot script. Specifically: I almost wrote
  `plot_shift_naive_vs_af.py` — deleted because it duplicated the
  existing rotated-frame plot infrastructure.
- **Adding the "Sumiya AF+ attraction baseline"**: don't. It's
  trivially the same model with sign flipped, and the data is
  suppression — attraction-mode fits give near-zero gains. Not
  informative beyond "yes, sign of gain matches sign of effect."

## Reminders for me

- The "AF+1" terminology refers to the `1 + ...` modulation
  formulation (Reynolds-Heeger AF+). Our model is in this family,
  with **gain-based** HP/LP asymmetry. Don't confuse with a
  "AF+1 baseline" — there's no such thing.
- Klein/Womelsdorf classic = unit-amplitude Gaussians, no gain — so
  the HP/LP asymmetry MUST live in σ. Size-based comparison model.
- The shift-prediction TSV is the canonical interchange format.
  Add columns when adding new models. Don't fork the format.
- Always do `Subject.get_prf_parameters_volume(type='conditionwise')`
  for the data — handles the nested
  `condition-<cond>/sub-XX_desc-X.nii.gz` layout.
