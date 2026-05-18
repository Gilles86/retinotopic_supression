# Conditionwise PRF shifts: TWO approaches, kept separate

This is the most important conceptual distinction in this part of the
project. They get confused easily (I have, multiple times). Read this
fully before writing any script that touches PRF shifts.

There are **two fundamentally different methodological approaches** to
how attention-driven PRF shifts get modeled.

---

## Approach 1 — Post-hoc shift fits (every other group's pipeline)

**What it is:** Fit vanilla PRFs (no attention model) separately per
condition, observe how the fitted PRF centers move, fit a
phenomenological "shift model" to those observed shifts.

**Pipeline:**

1. **Fit PRFs per condition with a vanilla PRF model — no AF, no
   attention modulation.** For retsupp: `m4 conditionwise` (4 fits
   per voxel, one per HP-distractor condition):
   ```
   derivatives/prf_conditionfit/model4/sub-XX/condition-<cond>/sub-XX_desc-{x,y,sd}.nii.gz
   ```
   Each voxel gets 4 independent Gaussian PRFs, each with its own
   (x, y, σ). The fit sees the stimulus AS IS — no AF modulation in
   the forward model.

2. **Compute observed PRF-center shifts** as the empirical movement of
   the fitted centers:
   ```
   shift_voxel_v_condition_c = (x_c - x_baseline, y_c - y_baseline)
   ```
   where `baseline` = mean over the 4 conditionwise PRFs (per voxel),
   or alternatively the m4 mean fit.

3. **Fit a SHIFT MODEL** on those observed shifts post-hoc. The shift
   model is **phenomenological** — it doesn't model the BOLD time
   series. It only fits parameters that explain how location vectors
   move. Examples:
   - **Klein/Womelsdorf**: 4 attention-field Gaussians at the 4 ring
     positions, **per-ring σ** (σ_HP, σ_LP), unit amplitude, no gain.
     HP/LP asymmetry comes from σ alone.
   - **Sumiya analytical**: precision-weighted mean shift, expresses
     the modulated PRF center as a weighted average of (true PRF
     center, ring positions).

4. **Optimise the shift-model parameters** by minimising e.g. Σ ||obs
   shift − predicted shift||² across (voxel, condition) pairs.

**Key property:** the PRFs being fit DO NOT see any AF modulation.
The "shift" is an emergent thing that the post-hoc model
*describes*, not *causes*. The single voxel's PRF is allowed to be
literally a different PRF in each condition.

**retsupp scripts that implement Approach 1:**
- `Subject.get_prf_parameters_volume(model=4, type='conditionwise')` —
  loads the 4-condition PRFs.
- `retsupp/visualize/paper/fit_klein_static_shift.py` — Approach 1
  with Klein's 4-AF, per-ring σ shift model.

---

## Approach 2 — Joint AF + PRF BOLD-level fitting (ours)

**What it is:** Fit a joint AF + single-PRF-per-voxel model on the
BOLD time series. The AF modulates the visual stimulus drive that
the (single) PRF sees per condition. The PRF stays put — what
changes is the effective stimulus drive integrated by the PRF.

**Pipeline:**

1. **Joint forward model** on BOLD, all conditions in one fit:
   ```
   M_C(g) = 1 + sign · Σ_ℓ g_ℓ · A_ℓ(g)            # AF modulation
   neural_v(t) = Σ_g paradigm(g, t) · M_{c(t)}(g) · S_v(g)
   bold_v(t)   = HRF * neural_v(t)
   ```
   `S_v` = single per-voxel Gaussian PRF (x, y, σ). `A_ℓ` =
   per-ring AF Gaussian, **shared σ_AF** across rings,
   **per-ring gain** (g_HP at HP, g_LP at 3 LP rings).

2. **Joint optimisation** of per-voxel PRF (x, y, σ, ...) AND shared
   AF parameters (σ_AF, g_HP, g_LP, sometimes σ_dyn, g_HP_dyn, etc.)
   on BOLD residuals.

3. **Predicted shifts are derived POST-HOC from the fit**: apply the
   AF modulation to the fitted PRF, compute CoM of the modulated
   response per condition, subtract baseline. This gives a "shift
   per voxel per condition" that can be compared with Approach 1
   data.

**Key property:** there is ONE PRF per voxel; "shifts" are not real
movements of the PRF, they're center-of-mass changes of the
(PRF × attention_field) product. The PRF itself stays put. The fit
is constrained by every BOLD timepoint, not just the per-condition
centers.

**retsupp scripts that implement Approach 2:**
- `retsupp/modeling/fit_dog_dynamic_af_braincoder.py` — joint AF +
  DoG-PRF fit on BOLD.
- `retsupp/modeling/local_models.py`:
  `DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma`
  (and ~10 sibling variants).
- `retsupp/visualize/paper/predict_shifts_from_af.py` — takes the
  Approach-2 fit pickle, computes the predicted conditionwise CoM
  shifts under the AF modulation.

---

## What the comparison figure shows

The talk-figure compares three things in the same rotated visual-field
frame:

| Panel | Approach | What it is |
|---|---|---|
| **DATA** | 1 | Observed shifts: m4 conditionwise PRF centres minus baseline |
| **Klein 4-AF** | 1 | Fitted size-based shift model on the data |
| **Our AF (predicted)** | 2 | Our BOLD-level AF fit's predicted shifts |

It establishes:
1. **Observed shifts are LOCAL**, not global (visible in either Approach
   1 panel) — answers the "your shift model shifts every voxel"
   confusion at VSS.
2. **Klein (Approach 1, size-based)** and our AF (Approach 2,
   gain-based) make similar conditionwise-shift predictions, despite
   fitting on different objectives at different levels of the
   pipeline.
3. **The two approaches converge** on the spatial pattern — both
   say shifts concentrate near HP and decay away.

---

## Trap I keep falling into

> Don't use Approach-2 outputs (AF-fit-derived baseline, AF-fit-derived
> σ) as inputs to an Approach-1 fit. Approach 1 is supposed to be
> AF-free. Its baseline PRF comes from the AF-free conditionwise
> fits. Its σ comes from the same fits.

This is what `feedback_audit_before_building.md` and the
"don't conflate approaches" memory are about. If a Klein-style fitter
needs σ_PRF, it should load it from the m4 conditionwise (or m4 mean)
NIfTI — NEVER from the AF fit pickle.

---

## Sister scripts (related, useful)

- `plot_rotated_shift_fields.py` — vector-field viz in the rotated
  frame. Reads the long-format TSV from `predict_shifts_from_af.py`
  (Approach 2) and/or `fit_klein_static_shift.py` (Approach 1) — the
  TSV schema is shared (subject, roi, condition, voxel_idx, base_x/y,
  obs_x/y, plus model-specific pred_*).
- `plot_rotated_shift_heatmaps.py` — heatmap version.
- `compare_amplitude_vs_position_shift.py` — within-Approach-2
  comparison: amplitude-modulation (the standard formulation) vs
  position-shift (Sumiya analytical). Both Approach 2.
- `demo_foveal_af_influence.py` — illustration that AF reach (Gaussian
  tails) extends inside the aperture.

---

## When to extend, when to leave alone

- **Want a new shift model in the Klein family?** Add to
  `fit_klein_static_shift.py` (Approach 1). Local fit, ~seconds.
- **Want a new AF model variant fit on BOLD?** Add a class to
  `retsupp/modeling/local_models.py` (Approach 2). Cluster fit, hours.
- **Want a new visualisation?** Extend `plot_rotated_shift_fields.py`
  or `plot_rotated_shift_heatmaps.py`. **Don't** write a parallel
  plot — they handle Approach-1 and Approach-2 outputs in the same
  TSV schema.
