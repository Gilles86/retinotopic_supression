# Cluster analysis plan — retsupp PRF & suppression follow-ups

Discussion goals for this afternoon's meeting with Dock & Jan: which of the
four proposed analyses are worth running on the cluster, in what order,
and what we expect them to add over the model-free results we already
have.

Current model-free position (recap, see meeting_report.pdf):
- HP-specific PRF push-away in V3AB (p ≈ 0.05) and hV4 (p ≈ 0.005–0.010)
  using the projection-onto-anchor metric (no Jensen).
- Permutation null (per-subject condition shuffle) confirms hV4 effect at
  p = 0.010, V3AB borderline at p = 0.050.
- Linear projection vs distance shows:
   * V3AB and TO carry the bidirectional pattern (positive near, negative
     far — local suppression + global attraction).
   * hV4 carries pure local suppression only.
   * VO shows a weaker effect at ~2° distance.
- DN model 6 already shows the expected hierarchy: `srf_amplitude` an
  order of magnitude larger from V3AB onward; `excitatory_sd` doubles.
- AF/AF+ fits at the per-condition level have not converged on
  interpretable AF positions — too many free parameters relative to a
  small condition-specific signal on top of a common attentional state.

Important design caveat:
- The HP distractor locations sit at 4° eccentricity, which is OUTSIDE the
  PRF mapping bar aperture (radius ≈ 3.17°). This means voxels with PRFs
  near HP — exactly the ones carrying the suppression signal — necessarily
  have PRFs that wander past the aperture boundary. A naive "exclude PRFs
  outside the aperture" filter at 50% mass-outside removes 7.9% of voxels
  AND most of the signal. Compromise filter at 75% mass-outside drops
  only 0.3% and preserves the effect.

---

## Analysis 1 — Four competing AF+ fields, one per distractor location

### Idea
All four ring positions always have an AF (because all four are real
distractor locations with non-zero distractor probability). Across
conditions, only the *strength* (size) of the AF at the HP location
varies — it is *larger* (or sharper) than the AFs at the three LP
locations. Centers are fixed at the four ring positions; we fit one
shared `σ_AF_HP`, one shared `σ_AF_LP`, and one `b` per voxel.

### Concrete model
For each voxel `v` in condition `C` with HP at quadrant `H_C`:

```
R_v_C(x) = (b − Σ_{ℓ ∈ ring} g_ℓ_C · A_ℓ(x; ℓ, σ_ℓ_C)) · S_v(x)
```

where
- `g_ℓ_C = g_HP` if `ℓ == H_C` else `g_LP`,
- `σ_ℓ_C = σ_HP` if `ℓ == H_C` else `σ_LP`,
- `S_v(x) = N(x; base_v, σ_base_v)`,
- `b > Σ g_ℓ` enforces positive responses.

Center of mass closed form follows the same overlap-integral derivation
as my single-AF+ model. With four AFs the formula has four overlap
terms; still fully closed-form, no numerical integration.

### Free parameters (per ROI per subject)
- Shared: `g_HP`, `g_LP`, `σ_HP`, `σ_LP`, `b` (5 params).
- Per voxel: `(x0_v, y0_v)` = base PRF position (2N params, closed form).

### Hypotheses
- `g_HP > g_LP` and/or `σ_HP > σ_LP` would be the model statement of
  "HP suppression is stronger than LP".
- Fit per-subject; collect distributions of `g_HP / g_LP` ratios across
  V1/V2/V3/V3AB/hV4/LO/TO/VO; expect ratio significantly > 1 in V3AB/hV4
  (where we already see HP-specific effects).

### Implementation notes (re-read of `retsupp/modeling/af_model.py`)
- `_solve_base` is the closed-form per-voxel base assuming attractive AF
  and `Σ_C R_C = 0`. For 4 fixed AFs at the ring corners we'd need a
  modified `_solve_base` that respects the new model form (each voxel
  sees the SAME 4-AF structure across conditions, just with the labels
  rotated).
- Reuse `_within_voxel_r2`, `prepare_voxel_data`, and
  `fit_af_rotated_canonical` infrastructure.
- Add `fit_af_four_competing` with bounds: `σ_HP, σ_LP ∈ [0.5, 15]°`,
  `log(g_HP), log(g_LP) ∈ [-3, 3]`, `b ∈ [1, 50]`.
- Multi-start over a small grid of `(σ_HP, σ_LP, g_HP/g_LP)` initial
  guesses.

### Effort + cluster requirement
~30 min compute on login node (closed-form, ~10 sec/subject × 28).
No GPU needed. Drop in as `retsupp/modeling/fit_af_four.py`.

### Stop criteria
If `g_HP/g_LP` is ~1 across V3AB/hV4 with no hierarchy effect, the
model-free results don't translate to identifiable model parameters
and we drop the framework.

---

## Analysis 2 — PRF model with extended design matrix (distractors included)

### Idea
The current PRF stimulus includes the bar that sweeps through the
aperture but **NOT** the distractor stimuli that are presented during
the search task. The search task is happening *concurrently* with the
PRF mapping (per `Subject.get_stimulus`). Distractors at the 4 ring
locations are consistently active and *outside* the bar aperture (they
sit on the 4° distractor ring; the bar aperture is ~3° eccentricity per
`get_experimental_settings`). For voxels with PRFs at or near a ring
location, the distractors are a real visual stimulus we're ignoring.

### Plan
Build an extended design matrix that includes:
- The original bar stimulus (current `Subject.get_stimulus`).
- Four "distractor on/off" regressors, one per ring quadrant, time-locked
  to actual trial presentations from `Subject.get_onsets`.

Fit two PRF models on the same data:
- **Model A** = current bar-only design (model 4).
- **Model B** = bar + 4 distractor regressors.

Compare per-voxel R² (cross-validated across runs) and per-voxel
parameter changes. Distractor inclusion should improve fits for voxels
with PRFs near the ring; the size of the improvement is a direct
estimate of how much variance the distractors carry.

### Implementation
- Modify `Subject.get_stimulus` (or write a new method) to add 4
  binary on/off channels to the design representing each ring-position
  distractor's presence. The braincoder model class already supports
  multi-channel stimulus inputs. Re-fit in `modeling/fit_prf.py` with
  an extended `paradigm` array.
- Use cross-validation: hold out one run, fit on the rest, score on
  held-out. Run on the cluster (GPU env, `tms_risk_cuda` style or
  `neural_priors2`).

### Effort + cluster requirement
GPU env needed (fits ~5 min/subject × 28 = ~3 hours wall if serialized;
parallel SLURM array brings it under 10 min). Re-uses
`retsupp/modeling/fit_prf.sh` SLURM template.

### Open question for the meeting
Does the GLM regress out distractor activity during conditionwise PRF
fits already? If `derivatives/cleaned/` already removes the search-task
GLM residuals, the bar-only PRFs are clean and this analysis is moot.
**Need to check `preprocess/clean.py` and `eccentric_glm/clean_data.py`
before running.**

---

## Analysis 3 — Location localizer + single-trial GLM (Richter 2025 replication)

### Idea
This is the most directly publishable analysis. Richter et al. 2025
showed in EVC: BOLD at HPDL ↓ < BOLD at NL-near < BOLD at NL-far for all
stimulus types (target, distractor, neutral), even during omission
trials (proactive suppression). We have the data to do this in V1–LO.

### Pipeline
1. **ROI definition** per voxel-distractor-location:
   - Use the existing `retsupp/eccentric_glm/get_rois.py` machinery,
     which already builds ROI masks per ROI × hemisphere × quadrant
     using neuropythy retinotopic atlas. Output is in
     `derivatives/stimulus_rois`.
   - Read these as 4-quadrant ROIs for each visual area.
2. **Single-trial GLM**:
   - Modify `retsupp/eccentric_glm/fit_glm.py` (currently fits an
     across-trial GLM with one regressor per distractor location) to
     fit a per-trial GLM (LSS or LSA — least-squares-single).
     `nilearn.glm.first_level.FirstLevelModel` with one regressor per
     trial works directly.
   - Per voxel, get a trial × value vector. Average within each ROI
     mask (so per ROI per trial: one number).
3. **Contrast**:
   - Per condition (HP location), per stimulus type (HP-distractor /
     LP-distractor / target / neutral / omission), per ROI (V1...VO),
     get the mean trial-wise BOLD.
   - Report the same plots as Richter Fig 4: BOLD as a function of
     stimulus type and location, separately for HPDL vs NL-near vs NL-far
     bins.

### Hypotheses
- **Search trials**: BOLD at HPDL < NL-far across stimulus types in V1
  (Richter found this in V1+V2 combined; we have higher SNR / more subjects).
- **Omission trials**: same pattern proactively.
- **By ROI**: predicted BOLD-suppression hierarchy might differ from the
  PRF-shift hierarchy. We can test the same six visual maps and compare.

### Why this is high-priority
- Direct, comparable result with a published paper.
- Independent of PRF-shift modeling (which has been hard to interpret).
- Single-trial GLMs let us also do **trial history** analyses (was the
  HPDL active on trial t given that trial t-1 had HPDL distractor?).

### Implementation notes from `retsupp/eccentric_glm/`
- `clean_data.py`: regresses out the PRF prediction from BOLD time
  series → `derivatives/prf_regressed_out`. Use this as the input.
- `fit_glm.py`: currently does *contrast* GLMs over the four distractor
  locations. Need to switch to single-trial regressors (one per trial).
- `summarize_glms.py`: aggregates beta/zmaps over ROI×quadrant masks,
  drop-in for the summary step.
- `get_rois.py`: produces the quadrant ROIs in BOLD space; reuse.

### Effort + cluster requirement
Modest. CPU env. Modify existing scripts; per subject ~15 min, 28
subjects in parallel via SLURM array → ~30 min wall.

---

## Analysis 4 — Joint AF + PRF model in the braincoder framework

### Idea
Currently we fit PRFs (model 4 / 8) without an AF, then post-hoc fit an
AF on top of the PRF parameters. This forces the AF to explain
condition-specific differences in already-noisy PRF estimates. A cleaner
approach is to **fit the AF and PRFs jointly** at the BOLD-signal level,
within a single ROI, using braincoder's model machinery.

### Concrete model
For each voxel `v`, response time series `BOLD_v(t)` is predicted by the
HRF-convolved product of the bar stimulus with the voxel's PRF, *with
the PRF center shifted by an AF effect that depends on the current HP
condition*:

```
predicted_PRF_v(x; t) = AF(x; μ_AF_C(t), σ_AF, b) · S(x; x0_v, y0_v, σ_v)
predicted_BOLD_v(t)   = HRF(t) ⊛ ∫ predicted_PRF_v(x; t) · stimulus(x; t) dx
```

where `μ_AF_C(t)` is a step function that takes one of 4 values
depending on which condition's run we're in.

### Free parameters
- Per voxel: `(x0_v, y0_v, σ_v)` — base PRF parameters (3 params).
- ROI-shared: `σ_AF`, `b`, and 4 free `μ_AF_C` (or 1 shared via
  rotation symmetry) — 3 + 8 = 11 params per ROI per subject.

### Why braincoder
- `GaussianPRF2DWithHRF` already has the closed-form HRF + 2D Gaussian
  pipeline; the AF multiplication is a straightforward extension.
- braincoder backends (TF/JAX) handle parameter sharing across voxels
  natively (the design pattern is in `libs/braincoder/braincoder/models.py`).
- Gradient-based fitting at the BOLD level avoids the
  noise-amplification of post-hoc PRF-parameter fits.

### Methodological innovation
This is the part Dock and Jan would publish independently. Currently no
existing PRF model jointly accounts for spatial attention/suppression at
the level of the BOLD signal — Klein 2014, Tunçok 2025, Sumiya thesis,
and our own work all do post-hoc fits. A braincoder-native joint AF+PRF
model would be reusable across `value_capture`, `abstract_values`, and
`tms_risk` projects.

### Implementation notes
- Inherit from `GaussianPRF2DWithHRF` (or `DivisiveNormalizationGaussianPRF2DWithHRF`)
  and add an `attention_field` channel that multiplies the spatial PRF
  before the HRF convolution.
- New parameter group: `μ_AF_canonical, σ_AF, log(b)` — shared across
  voxels in the ROI.
- Use the rotated-frame trick (HP at canonical (0, 4°)) so per-condition
  variation reduces to one shared `μ_AF`.
- Fit on cluster GPU (4–6 minutes per subject for a ~5000-voxel V3AB+hV4).

### Effort + cluster requirement
Substantial — 1–2 weeks of dev. New braincoder model class, tests, fit
script. GPU CUDA env (`neural_priors2` works). Cluster jobs ~1 hour
total.

### Stop criteria
- If the simpler four-competing-AFs (Analysis 1) already gives clean
  hierarchy, joint modeling adds polish but not new findings — defer.
- If Analysis 1 is noisy at the parameter level, joint modeling has the
  best chance of stabilizing things and is worth the dev cost.

---

## Recommended order

| # | Analysis | Risk/Reward | Days |
|---|----------|-------------|------|
| 1 | Location localizer + single-trial GLM (Richter replication) | Low risk, high publication value | 1–2 |
| 2 | Extended design matrix PRF | Diagnostic only; rules out a confound | 1 |
| 3 | Four competing AF+ (Analysis 1) | Moderate risk, makes HP vs LP a quantitative test | 2–3 |
| 4 | Joint AF + PRF in braincoder | High risk, high reward, methodological paper | 10–14 |

### Why this order
- **#1 first** because it's the most direct counterpart to the Richter
  result and gives us a publishable analog without any modelling. Cheap.
- **#2 in parallel** because if `derivatives/cleaned/` already regresses
  out distractor activity, our PRF effects can't be a distractor-driven
  artefact and #1's BOLD-suppression interpretation is cleaner.
- **#3 third** because it formalizes the HP-vs-LP framing we've been
  testing model-free; adds quantitative parameter to the talk.
- **#4 only if** #3 leaves the AF parameters poorly identified. The
  joint braincoder model is the right answer if the post-hoc fits keep
  hitting bounds, but it's a big dev investment.

**My recommendation:** start with #1 and #3 in parallel (cheap, fast to
set up, addresses real questions). Then #2 once #1 results are in. #4
only if #2 leaves the model-fit story unsatisfying.

---

## Things to verify before any of these run

1. **Does `derivatives/cleaned/` already regress out the search-task
   distractor activity?** Check `preprocess/clean.py`. If yes, Analysis 2
   is moot; if no, Analysis 2 is mandatory before any conclusion about
   PRF-shift origin.
2. **Are the existing `derivatives/stimulus_rois/` quadrant masks still
   correct after the recent neuropythy registration changes?** Run a
   sanity check (count voxels per ROI×quadrant per subject).
3. **Single-trial GLM choice — LSS vs LSA**: use LSS by default for
   trial-wise estimates (less colinearity in fast event designs).
   `nilearn` doesn't ship LSS directly; nilearn-extensions or `pyglm`
   does. `glmsingle` (already in the conda env) is a third option and
   is industry-standard.

---

## What I'd like Dock & Jan's input on

1. Is the **Richter-style single-trial replication** in scope for the
   VSS talk, or is this a separate paper? Tactically, those analyses
   require running on the cluster and ~2 days of work.
2. The **HP vs LP framing** vs Tunçok's distributed/neutral baseline:
   is there a behavioral/protocol way to add a distributed condition in
   future scans, or do we accept the HP/LP design as is?
3. **Ranking of analyses** — what's the minimum viable result for VSS
   poster / talk? My guess: Fig 1, Fig 3 (model-free) for the talk, with
   Analysis 1 + Analysis 3 results as supporting material.
