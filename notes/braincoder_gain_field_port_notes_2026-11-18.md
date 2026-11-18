# Porting the retsupp gain-field models to braincoder: what to take into account (18 Nov 2026)

For the agent moving the gain-field models into braincoder. The goal is that the paper's main model and fitting
procedure can be run from a released braincoder, and that the result **reproduces the published fits**. Everything
below was checked against the code on 18 Nov 2026.

## 1. Where things are now

| What | Where | On braincoder main? |
|---|---|---|
| Generic attention-field pRFs (`AttentionFieldPRF2D`, `DoGAttentionFieldPRF2D`, `(DoG)DynamicAttentionFieldPRF2D_v2/_v3`, all `WithHRF`), 14 classes, test `braincoder/tests/test_attention_field_prf.py` | `libs/braincoder/braincoder/models.py` on `feature/f-beta-r2-mixture` (merge of `feature/attention_field_prf` + the R² mixture), submodule HEAD f82f316 | **No** |
| v3 + target gain, sharedSigma, the factorial sign patterns, the **item scale** (`..._sharedSigma_factorial_items`, the class the paper uses), ~20 other variants | `retsupp/modeling/local_models.py` (3.9k lines) | No |
| Building the model, paradigm, indicators, folds | `retsupp/modeling/fit_af_prf_cv_v2.py` (`build_model`, `build_data_and_paradigm`, `shared_pars_for`, `fixed_pars_for`, `get_ring_positions`) | No |
| Ladder of nested models, bounds, **zero-sum constraint**, starts | `retsupp/modeling/fit_cv_ladder_nested.py` (`LADDER`, `SHARED_BOUNDS`, `zs_bounds`, `ZERO_SUM_*`, `zs_fit`) | No |
| **Joint Levenberg–Marquardt** of per-voxel pRFs + shared terms (Schur complement) | `retsupp/modeling/joint_lm.py` | No |
| Per-area HRF | `retsupp/modeling/roi_hrf.py` (SPM HRF with fitted delay / dispersion per subject × area) | SPM HRF yes |
| DN and Gaussian backbones | `retsupp/modeling/dn_attention_field.py`, `gauss_attention_field.py` | No |
| Rationale | `notes/joint_optimisation_2026-11-13.md`, `notes/zero_sum_refit_2026-11-13.md` | – |

**The two braincoders have diverged.** Main is 0.7.1, Keras 3 (`keras>=3.0`, TF ≥ 2.16 optional), with `models/` split
into `base.py`, `prf_2d.py`, … and an `ops`/`utils/backend.py` layer. 93 commits on main are not in our submodule, and
25 submodule commits are not on main. The submodule is TF 2.14, one `models.py`, `@tf.function` and `tfp` throughout.
So this is a **port to Keras 3 ops**, not a merge: no `tf.*` / `tfp.*` calls in the ported classes.

Check which submodule fixes main already has before porting on top of it:
- `gamma_pdf` in log space (f82f316): on main (`hrf.py` uses `lgamma`).
- `rf_normalization` option (a4fc80b): on main. The fits used `'legacy'`; keep that the default.
- DN `WithHRF` passing only HRF columns (2849288) and signed DN `rf_amplitude` (38754e5): **check**.
- `@tf.function` fit steps (4acbd4b, 75034c1, bb0eac7): main has its own backend-agnostic version; just make sure
  `shared_pars` still works (it is on main, f229118 / 1baddaf).

## 2. The model the paper uses (main model, `m5_TDS_dS_zs`)

Per voxel (7): DoG pRF `x, y, sd, baseline, amplitude, srf_amplitude, srf_size`.
Shared within subject × area: `sigma_AF`, `g_HP`, `g_LP`, `sigma_dyn`, `g_HP_dyn`, `g_LP_dyn`, `g_T_dyn`,
`sigma_T_dyn` (tied := `sigma_dyn`), `item_scale`. HRF: fixed per subject × area (not fitted here).

The drive is the paradigm times a multiplicative field, then the DoG pRF, then the HRF:

```
S_eff(x, t) = S(x, t) · (item_scale if ‖x‖ > 3.25° else 1)          # 8 search items vs the bar
M(x, t)     = 1 + g_HP · A_HP(x)  + g_LP · Σ_{ℓ≠HP} A_ℓ(x)            # sustained, σ_AF, HP of the current run
                + g_HP_dyn · d_HP(t) A^dyn_HP(x) + g_LP_dyn · Σ_{ℓ≠HP} d_ℓ(t) A^dyn_ℓ(x)   # distractor on, σ_dyn
                + g_T_dyn · Σ_ℓ tgt_ℓ(t) A^dyn_ℓ(x)                    # target on, σ_T_dyn := σ_dyn
pred_v(t)   = HRF * [ baseline_v + amplitude_v · Σ_x DoG_v(x) S_eff(x, t) M(x, t) ]
```

A are unit-peak Gaussians at the four ring positions (4°, diagonals); `d_ℓ(t)`, `tgt_ℓ(t)` are per-TR on-fractions.

Constraints in the main model:
- **Zero-sum sustained field:** `g_HP + 3·g_LP = 0`, so `dS = g_HP − g_LP = 4/3·g_HP`. Optimised as one free
  coordinate (`joint_lm.shared_map`: g_HP free, g_LP follows). It is a constraint on the parametrisation, not a
  penalty. Reason: with pRFs free, a level shared by all four locations is a reshaping of every pRF and is not
  identifiable (`notes/zero_sum_refit_2026-11-13.md`). **This must be expressible in braincoder** (linear map
  s = Aφ + b on the shared terms, or a dedicated `zero_sum=True` option).
- **No transient HP − LP term:** `g_HP_dyn` tied to `g_LP_dyn` (sign tag `'tied'`). `m7_full_zs` frees it (supplement).
- Sign tags per gain (`plus`, `minus`, `zero`, `free`, `tied`): the paper's models use only `free`, `zero`, `tied`.
- Bounds (`zs_bounds`), in natural units: `g_HP ∈ [−1, 3]`, `g_LP ∈ [−1, 1/3]` (field stays ≥ 0), transient gains
  `∈ [−1, 5]`, `item_scale ∈ [0.05, 50]`, `sigma_AF, sigma_dyn ∈ [0.5, 8]` (σ, not FWHM). Bounds act in raw space by
  projection.

## 3. Traps

1. **Sign convention.** The braincoder classes default to `mode='suppression'`, i.e. `M = 1 − (…)`. The paper's
   gains are signed with `M = 1 + (…)`; the factorial subclass forces `mode='signed'` and owns the transforms. An
   earlier bug double-negated the signed mode. Make `signed` the only mode in the port (or the default) and test
   the sign explicitly: a positive `g_T_dyn` must *raise* the predicted response at the target.
2. **Item scale and the paradigm.** The paradigm paints the bar *and all 8 search-item rectangles* as 1 (not only the
   distractor/target). `item_scale` multiplies pixels with ‖x‖ > 3.25° (`item_radius_min`), i.e. outside the
   3.17° bar aperture. Don't make the radius a hard-coded constant in braincoder; take a mask or radius argument.
3. **σ_T_dyn is tied to σ_dyn** inside the forward transform; its raw slot gets zero gradient. Either drop the slot
   in the port or keep it and document it. If you drop it, the parameter vectors change and the comparison with
   saved fits needs a column map.
4. **Parameter order matters** for loading saved fits: `[x, y, sd, baseline, amplitude, srf_amplitude, srf_size,
   sigma_AF, g_HP, g_LP, sigma_dyn, g_HP_dyn, g_LP_dyn, g_T_dyn, sigma_T_dyn, item_scale]`. Keep a loader or a
   label-based mapping.
5. **Precision.** The model runs in float32; the SSE, JᵀJ, Schur complement and solves are float64. Summing the SSE
   in float32 gave gradient noise large enough to stop L-BFGS early. Keep this split under Keras 3 (cast predictions
   to float64 before the residual).
6. **Memory.** The dynamic terms make the field time-varying, so the naive forward builds a `(V, T, G)` buffer. A
   separable Gx·Gy factorisation of isotropic Gaussians on a rectangular grid cut VRAM ~50× and time ~13× for the
   dynamic models (exact to 1e-14). Worth using; the per-ROI fits run on CPU.
7. **HRF.** The paper fixes the HRF per subject × area from a separate joint pRF + HRF fit (`roi_hrf.py`, SPM HRF,
   canonical delay 4.5 / dispersion 0.75 as fallback). The `_items` class does not support a flexible HRF tail; that
   is fine, but say so.
8. **Voxel-level `sd_min`.** Mean-fit templates used `sd_min = 0.2` at model level. Keep `sd_min` a model attribute
   (as on main), not a reader-side clamp.
9. **Starts matter.** Each LM fit is run from two starts and the lower training SSE kept (cold: the clamped fit
   projected onto zero-sum, g_HP = ¾·dS, g_LP = −¼·dS; warm: the parent with the new term at 0). A start at the
   parent's optimum must reproduce the parent's prediction exactly; the ladder checks this. The port should keep
   that property (new term at its null value ⇒ identical prediction).

## 4. The fitter

`joint_lm.lm_fit` is damped Gauss–Newton over ~7 × V per-voxel parameters + ~5–10 shared ones, with the voxel blocks
eliminated by the Schur complement (bundle adjustment). Jacobians by forward-mode JVPs (one per per-voxel column, all
voxels at once, plus one per shared term). Steps accepted only if SSE drops; Marquardt damping `D = diag(JᵀJ)`.
Bounds on shared terms by projection (a term at a bound whose step points outward is held). Also
`schur_hessian` (λ = 0) for Laplace covariances of the shared terms, used for the identifiability notes.

In braincoder this is a new fitter next to `ParameterFitter` (e.g. `JointLMFitter`), generic over any model with
per-voxel + shared parameters, not specific to attention fields. Under Keras 3 the JVPs need a backend-specific path
(`jax.jvp`, `tf.autodiff.ForwardAccumulator`, `torch.func.jvp`); the current code uses the TF accumulator. Defaults
in the paper runs: `max_iter = 300`, `ftol = 1e-10`, `lam0 = 1e-3`.

## 5. Acceptance test: reproduce the paper's fits

Before calling the port done, load saved fits and compare predictions, not just parameters:
- Fits: `/shares/zne.uzh/gdehol/ds-retsupp/derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm`
  (models `m5_TDS_dS_zs`, `m7_full_zs`, `m4_TDS_zs`; 241 subject × area cells each). The view root `..._zs_lm_ref`
  maps `m7_full` → `m5_TDS_dS_zs`.
- For a few cells (e.g. one good early-visual cell and one TO cell, where the item scale sits at its floor), build the
  data with retsupp's `build_data_and_paradigm`, predict with the old class and with the port from the same
  parameters: max |Δpred| at float32 tolerance.
- Then refit one cell with the port's LM from the saved start: training SSE equal or lower, dS within ~1e-3.
- Unit tests in braincoder on synthetic data: sign (trap 1), zero-sum invariance (adding c to every sustained gain
  ≈ rescaling pRFs when the pRFs are free), null-term reproduction (trap 9), separable vs dense forward.

Do not overwrite or rewrite any existing derivative; any refit goes to a new folder name.

## 6. Scope: what not to port

- Archived conditionwise-shift and Klein-shift models (`archive/conditionwise_shifts/`, the Klein classes in
  `local_models.py`): not used by the paper.
- The many exploratory variants in `local_models.py` (runPosition, sharedDynGain, allSharedSigma, oversampled,
  Gaussian drive/shift A/B/C, target/LP splits, DN baseline fields, centre-surround κ): leave in retsupp unless the
  paper's supplement needs one. The supplement uses `m7_full_zs` (dD free) and the Gaussian/DN backbones for
  robustness.
- Paper-specific glue (ROI loops, folds, ladder bookkeeping) stays in retsupp; braincoder gets the model, the
  zero-sum/tied parametrisation and the LM fitter.

## 7. After the port

- retsupp keeps working on the old submodule until the port reproduces the fits; then switch `libs/braincoder` to a
  release tag and update `fit_cv_ladder_nested.py` imports. The submodule must keep both the attention-field classes
  and the R² mixture code (an earlier bump lost the AF classes and every AF job failed on import).
- The public braincoder repo and its skill must not contain unpublished retsupp results, numbers or private paths.
