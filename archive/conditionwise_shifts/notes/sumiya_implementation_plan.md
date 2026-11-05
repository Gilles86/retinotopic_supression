# Implementing Sumiya Sheikh Abdirashid's attention-field model in retsupp

Reference: *A Light of Our Own — Illuminating Interactions Between Attention and Visual Input* (VU Amsterdam, 2026), chapters 1–2. Promotor S.O. Dumoulin, copromotor T.H.J. Knapen.

## What chapter 2 says, in one paragraph

The classical attention-field (AF) model (Reynolds & Heeger 2009; Womelsdorf 2008; Klein 2014) describes the attention-driven population receptive field (AD-pRF) as the product of two Gaussians, the attention field A(μ_AF, σ_AF) and the stimulus drive S(μ_SD, σ_SD):

  R(μ_AD, σ_AD) = A(μ_AF, σ_AF) · S(μ_SD, σ_SD)

Because both factors are Gaussian, the product is Gaussian and the AD-pRF center has a closed form:

  μ_AD = (μ_AF·σ_SD² + μ_SD·σ_AF²) / (σ_AF² + σ_SD²)
  σ_AD² = (σ_AF² · σ_SD²) / (σ_AF² + σ_SD²)

Sumiya's chapter 2 contribution is to show that **adding a single additive offset** — either to the AF or to the SD-pRF — qualitatively changes the model's predictions:

- **AF model** (no offset): predicts *global* attentional attraction. Every PRF, regardless of distance from the attended locus, is pulled toward it. Amplitudes are irrelevant (Gaussians normalize out).
- **AF+ model** (offset on AF): R = [a_AF · A(μ_AF, σ_AF) + b_AF] · S(μ_SD, σ_SD). Predicts *local* attraction only — when AF and SD-pRF tails barely overlap (b_AF ≈ AF tail), the SD-pRF is multiplied by ~b_AF and is effectively unaffected. Amplitude of the AF matters; amplitude of the SD-pRF does not.
- **SD+ model** (offset on SD-pRF): R = A(μ_AF, σ_AF) · [a_SD · S(μ_SD, σ_SD) + b_SD]. Predicts the AD-pRF returns to the AF when AF and SD-pRF are far apart (i.e. PRFs migrate fully to the attended locus, which is biologically implausible). Amplitude of the SD-pRF matters; amplitude of the AF does not.

In chapter 2 a single Gaussian is then *fit* to each AD-pRF to recover an effective center/size estimate (since AF+ and SD+ are no longer Gaussian).

The empirical finding in our group's prior work (Abdirashid et al. 2025) — that attention-driven attraction is observed *only when the population RF overlaps with the attended locus* — is uniquely consistent with **AF+** and rules out AF and SD+. That makes AF+ the natural candidate for retsupp's history-driven suppression analysis (replacing "attention" with "HP-distractor" gives a suppression analog: receptive fields shift only if their pRF overlaps with the HP distractor location).

## How this maps onto retsupp's existing code

The current `retsupp/modeling/fit_attention_model.py` already implements **the no-offset AF model** for the conditionwise PRFs:

- `compute_net_shifts(prf_x, prf_y, prf_sd, current_condition, attention_sd, ratio)` performs a precision-weighted-mean update (4 attention Gaussians × PRF Gaussian).
- The `ratio` parameter scales the attention-field σ for the *current* HP location relative to the other three (so it implicitly captures the asymmetry between attended and unattended).
- Free parameters per ROI per subject: `attention_sd` (base) and `log_ratio`. Optimized with `scipy.optimize.minimize(L-BFGS-B)`.
- Output: per-voxel predicted dx/dy/shift; per-ROI fitted `(attention_sd, ratio)`; PDF of hexbin plots per condition.

This is **exactly the AF model from chapter 2, generalized to four attention sources.** It has no offsets, so by Sumiya's analysis it predicts global effects with no amplitude dependence.

## Implementation plan: add AF+ and SD+ variants

### Step 1 — refactor `fit_attention_model.py` to a model-agnostic core

Right now `compute_net_shifts` hard-codes the precision-weighted-mean shortcut, which is only valid for AF. For AF+ and SD+ that shortcut breaks (the product is no longer a single Gaussian). Restructure as:

1. `eval_response_1d(x_grid, mu_AF, sigma_AF, mu_SD, sigma_SD, model='AF', a_AF=1, b_AF=0, a_SD=1, b_SD=0)` returns the AD-pRF response on a 1-D grid.
2. `eval_response_2d(grid_x, grid_y, ...)` does the same in 2-D (since our distractors live on a 4° ring, not on a line).
3. `fit_gaussian_to_response(response, grid)` returns (μ_x, μ_y, σ) of a Gaussian fit to the response — Sumiya does this with a closed-form first/second-moment estimate ("center of mass" of the positive part); copy that.
4. `predict_ad_prf_center(prf_x, prf_y, prf_sd, condition, params, model)` wraps (1)+(3): builds 4 attention Gaussians from the HP locations, multiplies by SD-pRF, applies offsets per `model`, fits Gaussian, returns predicted (x, y).

This makes adding AF+/SD+ a one-line change at the dispatch site.

### Step 2 — fit AF, AF+, SD+ side-by-side

For each subject × ROI:

- AF (current): free params = `(attention_sd, log_ratio)`.
- AF+: free params = `(attention_sd, log_ratio, log_a_AF, log_b_AF)` — 4 params. `log_` to enforce positivity. The chapter 2 simulations explore `b_AF` in the range [0, 1].
- SD+: free params = `(attention_sd, log_ratio, log_a_SD, log_b_SD)`. Same shape.

Use the same loss as currently (sum of squared empirical-vs-predicted shifts in 2-D), `L-BFGS-B`, multi-start from a small random grid to escape local minima. Keep `log_ratio` bounded (current bounds `[-1000, 1000]` are absurd; tighten to e.g. `[-3, 3]` — `ratio` ∈ [0.05, 20]).

### Step 3 — model comparison

Per ROI per subject: AIC / BIC comparison across {AF, AF+, SD+}. Aggregate counts of best-fitting model by ROI (V1, V2, V3, V3AB, hV4, LO, TO). The chapter 2 prediction (and the prior empirical finding) is that **AF+ wins**, especially in early visual areas where pRFs are small and overlap with the HP-distractor only when nearby.

For retsupp specifically there is one twist: because HP-distractors are *suppressed*, the sign of the effect is inverted relative to attention. We expect either:
- **`ratio` < 1 in AF+ for the HP location** (attention σ is wider for HP → less precise → less attraction → effective repulsion), or
- A sign flip on `b_AF` (negative offset → repulsion). Decide upfront whether to allow `b_AF < 0` (Sumiya's chapter only considers ≥ 0). I'd allow it, but flag clearly in the writeup that this departs from her parameterization.

### Step 4 — visualization

Reuse the `create_hexbin_plots` PDF output. Add a row of "predicted under AF / AF+ / SD+" panels per condition so the qualitative behavior of the three models is visible side by side. The figure-2 schematic in Sumiya ch.2 is a great template.

### Step 5 — publish to the AD-pRF residual map

The empirical "attractor / repulsor" map is the Δ-position field per condition. After fitting AF+ to the conditionwise data, we can also evaluate the predicted AF+ response *as a map* on a fine grid, then visualize it as a 2-D vector field that the audience can compare to the empirical arrows from VSS-figure B1.

## File-level changes

- Edit `retsupp/modeling/fit_attention_model.py`:
  - Add `eval_response_1d`, `eval_response_2d`, `fit_gaussian_to_response` helpers.
  - Add `model` arg to `process_subject(...)` and `predict_ad_prf_center(...)`; default to `'AF'` for backwards compat.
  - Add CLI flag `--model {AF,AF+,SD+}` and a `--multistart N` flag.
  - Output dir: `derivatives/attention_model/model-{prf_model}/sub-XX/sub-XX_attmodel-{AF,AFplus,SDplus}_*.tsv` (avoid `+` in filenames).

- New SLURM submitter `retsupp/modeling/fit_attention_model.sh` to array over subjects × {AF, AF+, SD+}.

- New notebook (or extend `notebooks/fit_dynamic_model.ipynb`) for the model-comparison plot: AIC ΔAIC heatmap (subjects × ROI × model) and the per-ROI winner-takes-all bar chart.

## Open questions for Gilles

1. Should `b_AF` be allowed to go negative (i.e. interpret as suppression, departing from Sumiya's positive-offset interpretation), or do we keep the standard parameterization and flip the sign of the effect via the AF amplitude `ratio` instead?
2. Do we want to fit the four HP-condition responses *jointly* with shared parameters (current behavior) or per condition (one (σ, ratio) per HP location)? Joint fit is more robust; per-condition is more diagnostic if the four locations differ systematically (e.g. upper vs. lower visual field).
3. Sumiya rejects SD+ on biological-plausibility grounds in the discussion. Are we still required to fit it for completeness in the paper, or only AF and AF+?
