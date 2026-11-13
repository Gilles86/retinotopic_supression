# Joint optimisation of gain fields and pRFs (2026-11-13)

Question: the alternating polish does not reach the joint optimum and the sustained LP gain (g_LP) depends on the
start. Can gain fields and pRFs be optimised jointly, and is g_LP identified?

Canaries: sub-03 V1 (421 voxels), TO (123), IPS (120); m4_TDS and m7_full; the voxels of the reference root
`af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`; DoG, bounded shared terms, per-area HRF
(`roi_hrf_dog_final2.tsv`), all runs. Starts: **warm** = reference polished fit (clamped + 2 alternating rounds);
**cold** = clamped fit (pRFs frozen at m0, gains fitted on top). Gains are fractions (−0.62 = −62 %). σ² below is
the residual variance SSE/N (white noise; fMRI noise is autocorrelated, so Δχ² values overstate the evidence by the
autocorrelation factor, but not by orders of magnitude).

## Short answer

1. **The optimiser was the bottleneck: the polished fits were far from the joint optimum.** A joint
   Levenberg–Marquardt fit with the Schur complement lowers training SSE from the reference fit by 0.7–0.9 % (IPS),
   1.7–2.1 % (TO) and 0.08–0.18 % (V1), i.e. by ~3 000–8 000 σ² — 10–100× more than six further alternating rounds in
   the convergence run (−0.08 % IPS, −0.18 % TO, −0.008 % V1, still falling 1–4e-4 per round). From warm and cold
   starts it now ends at the same point (SSE within 0.01 %; same g_LP, σ_AF, item scale, dS; table below).
2. **g_LP is not flat — it is used for something else.** Freed together with the pRFs, the common sustained gain
   runs to the edge of its box: g_LP → −10 in IPS and TO, where `max(1 + gA, 0)` turns the field into a mask that
   zeroes all drive within ~2° of the four ring positions, with pRF amplitudes ×3–30. With the gains bounded at −1 (the
   field may not cut drive below zero) the joint optimum is g_LP = −0.83…−0.88 in IPS (σ_AF 2.4°), −1.00 (the bound)
   in TO and −0.87…−0.94 in V1 (m4_TDS), with median pRF amplitudes ×10–50 (IPS 36 → 520–780, TO 64 → 1 900). The
   earlier −17 %, −35 %, −61 % were stops on the way there, which is why they depended on the start.
3. **Why: a condition-invariant multiplicative field is a reshaping of every pRF.** The tied part of the sustained
   field, 1 + g_LP·Σ_ℓ A_ℓ(x), is the same in every run, so rf_v(x)·(1 + g_LP·ΣA(x)) is just another receptive field.
   With unrestricted pRFs g_LP is exactly unidentifiable; with DoG pRFs it is "identified" only through what a DoG
   cannot express. At the joint optimum the field suppresses the drive almost everywhere inside the mapped disc
   except far from the four diagonal ring positions (mod ≈ 0.1 in the fovea and ≈ 0.05 at the ring centres for
   g_LP = −0.83, σ_AF = 2.5°), and a 15–50× amplitude restores the response: the data use g_LP, σ_AF, the item scale and the
   amplitudes as extra pRF-shape terms. Held-out runs do not separate this from attention: shape terms generalise across
   runs just as well (the held-out SSE of each run3 fold also falls by ~900–1 500 σ² from g_LP = 0 to the joint
   optimum). Its value says how wrong the DoG is around the ring, not how attention is distributed.
4. **dS = g_HP − g_LP is identified and stable**, because HP moves between runs: it is the only part of the
   sustained field that differs across conditions. Profiled Hessian at the reference fit (IPS m7_full): the flattest
   direction is exactly g_HP + g_LP (smallest eigenvalue of the correlation-scaled Hessian 8e-4, condition number
   3e3; corr(g_HP, g_LP) = +0.998), SE(dS) = 0.008 vs SE(g_LP) = 0.14. Under the −1 bound dS = 0.012 (warm) / 0.017
   (cold) in IPS and 0.0096 / 0.0096 in TO, vs 0.023 / 0.033 (IPS) in the reference fits — the same within ~1–2 SE.
   The dS profiles have a sharp interior minimum near 0–0.03 in all splits (IPS all runs: +1 400 SSE ≈ 600 σ² at
   dS = 0.1), and the held-out minimum sits at the same place.

So: no optimiser can make g_LP an attention parameter; the model needs a constraint. The recommendation is at the end.

## Profiles (`notes/figures/joint_optimisation_2026-11-13/`)

g_LP fixed on a grid (−10 … +0.2), everything else (pRFs, other shared terms) refitted by LM (150 iterations per
point, warm-started along the grid from the free optimum outward), on all runs (training SSE) and on the three run3
folds (training and held-out SSE); dS likewise with the gains bounded at −1 (m7_full). `profile_*.pdf` per area ×
model × term; data `notes/data/joint_optimisation_2026-11-13/*profile*.tsv`.

What they show:

- **IPS, TO (g_LP):** training SSE is lowest in two places — the mask regime (g_LP ≤ −2) and a high-amplitude basin at
  g_LP ≈ −0.8…−0.9 — and jumps by 4 000–9 000 (≈ 2 000–4 000 σ²) between −0.8 and −0.7, where the warm-started fit
  drops back into the ordinary-amplitude basin (median amplitude 1 000 → 300 → 130). Above −0.7 SSE rises smoothly
  to g_LP = 0. The landscape is non-convex with several basins linked by the field ↔ amplitude trade-off; a 150-step
  profile is an upper envelope (the free fit at the same g_LP was sometimes lower).
- **V1 (m4_TDS):** one smooth interior minimum at g_LP ≈ −0.8…−1.0 in all splits (training and held-out); g_LP = 0
  costs ~4 800 SSE (≈ 1 100 σ²) in training and ~3 000 (≈ 900 σ²) held out per fold. Same amplitude inflation
  (×1.5–2 here) and the item scale falls.
- **dS (m7_full, gains ≥ −1):** sharp minimum at 0.00–0.03 in IPS and TO, training and held-out; the held-out
  minimum is at dS = 0 (IPS fold 2) or −0.1…0 (TO fold 2).

V1 m7_full is a warning about multiple optima: from the warm start LM ends in a positive-gain basin (g_LP = +2.6,
σ_AF = 4.3°) whose SSE (4 363 730) is *worse* than the nested m4_TDS optimum (4 359 489, g = −0.87). A joint fit has to
be started from the nested model's joint optimum as well, as the clamped chain does.

## The optimiser (`retsupp/modeling/joint_lm.py`; ladder `--polish-method lm`)

Levenberg–Marquardt over all free parameters (7 per voxel + 6–8 shared), with the block structure of bundle
adjustment: JᵀJ is block-arrowhead (one 7×7 block per voxel, voxel–shared couplings W_v, one dense shared block); each
damped step eliminates the voxel blocks with the Schur complement, solves a q×q system for the shared terms and
back-substitutes per voxel. Jacobians by forward-mode autodiff of the ladder's own model class (one pass per per-voxel
column for all voxels at once, one per shared term). Model in float32 (unchanged); residuals, JᵀJ, Schur complement and
solves in float64. Steps are kept only if training SSE drops (Marquardt damping, λ adapted), so the fit never ends above
its start. Shared terms boxed by projection (SHARED_BOUNDS). Two safeguards were needed on real data: per-voxel raw
floors at −15 for sd, surround amplitude and surround width (below that the surround's derivatives underflow to NaN —
the first version crashed on two IPS voxels whose surround width had gone to 1e-31), and per-voxel steps clipped to
max(1, 0.25·|raw|) (a nearly flat surround direction otherwise jumps to raw −100). Checked on simulated data (recovers
the shared terms) and against central differences on the IPS data (Jacobian error ≤ 4e-4 relative).

Diagnostics (`retsupp/modeling/joint_identifiability.py`): `dump` (data via the ladder's `build_data_and_paradigm`),
`fit` (LM from warm and cold), `hessian` (profiled Gauss–Newton Hessian, i.e. Schur complement, at the starts and the
LM ends: eigenvalues/vectors, correlations, SEs), `profile`, `check` (Jacobian vs differences); SLURM wrapper
`retsupp/modeling/slurm_jobs/joint_identifiability.sh`; plots `retsupp/visualize/plot_joint_identifiability.py`.
Outputs on the cluster under `derivatives/joint_optimisation_2026-11-13/`.

Speed (4 CPUs, standard partition, peak RSS 3.6 GB for IPS): ~2.5 s per iteration for ~120 voxels, ~6.6 s for V1
(421 voxels): ≈ 0.018 s per voxel per iteration. Most of the SSE drop comes in the first 100–300 iterations; after
that it creeps (relative decrease 2–8e-8 per iteration after 600–1 000 iterations, ≈ 0.01–0.03 σ² per step, along
nearly flat per-voxel DoG directions such as amplitude ↔ surround amplitude). Not tried: variable projection (exact
per-voxel inner solves); with warm and cold starts already agreeing it would only shorten the tail. Ladder test: sub-03
V1 m7_full, `--polish-method lm --warm-polish-root <reference>`, 300 iterations in 35 min,
`..._polished_lmtest/m7_full/sub-03/`.

## Warm vs cold (LM end points; 1 000 iterations for g ≥ −10, 600 for g ≥ −1)

| area | model | gains | start | SSE start | SSE end | g_LP | dS | σ_AF | item scale | median amplitude |
|---|---|---|---|---|---|---|---|---|---|---|
| IPS | m7_full | ≥ −10 | warm | 858 242 | 850 397 | −10 | 3.09 | 0.93 | 1.63 | 108 |
| IPS | m7_full | ≥ −10 | cold | 859 147 | 850 500 | −10 | 3.17 | 0.93 | 1.54 | 102 |
| IPS | m7_full | ≥ −1 | warm | 858 242 | 851 872 | −0.832 | 0.012 | 2.48 | 1.27 | 780 |
| IPS | m7_full | ≥ −1 | cold | 859 147 | 851 880 | −0.861 | 0.017 | 2.42 | 0.95 | 604 |
| IPS | m4_TDS | ≥ −1 | warm | 858 820 | 852 632 | −0.875 | — | 2.38 | 1.59 | 522 |
| IPS | m4_TDS | ≥ −1 | cold | 859 563 | 852 596 | −0.866 | — | 2.40 | 1.83 | 532 |
| TO | m4_TDS | ≥ −10 | warm | 854 306 | 836 253 | −10 | — | 0.91 | 2.04 | 1 080 |
| TO | m4_TDS | ≥ −10 | cold | 855 220 | 836 338 | −10 | — | 0.91 | 1.88 | 1 164 |
| TO | m7_full | ≥ −1 | warm | 854 272 | 839 873 | −1.000 | 0.0096 | 2.13 | 2.41 | 1 927 |
| TO | m7_full | ≥ −1 | cold | 855 195 | 839 892 | −1.000 | 0.0096 | 2.13 | 2.41 | 1 930 |
| TO | m4_TDS | ≥ −1 | warm | 854 306 | 840 271 | −1.000 | — | 2.13 | 2.28 | 1 918 |
| TO | m4_TDS | ≥ −1 | cold | 855 220 | 840 279 | −1.000 | — | 2.13 | 2.29 | 1 908 |
| V1 | m4_TDS | ≥ −10 | warm | 4 367 151 | 4 359 489 | −0.868 | — | 2.06 | 0.55 | 516 |
| V1 | m4_TDS | ≥ −1 | warm | 4 367 151 | 4 359 784 | −0.925 | — | 1.80 | 0.62 | 378 |
| V1 | m7_full | ≥ −10 | warm | 4 367 153 | 4 363 730 | +2.64 | −0.36 | 4.28 | 0.05 | 56 |

Reference (warm start) values for comparison: IPS m7 g_LP −0.59, dS 0.023, σ_AF 1.22, item scale 29, amplitude 36;
cold start g_LP −0.33, dS 0.033. V1 cold-start fits were still running at the time of writing.

## Cost of rerunning all 241 cells × 2 models

45 405 voxels over the 241 cells (median 131, max 500). At 0.018 s per voxel per iteration and 600 iterations: ≈ 136
task-hours per model on 4 CPUs (≈ 545 CPU-h), ≈ 1 100 CPU-h for both models, plus data loading (~5 min per subject ×
area task). Longest task (500 voxels, both models) ≈ 3 h. At the cluster's prices (4 vCPU + 16 GB) that is ~3 CHF;
wall time ~3–4 h with ~100 tasks running at once. Rerunning is cheap — but it is not worth doing with the model as it
is (below).

## Recommendation

1. **Do not report g_LP or g_HP (the level of the sustained field) from fits in which the pRFs are free.** The level of the condition-invariant field is a pRF-shape term: its joint optimum sits at
   complete suppression with ×10–50 amplitudes, and the values in the current polished root are arbitrary points along
   the optimiser's path. This also applies to the current reference (`..._polished`) Fig. 5 / key-number entries that
   take g_LP or g_HP from the polished m7_full. The item scale moves with it (29 → 1–2 in IPS) and the transient terms
   shift too (IPS g_T_dyn 0.11 → 0.02–0.03), so the item scale and the T, D gains of the polished fits need the same
   re-check once the sustained field is reparametrised.
2. **Make the model identified by removing the condition-invariant part of the sustained field**: replace
   1 + g_HP·A_HP + g_LP·Σ_{LP}A by 1 + dS·(A_HP − ¼·Σ_ℓ A_ℓ) (zero-sum over the four positions, so the field averages out
   over HP conditions), or equivalently fix g_LP = 0 and keep dS. Everything condition-invariant then belongs to the
   pRF (and to the item scale), and dS is the only sustained term. dS's estimate barely moves under all of the above
   (0.01–0.03 in IPS whatever g_LP does; sharp profile minimum), so this costs nothing for the paper's key test.
   Bound the gains so that the field stays positive (g ≥ −1, or a softplus parametrisation) so the `max(·, 0)` clamp
   can never produce the mask regime.
3. **If a sustained LP level is wanted descriptively**, report it only from the clamped fit (pRFs held at the field-free
   bar + item fit), defined as "the condition-invariant field needed on top of the field-free pRFs", and say that it is
   not separable from pRF shape — not as an attentional gain.
4. **Optimiser**: use `--polish-method lm` for any all-free refit (it reaches the joint optimum the alternating polish
   never got near, and warm/cold starts agree). Start every all-free model also from its nested parent's joint optimum
   (V1 m7_full otherwise lands in a worse basin than m4_TDS). Polish-tol "convergence" of the alternating method is not
   convergence (sub-03 V3 stopped after one round at 9e-6 while LM keeps finding 0.1–2 % elsewhere).
5. Before rerunning all 241 cells, refit the canaries with the reparametrised field (point 2) with LM, check dS from
   warm/cold/parent starts and the held-out fold, then run the full set (~3 CHF, ~4 h wall).
