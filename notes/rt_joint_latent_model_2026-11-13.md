# Joint latent-gain model of single-trial log RT (addendum 6 of `notes/rt_power_plan_2026-11-13.md`)

13 Nov 2026. Fixed in the plan before any result (commit 196e3e3); input choices recorded as a dated line in the plan
(commit 7cf5070) before any gain–RT relation was computed. Complements the main analysis (steps 1–5, run separately);
does not replace it.

## Inputs: per-trial gains with their sampling covariance

- **Estimates.** `retsupp/modeling/fit_trialwise_af_gains_trialse.py` (SLURM `retsupp/modeling/slurm_jobs/fit_trialwise_af_gains_trialse.sh`):
  same BOLD-level method and regressors as `fit_trialwise_af_gains.py` (refitted root
  `af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`, m7_full held fixed, HRF table
  `roi_hrf_dog_final2.tsv`, each area's signal voxels, nine areas), but per-trial gains by weighted OLS (no ridge, so
  no shrinkage toward the canonical gains). One target (T), distractor (D, distractor trials only) and global (G)
  gain per trial; the previous-distractor regressor (P) stays in the design as before.
- **Uncertainty.** Circular-shift placebo: the residual of the canonical model (all voxels of the area) is shifted in
  time within each run by a random offset of ≥ 20 volumes and projected through the same estimator; 200 shifts, same
  offsets in every area of a participant. The spread of the placebo estimates is each trial's 3 × 3 sampling
  covariance under the data's own spatial and temporal noise (noise shared across voxels included).
- **Pooling (main).** Per participant, each area's deviation from its canonical gain, averaged over the nine areas
  with fixed weights 1 / median SE² (participant × area × kind); the pooled covariance is the same weighted mean of the
  placebo estimates (`retsupp/modeling/collect_trialse.py`). Robustness: the same pooled within each pathway (early
  V1–V3; dorsal V3AB, IPS; occipitotemporal hV4, LO, TO, VO).
- Output: cluster `derivatives/af_trialwise_trialse_polished_2026-11-13/` (per area: `*_trialse.tsv`,
  `*_trialse_shifts.npz`); pooled table `notes/data/rt_joint_latent_2026-11-13/trialse_pooled.tsv` (+ `_area_summary.tsv`).
  27 tasks (sub-03 … sub-30), 4–7 min each, 1 CPU, MaxRSS 8–26 GB.

## Model (`retsupp/visualize/paper/rt_joint_latent_model.py`)

17,292 correct trials with an RT (as Fig. 1; 13,707 with a distractor), n = 28 (sub-01, sub-02 excluded), behaviour from
`notes/data/behaviour_trials_2026-11-11.tsv.gz` (merged on subject × session × run × trial; trial types checked
identical). Gains in units of each kind's SD over all included trials.

    latent gains   z_k = m_k + diag(τ_s) u_k,      u_k ~ N(0, R)                  R: 3 × 3 LKJ(2) correlation
                   m_kj = α_sj + C_k (γ_j + δ_sj)                                  C: distractor HP / LP, target at HP,
                                                                                    target repeat, distractor repeat, session
    observed       ĝ_k ~ N(z_k, Σ_k)                                               Σ_k: placebo covariance of trial k
    log RT         y_k ~ N(β0 + X_k β + a_s + b_s · u_k, σ_s)
                   X: distractor HP, LP (vs absent), target at HP, target repeat, distractor repeat,
                      trial number (z, and its square), run 2–6, session 2
                   b_s ~ MVN(b_μ, diag(σ_b) L L' diag(σ_b))                        correlated participant slopes, LKJ(2)

- u_k is the trial's latent deviation from its condition mean in units of the participant's latent SD, so **b_μ is the
  group change in log RT per SD of latent gain** (holding the other two gains fixed). On trials without a distractor
  there is no distractor-gain term.
- Priors: β0 ~ N(log 0.85, 0.3); β ~ N(0, 0.2); b_μ ~ N(0, 0.05); σ_b ~ HalfNormal(0.03); participant intercepts,
  log σ_s, α, δ and log τ hierarchical (HalfNormal SDs), all non-centred.
- The latent gains are integrated out analytically: given the parameters, (ĝ_k, y_k) is jointly Gaussian with
  Cov(ĝ) = diag(τ) R diag(τ) + Σ_k, Cov(ĝ, y) = diag(τ) R b_s, Var(y) = b_s' R b_s + σ_s². Same model, ~52k fewer
  sampled dimensions, no latent-level funnel.
- PyMC 5.28, NUTS via NumPyro 0.20 (JAX, CPU), 4 chains.

### Bambi cross-check (observed gains, no measurement error)

    log_rt ~ dist_cond + target_cond + target_repeat + dist_repeat + trial + trial2 + C(run) + C(session)
             + g_target + g_dist + g_global + (1 + g_target + g_dist + g_global | subject)

g_* are the observed pooled gains, per participant residualised on the same condition design as the latent means and
divided by their SD (g_dist = 0 on trials without a distractor); dist_cond reference = absent; target_cond: target at
the HP location or not (all trials; Bambi's reference level is HP, so `target_cond[LP]` = −(target at HP)). Bambi
0.17.2, NumPyro. The PyMC model with Σ_k set to 10⁻⁴ I (SE 0.01 SD) is the same regression and should reproduce it.

## Checks

**Input calibration (gains only, before any RT relation).** Within every area the per-trial estimate divided by its
placebo SE has SD 0.96–1.06 (`trialse_pooled_area_summary.tsv`): the placebo SEs account for essentially all of the
trial-to-trial spread. Placebo SEs are ~3× the independent-voxel SEs (noise shared across voxels). So single-trial
target and distractor gains carry almost no detectable trial-to-trial variation; the global gain carries a little.
The model agrees: latent reliability τ² / (τ² + median SE²) is 0.03 (target), 0.02 (distractor), 0.15 (global).

**Convergence (main).** 4 chains × 1000 draws (1000 tune), target_accept 0.9: 0 divergences, max R̂ 1.01, min bulk ESS
379 (the RT intercept; every group slope, condition effect and latent-gain parameter ≥ 700), min tail ESS 789. 51 min on 4 CPUs.
Tree depth was saturated in an early version with a batched MvNormal likelihood (8.7 ms/gradient); the likelihood is
now written out elementwise (identical log density to machine precision, 2.1 ms/gradient).

**Posterior predictive checks** (200 draws, data simulated from the model incl. latent gains and measurement noise;
`main_ppc_conditions.tsv`, `main_ppc_fig1_contrasts.tsv`):
- Fig. 1 contrasts (ms, mean over participants of within-participant condition differences), observed vs predicted
  mean [95% band]: HP − LP distractor −26.6 vs −26.0 [−33.5, −17.7]; LP − absent +40.0 vs +39.2 [+29.7, +48.7];
  HP − absent +13.4 vs +13.3 [+3.9, +23.1]; target at HP − not (distractor trials) +27.3 vs +28.2 [+19.5, +37.1];
  target repeat −37.2 vs −34.9 [−44.2, −26.7]; distractor repeat (distractor trials) −24.5 vs −23.6 [−32.3, −15.2].
  All inside. (Fig. 1 reports medians over participants, −25 / +43 ms; the means are the comparable statistic here.)
- Participant × distractor-condition mean RT: 97.6% of 84 cells inside the 95% band.
- Per condition cell (5 cells): mean RT and SD of log RT inside in all; the RT median (1/5 inside) and 90th percentile
  (1/5) are not: observed RTs are more right-skewed than log-normal (median 8–19 ms lower, 90th percentile 16–50 ms
  higher than predicted; 10th percentile inside in 4/5). The Gaussian log-RT likelihood misses the shape of the
  distribution, not its location.
- Observed gains (13 cell × gain combinations): SD inside in 13/13, mean inside in 11/13 (target-gain mean in two
  cells, offsets ≤ 0.06 SD).

**Check against Bambi.** With Σ_k → 10⁻⁴ I the PyMC model gives group slopes +0.000, +0.002, −0.002 log RT per SD
(target, distractor, global) vs Bambi +0.001, +0.001, −0.002, and the same condition effects (within 0.002 log units); its own fit
has max R̂ 1.03, min bulk ESS 129 (latent SDs, which are irrelevant to the slopes there).

**Parameter recovery** (`recovery_recovery.tsv`): data simulated from the fitted model (posterior-mean parameters,
the fitted latent SDs, each trial's own Σ_k, participant slope deviations as fitted) with known group slopes, refitted
with the same pipeline; one data set each.
- Known slopes (+0.02, −0.02, −0.02 log RT per SD for target, distractor, global): recovered +0.026 [−0.009, +0.057],
  −0.032 [−0.068, −0.003], **−0.020 [−0.036, −0.005]**; latent SDs, condition effects and R covered (trial linear and
  quadratic slightly off, −0.004 vs −0.001). Refit: 0 divergences, max R̂ 1.02, min bulk ESS 338.
- Null slopes (0, 0, 0): +0.022 [−0.027, +0.066], −0.026 [−0.060, +0.006], −0.014 [−0.043, +0.015], all covering 0
  (2 divergences, max R̂ 1.01, min ESS 408).
- Participant slopes: r(true, estimated) 0.72–0.80 for the global gain, 0.10–0.47 for target and distractor.
- Reading: the global-gain slope is recovered; the target and distractor slopes per SD of latent gain are only weakly
  determined (their latent SD is ~0.03 of the observed spread), so their intervals are wide and a single null data set
  gives point estimates of ±0.02–0.03.

## Results

Group slopes on log RT per SD of the trial's latent gain (deviation from its condition mean, other gains held fixed);
ms at the mean RT over participants (0.88 s). P = posterior probability of the sign shown. "Gain per SD of log RT" is
the same relation in the direction of Fig. 7b (latent gain change per SD of log RT beyond the design, raw gain units
× 100: Cov(z, y) / SD(y)), which is what the data pin down even when the latent SD is small. **Direction:** Fig. 7b
regresses gain on RT, this model regresses RT on gain. A negative value means the same thing in both: higher gain goes
with faster responses (lower gain on slower trials).

| Gain | Main: log RT per SD of latent gain [95% HDI], ms, P | Main: gain × 100 per SD of log RT [95% HDI], P | Bambi, observed gains: log RT per SD [95% HDI], P | Fig. 7b baseline: gain × 100 per SD of log RT |
|---|---|---|---|---|
| Target | +0.015 [−0.009, +0.042], +13 ms [−8, +38], P(+) = .878 | +0.45 [−0.83, +1.73], P(+) = .759 | +0.001 [−0.003, +0.005], P(+) = .738 | +0.9 [−0.9, +2.9], P = .833 |
| Distractor | −0.009 [−0.036, +0.017], −8 ms [−31, +15], P(−) = .763 | −0.19 [−1.46, +1.15], P(−) = .608 | +0.001 [−0.003, +0.005], P(+) = .739 | −2.0 [−3.7, −0.2], P = .990 |
| Global | **−0.017 [−0.034, −0.001], −15 ms [−30, −1], P(−) = .985** | **−2.40 [−3.70, −1.01], P(−) > .999** | −0.002 [−0.006, +0.002], P(−) = .837 | −1.9 [−4.7, +0.6], P = .928 |

Condition effects on log RT (main model, ms at the mean RT): HP − LP distractor −11.6 [−19.2, −3.3]; LP − absent
+34.9 [+26.9, +43.3]; HP − absent +22.8 [+15.0, +30.8]; target at HP +14.8 [+6.3, +22.1]; target repeat −33.4
[−38.9, −28.0]; distractor repeat −10.7 [−17.4, −4.2]. These are adjusted for each other (an HP distractor trial
never has its target at HP, and HP distractors repeat more often), so they are smaller than the raw Fig. 1
differences; the model's predicted raw differences reproduce Fig. 1 (PPC above).

Robustness, pathway-level gains (each pathway its own fit, same model):

| Pathway | Converged | Target | Distractor | Global |
|---|---|---|---|---|
| Early (V1–V3) | 1 divergence, R̂ 1.01, min ESS 339 | +0.012 [−0.012, +0.036], P(+) = .84; gain/SD RT +1.2 [−1.3, +3.7] | −0.005 [−0.039, +0.030], P(−) = .61; +0.1 [−2.7, +2.7] | **−0.014 [−0.026, −0.002], P(−) = .984; −3.0 [−4.6, −1.3], P > .999** |
| Occipitotemporal (hV4, LO, TO, VO) | 0 divergences, R̂ 1.01, min ESS 237 | +0.024 [−0.004, +0.055], P(+) = .958; **+1.2 [+0.07, +2.3], P(+) = .987** | −0.001 [−0.024, +0.023]; +0.9 [−0.5, +2.3], P(+) = .90 | −0.012 [−0.031, +0.006], P(−) = .904; **−1.9 [−3.4, −0.4], P(−) = .994** |
| Dorsal (V3AB, IPS) | **no** (60 divergences, R̂ 4.6, ESS 4; two participants have no dorsal pool; not diagnosed further) | not interpretable | | |

Model-free check of the global-gain relation (not part of the plan, for orientation): per participant, the precision-
weighted regression of the pooled global gain on residual log RT (design as above) gives −2.9 (gain × 100 per SD),
20/28 participants negative, Wilcoxon p = .019; target +1.6 (p = .41), distractor −0.4 (p = .28).

**In plain language:** in one model that treats every trial's gain estimate as the noisy measurement it is, trials
with a higher global gain (the whole stimulus drive scaled up) were answered faster: about 15 ms per SD of the latent
global gain, or −2.4% global gain per SD of log RT, credible (95% HDI excludes 0) and the same direction and size as
the non-credible baseline in Fig. 7b (−1.9%, P = .928). The target and distractor gains vary too little from trial
to trial, relative to their measurement noise (latent reliability ~0.02–0.03), for this model to say anything about
their link to RT; the distractor-gain effect of Fig. 7b (−2.0%, P = .990) is not reproduced here (−0.2%, P = .61) —
not a contradiction, the per-trial route is much less sensitive for it than the parametric model of Fig. 7b.

**Caveats.**
- Single-trial target and distractor gains are close to pure noise (z-SD ≈ 1.0 in every area); their per-SD slopes
  are weakly identified (recovery above), and their posteriors are wide.
- The placebo SEs include other trials' true gain variation, so they lean conservative; latent SDs are then a lower
  bound and per-SD slopes an upper bound in magnitude. The gain-per-SD-RT column does not depend on that.
- Measurement errors of neighbouring trials (overlapping BOLD windows) are correlated; the model treats trials as
  independent given their 3 × 3 within-trial covariance.
- The Gaussian log-RT likelihood misses the right skew of the RT distribution (PPC); location effects are fine.
- The pooled weights are 1 / median SE² per area, so the pooled gain is dominated by the areas with the most precise
  per-trial estimates; the baseline averages the nine areas equally.

## Files and jobs

- Code: `retsupp/modeling/fit_trialwise_af_gains_trialse.py`, `retsupp/modeling/slurm_jobs/fit_trialwise_af_gains_trialse.sh`,
  `retsupp/modeling/collect_trialse.py`, `retsupp/visualize/paper/rt_joint_latent_model.py`,
  `retsupp/visualize/paper/slurm_jobs/rt_joint_latent_model.sh`.
- Per-trial gains: SLURM 8339580 (sub-03 canary), 8339739 (sub-04 … sub-30); cluster clone `~/retsupp_rtlatent`
  (own clone of the branch so other agents' untracked files in `~/retsupp_plan` stay untouched).
- Model fits (4 CPUs each): main 8341020 (51 min), Σ → 0 check 8341021, pathways 8341022–8341024, recovery
  8342401/8342402, report 8342403, Bambi 8341039. Traces (`*.nc`) and simulated data (`*.npz`) on the cluster in
  `~/retsupp_rtlatent/notes/data/rt_joint_latent_2026-11-13/` (not committed); tables here in
  `notes/data/rt_joint_latent_2026-11-13/` (the 26 MB `trialse_pooled.tsv` is not committed either).
- Rerun the tables: `python retsupp/visualize/paper/rt_joint_latent_model.py report --gains <pooled tsv> --out-dir <dir> --tag main --traces main --bambi-traces bambi_observed --ppc`
  (bauer env), and `recovery --sims sim_known.npz sim_null.npz --traces rec_known rec_null`.
