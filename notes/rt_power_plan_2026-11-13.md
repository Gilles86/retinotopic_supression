# RT links: analysis plan to gain power (fixed 13 Nov 2026, before any of these analyses were run)

**Goal.** Increase the precision of the trial-wise gain–RT relations (Fig. 7b) without forking paths. Every variant
below is run and reported, whatever the outcome.

**Data.** Trial-wise gain estimates on the refitted fits: `af_trialwise_gains_signal_se_polished_2026-11-12`, signal
voxels, with leave-one-run-out jackknife SEs; n = 28, nine areas. The current result is the comparison baseline:
- distractor gain per SD of log RT −2.0% [−3.7, −0.2], P = .990;
- global gain −1.9%, P = .928;
- target gain +0.9%, P = .833.

## Main analysis (1 + 2 + 3 together)

**1. Residualise RT on the design.** Within each participant, regress log RT on:
- distractor condition (HP / LP / absent);
- target location relative to HP (HP / LP);
- target repeat and distractor repeat (vs switch);
- trial number within run (linear and quadratic);
- run and session.

Use correct trials only, as before, then z-score the residual within participant. The trial-wise gains are related to
this residual RT instead of z(log RT). The design factors are the same ones used in Fig. 1 and in the history terms
of Fig. 7a.

**2. One joint hierarchical model** for the three RT slopes (target, distractor, global gain):
- per participant × area, the three slope estimates are modelled jointly;
- area and participant random effects, correlated across the three terms (LKJ prior);
- group means as in the current meta-analysis.

**3. Measurement error in the likelihood.** Each per-participant × area slope estimate enters with its jackknife SE as
known observation noise. The estimate is distributed as the latent slope plus noise with that SE, instead of being
weighted by it.

**Reported for each term:**
- the posterior mean, 95% HDI and P(sign);
- the same with step 1 only, step 2 only and step 3 only, so the contribution of each change is visible;
- the baseline above;
- convergence (R̂, ESS, divergences) and posterior predictive coverage.

## Robustness (reported, not the main result)

**4. RT quintile bins.** Gains per residual-RT quintile per participant × area, then a hierarchical monotonic-trend
test.

**5. Voxel selection.** The main analysis on the r > 0.2 selection (own voxels), as a check only.

## Interpretation fixed in advance

- **Credible:** an effect counts as credible if its 95% HDI excludes 0, as before.
- **Wording:** an RT effect that becomes credible only under the new model is reported as such, with the baseline
  alongside. It is not presented as the only analysis.
- **Fig. 7:** is updated only if the main analysis changes the conclusion for a term; in that case both versions go in
  the supplement.

## Addendum, added at the author's request before any result of this plan was seen: joint latent model on log RT

**6. One Bayesian model of single-trial log RT.** This complements the main analysis; it does not replace it.

**Fixed effects** (correct trials only):
- distractor condition (HP / LP / absent);
- target location (HP / LP);
- target and distractor repeats;
- trial number (linear and quadratic);
- run and session.

**Neural predictors.** The trial's latent target, distractor and global gains, per area or pooled over areas as
defined below, each with:
- a participant-level slope;
- a group mean;
- a correlated random effect.

**Measurement model.** Each latent gain is observed through its BOLD estimate, with that estimate's standard error:
observed gain ~ Normal(latent gain, SE). The latent gains get a hierarchical prior per participant × area: their mean
and SD are estimated, and they may depend on the conditions, so the gains' own condition effects are modelled too.

**Area pooling (fixed now).**
- **Main version:** per participant, a precision-weighted mean of each gain over the nine areas, so one latent value per
  trial and gain.
- **Robustness version:** pathway-level gains (early, dorsal, occipitotemporal), if the model converges.

**Inputs.** Per-trial (or per-short-bin) gain estimates with SEs on the refitted fits, from the trial-wise pipeline.
If per-trial estimates are not available, they are produced with the same BOLD-level method; the binning is decided
before the RT relation is looked at.

**Reported.**
- For each gain: the group slope on log RT (ms per SD of gain) with 95% HDI and P(sign).
- The condition effects on log RT, which should reproduce Fig. 1.
- Convergence: R̂, ESS, divergences; a non-centred parameterisation where needed.
- Posterior predictive checks of the RT distribution per condition.
- A parameter-recovery check on data simulated from the fitted model.

**Interpretation.** As for the main analysis: credible if the 95% HDI excludes 0. Reported whatever the outcome, next to
the main analysis and the baseline.

**13 Nov 2026, addendum 6 inputs (decided before any gain–RT relation of this step was computed or looked at).**
No binning: one gain estimate per trial and kind (target, distractor, global), by weighted OLS on the per-trial
regressors of `fit_trialwise_af_gains` (no ridge), nine areas, signal voxels, refitted root. Sampling covariance per
trial (3 × 3, across kinds) from 200 circular-shift placebo projections of the canonical model's residual (same
offsets in every area). Area pooling: per participant, the deviation from each area's canonical gain, averaged over
the nine areas with fixed weights 1 / median SE² per participant × area × kind; its covariance from the same weighted
mean of the placebo estimates (so cross-area error correlation is kept). Code: `retsupp/modeling/fit_trialwise_af_gains_trialse.py`.
