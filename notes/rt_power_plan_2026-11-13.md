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
