# RT links with more power: results (13 Nov 2026)

Implements `notes/rt_power_plan_2026-11-13.md` (commit ff3f2c0), which was fixed before any of these analyses were run.
All variants are reported. Fits: the refitted fits (pRFs refitted together with the gain fields,
`..._reliable_roihrf_polished`), each area's signal voxels, n = 28, nine areas, 230 subject × area estimates per term.
Gains are per SD of (residual) log RT, × 100 (percentage points of gain). Posterior mean [95% HDI], P = posterior
probability of the sign shown. Following the plan, an effect is **credible** when its 95% HDI excludes 0.

## What was run

1. **Residual RT** (`fit_trialwise_af_gains --rt-modulator residual`). Within each participant, log RT of the correct
   trials (0.1–2 s, as before) was regressed (OLS) on:
   - distractor condition (HP / LP / absent);
   - target location relative to HP (HP / LP);
   - target repeat and distractor repeat;
   - trial number within run (linear and quadratic);
   - run (one level per session × run, which includes session).

   The residual was z-scored within participant and replaces z(log RT) as the RT modulator of the trial-wise gains.
   Everything else is unchanged: BOLD-level re-estimation on the refitted m7, signal voxels, leave-one-run-out
   jackknife SEs. The design explains 7.5% of the log-RT variance in sub-03 (19 columns).
2. **Joint model**: target, distractor and global slopes in one hierarchical model. Area and participant effects are
   multivariate normal with LKJ(2) correlations. Group means and priors are as in the Fig. 7 meta-analysis.
3. **Measurement error**: latent slope λ ~ Normal(θ, τ), observed slope ~ Normal(λ, jackknife SE), with λ sampled
   explicitly.

   **Note:** integrating λ out gives exactly the current likelihood, Normal(θ, √(SE² + τ²)). Step 3 is therefore the
   same model written differently, and it cannot change the result beyond Monte Carlo error. The table confirms this.

Sampling: nutpie, 4 chains, 2000 draws after 2000 tuning, target_accept .99.

- The main analysis and check 5 did not converge well enough at 2000 draws (max R̂ 1.02, bulk ESS 285).
- They were rerun unchanged with 6000 draws and 6000 tuning steps. Those runs are the ones reported here.
- The 2000-draw run gave the same means: main analysis target +0.99, distractor −1.65 (P = .976), global −2.55
  (P = .977).

## Main table

| Analysis | Target gain | Distractor gain | Global gain | Div. | Max R̂ | Min bulk ESS | PPC: 95% intervals cover |
|---|---|---|---|---|---|---|---|
| Baseline (as fixed in the plan) | +0.9, P = .833 | −2.0 [−3.7, −0.2], P = .990 | −1.9, P = .928 | — | — | — | — |
| Baseline, rerun in this code | +0.9 [−0.9, +2.9], P = .843 | **−2.0 [−3.7, −0.2], P = .986** | −1.9 [−4.5, +0.5], P = .930 | 0 | 1.01 | 847 | 96–97% |
| Step 1 only: residual RT | +1.1 [−0.7, +2.9], P = .885 | **−1.6 [−3.3, −0.1], P = .979** | −2.5 [−5.1, +0.0], P = .975 | 1 | 1.01 | 903 | 95–98% |
| Step 2 only: joint model | +0.9 [−1.1, +2.9], P = .815 | **−2.0 [−3.7, −0.1], P = .987** | −2.0 [−4.5, +1.0], P = .926 | 0 | 1.01 | 782 | 95–97% |
| Step 3 only: measurement error | +0.9 [−0.9, +2.8], P = .835 | **−2.0 [−3.8, −0.4], P = .987** | −1.9 [−4.4, +0.9], P = .923 | 0 | 1.01 | 1105 | 96–97% |
| **Main (1 + 2 + 3)** | +1.0 [−0.8, +2.9], P = .863 | **−1.6 [−3.3, −0.04], P = .977** | **−2.6 [−5.1, −0.04], P = .976** | 0 | 1.00 | 1386 | 95–98% |

Bold: credible (HDI excludes 0). In every analysis the posterior predictive band holds the observed mean of every
area (9 of 9).

In plain words:
- Slower trials still come with a lower distractor gain under every variant. Removing the design effects from RT
  makes the effect slightly smaller (−1.6 instead of −2.0) without changing the conclusion.
- The global gain becomes credible in the main analysis (−2.6 [−5.1, −0.04]), but only just. The change comes from
  the residual RT (step 1; P = .975 on its own), not from the joint model or the measurement-error form.
- The target gain does not change with RT under any variant.

**Correlations in the joint model** (main analysis, posterior mean [95% HDI]):
- Participant effects, target–distractor: +.43 [−.05, +.86]. Participants whose target gain rises on slow trials tend
  to show the same for the distractor gain.
- All other pairs, participant or area: near 0 with wide intervals (area: nine areas cannot inform a correlation).

In plain words: the joint model barely borrows strength across the three terms, which is why step 2 alone changes
nothing.

**Groups** (main analysis):

| Term | Early (V1–V3) | Dorsal (V3AB, IPS) | Occipitotemporal |
|---|---|---|---|
| Target | +2.2 [−1.4, +5.9], P = .891 | −1.1 [−4.7, +2.7], P = .762 | +1.2 [−1.6, +4.1], P = .829 |
| Distractor | −2.1 [−4.8, +0.7], P = .939 | −2.5 [−5.3, +0.4], P = .962 | −0.8 [−3.0, +1.3], P = .790 |
| Global | −3.2 [−6.4, +0.1], P = .974 | −0.4 [−4.0, +3.7], P = .584 | −3.2 [−6.1, +0.02], P = .980 |

In plain words: no single group is credible on its own. The global effect comes from the early and occipitotemporal
areas, not from the dorsal ones.

## Robustness (reported, not the main result)

**4. Residual-RT quintiles.** Gains per quintile of residual RT, per subject × area (scheme rq5 of the trial-wise
fits), centred within subject × area. The hierarchical monotonic-trend model uses a shared monotonic shape (Dirichlet
steps) scaled by a slope with group, area and participant effects, with noise SD per area. β is the change from the
fastest to the slowest quintile.

| Term | β (fastest → slowest quintile) | Div. | Max R̂ | Min bulk ESS | PPC cover |
|---|---|---|---|---|---|
| Target | +3.5 [−19.7, +24.9], P = .634 | 0 | 1.01 | 833 | 95% |
| Distractor | +1.7 [−14.2, +16.7], P = .582 | 0 | 1.01 | 728 | 94% |
| Global | −6.8 [−19.6, +7.0], P = .833 | 0 | 1.00 | 1048 | 95% |

In plain words: binning gives no credible trend for any gain. The bin gains have no per-estimate SEs and are far
noisier, so this test has much less power. It does not confirm the slopes, but it does not contradict them either
(the global gain points the same way).

**5. Main analysis on the r > 0.2 selection (own voxels, areas with ≥ 10 voxels; 238 estimates; 6000 draws).**

| Term | Estimate | Div. | Max R̂ | Min bulk ESS | PPC cover |
|---|---|---|---|---|---|
| Target | **+1.5 [+0.00, +2.8], P = .977** | 0 | 1.00 | 2019 | 95% |
| Distractor | −1.3 [−2.7, +0.2], P = .961 | 0 | 1.00 | 2019 | 97% |
| Global | −2.3 [−4.7, +0.02], P = .975 | 0 | 1.00 | 2019 | 98% |

In plain words: with the looser voxel selection the distractor and global effects keep their direction but are not
credible (their HDIs just include 0). The target effect becomes just credible in the other direction (higher target
gain on slower trials; its HDI only just excludes 0). None of the three RT effects is stable against voxel
selection.

## Interpretation (rules fixed in the plan)

- **Distractor gain:** credible in the baseline and in the main analysis. The conclusion does not change (the
  estimate is a little smaller: −1.6 vs −2.0).
- **Global gain:** not credible in the baseline (P = .928), credible in the main analysis (−2.6 [−5.1, −0.04],
  P = .976). The plan's wording rule applies: report it as credible only under the new analysis, with the baseline
  alongside, and not as the only analysis. It is borderline (the HDI ends at −0.04), comes from the residual-RT step,
  is not credible with the r > 0.2 selection (P = .975), and shows no credible quintile trend (P = .833).
  - Suggested sentence: "With RT residualised on the design and the three gains in one model, the global gain is
    also lower on slower trials (−2.6% [−5.1, −0.04] per SD; with z(log RT) −1.9%, P = .93), but this does not
    survive a looser voxel selection."
- **Target gain:** not credible in the baseline or the main analysis.
- **Fig. 7:** the main analysis changes the conclusion for one term (global), so under the plan Fig. 7 gets a second
  version and both go to the supplement:
  - `notes/figures/paper_2026-11/fig7_compact_polished_n28_palA.*` (earlier, z(log RT));
  - `notes/figures/paper_2026-11/figS_fig7_residrt_joint_polished_n28_palA.*` (new; panel b from the main analysis,
    panel a unchanged, palette A).

  The figure follows the paper's convention, solid when P(sign) > .975. In the new panel b this marks the
  occipitotemporal global gain (P = .980), whose HDI just includes 0 (upper bound +0.0002). By the plan's HDI rule
  it is not credible; the legend gives the numbers.

## Files

- Trial-wise fits with the residual RT:
  - `derivatives/af_trialwise_gains_signal_se_residrt_polished_2026-11-13`: SLURM 8338581 (canary, sub-03), 8338858;
    ~15 min, ≤ 26 GB.
  - `derivatives/af_trialwise_gains_ownvox10_residrt_relnull_r0.2_polished_2026-11-13`: SLURM 8338859.
- Collected tables: `notes/data/trialwise_signal_se_residrt_polished_2026-11-13/`,
  `notes/data/trialwise_ownvox10_residrt_relnull_r0.2_polished_2026-11-13/`.
- Models: `retsupp/visualize/paper/rt_power_meta.py`. Outputs:
  - `notes/data/rt_power_2026-11-13/`: baseline, steps 1–3, quintiles; also the 2000-draw main runs, superseded.
  - `notes/data/rt_power_2026-11-13_main6000/`: main and check 5.
  - Combined table (overall, group and area rows): `notes/data/rt_power_results_2026-11-13.tsv`.
  - Each folder holds `summary.tsv`, `diagnostics.tsv` and `correlations.tsv`.
- Commands:

```bash
P=notes/data; R=$P/trialwise_signal_se_residrt_polished_2026-11-13/trialwise_af_signal_se_residrt_polished_2026-11-13
~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.rt_power_meta \
    --zlogrt $P/trialwise_signal_se_polished_2026-11-12/trialwise_af_signal_se_polished_2026-11-12_parametric.tsv \
    --resid ${R}_parametric.tsv --resid-bins ${R}_bins.tsv \
    --resid-r02 $P/trialwise_ownvox10_residrt_relnull_r0.2_polished_2026-11-13/trialwise_af_ownvox10_residrt_relnull_r0.2_polished_2026-11-13_parametric.tsv \
    --out $P/rt_power_2026-11-13
# main and check 5 with more draws
... --only "main (1 + 2 + 3)" "5: main on r > 0.2 (own voxels)" --draws 6000 --out $P/rt_power_2026-11-13_main6000
```
