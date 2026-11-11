# Discussion checks, 2026-11-12 (n = 28, sub-03–30)

Script: `retsupp/visualize/paper/discussion_checks.py` (reuses `richter_followup.py` loaders). No fits.
Medians of within-subject differences with percentile-bootstrap 95% CIs over subjects (10 000 resamples), Wilcoxon
signed-rank; means ± SEM where stated.

## 1. Behaviour next to vs opposite HP, by history

Sources: `notes/data/discussion_behaviour_by_block_n28_2026-11-12.tsv`,
`notes/data/discussion_behaviour_trial_counts_n28_2026-11-12.tsv`, `notes/data/discussion_behaviour_anova_n28_2026-11-12.tsv`
(trials: `notes/data/behaviour_trials_2026-11-11.tsv.gz`; correct trials, RT in ms). Session 1 block 1 (s1b1) = the
first three runs, where no location had been HP yet.

| Contrast (RT, ms) | All blocks | Session 1 block 1 only | s1b1 − other three blocks |
|---|---|---|---|
| Distractor next − opposite | +37.8 [+24.2, +45.5], p < .001 | +34.4 [+5.9, +53.2], p = .025 | −6.5 [−26.6, +16.2], p = .58 |
| Distractor HP − next | −37.8 [−41.9, −30.6], p < .001 | −35.7 [−57.2, −8.2], p < .001 | +3.8 [−20.1, +26.4], p = .90 |
| Distractor HP − opposite | −3.8 [−15.9, +3.7], p = .34 | −4.3 [−43.6, +15.2], p = .57 | +8.7 [−33.5, +20.5], p = .83 |
| Target next − opposite | +10.4 [+0.2, +18.4], p = .005 | +16.9 [+2.5, +34.0], p = .081 | +11.3 [−11.2, +28.9], p = .32 |
| Target HP − next | +20.6 [+15.7, +29.7], p < .001 | +19.5 [−1.7, +50.3], p = .048 | −10.4 [−28.9, +13.9], p = .51 |
| Target HP − opposite | +30.2 [+22.8, +48.3], p < .001 | +31.9 [+17.1, +58.9], p = .004 | +1.5 [−30.5, +29.7], p = .92 |

- Correct trials per subject in s1b1 (median [min–max]): distractor at HP 77 [59–87], next 30 [20–36], opposite
  15.5 [11–18], absent 33 [23–36]; target (distractor trials) at HP 29.5, next 61.5, opposite 32. The other blocks are
  similar (per-block values in the counts table). The opposite cell has ~16 trials per subject per block.
- Interaction, repeated-measures ANOVA relation (HP / next / opposite) × session-block (4), n = 28: distractor
  relation F(2, 54) = 27.5, p < .001; session-block F(3, 81) = 4.86, p = .004 (practice); relation × session-block
  F(6, 162) = 0.27, p = .95. Target: relation F(2, 54) = 31.4, p < .001; relation × session-block F(6, 162) = 0.52, p = .79.
- Per session × block (all in the by-block table): distractor next − opposite +17 to +47 ms in every block (p ≤ .025);
  HP − opposite between −11 and +5 ms in every block (all p ≥ .45).

**Plainly:** where no carry-over is possible (session 1, block 1) the pattern is already there: a distractor next to HP
is the slowest (+34 ms vs opposite, p = .025), a distractor opposite HP is about as fast as at HP (−4 ms, p = .57),
and the target cost is graded HP > next > opposite. Nothing differs reliably between blocks (relation × block
p = .95 and .79; s1b1 vs the rest all p ≥ .32), though the s1b1 cells are small (~16 opposite trials per subject),
so the s1b1 estimates are wide.

## 2. Model vs measured HP reduction at the item locations (Fig. 3d, e)

Sources: `notes/data/discussion_model_vs_measured_n28_2026-11-12.tsv`, `notes/data/discussion_model_vs_measured_corr_n28_2026-11-12.tsv`
(measured: `notes/data/event_deconv_relative_final2_2026-11`; predicted: noise-free m7_full predictions through the
same pipeline, `notes/data/kappa_neighbour_2026-11/event_deconv_m7`). % of the subject's mean response, mean ± SEM
[bootstrap 95% CI of the mean], Wilcoxon.

| Early areas | Measured | m7 predicted | Measured − predicted | Between-subject ρ (measured, predicted) |
|---|---|---|---|---|
| HP − LP | −11.7 ± 4.0 [−19.4, −3.9], p = .009 | −5.4 ± 3.7 [−12.2, +2.2], p = .036 | −6.3 ± 4.7 [−15.0, +3.1], p = .095 | .20, p = .31 |
| HP − opposite | −21.6 ± 6.8 [−34.9, −8.9], p = .005 | −6.3 ± 3.8 [−13.2, +1.4], p = .023 | −15.4 ± 7.1 [−28.7, −1.3], p = .026 | .20, p = .32 |
| Next − opposite | −14.9 ± 6.0 [−26.8, −3.8], p = .022 | −1.3 ± 0.8 [−2.8, +0.1], p = .16 | −13.6 ± 6.2 [−25.9, −2.4], p = .040 | −.14, p = .47 |
| HP − next | −6.7 ± 3.7 [−13.7, +0.3], p = .095 | −4.9 ± 3.7 [−11.6, +2.6], p = .045 | −1.8 ± 4.5 [−10.5, +7.0], p = .70 | .16, p = .43 |

All areas: HP − LP measured −9.7 ± 3.1 vs predicted −4.1 ± 3.0, gap −5.7 ± 3.6 (p = .081); HP − opposite gap −10.8 ± 5.6
(p = .040); next − opposite gap −7.8 ± 4.7 (p = .081); ρ for HP − LP = .22 (p = .26).

**Plainly:** the model's HP − LP (−5.4%) lies inside the 95% CI of the measured effect, and the gap to the measured
−11.7% is not significant (p = .095): the model's reduction at HP relative to HP's own neighbours (HP − next) matches
the data. What the model does not reproduce is the gradient over the LP locations: the measured lower response next to
HP than opposite (gap p = .040), which also makes the HP − opposite gap significant (p = .026). The model does not
capture which participants show the effect (ρ ≈ .2, n.s.), but with per-subject noise of this size that test has
little power.
