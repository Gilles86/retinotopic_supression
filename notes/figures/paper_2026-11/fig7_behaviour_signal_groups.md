**Figure 7 (prototype). The history effects of Fig. 1 in the gains, and gain and behaviour trial by trial.** a, History (Fig. 1e, f; the location effects of Fig. 1c, d are in Fig. 5d): target gain on target-repeat minus other trials, and distractor gain on distractor-repeat minus distractor-switch trials (the difference taken within HP and within LP and averaged, as for the response times in Fig. 1f); the display icons are those of Fig. 1e, f (four possible locations; filled, the item on this trial; hollow, on the previous trial; no HP ring, since HP and LP trials are pooled), each series the gain in the first condition minus that in the second. Points and bars are the median over subjects (n = 30) and its bootstrapped 95% CI; Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the three ROI groups within each series; estimates with q < .05 are drawn solid, the others pale (key at the top); stars: * q < .05, ** q < .01, *** q < .001; all values below. b, Within subjects, trial by trial: Bayesian hierarchical linear model of log RT (z within subject) per ROI group on the per-trial target gain (gain field at the target), distractor gain (gain field at the distractor) on HP and on LP trials, and global gain (over the whole display), each z-scored within run and trial type, with trial number, target repeat, distractor repeat and target-at-previous-distractor as covariates, and random intercepts and gain slopes per subject (bambi, 4 × 1000 draws, no divergences, R̂ ≤ 1.01); the icons show each gain field (illustrative); up is slower. The distractor gain is shown as the slopes on HP and LP trials (entered separately) averaged per posterior draw; per trial type and their difference in the text and table. Points are posterior means, bars 95% HDIs, solid when the HDI excludes 0 and pale otherwise; stars give the posterior probability that the effect has the sign shown: * > .975, ** > .995, *** > .9995 (not corrected across ROIs). Both panels use the same fits: trial-wise and bin-wise gains estimated on top of each subject's final joint gain-field fit (m7, DoG pRFs, one HRF per subject × ROI). Voxels: per ROI the voxels whose held-out signal beats a model-free null (as for the LOSO tests), BOLD SD ≤ 20%; pRFs of voxels outside the gain-field fit come from the DoG fold fits with the same per-ROI HRF, their baseline and amplitude by least squares. ROI groups: early (V1, V2, V3), dorsal (V3AB, IPS), occipitotemporal (hV4, LO, TO, VO); one set of gains per group, estimated as a shared deviation from each ROI's own fitted gains (the regressors of all the group's voxels in one fit).

**Per-ROI statistics (a).** Median [bootstrapped 95% CI], Wilcoxon signed-rank p against 0, Benjamini-Hochberg q across the 3 ROI groups within each series.

*Target gain, repeat − switch*

| ROI | n | Median [95% CI] | p | q |
|---|---|---|---|---|
| early | 30 | −0.050 [−0.141, +0.016] | 0.0879 | 0.0879 |
| dorsal | 28 | −0.030 [−0.181, +0.004] | 0.0204 | 0.0612 |
| occipitotemporal | 30 | −0.074 [−0.112, −0.017] | 0.0497 | 0.0746 |

*Distractor gain, repeat − switch*

| ROI | n | Median [95% CI] | p | q |
|---|---|---|---|---|
| early | 30 | −0.003 [−0.149, +0.153] | 0.584 | 0.63 |
| dorsal | 28 | +0.056 [−0.081, +0.134] | 0.63 | 0.63 |
| occipitotemporal | 30 | −0.026 [−0.175, +0.078] | 0.36 | 0.63 |
