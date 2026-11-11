**Figure 7 (prototype). The history effects of Fig. 1 in the gains, and gain and behaviour trial by trial.** a, History (Fig. 1e, f; the location effects of Fig. 1c, d are in Fig. 5d): target gain on target-repeat minus other trials, and distractor gain on distractor-repeat minus distractor-switch trials (the difference taken within HP and within LP and averaged, as for the response times in Fig. 1f); the display icons are those of Fig. 1e, f (four possible locations; filled, the item on this trial; hollow, on the previous trial; no HP ring, since HP and LP trials are pooled), each series the gain in the first condition minus that in the second. Points and bars are the median over subjects (n = 30) and its bootstrapped 95% CI; Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs within each series; estimates with q < .05 are drawn solid, the others pale (key at the top); stars: * q < .05, ** q < .01, *** q < .001; all values below. b, Within subjects, trial by trial: Bayesian hierarchical linear model of log RT (z within subject) per ROI on the per-trial target gain (gain field at the target), distractor gain (gain field at the distractor) on HP and on LP trials, and global gain (over the whole display), each z-scored within run and trial type, with trial number, target repeat, distractor repeat and target-at-previous-distractor as covariates, and random intercepts and gain slopes per subject (bambi, 4 × 1000 draws, no divergences, R̂ ≤ 1.01); the icons show each gain field (illustrative); up is slower. The distractor gain is shown as the slopes on HP and LP trials (entered separately) averaged per posterior draw; per trial type and their difference in the text and table. Points are posterior means, bars 95% HDIs, solid when the HDI excludes 0 and pale otherwise; stars give the posterior probability that the effect has the sign shown: * > .975, ** > .995, *** > .9995 (not corrected across ROIs). Both panels use the same fits: trial-wise and bin-wise gains estimated on top of each subject's final joint gain-field fit (m7, DoG pRFs, one HRF per subject × ROI). Voxels: per ROI the voxels whose held-out signal beats a model-free null (as for the LOSO tests), BOLD SD ≤ 20%; pRFs of voxels outside the gain-field fit come from the DoG fold fits with the same per-ROI HRF, their baseline and amplitude by least squares.

**Per-ROI statistics (a).** Median [bootstrapped 95% CI], Wilcoxon signed-rank p against 0, Benjamini-Hochberg q across the 9 ROIs within each series.

*Target gain, repeat − switch*

| ROI | n | Median [95% CI] | p | q |
|---|---|---|---|---|
| V1 | 30 | −0.098 [−0.198, +0.042] | 0.0919 | 0.276 |
| V2 | 30 | −0.027 [−0.055, +0.056] | 0.516 | 0.516 |
| V3 | 30 | −0.046 [−0.163, +0.033] | 0.205 | 0.308 |
| V3AB | 28 | −0.062 [−0.139, −0.015] | 0.00886 | 0.0797 |
| IPS | 19 | −0.090 [−0.249, −0.008] | 0.0323 | 0.146 |
| hV4 | 30 | −0.060 [−0.133, +0.013] | 0.245 | 0.315 |
| VO | 21 | −0.031 [−0.071, +0.003] | 0.137 | 0.308 |
| LO | 30 | −0.043 [−0.190, +0.017] | 0.205 | 0.308 |
| TO | 29 | −0.030 [−0.236, +0.054] | 0.325 | 0.366 |

*Distractor gain, repeat − switch*

| ROI | n | Median [95% CI] | p | q |
|---|---|---|---|---|
| V1 | 30 | +0.006 [−0.132, +0.332] | 0.184 | 0.984 |
| V2 | 30 | −0.007 [−0.100, +0.159] | 0.984 | 0.984 |
| V3 | 30 | −0.060 [−0.214, +0.158] | 0.67 | 0.984 |
| V3AB | 28 | +0.045 [−0.066, +0.141] | 0.508 | 0.984 |
| IPS | 19 | +0.060 [−0.110, +0.265] | 0.651 | 0.984 |
| hV4 | 30 | −0.078 [−0.203, +0.066] | 0.371 | 0.984 |
| VO | 21 | −0.047 [−0.160, −0.005] | 0.338 | 0.984 |
| LO | 30 | +0.063 [−0.213, +0.186] | 0.855 | 0.984 |
| TO | 29 | −0.015 [−0.092, +0.092] | 0.932 | 0.984 |
