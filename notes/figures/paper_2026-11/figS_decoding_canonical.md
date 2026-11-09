**Figure S. Decoding with pRFs that use the canonical HRF (robustness check of Fig. 6).** Everything as in Fig. 6, but the decoder's pRFs use one canonical HRF instead of an HRF per subject and ROI. The bar stimulus of every run was decoded from the BOLD activity of one ROI at a time with pRFs
(difference of Gaussians, canonical HRF, no gain terms) fitted on the other runs (cross-validation folds), so the decoded run
never entered the pRF fit; voxels with cross-run reliability > 0 on the training runs (at most 2000 per ROI); every pixel
of every frame decoded; the encoding model contains only the bar, not the search items. Only V1–V3 are shown. a, True bar (top) and decoded intensity in a single held-out run (middle;
sub-07 ses-1 run-3, V1, frame-wise r = 0.60: rank 3 of 1074 subject × run × ROI decodes, chosen as the
clearest of the five best; the median run has r = 0.46) and averaged over all runs and subjects (V1; bottom), at
seven time points of the run (the last one without a bar); decoded rows on a magma scale from 0 to each row's maximum. Grey circle, the 3.17° aperture of the bar. b, Per subject,
the spatial correlation between decoded and true frame inside the aperture, averaged over the frames with a bar and over
runs (blue dots, n = 30 subjects), against the same with the order of the true frames shuffled (100 permutations; grey).
Points and bars: median and bootstrapped 95% CI. Decoded vs shuffled: Wilcoxon signed-rank test, one-sided (*** p <
.001); V1 median r = 0.51 vs -0.003, p = 9.3 × 10^-10; V2 median r = 0.47 vs -0.004, p = 9.3 × 10^-10; V3 median r = 0.41 vs -0.005, p = 9.3 × 10^-10. The correlation peaks at lag 0 in V1–V3, so the decoded time course is not displaced. c, Each
run's visual field rotated so that the HP location (red dashed circle; grey, the LP locations, drawn with the item
footprint, radius 0.75°) lies at 12 o'clock (90° steps, then 45° with bilinear interpolation); shown is the decoded
intensity on the bar's pixels minus its average over the four rotations (in % of that average), averaged over runs,
V1–V3 and subjects, smoothed (Gaussian, σ = 0.4°). The dotted diagonals bound the quadrant facing HP. d, Per subject and ROI, the mean decoded intensity
on the bar's pixels in the quadrant facing HP minus the mean over the three other quadrants, averaged over runs; dots are
subjects, points and bars the median and bootstrapped 95% CI. Wilcoxon signed-rank test, one-sided (HP < other
quadrants), Benjamini-Hochberg across the ROIs per decoder; * q < .05. V1 median -0.0002, 50% of subjects below 0, n = 30, p = 0.202, q = 0.202; V2 median -0.0014, 73% of subjects below 0, n = 30, p = 0.015, q = 0.034; V3 median -0.0006, 67% of subjects below 0, n = 30, p = 0.022, q = 0.034. Open markers and the upper row of stars: the decoder with the bar outline given (only the pixels the bar covers are decoded, the rest fixed at 0; median and bootstrapped 95% CI), same tests: V1 median -0.0003, p = 0.232, q = 0.232; V2 median -0.0014, p = 0.012, q = 0.029; V3 median -0.0006, p = 0.019, q = 0.029. Filled markers and the lower row of stars: the unconstrained decoder. e, Decoded
intensity on the bar's pixels per quadrant of the aperture, in % of the mean of the four quadrants, in the same rotated
frame (icons: HP ringed in red at the top, the quadrant shaded): the quadrant facing HP, the two quadrants next to it
(left and right pooled; they do not differ, two-sided Wilcoxon p = 0.24) and the opposite quadrant;
per subject and ROI the decoded intensity per quadrant averaged over runs and then expressed in % of the mean of the four quadrants, then averaged over V1–V3; dots are subjects, points and bars the mean and within-subject SEM (Cousineau-Morey).
HP -0.89 ± 0.38%, next to HP +0.50 ± 0.18%, opposite -0.12 ± 0.24% (mean ± SEM); HP below next to HP, one-sided Wilcoxon p = 0.008 (**); HP below opposite, p = 0.17; Friedman test across the four quadrants p = 0.10. f, The difference of d by eccentricity, in %
of the decoded intensity in the other quadrants at the same eccentricity, mean of V1–V3 per subject; line and band, mean ± SEM over
subjects (rings of 0.25°), in the three bands shaded. Items (radius 0.75°) begin at 3.25°
eccentricity, just outside the aperture, and their responses are in the BOLD data but not in the encoding model; if item
responses leaking into the aperture produced the difference, it would grow toward the aperture edge. It does not: per
band (mean ± SEM; one-sided Wilcoxon signed-rank test, uncorrected, * p < .05) 0–1.5° -0.8 ± 0.5%, p = 0.052; 1.5–2.5° -1.4 ± 0.7%, p = 0.032; 2.5°+ -1.3 ± 0.6%, p = 0.055; whole aperture -1.2 ± 0.5%, p = 0.019; the three bands do not differ
(Friedman test, p = 0.67; outer minus inner band, two-sided Wilcoxon p = 0.61). Within 0.75° of
the fovea, where the four quadrants meet, there is no difference (+0.1 ± 0.4%). pRFs: canonical fold fits.
