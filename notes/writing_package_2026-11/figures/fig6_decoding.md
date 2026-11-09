**Figure 6. The bar stimulus can be read out of visual cortex, and it is read out at lower intensity in the quadrant facing the HP location.** The bar stimulus of every run was decoded from the BOLD activity of one ROI at a time with pRFs
(difference of Gaussians, one HRF per subject and ROI, fitted jointly with the pRFs, no gain terms) fitted on the other runs (cross-validation folds), so the decoded run
never entered the pRF fit; voxels with cross-run reliability > 0 on the training runs (at most 2000 per ROI); every pixel
of every frame decoded; the encoding model contains only the bar, not the search items. The higher areas (V3AB, IPS, hV4, VO, LO, TO) were decoded the same way; an area was skipped for a subject when fewer than 20 voxels passed the criteria (n subjects: V1 30, V2 30, V3 30, V3AB 25, IPS 19, hV4 29, VO 11, LO 29, TO 27). Their tests are exploratory. a, True bar (top) and decoded intensity in a single held-out run (middle;
sub-03 ses-1 run-4, V1, frame-wise r = 0.62: rank 1 of 2653 subject × run × ROI decodes, chosen as the
clearest of the five best; the median run has r = 0.30) and averaged over all runs and subjects (V1; bottom), at
seven time points of the run (the last one without a bar). Grey circle, the 3.17° aperture of the bar. b, Per subject,
the spatial correlation between decoded and true frame inside the aperture, averaged over the frames with a bar and over
runs (blue dots, n = 30 subjects), against the same with the order of the true frames shuffled (100 permutations; grey).
Points and bars: median and bootstrapped 95% CI. Decoded vs shuffled: Wilcoxon signed-rank test, one-sided (*** p <
.001); V1 median r = 0.51 vs -0.002, p = 9.3 × 10^-10; V2 median r = 0.47 vs -0.004, p = 9.3 × 10^-10; V3 median r = 0.40 vs -0.005, p = 9.3 × 10^-10; V3AB median r = 0.29 vs -0.004, p = 3.0 × 10^-8; IPS median r = 0.12 vs -0.008, p = 1.9 × 10^-6; hV4 median r = 0.28 vs -0.005, p = 1.9 × 10^-9; VO median r = 0.16 vs -0.003, p = 4.9 × 10^-4; LO median r = 0.25 vs -0.007, p = 1.9 × 10^-9; TO median r = 0.19 vs -0.007, p = 7.5 × 10^-9. The correlation peaks at lag 0 in V1–V3, so the decoded time course is not displaced. c, Each
run's visual field rotated so that the HP location (red dashed circle; grey, the LP locations, drawn with the item
footprint, radius 0.75°) lies at 12 o'clock (90° steps, then 45° with bilinear interpolation); shown is the decoded
intensity on the bar's pixels minus its average over the four rotations (in % of that average), averaged over runs,
V1–V3 and subjects, smoothed (Gaussian, σ = 0.4°). The dotted diagonals bound the quadrant facing HP. d, Decoded
intensity on the bar's pixels per quadrant of the aperture, in % of the mean of the four quadrants, in the same rotated
frame (icons: HP ringed in red at the top, the quadrant shaded): the quadrant facing HP, the two quadrants next to it
(left and right pooled; they do not differ, two-sided Wilcoxon p = 0.22) and the opposite quadrant; per subject
the mean over runs and V1–V3; dots are subjects, points and bars the mean and within-subject SEM (Cousineau-Morey).
HP -0.85 ± 0.38%, next to HP +0.46 ± 0.18%, opposite -0.08 ± 0.22% (mean ± SEM); HP below next to HP, one-sided Wilcoxon p = 0.015 (**); HP below opposite, p = 0.13; Friedman test across the four quadrants p = 0.06. e, The pre-specified test: per subject and ROI, the mean decoded intensity
on the bar's pixels in the quadrant facing HP minus the mean over the three other quadrants, averaged over runs; dots are
subjects, points and bars the median and bootstrapped 95% CI. Wilcoxon signed-rank test, one-sided (HP < other
quadrants), Holm across V1–V3 (the pre-specified test); * p_Holm < .05. V1 median -0.0002, 53% of subjects below 0, p = 0.320, p_Holm = 0.320; V2 median -0.0014, 73% of subjects below 0, p = 0.013, p_Holm = 0.039; V3 median -0.0009, 63% of subjects below 0, p = 0.027, p_Holm = 0.055. V3 is borderline (p_Holm = 0.055, same direction). Higher areas (exploratory): one-sided Wilcoxon, Benjamini-Hochberg across the six areas; V3AB median -0.0004, 64% below 0, n = 25, p = 0.043, q = 0.085; IPS median +0.0001, 42% below 0, n = 19, p = 0.601, q = 0.601; hV4 median -0.0002, 52% below 0, n = 29, p = 0.179, q = 0.268; VO median -0.0004, 73% below 0, n = 11, p = 0.034, q = 0.085; LO median +0.0001, 48% below 0, n = 29, p = 0.525, q = 0.601; TO median -0.0001, 67% below 0, n = 27, p = 0.031, q = 0.085. Open markers and the upper row of stars: the decoder with the bar outline given (only the pixels the bar covers are decoded, the rest fixed at 0; median and bootstrapped 95% CI), same tests: V1 median -0.0003, p = 0.299, p_Holm = 0.299; V2 median -0.0016, p = 0.010, p_Holm = 0.031; V3 median -0.0009, p = 0.030, p_Holm = 0.061; higher areas (exploratory, BH): V3AB median -0.0004, n = 25, p = 0.040, q = 0.120; IPS median +0.0001, n = 19, p = 0.570, q = 0.570; hV4 median -0.0003, n = 29, p = 0.163, q = 0.244; VO median -0.0003, n = 11, p = 0.120, q = 0.240; LO median +0.0001, n = 29, p = 0.500, q = 0.570; TO median -0.0001, n = 27, p = 0.031, q = 0.120. Filled markers and the lower row of stars: the unconstrained decoder. f, The difference of e by eccentricity, in %
of the decoded intensity in the other quadrants at the same eccentricity, mean of V1–V3 per subject; line and band, mean ± SEM over
subjects (rings of 0.25°); left, the three bands drawn on the display. Items (radius 0.75°) begin at 3.25°
eccentricity, just outside the aperture, and their responses are in the BOLD data but not in the encoding model; if item
responses leaking into the aperture produced the difference, it would grow toward the aperture edge. It does not: per
band (mean ± SEM; one-sided Wilcoxon signed-rank test, uncorrected, * p < .05) 0–1.5° -0.6 ± 0.5%, p = 0.103; 1.5–2.5° -1.4 ± 0.6%, p = 0.029; 2.5°+ -1.3 ± 0.6%, p = 0.040; whole aperture -1.1 ± 0.5%, p = 0.033; the three bands do not differ
(Friedman test, p = 0.53; outer minus inner band, two-sided Wilcoxon p = 0.38). Within 0.75° of
the fovea, where the four quadrants meet, there is no difference (+0.2 ± 0.5%). g, Without a model: the measured response
(cleaned BOLD, % signal) of V1–V3 voxels with cross-run reliability > 0 and a pRF centre within 2° of an item location,
locked to the moment the bar centre crosses the pRF centre along the bar's direction of motion (left: sketch, and the
mean response of these voxels, which peaks about 3 s after the crossing). Each voxel's location is classed relative to
the run's HP location as in d: HP (red), next to HP (the two neighbouring LP locations, dark grey) or opposite (light
grey); responses are averaged over passes and voxels per subject, ROIs pooled weighted by voxel count, and shown
relative to the mean of the four locations; lines and bands, mean ± SEM over subjects. HP minus the mean of the three LP
locations in 1.6–6.4 s (grey; * one-sided Wilcoxon p < .05): -0.060% signal (-2.6% of the LP response), 67% of subjects below 0, one-sided Wilcoxon p = 0.017 (vs the two neighbouring LP locations -3.1%, p = 0.021; vs the opposite one -1.4%, p = 0.085). Search items are not removed and the
response before the crossing is already rising (the bar enters the pRF earlier), so no pre-crossing baseline is
subtracted; with one (−6.4 to 0 s) the difference is -2.1%, p = 0.16, not significant. pRFs: jointhrf fold fits.
