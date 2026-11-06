# retsupp: pre-specified tests (2026-11-06)

Written before any output of the final HP-rotation null exists. The earlier null fits (rotation 1,
DoG, 259 subject × ROI fits, and one DN fit, all on the current inputs) were cancelled on 2026-11-06
before any test was run on them; they will not be used.

## Decisions (user, 2026-11-06)

1. **Model.** DoG pRFs are the main model and carry the primary test. DN pRFs are a robustness check
   in the supplement and do not change the story, whatever they show.
2. **Primary test.** Group dS = g_HP − g_LP against the HP-rotation null, one-sided (dS < 0), DoG.
   Subject value: the median over V1, V2, V3, V3AB, IPS, hV4, VO, LO, TO (nine ROIs; SPL1 and FEF
   are not used).
3. **Headline effect size.** The conditional estimate of dS (dS estimated alone on top of m4). The
   joint estimate (m7_full) is reported alongside as a check.
4. **HRF rule.** The gain fits use one HRF per subject × ROI if it improves held-out R² over the
   canonical HRF in the majority of subjects (Wilcoxon p < .05); otherwise the canonical HRF. Applied
   mechanically; the numbers are reported.
5. **Voxel selection.** Cross-run reliability only, plus the outlier rule (BOLD SD ≤ 20%).
6. **Decoding (gain-free converging test).** Decoded bar drive at the HP location lower than at the LP
   locations, one-sided (HP < LP), in V1, V2 and V3, all 30 subjects, on the final inputs; Holm
   correction across the three ROIs. Earlier decoding runs are not reused or cited.
7. **Trial-by-trial gain and behaviour (Fig. 7)** is a main result and is redone on the final fits.

## Operational details (filled in by Claude; to be confirmed)

- **Null construction.** Each run's HP label is moved 1, 2 or 3 ring steps; data, stimulus and events
  unchanged. A subject's four labellings are exchangeable under the null, so each null draw assigns
  every subject one of its four labellings at random, the same for all its ROIs. 100,000 draws.
- **Group statistic.** Median over subjects of the subject value. One-sided p = (1 + number of draws
  with statistic ≤ observed) / (1 + 100,000). Reported: observed value, null distribution, p, and
  the mean of the rotated values (a real effect leaks into the rotated labellings, which should lean
  positive by about a third of it).
- **Which dS enters the null (user, 2026-11-06): the conditional dS**, the headline estimate. Under
  each of the three rotations the conditional dS is recomputed on the final m4 fits with the HP label
  moved; the joint m5/m7 refits under rotation are not run.
- **HRF rule, operationalised.** Candidate per subject × ROI HRF: the median delay and dispersion of
  the ROI's voxels with training R² ≥ .1 in the fold's own flexible-HRF fit, held fixed (models 14 DoG,
  16 DN in `retsupp/modeling/cv/fit_prf_cv_roi_hrf.py`); held-out data play no part. The variant with
  the shared HRF fitted per ROI (24, 26) is tested on sub-03 first and replaces the median variant only
  if it predicts better there. Per subject: median over the nine ROIs of held-out R² (per subject ×
  ROI HRF) minus held-out R² (canonical HRF), DoG pRFs, reliable voxels, same folds. Rule met if more
  than 15 of 30 subjects are positive and the two-sided Wilcoxon signed-rank p < .05. Evaluated for
  DoG, which decides the HRF for all gain fits; the DN numbers are reported.
- **Decoding test, operationalised (definition of the HP location confirmed by the user, 2026-11-06).**
  The HP ring position (4°) lies outside the bar aperture (3.17°), so "bar drive at the HP location" is
  the decoded bar drive in the visual-field quadrant facing HP, inside the aperture; LP = the three
  other quadrants. Per subject and ROI: HP quadrant minus the mean over the three LP quadrants, from the gainless pRF model, pRFs fitted on runs other than
  the ones decoded. Wilcoxon signed-rank across the 30 subjects, one-sided (HP < LP), Holm across V1,
  V2, V3.
- **Secondary results, not corrected with the primary test:** dS per ROI (BH across the nine ROIs),
  dD, LOSO per ladder step, DN versions of everything.

## Outcome of the sub-03 HRF-variant test (2026-11-06)

sub-03, fold 0, reliable voxels, median held-out R² per ROI. The per subject × ROI HRF from the median of the
training fits (models 14, 16) beats the variant with a shared HRF fitted per ROI (24, 26) in 7 of 9 ROIs for DoG
(median over ROIs .225 vs .211; canonical .190, per-voxel .227) and in 8 of 9 for DN. The median variant therefore
stays the candidate, as specified above. Rule outcome on all 30 subjects: pending (job 8117340).
