# Prompt for the writing agent (retsupp, 2026-11-05)

Read `notes/results_overview_2026-11.md` first; it has every number and path. Plan the paper from it, but treat all gain-field numbers as provisional, because the fits may be redone (below). Scope is gain-field models only: pRF-shift analyses are archived and must not be cited.

**New since the last overview**

- **Behaviour, now n = 30 with the pilots:** singleton at HP vs LP −27.8 ms, all subjects (p = 1.9e-09); target at HP vs LP +28.4 ms, 87% (p = 3.9e-07).
- **dS (g_HP − g_LP) is negative under DoG and DN:** parameters p = .010 and .004; leave-one-subject-out 77%, p = .010 (DoG) and 63%, p = .23 (DN). The DN disagreement is unexplained.
- **Within-subject CV cannot test dS:** putting in an effect of our size (−.016) gives the same harm as no effect (10% vs 0% of subjects improved). Break-even is near −.05. Report this as lack of power, not as evidence against dS, and report every prediction test.
- **Gain-model inputs:** all gain fits use a fixed canonical HRF (delay 4.5, dispersion .75), while m4's median HRF is 3.8 and .43 (20-subject cache). Voxels are selected on m4, also for the DN fits. Starting values are m4 (DoG) and m5 (DN), with pRFs then refit.
- **The R² mixture "FDR" is not valid as FDR:** at its threshold (R² about .02), V1 is 74% contralateral, implying roughly half false positives, not 5%. At R² ≥ .1 it is 96%. Do not write "FDR-corrected" for this selection.
- **Voxels actually in the gain fits (`p_signal` > .5, top 500):** 83–91% contralateral. Below R² = .1: V1 15%, V2 17%, V3 18%, hV4 60%, V3AB 68%, IPS 83%. Restricting them to cross-run null cvR² > 0 or R² ≥ .1 gives 93–98%. The m0 pRF model's cvR² > 0 gives only 85–93%. Prediction scores already use the cross-run null, but the shared gains are fitted on all selected voxels.
- **V1 pRF size barely grows with eccentricity:** .020 °/° (V2 .067, V3 .14, hV4 .21, V3AB .27), about .57° throughout. Cause unknown: possibly fixation, stimulus limits, DoG centre σ or the HRF.
- **Figure 1 draft:** `notes/figures/fig1_paradigm_prf_occipital.pdf` (paradigm plus pRF sanity checks, all m4), script `retsupp/visualize/paper/plot_fig1.py`.

**Still running, no results yet**

- **HP-label permutation null, DoG and DN:** 260 of 1,980 fits done. This is what decides whether the parameter result can lead. It cannot catch a modelling problem tied to the true HP location.
- **One HRF per subject × ROI vs one per voxel vs canonical:** held-out R² for Gaussian, DoG and DN (`retsupp/modeling/cv/fit_prf_cv_roi_hrf.py`). Test fit queued.
- **Gaussian, DoG and DN sanity comparison:** contralaterality and V1 size vs eccentricity, for fixed and flexible HRF.
- **Within-condition CV (`run3`):** 275 of 330 done. Expect it to be underpowered too.
- **Whole-brain DN maps:** stalled at 19 of 30 subjects, needs resubmitting.

**Undecided, so write around these**

Primary test and pRF model; voxel selection (replacing the mixture); HRF in the gain fits. If the HRF or the voxel selection changes, the whole gain ladder, LOSO and null get rerun, and every gain number changes. Leave placeholders for those numbers, and don't write the dS claim as final.
