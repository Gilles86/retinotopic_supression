# retsupp: overview of all current results (2026-11-05)

Written as the starting point for planning the paper. It says what each result is, how far it can be trusted and where it lives. Detailed tables are in `notes/fitting_lessons_2026-11.md`.

Grades: **solid** holds across the checks we ran; **qualified** real but with a stated dependence; **open** not yet decidable; **superseded** do not cite; **not re-checked** older result not reproduced under the current pipeline.

## 1. Summary

Thirty participants did a visual search task with a high-probability (HP) distractor location during 7T fMRI, while a bar swept through the visual field for pRF mapping. The question is whether learned suppression of the HP location shows up as a change in spatial tuning.

- Behaviour shows strong learned suppression: 28 ms, in all 30 subjects.
- Visual cortex is well described by pRFs plus the search items as stimuli. Items improve out-of-subject prediction in 100% of subjects.
- Attention acts better as gain than as pRF shift. Shifts did not survive cross-validation.
- Target-evoked gain is strong and grows up the hierarchy. Distractor-evoked gain is positive, weaker and does not predict out of subject when tied across locations.
- **The key test, dS = g_HP − g_LP (sustained gain at HP minus LP), is negative under both pRF models, which means less sustained gain at the HP location.** Parameters: DoG −.016, p = .010; DN −.028, p = .004. Prediction: DoG 77% of subjects improved, p = .010; DN 63%, p = .23.
- Within-subject CV cannot test dS: a simulation with our effect size also fails it (section 7). This has to be explained in the paper, not left as a failed test.
- There is no evidence of learning over runs and no robust link between HP-specific gain and RT, the latter on an older fit.
- **Main open item:** the permutation null with HP labels rotated around the ring (section 8).

## 2. Design and terms

- 30 subjects (`retsupp/data/subjects.yml`), 2 sessions × 6 runs (sub-20 ses-1 and sub-24 ses-2 have 5), TR 1.6 s, 258 volumes, NORDIC, fMRIprep.
- Search array: 8 items on a ring at 4° (`project_paradigm_8items` memory); targets and singleton distractors at the 4 diagonal positions, with the HP location changing between runs.
- Bar aperture radius 3.17°, so **the HP location lies outside the mapped area**. This shapes which voxels can show the effect (section 7).

**Two sets of model names collide; keep them apart in the paper.**

| Whole-brain pRF labels | Model |
|---|---|
| m1, m3 | Gaussian, fixed and flexible HRF |
| m2, m4 | DoG, fixed and flexible HRF |
| m5, m6 | DN, fixed and flexible HRF |

| Gain-field ladder | Adds |
|---|---|
| m00_bar | pRF on the bar stimulus only |
| m0_prf | + search items with a free scale (`item_scale`) |
| m1_T, m2_D, m3_S | + target, distractor or sustained gain alone |
| m3b_TD, m4_TDS | Combinations, gains tied across locations |
| m5_TDS_dS | m4 + separate g_HP and g_LP (sustained) |
| m6_TDS_dD | m4 + separate g_HP_dyn and g_LP_dyn (distractor-evoked) |
| m7_full | Both HP-specific terms |

Terms: items = m0 vs m00; T, D, S = m1, m2, m3 vs m0; dS = m5 vs m4, or the parameter g_HP − g_LP; dD = m6 vs m4, or g_HP_dyn − g_LP_dyn. Negative dS means less gain at HP.

**Model.** Gain fields multiply the stimulus, so with DN they enter both activation and normalization pool (Reynolds & Heeger 2009). Activation-only variants were decided against (2026-11-03).

**Current fitting procedure (nested, bounded).** Per subject × ROI, at most 500 voxels (p_signal > .5). m00 and m0 fit per-voxel pRFs by gradient descent; every gain model then freezes those pRFs and fits only shared parameters by L-BFGS from its parent's optimum, so training SSE cannot rise. Bounds: σ .5–8°, |gain| ≤ 10, `item_scale` ≤ 50. Voxels above 20% BOLD SD are dropped. 30 subjects × 11 ROIs.

**LOSO test.** Effect size δ = median of the other subjects' own conditional estimate, so nothing is fitted to the subject tested. Scored as ΔR² on signal voxels (cross-run null cvR² > 0); subject value = median over ROIs; Wilcoxon across subjects.

## 3. Behaviour: solid

n = 30 including pilots sub-01 and sub-02, 21,480 trials, RTs from correct trials. Recomputed 2026-11-05 with `notes/scripts/results_overview_behaviour.py`. The pilots' events have no HP column, so each run's HP comes from the modal singleton location (`Subject.get_hpd_locations`); that rule matches the recorded column on every trial of the other 28.

| Contrast | Mean RT (ms) | Difference | Subjects in the expected direction | p |
|---|---|---:|---:|---:|
| Distractor HP vs LP | 877 vs 905 | −27.8 ms | 100% | 1.9e-09 |
| Distractor LP vs absent | 905 vs 862 | +43.0 ms | 93% | 2.6e-08 |
| Distractor HP vs absent | 877 vs 862 | +15.2 ms | 70% | .001 |
| Target at HP vs LP (distractor trials) | 908 vs 880 | +28.4 ms | 87% | 3.9e-07 |

Both pilots go the same way (HP vs LP −44.7 and −42.9 ms; target at HP vs LP +38.5 and +50.0 ms), with accuracy .98 and .89 against a median of .91. Without them: −26.6 ms (p = 7.5e-09), +40.0 ms (p = 1e-07), +13.4 ms (p = .004), +27.3 ms (p = 1.5e-06).

Figures: `notes/figures/rt_talk_distractor.pdf`, `notes/figures/rt_talk_target.pdf`.

## 4. pRF models: solid

Leave-one-run-set-out, 30 subjects × 3 folds, voxels selected per fold on training-run reliability. Figure `notes/figures/prf_cv_roi_reliable.pdf`, table `notes/data/prf_cv_roi_summary.tsv`.

Median held-out R² in reliable voxels: m1 .201, m2 .210, m3 .230, m4 .238, m5 .216, m6 .245.

| Contrast | Reliable voxels, median Δ | % subjects | All voxels, median Δ |
|---|---:|---:|---:|
| DoG vs Gaussian, flexible HRF | +.0061 | 97 | +.0041 |
| DN vs DoG, flexible HRF | +.0071 | 100 | −.0004 |
| Flexible HRF, DoG | +.0177 | 100 | +.0033 |
| Flexible HRF, DN | +.0200 | 100 | +.0032 |

DN beats DoG only in reliable voxels. Reliable share by ROI: V3 26%, LO 24%, V2 23%, V1 19%, hV4 15%, TO 10%, V3AB 8%, VO 3%, IPS 2%, SPL1 and FEF 0%. **SPL1 and FEF are not interpretable** (signal voxels 1.7% and 0.4%).

**DN parameters across the hierarchy vs Aqil et al. 2021: qualified.** Figures `notes/figures/dn_hierarchy_aqil.pdf`, `notes/figures/dn_hierarchy_ratios_supp.pdf`; notes `notes/dn_hierarchy_aqil.md`.

| Parameter | Aqil | Here (m6, ρ along hierarchy) |
|---|---|---:|
| σ centre (positive control) | Up | +.71 |
| Size ratio | Down | +.28, gone with eccentricity control (−.02) |
| b | Down | −.70 |
| d | Down | +.80 (opposite) |
| b/d | Not reported | −.90 |
| d/c | Not reported | +.82 |

Limits: a, b and d share a free scale, so raw b and d depend on the fit's path; b/d and the size ratio do not. One contrast in a 3.17° aperture; 23% of surrounds exceed the grid.

**Whole-brain DN maps: incomplete.** The May m6 predates the flexible-HRF fix and is invalid. Refit into `derivatives/prf.hrffix`: sub-01 to sub-19 merged, sub-20 to sub-30 not (32 of 110 chunks). The failures came from cluster nodes u24-chaihm0-651 and -636. This blocks group DN maps but not the dS tests, whose DN fits are per ROI and complete. First maps: `notes/figures/dn_maps/dn_maps_sub-19.pdf`.

**Figure 1 maps:** `notes/figures/fig1_r2low/`, `notes/figures/fig1_inflated/` (sub-18, sub-19, sub-23).

**Outlier voxels: solid, needs a Methods line.** Percent signal change divides by a raw mean that collapses in some runs: 53 voxels above 20% SD hold 36% of variance and are irreproducible. In the worst fit (sub-07 V2) they drove g_HP to its bound and flipped the sign of dS. Voxels of 3–20% SD have the best signal, so a tighter limit would be wrong. Outlier-free refits are used throughout below.

## 5. Shifts vs gain

**Gain beats shift in CV: solid but low-powered.** Leave-one-HP-condition-out, 30 subjects × 11 ROIs. Shift adds nothing over a fixed pRF in about 9 of 11 ROIs; gain ≥ shift in every ROI, significant by sign test in V1, V2, V3AB, VO, IPS. Table `derivatives/af_prf_cv_shiftvsgain/cv4_per_roi.tsv` (run 2026-06-01; the figure `cv4_verdict.pdf` is no longer in `notes/figures/`). Given section 7, read this as "shift never wins", not as evidence that shifts are absent.

**PRF fits per condition and their shifts: archived, not used** (decision 2026-11-05). Code, notes and notebooks are in `archive/conditionwise_shifts/`. This includes the observed and predicted shifts, projection and GAM curves, and the post hoc Klein fits.

## 6. Gain-field parameters

m7_full, full data. Subject value = median over ROIs; Wilcoxon across n = 30. Figure `notes/figures/gain_parameters_m7.pdf`; tables `notes/data/gain_field_parameters.tsv`, `notes/data/gain_field_parameter_stats.tsv`.

| Parameter | DoG median | % negative | p | DN median | % negative | p |
|---|---:|---:|---:|---:|---:|---:|
| g_HP | −.061 | 93 | <.0001 | −.134 | 97 | <.0001 |
| g_LP | −.039 | 97 | <.0001 | −.099 | 97 | <.0001 |
| **dS** | **−.016** | **73** | **.010** | **−.028** | **80** | **.004** |
| g_HP_dyn | +.099 | 23 | .004 | +.180 | 17 | .0002 |
| g_LP_dyn | +.131 | 13 | <.0001 | +.210 | 17 | <.0001 |
| dD | −.062 | 70 | .028 | −.024 | 57 | .44 |
| g_T_dyn | +.247 | 3 | <.0001 | +.487 | 3 | <.0001 |

dS in m5: DoG −.018 (p = .0005), DN −.034 (p = .0006). Gains are 1.6–1.8 times larger under DN. Median widths: σ_AF 2.2° (DoG), 1.9° (DN); σ_dyn 3.6°, 3.1°. σ_dyn sits at a bound in 44% (DoG) and 31% (DN) of fits.

Per ROI, m7, median (FDR q):

| ROI | dS DoG | dS DN | dD DoG | dD DN | g_T DoG | g_T DN |
|---|---:|---:|---:|---:|---:|---:|
| V1 | −.042 (.47) | −.047 (.50) | +.084 (.47) | +.138 (.38) | +.005 (.78) | +.208 (.011) |
| V2 | −.046 (.085) | −.053 (.076) | +.050 (.11) | +.078 (.45) | +.121 (.029) | +.233 (<.001) |
| V3 | −.052 (.085) | −.058 (.024) | +.039 (.14) | +.156 (.033) | +.300 (<.001) | +.487 (<.001) |
| hV4 | −.023 (.14) | −.025 (.062) | −.071 (.085) | −.039 (.74) | +.338 (<.001) | +.555 (<.001) |
| VO | −.021 (.47) | −.043 (.35) | −.078 (.16) | −.055 (.32) | +.350 (<.001) | +.538 (<.001) |
| LO | −.001 (.63) | −.022 (.50) | −.067 (.51) | −.077 (.45) | +.256 (<.001) | +.674 (<.001) |
| TO | −.007 (.63) | −.011 (.67) | −.036 (.79) | −.029 (.94) | +.253 (<.001) | +.777 (<.001) |
| V3AB | −.003 (.63) | −.025 (.32) | −.090 (.085) | −.080 (.13) | +.300 (<.001) | +.544 (<.001) |
| IPS | −.005 (.47) | −.022 (.076) | −.100 (.085) | −.098 (.16) | +.389 (<.001) | +.754 (<.001) |

So dS is mainly a V2–hV4 result at pooled level, and in m7 only DN V3 passes FDR in a single ROI. In m5 (original DoG fits, before outlier removal) q < .05 in V2, V3, hV4, VO, V3AB, IPS, with dS and dD sharing HP-specific variance. Not rerun on outlier-free fits in this form.

**Target gain: solid.** Grows from V1 to higher areas under both models; FEF is the exception and is not interpretable.

**Caveats for dS as a parameter:**

1. **No valid null yet** (section 8). HP is one ring position and LP three.
2. **Overstated by the joint fit.** Conditional estimate vs m5 group median: V2 −.005 vs −.042, V3 −.013 vs −.051, hV4 −.021 vs −.030, VO −.013 vs −.030, V3AB −.011 vs −.019, IPS −.009 vs −.016. Report the conditional size.
3. **Absolute sign of sustained gain is not a result.** g_HP and g_LP are negative here but were positive in earlier joint fits (`project_af_signs_attraction` memory). The tied term has no control condition. Only the difference is interpretable.
4. **dD is not robust.** Parameter p = .028 vs .44, weakens with outlier removal and noise weighting, and changes sign across the hierarchy (positive V1–V3, negative hV4 onward).

## 7. Prediction tests

### Leave one subject out: qualified

From actual predictions, on outlier-free fits (`retsupp/modeling/loso_fixed_group_exact.py`). Figure `notes/figures/fixed_group_mechanisms_loso_illustrated.pdf`.

| Term | DoG % improved | DoG p | DoG ΔR² (e-5) | DN % improved | DN p | DN ΔR² (e-5) |
|---|---:|---:|---:|---:|---:|---:|
| Items | 100 | <.0001 | 329.29 | 100 | <.0001 | 2190.72 |
| T | 90 | <.0001 | 7.00 | 97 | <.0001 | 7.75 |
| D | 50 | .81 | .01 | 53 | .84 | .03 |
| S | 90 | <.0001 | .31 | 90 | <.0001 | .14 |
| **dS** | **77** | **.010** | **.25** | **63** | **.23** | **.09** |
| dD | 63 | .088 | .10 | 67 | .084 | .05 |

- dS is smaller under DN (paired p = .0002), yet subjects agree across models (ρ = .92).
- Post hoc, each subject's own best dS is negative in 80% (DoG) and 83% (DN).
- By ROI, uncorrected, DoG vs DN: IPS 83% vs 28%, V3AB 76% vs 68%, hV4 62% vs 66%.
- Base m4_TDS R² in signal voxels: DoG .213, DN .216, in sample with two more parameters per voxel.
- **Unresolved:** DN items' own δ is at the grid edge (`item_scale` = 0) in 97% of cells, so that row is a lower bound, yet the fitted m7 `item_scale` is 1.6. These look inconsistent.

### Within-subject CV is underpowered for dS: solid as a diagnosis

Percent of subjects improved (DoG, before outlier removal):

| Term | Leave one HP location out (n = 30) | Leave one run per location out (n = 26, preliminary) | LOSO (n = 30) |
|---|---:|---:|---:|
| Items | 100 | 96 | 100 |
| T | 73 (.0016) | 65 (.15) | 90 (<.0001) |
| S | 83 (<.0001) | 77 (.0003) | 87 (<.0001) |
| D | 23 (.0019) | 27 (.067) | 50 (.94) |
| dS | 20 (<.0001) | 35 (.27) | 80 (.006) |
| dD | 7 (<.0001) | 12 (<.0001) | 73 (.011) |

All 30 subjects have a negative mean m5 − m4 in leave-one-HP-out, although 46% of signal voxels improve and the median change is −6.5e-6. The rotated map (`notes/figures/cv_ladder_nested_rotated_dcv_bounded.pdf`) looks uniformly negative because of means dominated by a minority of voxels.

Cause, checked 2026-11-05 (`notes/scripts/results_overview_cv_power.py`, outlier-free fits). A conditional dS estimated from one HP condition has an SD of about .057 (median subject × ROI) against an effect of about −.009 to −.016. A noisy δ costs δ²a in SSE and gains only 2δb, so noise above the effect guarantees a loss. A fitter-free version (δ from three conditions, scored on the fourth) reproduces the loss: 3% of subjects improved, ΔR² −4.3e-5. This points away from the optimizer or refitting of other terms as the main cause; nested fits also never end worse than their parent on training SSE.

Power: every subject × ROI recentred on a chosen dS, real across-condition variation kept.

| dS put in | Subjects improved | p | ΔR² (e-5) |
|---:|---:|---:|---:|
| 0 | 0% | <.0001 (harm) | −7.1 |
| −.005 | 0% | <.0001 (harm) | −6.9 |
| **−.016** | **10%** | **<.0001 (harm)** | **−6.6** |
| −.03 | 20% | <.0001 (harm) | −4.3 |
| −.06 | 67% | .019 | +3.4 |
| −.10 | 90% | <.0001 | +20.7 |
| −.20 | 100% | <.0001 | +121.9 |

Break-even is near −.05, about three times our estimate; a rough scaling suggests about nine times more data per subject. **A true zero and an effect of our size give the same result, so this neither supports nor undermines dS.**

Limits: the inserted effect is linear and only moves g_HP. The kept variation may include real differences between HP locations. A simulation with a known dS in the BOLD, refit through the pipeline, would be the stronger form. This power check was run on leave-one-HP-out; I expect the same for run3 but did not test it.

The within-subject run3 job is still finishing (247 of 330 tasks when last counted; four timeouts: sub-06 IPS, sub-14 V3, sub-17 V1 and V2).

### Voxel-level views: descriptive

LOSO dS by distance of pRF centre from HP (DoG, signal voxels, post hoc bins, uncorrected; `notes/scripts/results_overview_zoom.py`):

| Distance | Voxels | % improved | p | ΔR² (e-5) | Own dS | Own dS p |
|---:|---:|---:|---:|---:|---:|---:|
| 0–1° | 3,562 | 62 | .77 | +.81 | −.009 | .15 |
| 1–2° | 8,852 | 73 | .009 | +2.27 | −.013 | .015 |
| 2–3° | 18,215 | 80 | .0006 | +1.76 | −.027 | <.0001 |
| 3–4° | 56,734 | 77 | .019 | +.13 | −.020 | .003 |
| 4–6° | 94,112 | 67 | .088 | +.03 | −.017 | .005 |
| >6° | 29,130 | 77 | .008 | +.22 | −.020 | .006 |

By sensitivity to the gain (`a`, blind to the sign of the effect), per subject × ROI × condition:

| Sensitivity percentile | Share of leverage | % improved | p | ΔR² (e-5) | Own dS |
|---:|---:|---:|---:|---:|---:|
| 0–50 | 6% | 60 | .97 | +.01 | −.009 |
| 50–75 | 11% | 67 | .13 | +.06 | −.016 |
| 75–90 | 18% | 83 | .002 | +.39 | −.017 |
| 90–100 | 65% | 73 | .009 | +.82 | −.012 |

- Zooming raises the effect size, not its reliability.
- The sign is the same everywhere, and a flat own dS beyond 2° is what one shared field predicts. This is not evidence of focality.
- **Voxels centred on HP are a poor test:** their centres lie at 3–5°, at or beyond the bar aperture, and that is also where the distractor is most frequent. Sensitivity is the better way to define voxels.
- My unbinned check over 9 ROIs without a voxel minimum gives 70%, p = .004, against 77%, p = .010 as reported. These tables are not the same selection as the main LOSO.
- DN not run in this form.

ΔcvR² by voxel reliability (unbounded nested, ×10⁻³): dS and dD are negative at every reliability level; see `notes/fitting_lessons_2026-11.md`.

## 8. Permutation null: running

Every run's HP label is moved 1, 2 or 3 ring steps; data, stimulus and events stay unchanged. Under the null, a subject's four labellings are exchangeable, so each subject is assigned one at random, the same across ROIs, 100,000 draws. Fits for m5, m6, m7 on 30 subjects × 11 ROIs, DoG and DN. Code: `retsupp/modeling/hp_rotation_null.py`, `--hp-rotation` in `retsupp/modeling/fit_cv_ladder_nested.py`, branch `hp-rotation-null`.

**Results: pending.** Canary fits are running; the full set follows once they pass.

What it can and cannot show:

- It tests whether the true HP position differs from the other relative positions, which covers the 1-vs-3 asymmetry.
- It does not catch a problem tied to the true HP location. The distractor is most frequent there, so any mismatch in how items or salience are modelled could load onto g_HP, and relabelling removes that link.
- A real effect leaks into the rotated labellings, since the true HP then sits among the "LP" positions. In a linear approximation the four contrasts of a subject sum to zero, so the rotated values are not independent of the real one: with a real dS they should lean positive by about a third of it.

## 9. Learning and behaviour links

**Learning: no evidence.** Gains re-estimated linearly per run, block position and parametric slope on the full-bounded m7 fit. HP−LP sustained slope within block p = .86, across blocks p = .17, no ROI q < .7. Distractor HP−LP drifts negative over block positions (−.01, −.05, −.08), p = .24. Figure `notes/figures/learning_gains_nested_full_bounded.pdf`. With the power above, this is not evidence against learning.

**Trial-wise gains and RT: mostly null, not rerun on current fits.** October 2026, on the older canonical fit (`project_trialwise_af_rt` memory, tables `notes/data/trialwise_af_*.tsv`, figure `notes/figures/trialwise_af_story.pdf`).

- Overall trial amplitude vs RT flips along the hierarchy: V1, V2 larger → faster (q < .02); IPS, SPL1, FEF larger → slower (t ≈ 6.5).
- Per-trial T, G, D gains are collinear (r ≈ .92–.98) in IPS, SPL1, FEF, so components cannot be separated there.
- Target gain is lower on target-repeat trials in V3AB, VO, IPS (q < .05).
- Run-level and between-subject brain–behaviour: null. HP−LP differences are below the 80%-power minimum detectable effect in every ROI.

**Decoding:** bar reconstruction works (`notes/docky_vss2026/4_decoding/`); a talk demonstration with no statistical result behind it for the paper.

## 10. Superseded: do not cite

| Claim | Where | Replaced by |
|---|---|---|
| Sustained HP suppression V2 p = .0002, V3 p = .0002 (Gaussian static) | `notes/STATUS.md`, May | Sections 6, 7. Effect far smaller in current fits |
| "5 free gain parameters", hV4 sustained null etc. | `notes/GAIN_FINDINGS.md`, May | Section 6 |
| Paradigm with only the singleton | Fits before 2026-05-08 | 8-item paradigm |
| HP ≠ LP "hurts", richer models fit worse in training | CV ladder, 28–29 Oct | Section 7. Metric and power; nested fits are monotonic in pooled SSE |
| Both gains positive, 89% (attraction) | `project_af_signs_attraction` memory, May–June | Sign is fit-dependent, section 6 caveat 3 |
| May whole-brain m6 and anything using it | Before braincoder 2849288 | Refit in section 4 |
| Conditionwise pRF shifts: hV4 and V3AB push-away, predicted vs observed shifts, AF and AF+ fits to conditionwise parameters | `archive/conditionwise_shifts/` | Archived, not to be used |

## 11. Inconsistencies to settle before writing

1. **Pilots.** All analyses now use n = 30, including sub-01 and sub-02. They have a different block order (`AA BBB CCC DDD A`), so Methods should state it.
2. **Primary test and pRF model.** Choose before seeing the null, or report openly how the choice was made.
3. **Which ROIs count.** Parameter tables use 11; several tests use V1–IPS. SPL1 and FEF have almost no signal voxels.
4. **`item_scale` under DN** (section 7).
5. **`distractor_radius`** in `Subject.get_stimulus_with_distractors`: `CLAUDE.md` flags that it may not match the rendered size. The fits use rectangles (1.5 × .375°), so check this only for older outputs.
6. **Unreviewed:** uncommitted changes to the GLMsingle quadrant outputs (`notes/figures/glmsingle_quadrants.tsv`, `.pdf`). Not included here. There are many other uncommitted changes in the tree, including deleted talk figures and a modified `libs/braincoder`.

## 12. Input for the paper plan

Not a decision, only what the results above imply.

- **Lead with parameters, not model comparison.** Group tests suit an effect this small; within-subject prediction demonstrably cannot detect it. Report every prediction test anyway, and state that this order was chosen after seeing them.
- **Strong anchors:** behaviour, pRF fits, items, target gain.
- **Suggested sequence:** (1) behaviour; (2) maps and pRF models; (3) items and gain-field model, target and distractor; (4) dS with the null, conditional effect size and sensitivity analysis; (5) generalization: LOSO, DoG vs DN, power analysis for within-subject CV; (6) shift-model CV, learning and behaviour links as secondary results.
- **What the claim can be, if the null holds:** a small, consistently signed group-level reduction of sustained gain at the HP location, clearest in V2–hV4, whose out-of-subject generalization depends on the pRF model. Not robust at the level of one subject.
- **Remaining work:** the null; run3 completion; whole-brain DN maps (needs the 78 missing chunks); optional simulation with dS in the BOLD; rerun of the RT link on current fits if it stays in the paper.
