# retsupp paper plan (2026-11-05)

Results lead with group-level gain parameters, LOSO is the model-comparison metric, and the selection-history claim rests on dS = g_HP − g_LP against the HP-rotation null. Within-subject CV is reported as underpowered, not as evidence against dS. Scope is gain-field models, plus the gain-free decoding test below as converging evidence; pRF-shift analyses are archived. All gain numbers are provisional until Phase C.

**Rules.** Nothing overwritten; new outputs carry a 2026-11 suffix. Commit only own new files, on a branch. Before any cluster submission, list models, subjects × ROIs, tasks and core-hours, and wait for OK. n = 30. DoG and DN both. Never call the R² mixture "FDR-corrected".

## Phase A: inputs to the gain fits, then stop for a decision

- **A1** Cost of finishing the null vs rerunning it; further null submissions paused (arrays held).
- **A2** HRF: fitted, but never independently per voxel (user, 2026-11-05). Candidates are one per subject × ROI and possibly one per subject. Canonical and the existing per-voxel fits are reference points only. Held-out R², Gaussian, DoG, DN (`retsupp/modeling/cv/fit_prf_cv_roi_hrf.py`); cache on all 30 subjects.
- **A3** Voxel selection: the working choice is **cross-run reliability only** (model-free, per fold on training runs, `retsupp/modeling/cv/prf_cv_reliability.py`). Compared for the memo against the current rule, the current rule restricted to reliable voxels, and R² ≥ .1. Voxel count, % contralateral, share below R² = .1, dS. Whether fitting and scoring use the same set.
- **A4** pRF sanity per model and HRF (contralaterality, V1 σ vs eccentricity); test DoG centre σ, HRF, aperture, fixation.
- **A5** `notes/decision_memo_gain_inputs_2026-11.md`, then STOP.

## Phase B: pre-specify, before any new null output

`notes/primary_test_2026-11.md`:

- **Primary:** group dS from m7_full against the HP-rotation null, one-sided (dS < 0), DoG and DN. Subject value is the median over V1, V2, V3, hV4, VO, LO, TO, V3AB, IPS. Claimed only if both models are negative and DoG passes; the DN threshold is stated in advance.
- **Secondary:** per-ROI dS, dD, LOSO.
- **Secondary, gain-free:** bar-drive β at HP vs LP from decoding with a gainless pRF model. Test, ROI grouping (early, V1-V3) and direction fixed here, before the rerun.
- The HRF and voxel selection chosen in Phase A.

## Phase C: one final rerun on the chosen inputs

- **C1** Ladder m00-m7, voxels ≤ 20% BOLD SD, DoG and DN.
- **C2** LOSO for every step.
- **C3** HP-rotation null for m5, m6, m7, 100,000 draws; observed, null, p, mean of rotated values.
- **C4** Within-condition CV (`run3`), and the power simulation on it.
- **C5** DN vs DoG disagreement in LOSO for dS, and `item_scale` under DN.
- **C6** Decoding with the gainless pRF model on the chosen inputs, all 30 subjects, with the pre-specified test and a rotation null, using pRFs fitted on other runs than those decoded. Proposed design, not yet agreed: stimulus outline fixed and pixel intensities free, as one intensity map per HP condition (test), with an unconstrained per-TR decode for the example frames (the existing method, about 3,240 decodes, 4,300 core-hours; needs OK). First explain why the May analysis has n = 22-26 and why its figure and table differ.

## Phase D: analyses that shape the story

- **D1** D on top of T: LOSO for m3b_TD vs m1_T.
- **D2** Target gain split by target at HP vs LP: parameter difference and LOSO against m7.
- **D3** Hierarchy profile: per-ROI table, joint and conditional; hierarchical term × ROI model, ordered by hierarchy and by pRF size; per-ROI Bayes factors or equivalence bounds for dS.

## Phase E: figure prototypes, in parallel with Phase A

From current fits, gain panels stamped "Provisional: current inputs", input fits as one argument. Scripts in `retsupp/visualize/paper/`, output in `notes/figures/paper_2026-11/` (180 or 88 mm), legend `.md` beside each, combined `all_figures.pdf`.

- **Fig. 1** Paradigm and behaviour.
- **Fig. 2** Forward model, gain vs effective shift, ladder.
- **Fig. 3** pRF maps, held-out R² per ROI, σ vs eccentricity and % contralateral, example voxel, LOSO per step.
- **Fig. 4** Hierarchy profile: terms over ROIs, term × ROI, Bayes factors, σ_AF and σ_dyn.
- **Fig. 5** Sustained gain at HP: a dS per ROI; b dS against the null; c dS and LOSO by sensitivity bin; d power simulation.
- **Fig. 6** Decoding with a gainless pRF model: actual decoded stimulus frames, then decoded intensity at HP vs LP. Design to settle: see C6.
- **Table 1** The ladder.
- **Supplement** S2 outlier voxels; S4 dD per ROI; S8 σ vs eccentricity across pRF models and HRFs; S9 contralaterality by R² bin.

## Phase F: optional

Simulate a known dS in the BOLD and recover it; trial-wise gain vs RT on the Phase C fits; whole-brain DN maps (19 of 30, avoid nodes u24-chaihm0-651 and -636).
