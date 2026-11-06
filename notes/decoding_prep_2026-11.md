# Decoding (step 7): preparation, 2026-11-06

Investigation only; nothing submitted. Pre-specified test: `notes/primary_test_2026-11.md`.

## 1. Why the May analysis had n = 22-26, and why its figure and table differ

The May table (`notes/figures/decode_sweep/m4/group_bar_quadrant_stats_vox2000_psig0.5.tsv`, made by
`notes/test_static_quadrant.py`) was computed **locally** from `/data/ds-retsupp/derivatives/decoded/model4/`,
which was an incomplete copy of the cluster outputs. Three causes stack:

| Cause | Subjects | Effect |
|---|---|---|
| Excluded by design in `notes/run_static_dynamic_glm.py` (`EXCLUDE = {1, 2}`, old block-order worry; `Subject.get_hpd_locations` now handles their HP labels) | sub-01, sub-02 | −2 everywhere |
| Never decoded: their m4 refit (R², 2026-05-15) and `p_signal` (2026-05-16) finished after the decode jobs ran on 2026-05-15, and the decoder needs `p_signal` | sub-05, sub-10 | −2 everywhere |
| Decoded on the cluster, but the first decode attempts failed (191, 110, 77 failed tasks on 2026-05-15) and the reruns finished after the local sync, so the local copy lacks them | sub-07 (V1, V2, V3), sub-12 (V1, V2), sub-14 (V1) | V1 −3, V2 −2, V3 −1 |

Result: V1 n = 23, V2 n = 24, V3 n = 25, matching the table. On the cluster, 28 subjects have all 12 runs
(11 for sub-20 and sub-24) for V1, V2 and V3; only sub-05 and sub-10 are missing.

**Figure vs table.** The overlay figure (`group_bar_quadrant_overlay_vox2000_psig0.5.pdf`, 17:43) was
made 12 minutes after the table (17:31) by `notes/render_quadrant_pies.py`, which smooths each subject's
β-map spatially before taking the quadrant means. The table uses unsmoothed maps. Same subjects and same
t-test, different input maps, hence V1 orthogonal p = .020 (figure) vs .038 (table).

## 2. Can StimulusFitter fix the outline and leave intensities free?

Yes. `StimulusFitter.fit(..., mask=...)` in `libs/braincoder/braincoder/optimize.py` (submodule at
a4fc80b) accepts a 3-D boolean mask of shape (frames, R, R): only pixels inside the mask are free
parameters; every other pixel is fixed at 0 (`tensor_scatter_nd_add` into an empty stimulus). So the
true bar footprint per frame (from `Subject.get_bar_stimulus`) as the mask gives "outline fixed,
intensities free". Limits: masking is not implemented for stimuli with several bijectors (not our case),
and the 2-D mask branch has a shape-check bug (`len(mask) == init_pars.shape`); use the 3-D branch.

## 3. What the final pipeline needs

| Piece | Status |
|---|---|
| pRFs fitted on runs other than those decoded | **Exists**: `derivatives/prf_cv_roi/` has per-fold parameters (3 folds, each holding out one run per HP condition, about 4 of 12 runs) for models 1-6, all 30 subjects (90 pickles per model). Decode each held-out run with its fold's pRFs. Caveat: the per-fold fits are warm-started from the all-run mean fit (README of `modeling/cv`), a small optimism. |
| Chosen HRF | Pending the HRF test. Per subject × ROI HRF fits (models 23/24/26 in `prf_cv_roi_sharedhrf`) exist for sub-03 only; if the rule picks them, decoding needs those per-fold fits for all 30 subjects. Canonical HRF: model 2 (DoG) per-fold fits are ready. |
| Voxel selection: cross-run reliability | **Exists**: `derivatives/prf_cv_roi_reliability/sub-XX_fold-K_reliability.tsv.gz`, 30 subjects × 3 folds, computed on training runs. `decode.py` currently selects by `p_signal` or top-N R²; it needs a reliability option and a loader for per-fold parameters (`decode.py` reads the whole-brain m4 NIfTIs). |
| Bar-only paradigm, `sd_min = 0.2`, L2 = 1.0, lr 0.5, ResidualFitter 2000 iterations | Existing calibrated defaults in `retsupp/decode/decode.py`; keep them. `sd_min` must match the pRF fits used: check the `prf_cv_roi` fits' floor before decoding. |
| Outline-fixed decoder | Needs the 3-D mask passed from `decode.py` (not wired in yet). |
| Readout for the test | Define "bar drive at the HP location": the ring (4°) lies outside the bar aperture (3.17°), so the May analysis used the quadrant towards HP within r < 5°. The pre-specification should name the region (quadrant towards HP vs the three others, within the aperture) and whether drive is normalised by the true bar presence. |
| sub-01, sub-02 | Include (their HP labels come from `get_hpd_locations`). |

## 4. Cost

May timing of the validated pipeline (`retsupp/decode/decode.py`, one task = one subject × run × ROI,
ResidualFitter + StimulusFitter, 8 CPUs, ≤ 2,000 voxels): 5,135 completed tasks, mean 1.4 min, about
0.19 core-hours per task (sacct, 14-17 May).

| Design | Tasks | Core-hours |
|---|---|---|
| One decoder, 30 subjects × all runs (358) × V1, V2, V3 | 1,074 | about 200 |
| Both decoders (unconstrained + outline-fixed) | 2,148 | about 400 |
| Cheapest that answers the pre-specified test: outline-fixed only, plus unconstrained for a few example runs (3 subjects × 1 run × 3 ROIs) | about 1,083 | about 200 |

The "4,300 core-hours per decoder" in the plan does not match the May sacct record; it may have assumed
the slower `decoder.py` path (about 17 min per run, per `decode_drive_fast.py`) or all eight ROIs. Expect
some variance with the voxel counts that reliability selection gives (May used ≤ 2,000 voxels).
