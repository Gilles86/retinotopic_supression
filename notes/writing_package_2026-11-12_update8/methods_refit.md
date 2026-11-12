# Methods: pRFs refitted together with the gain fields (update 8)

Replace the sentence(s) saying that the gain fields were fitted with each voxel's pRF held at the fit without gain
fields. Suggested text:

"The gain-field models were fitted in two steps. First, the shared gain-field terms were fitted with each voxel's pRF
(centre, size, amplitude, baseline, and surround amplitude and size) held at its fit without gain fields. Second,
starting from that optimum, the per-voxel pRF parameters were refitted together with the shared terms: two rounds of
alternating optimization, each round first updating the per-voxel pRF parameters with the shared terms fixed (Adam),
then the shared terms with the pRFs fixed (bounded L-BFGS), keeping a step only when it lowered the training error. The
HRF (one per participant and area) stayed fixed. This was done for the model with all terms (the joint fit) and for the
model with item weight, target, distractor and sustained gain on which each HP − LP term was estimated alone, each from
its own optimum. The refit lowered the training error in every participant × area fit (median −0.03%; the smallest
change −0.002%) and moved the pRF centres little (median 0.009°; 0.04° for voxels with a centre within 2° of an item
location)."

Notes for the authors (not for the text):
- The refit was not run to full convergence (the training error was still falling slightly after two rounds); the
  changes in the second round were small.
- The models without HP − LP terms (bar only, pRF only, target, distractor, sustained, target + distractor) were not
  refitted; the leave-one-subject-out tests, the trial-wise gains (Fig. 7) and the Gaussian and DN robustness fits use
  the earlier fits with the pRFs held fixed (see UPDATE_NOTES.md, "Not regenerated").
- Source: `notes/prf_refit_with_gains_2026-11-12.md`; data `notes/data/prf_refit_2026-11-12_shared.tsv`,
  `notes/data/prf_refit_2026-11-12_prf_shift.tsv`; code `retsupp/modeling/fit_cv_ladder_nested.py --polish-method alt`.
