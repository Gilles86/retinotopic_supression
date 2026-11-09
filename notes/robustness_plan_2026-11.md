# Robustness analyses across pRF models (plan, 2026-11-09)

User request (2026-11-09): for every main figure, a robustness version with the same analysis but
**Gaussian** and **DN** pRF models, each with a **per-ROI HRF** (one HRF per subject × ROI). DoG with the
per-ROI (joint) HRF stays the main model. Robustness results go in the supplement, one figure per main
figure where something changes, otherwise a table.

| Figure | What depends on the pRF model | Needed for Gaussian / DN |
|---|---|---|
| 2 pRF fits | pRF maps, held-out R², size vs eccentricity | Fold fits with per-ROI HRF exist for DoG (model 14) and DN (model 16, `prf_cv_roi_sharedhrf`); Gaussian per-ROI-HRF fold fits to check / fit |
| 3 Item locations | Voxel selection (pRF centre within 2° of an item) | Re-export + deconvolution with Gaussian / DN pRF centres (cheap) |
| 4 Model illustration | none | — |
| 5 Gain fields | The whole gain-field ladder | Ladder rerun per model with per-ROI HRF: DN needs its own joint HRF fit first; Gaussian ladder family to add/check |
| 6 Decoding | Encoding model inverted | DN decode running (job 8200840, model 16); Gaussian per-ROI-HRF decode to add |
| 7 Gains and behaviour | Trial-wise gains from the ladder | Follows from the Fig. 5 ladders |
| Supplements | LOSO, split models, field widths, decoded bars | Follow from the ladders / decodes |

Order: (1) per-ROI HRF fits for Gaussian and DN (DN joint HRF fit was already planned after the DoG rerun);
(2) ladders; (3) decodes; (4) re-export for Fig. 3; (5) supplementary figures. Every cluster job is listed
for approval first.
