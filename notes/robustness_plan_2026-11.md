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

## Started 2026-11-09: Gaussian and DN with their own joint per-ROI HRF

Gaussian is a third pRF family "just to have it". Both Gaussian and DN get their **own** joint per-ROI HRF
(the DoG "final2" recipe), not the DoG table and not the median-of-voxels HRF of model 13.

Code (commit 0f8bd7f): `fit_cv_ladder_nested.py --family gauss` (`gauss_attention_field.py`: the DoG gain-field
classes with the surround amplitude fixed at 0; unit check: zero gains reproduce braincoder's
`GaussianPRF2DWithHRF` to 1e-7, free gains equal the DoG class with srf_amplitude 0 exactly; m00 starts from x, y, sd
of the joint-HRF Gaussian fold fits, fold mean); `cv/fit_prf_cv_roi_hrf_joint.py --family gauss` (canonical parent
model 1, flexible model 3) + `cv/slurm_jobs/fit_prf_cv_roi_hrf_joint.sh`; `decode_final.py --prf-source
gauss_jointhrf | dn_jointhrf` (stage-3 joint fold fits, HRF lower bounds as `jointhrf`), also `gauss_roihrf` (model 13).

Jobs (account hare.econ.uzh; task id of the joint fits = 10·subject + fold; ladders: one task per subject × ROI,
nine ROIs V1 V2 V3 V3AB hV4 LO TO VO IPS, `CV_SCHEME=full BOUNDED=1 MAX_BOLD_SD=20 RELIABLE_ONLY=1`; decodes: 360
subject × run tasks, nine ROIs, free + outline decoders, 4 CPUs, 12 GB, 45 min, as the DN run 8200840):

| What | Job | Tasks | Depends on | Output |
|---|---|---|---|---|
| Joint HRF, Gaussian, canary sub-03 fold 0 | 8206627 | 1 | — | `prf_cv_roi_jointhrf/sub-XX/*_gauss_final2_*` |
| Joint HRF, Gaussian, rest | 8207193 | 89 (16 CPUs, 24 GB, 4 h) | — | same |
| Joint HRF, DN, canary sub-03 fold 0 | 8206628 | 1 | — | `*_dn_final2_*` |
| Joint HRF, DN, rest | 8207235 | 89 (16 CPUs, 24 GB, 4.5 h) | — | same |
| HRF tables (`build_roi_hrf_table --fill-canonical`) | 8207248 (gauss), 8207249 (dn) | 1 each | afterany on the joint fits | `paper_2026-11/roi_hrf_{gauss,dn}_final2.tsv` |
| Ladder canary, Gaussian, sub-03 | 8208242 (table 8208239) | 9 | sub-03 joint folds | `af_prf_cv_ladder_nested_full_bounded_maxsd20_gauss_reliable_roihrf` |
| Ladder, Gaussian | 8208296 | 261 | table 8207248 + canary | same root |
| Ladder canary, DN, sub-03 (warm start from the `..._maxsd20_dn` root) | 8208299 (table 8208298) | 9 | sub-03 DN joint folds | `af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_reliable_roihrf` |
| Ladder, DN | 8208300 | 261 | table 8207249 + canary | same root |
| Decode canary + `--check-prfs`, Gaussian, sub-03 ses-1 run-1 | 8208260, 8208261 | 1 + 1 | sub-03 joint folds | `decoded_final_2026-11/gauss_jointhrf/` |
| Decode, Gaussian | 8208297 | 359 | joint fits + canary | same |
| Decode canary + check, DN | 8208301, 8208302 | 1 + 1 | sub-03 DN joint folds | `decoded_final_2026-11/dn_jointhrf/` |
| Decode, DN | 8208303 | 359 | joint fits + canary | same |

The sub-03 canary ladders use a table built from the same fits (`roi_hrf_*_final2_sub03canary.tsv`; the sub-03 rows
equal those of the final table), so their outputs are the final sub-03 fits. The model-13 (median-of-voxels HRF)
Gaussian fold fits (8206357) were stopped after the user chose joint HRFs; only sub-01 and sub-03 fold 0 exist, so
`--prf-source gauss_roihrf` is not run.

Remaining for Gaussian (and DN) after the jobs: check the joint-HRF summaries (stage-1 mismatch 0, HRF bounds, pRF
shifts), compare the three HRF tables; collect the ladders (the DoG collect scripts with the `_gauss` / `_dn` roots)
and decodes (`compare_decoding_prf_models.py` with `gauss_jointhrf`, `dn_jointhrf`); Fig. 3 re-export + deconvolution
with Gaussian (and DN) pRF centres (joint-HRF fold fits); supplementary figures per main figure where something
changes, otherwise a table.
