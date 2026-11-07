# Paper figures (2026-11)

Current versions only; each has a PDF, a PNG and its legend (`.md`). Superseded versions are in `_archive/`.
Scripts are in `retsupp/visualize/paper/`. "Provisional" figures use the current gain fits and are redrawn after the
final refit.

| Figure | File | Script | Status |
|---|---|---|---|
| 1 Paradigm and behaviour | `fig1_paradigm_behaviour` (c, d as RTs) / `fig1_paradigm_behaviour_diff` (c, d as within-subject differences) | `fig1_paradigm_behaviour.py` (`--diff`) | final; choose the c, d version |
| 2 Gain-field model | `fig2_model_hp_blocks` | `fig2_model.py` | final |
| 3 pRF fits | `fig3_prf_fits` | `fig3_prf_fits.py` | provisional (c: HRF choice) |
| 4 Target and distractor gain along the hierarchy | `fig4_hierarchy` | `fig4_hierarchy.py` | provisional |
| 5 Sustained gain at HP | `fig5_sustained_hp` | `fig5_sustained_hp.py` | provisional; null panel pending |
| Responses at a pRF's item location (model-free; place to be decided) | `fig_event_responses_2x3_deconv` | `fig_event_responses_2x3.py --deconv notes/data/event_deconv_2026-11` | final apart from layout |
| 6 Decoding | not made yet | `fig6_decoding.py` | pending (final pRF fold fits) |
| 7 Trial history and behaviour | `fig7_behaviour` | `fig7_behaviour.py` | prototype on the October fits |
| Table 1 The ladder | `table1_ladder` | `table1_ladder.py` | provisional |
| Table S Parameters | `table_parameters_2026-11` | written by `fig4_hierarchy.py` | provisional |
| S Field sizes | `figS_field_size` | `fig4_hierarchy.py` | provisional |
| S Within-subject CV power | `figS_cv_power` | `fig5_sustained_hp.py` | provisional |
| S Fitted gain fields | `figS_gainfield_size` | see its legend | provisional |

`_archive/`: earlier Fig. 2 (pRF-location version), Fig. 3 versions with the example voxel, the voxel scans, the
event-average prototypes (plain averages, before deconvolution) and the joint bar-and-items variant of the event figure.
