# Paper figures (2026-11)

Current versions only; each has a PDF, a PNG and its legend (`.md`). Superseded versions are in `_archive/`.
Scripts are in `retsupp/visualize/paper/`. "Provisional" figures use the current gain fits and are redrawn after the
final refit.

Order since 2026-11-09 (user): measured results before the modelling, so pRF fits are Fig. 2, the item-location
responses Fig. 3 and the model Fig. 4.

| Figure | File | Script | Status |
|---|---|---|---|
| 1 Paradigm and behaviour | `fig1_paradigm_behaviour` (c, d response times by HP/LP; e, f repeat vs switch) | `fig1_paradigm_behaviour.py` | final |
| 2 pRF fits | `fig2_prf_fits` | `fig2_prf_fits.py` | provisional (d: HRF choice) |
| 3 Responses at the item locations (model-free): target and distractor vs neutral item, HP vs LP per item type | `fig3_item_locations` | `fig3_item_locations.py` (statistics for the DCT basis in its legend) | final apart from layout |
| 4 Gain-field model | `fig4_model_hp_blocks` | `fig4_model.py` | final |
| 5 Gain fields: target and distractor gain, sustained and distractor HP − LP gain, the field to scale, leave-one-subject-out test per ROI (e) | `fig5_gain_fields` | `fig5_gain_fields.py` | provisional |
| Responses at a pRF's item location (model-free; place to be decided) | `fig_event_responses_2x3_deconv` | `fig_event_responses_2x3.py --deconv notes/data/event_deconv_2026-11` | final apart from layout |
| 6 Decoding | not made yet | `fig6_decoding.py` | pending (final pRF fold fits) |
| 7 The behavioural effects of Fig. 1 in the gains (a location HP − LP, b history repeat − switch; target and distractor side by side), and gain vs response time trial by trial (c) | `fig7_behaviour` | `fig7_behaviour.py` | prototype: a provisional (current fits), b, c on the October fits |
| Table 1 The ladder | `table1_ladder` | `table1_ladder.py` | provisional |
| Table S Parameters | `table_parameters_2026-11` | written by `fig5_gain_fields.py` | provisional |
| S Gain against pRF size (was Fig. 5a) | `figS_gain_vs_prf_size` | `fig5_gain_fields.py` | provisional |
| S Field sizes | `figS_field_size` | `fig5_gain_fields.py` | provisional |
| S Within-subject CV power | `figS_cv_power` | `fig5_gain_fields.py` | provisional |
| S dS and dD pooled over ROIs | `figS_pooled_dS_dD` | `fig5_gain_fields.py` | provisional |
| S Fitted gain fields | `figS_gainfield_size` | see its legend | provisional |

`_archive/` (numbers there are the order before 2026-11-09: old Fig. 2 = model, 3 = pRF fits, 4 = item locations): Figs. 4 and 5 before the split by question (`fig4_hierarchy`, `fig5_sustained_hp`), earlier Fig. 2 (pRF-location version), Fig. 3 versions with the example voxel, the voxel scans, the
event-average prototypes (plain averages, before deconvolution) and the joint bar-and-items variant of the event figure.
