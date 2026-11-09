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
| 3 Responses at the item locations (model-free): target and distractor vs neutral item; d, e by location type relative to HP (HP, next to HP, opposite; e item types pooled, per item type and HP vs all LP in the legend) | `fig3_item_locations` | `fig3_item_locations.py`; d, e from `deconvolve_location_responses.py --relative` (`notes/data/event_deconv_relative{,_dct}_2026-11/`; statistics for the DCT basis in its legend) | final apart from layout |
| 4 Gain-field model | `fig4_model_hp_blocks` | `fig4_model.py` | final |
| 5 Gain fields: top row a target and distractor gain, b the sustained field to scale; bottom row c sustained HP − LP gain, d distractor and target HP − LP gain (as Fig. 1c, d) | `fig5_gain_fields` | `fig5_gain_fields.py` | provisional |
| Responses at a pRF's item location (model-free; place to be decided) | `fig_event_responses_2x3_deconv` | `fig_event_responses_2x3.py --deconv notes/data/event_deconv_2026-11` | final apart from layout |
| 6 Decoding: the bar decoded from held-out runs in nine ROIs (b), lower decoded intensity in the quadrant facing HP (c, map and three decoded-bar differences; d, HP − other quadrants per ROI, BH across the nine ROIs, bar-outline decoder as open markers; e, per quadrant), by eccentricity (f) | `fig6_decoding` | `fig6_decoding.py` (collect: `fig6_collect.py`) | final inputs (pRFs with an HRF per subject × ROI) |
| 7 Trial history in the gains (a, repeat − switch, as Fig. 1e, f; the location effects are Fig. 5d), and gain vs response time trial by trial (b) | `fig7_behaviour` | `fig7_behaviour.py` | prototype on the October fits |
| Table 1 The ladder | `table1_ladder` | `table1_ladder.py` | provisional |
| Table S Parameters | `table_parameters_2026-11` | written by `fig5_gain_fields.py` | provisional |
| S Gain against pRF size (was Fig. 5a) | `figS_gain_vs_prf_size` | `fig5_gain_fields.py` | provisional |
| S Field sizes | `figS_field_size` | `fig5_gain_fields.py` | provisional |
| S Within-subject CV power | `figS_cv_power` | `fig5_gain_fields.py` | provisional |
| S dS and dD pooled over ROIs | `figS_pooled_dS_dD` | `fig5_gain_fields.py` | provisional |
| S Leave-one-subject-out per ROI and term (was Fig. 5b) | `figS_loso_per_roi` | `fig5_gain_fields.py` | provisional (earlier fits) |
| S Fitted gain fields | `figS_gainfield_size` | see its legend | provisional |
| S Decoding with canonical-HRF pRFs (robustness of Fig. 6) | `figS_decoding_canonical` | `fig6_decoding.py --source canonical --rois V1 V2 V3 --name figS_decoding_canonical` | final |
| S Decoded bars per position, HP rotated to the upper right; vertical and horizontal bars separately, minus the mean over HP locations | `figS_decoded_bars_rotated` | `figS_decoded_bars.py plot_split --n-bins 11` | final |
| S The same per ROI, orientations pooled | `figS_decoded_bars_per_roi` | `figS_decoded_bars.py plot_per_roi --n-bins 11` | final |
| S The bar decoded from each of the nine areas (group-mean frames, as Fig. 6a) | `figS_decoding_all_rois` | `fig6_decoding.py plot_all_rois` | final |
| S The same in example subjects per area (75th, 50th, 25th percentile of r; run means, three time points) | `figS_decoding_all_rois_examples` | `fig6_decoding.py plot_all_rois_examples` | final |
| S Measured response when the bar crosses a pRF near an item location (was Fig. 6g) | `figS_bar_crossing_response` | `fig6_decoding.py plot_bar_crossing` | final; test depends on the baseline choice |

`_archive/` (numbers there are the order before 2026-11-09: old Fig. 2 = model, 3 = pRF fits, 4 = item locations): Figs. 4 and 5 before the split by question (`fig4_hierarchy`, `fig5_sustained_hp`), earlier Fig. 2 (pRF-location version), Fig. 3 versions with the example voxel, the voxel scans, the
event-average prototypes (plain averages, before deconvolution) and the joint bar-and-items variant of the event figure.
