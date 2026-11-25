## Audit: figures, tables and numbers not on the joint zero-sum fits

Rule (fixed 25 Nov): joint, zero-sum gain-field fits only (main model m5_TDS_dS_zs: item weight, target, distractor, zero-sum sustained field, pRFs and gains fitted jointly by LM; view root `..._zs_lm_ref`). Estimated-alone terms and non-zero-sum fits are dropped. I checked the update 12 to 14 packages, the figure legends (.md), the generating scripts and their default roots. Costs are rough: they are scaled from item 3a (8414003/8414004: 705 + 381 CPU-h for a four-model ladder over 241 cells) and from the zero-sum refit. "OK" = already on the joint zero-sum fits.

### Main figures

| Figure | What it uses now | Zero-sum joint version exists? | Rerun needed |
|---|---|---|---|
| Fig. 1 v6 (behaviour) | Behaviour only | n/a | none |
| Fig. 2 (pRF fits) | pRFs without gain fields (DoG, Gaussian, DN m0) | n/a (not a gain-field fit) | none |
| Fig. 3 (item locations) | Data only: deconvolved responses, pRFs used only to select voxels | n/a | none. The Discussion's "model predicted −4.9 / −1.3%" next-vs-opposite numbers come from the earlier polished m7_full (`richter_followup_2026-11-11.md`) and need the item-2 pipeline on m5_TDS_dS_zs: `predict_location_timeseries_polished.py` on the view root, minutes per task |
| Fig. 4 (model schematic) | Illustrative gains | n/a | none |
| Fig. 5 update 13 (`fig5_gain_fields_v11_abc_zs_n28_palA`) | Maps and panel c draw dS **estimated alone** (af_fixed_group_mechanisms_zs_lm on m4_TDS_zs); b joint zs | joint zs gains exist | Superseded by the `_bayes` version, but see the next row |
| Fig. 5 update 14 (`..._bayes`, `..._hpdiag_bayes`) | Joint-fit gains **re-solved with g_HP, g_LP and the two distractor gains freed** (`dS_meta_zs_lm_2026-11-25`), so not the zero-sum, tied-distractor main model. The legend's "Values used" also still says "dS estimated alone ..., joint-fit dS ..." | Yes: the zero-sum constrained re-solve of the same sufficient statistics (being redone) | Rerun `dS_meta` with the constraint (local, minutes; Bayes ~10 min in the bauer env), re-render, fix the legend's "Values used" sentence |
| Fig. 6 a to d, c per area (decoding) | Decoder pRFs without gain terms (fold fits) | n/a | none for the data panels |
| Fig. 6 e, f (gain-field model prediction) | Simulation from the **polished m7_full** (non-zero-sum, with dD); `simulate_decode.py --mode fitted`, `simulate_decode_fitted.sh` default `LADDER_ROOT=..._roihrf_polished` | No | Rerun `simulate_decode_fitted.sh` with `LADDER_ROOT=<zs view root>` (model m5_TDS_dS_zs: needs the `m7_full` lookup pointed at the main model, which the view root already does), then `collect_simulate_decode --fitted` and `fig6_decoding plot`. 28 × 12 runs × V1 to V3, low tens of CPU-h. Manuscript sentences "−1.09 ± 0.44% predicted", "r = .57", "−0.34 ± 0.18% predicted" change |
| Fig. 7 update 13 / `_studentt` | Trial-wise gains on the zero-sum main model | OK | none |

### Supplementary figures and tables

| Item | What it uses now | Zero-sum joint version exists? | Rerun needed |
|---|---|---|---|
| figS_transient_hp_lp_zs_n28_palA (old Fig. 5d) | dD and dT **estimated alone** | Only m7_full_zs (joint dD, with dS); no zero-sum fit with dT | Drop, or show joint dD from m7_full_zs (exists). A joint dT needs a zero-sum m8_full_dT ladder model: code change plus ~150 to 250 CPU-h |
| figS_pooled_dS_dD_zs_lm_2026-11-13 | Mixes estimated-alone dS/dD with joint dS/dD | Joint parts exist | Re-render with joint only (`fig5_gain_fields --outputs pooled` minus the alone columns), local |
| figS_hierarchy_gain_terms_zs / table_hierarchy_gain_terms_zs | DoG target, distractor and joint dS are zs; rows "dS, dD, dT estimated alone" and "shared sustained gain estimated alone" use estimated-alone terms; **Gaussian and DN columns are earlier non-zero-sum fits** (pRFs held, estimated alone) | DoG joint yes; Gaussian/DN no | Drop the alone rows; the Gaussian/DN columns need zero-sum joint fits for those families (check 4 of update 3) |
| figS_gain_vs_prf_size_zs | Joint zero-sum target/distractor | OK | none |
| figS_width_profiles_zs | Zero-sum fits | OK | none |
| table_parameters_2026-11_zs | DoG joint zs (ignore its "LP joint" and "dD joint" rows); **DN columns: DN fits with pRFs held, not refitted, not zero-sum**; "DoG conditional" column = estimated alone | DoG yes; DN no | Drop the conditional column; replace the DN columns after the DN zero-sum fits |
| figS_dS_overview_n28 (update 14) | All 19 versions, deliberately including alone and non-zero-sum | n/a | Under the new rule it is a provenance supplement only, or drop it |
| figS_gain_fields_gauss / figS_gain_fields_dn (S10, S11; update 12) | m7_full on Gaussian / DN pRFs, non-zero-sum, plus estimated-alone dS | No | Gaussian and DN zero-sum joint ladders (m4_TDS_zs, m5_TDS_dS_zs) per family: ~2 × (the DoG zero-sum refit), several hundred CPU-h each, if the ladder supports zero-sum for those families (DN uses `dn_attention_field.py`, to check) |
| figS_dn_gain_locus (S12, DN baseline variants) | DN m7_full + baseline-field variants, earlier, n = 30 | No | Zero-sum DN base model first, then `dnb/dnd/dnbd` variants on it (~1,100 CPU-h as before) |
| figS_baseline_vs_gain (S13) | Earlier joint fit (pre-zero-sum, `fit_baseline_terms.py`) | No | Rerun `fit_baseline_terms.py` on the zs view root, 239 cells, ~1 to 2 h each |
| figS_decode_simulation, figS_decoded_map_model_vs_measured, figS_decoding_eccentricity, figS_decoding_sweep_per_area (model bands) | Simulation from the polished m7_full | No | Same rerun as Fig. 6 e, f |
| figS_field_widths_v3 | Polished fits, free g_HP/g_LP | Replaced by figS_width_profiles_zs | Drop (update 13 already says so) |
| figS_gain_field_map_search_v3 | Joint dD from non-zero-sum fits, alone dS | No | Drop (update 13 already says so) |
| figS_dog_kappa | Earlier joint fit | No | Rerun on the zs view root, small, or drop |
| figS_item_locations_voxel_selection | m7_full refitted (polished) voxel overlap | Partly (items_voxel_overlap_n28_zs_lm_2026-11-16.tsv) | Re-render from the zs table, local |
| figS_loso_per_roi (LOSO) | final2 fits, estimated-alone terms | No | Drop: LOSO goes to Methods/supplement only, and it rests on estimated-alone steps |
| figS_fig7_relnull_r01 / _r02 (update 12) | Polished fits refitted per voxel selection | Tables yes (`trialwise_meta_bayes_relnull_r0.{1,2}_zs_lm_2026-11-24`), figures no | Re-render `fig7_compact` from the zs tables, local |
| LP split variant (item 3, m8_full_adjopp) | final2 m7_full, non-zero-sum | No | Code change plus zero-sum ladder run, ~280 CPU-h (update-1 report, item 3) |
| Items 2/5b, Table 2d (SDT, SDT0) | m8_full_dT, earlier fits | No | Needs zero-sum dT model, see above |
| Items 1a/1c | Linearised on the zs main model (OK), but report "estimated alone" dD rows | OK for the joint rows | Drop the alone rows |
| Hierarchy "dS estimated alone" slope (manuscript −0.7 pp, p = .61) | Estimated alone | Joint zs slope exists (−0.4 pp, p = .75) | Use the joint row or the update-3 hierarchical version |
| **Update 2 hierarchical numbers** (`notes/data/dS_meta_zs_lm_2026-11-25`, handoff update 2, Fig. 5 `_bayes`) | Re-solve with g_HP, g_LP and g_D_HP, g_D_LP free (not zero-sum, distractor not tied) | Zero-sum constrained re-solve possible from the same npz | Being redone (update 3) |

**Already on joint zero-sum fits (no action):** Fig. 7 (both versions), the trial-wise RT analyses (residual RT, joint latent model, the r > 0.2 / 0.1 reruns), items 1a/1c joint rows, 2 and 5b S/SD rows (SD = m7_full_zs), item 3a, figS_width_profiles_zs, figS_gain_vs_prf_size_zs, the DoG columns of the hierarchy and parameter tables.
