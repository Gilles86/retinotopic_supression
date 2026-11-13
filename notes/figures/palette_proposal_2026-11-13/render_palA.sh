#!/bin/bash
# Re-render every current paper figure and supplement in palette A as new files with the suffix _palA (n = 28).
# Run from the repo root on the laptop. Inputs are the local tables of writing package update 8; nothing is
# overwritten except earlier _palA renders. Flat maps of Fig. 2 (pycortex2 env, ~20 s): notes/figures/fig1_r2low_palA.
set -eo pipefail
PY=${PY:-$HOME/mambaforge/envs/retsupp/bin/python}
EX="$PY -m retsupp.visualize.paper.run_excluding --exclude 1 2 --"
D=notes/data
TAG=polished_2026-11-12
SCR=${SCR:-$(mktemp -d)}          # side tables the figure scripts write; not kept
P=retsupp.visualize.paper

# Fig. 2 flat maps (areas only; angle and eccentricity unchanged)
if [ -n "$FLAT" ]; then
    mkdir -p notes/figures/fig1_r2low_palA
    cp notes/figures/fig1_r2low/sub-23_{angle,ecc}_flat.png notes/figures/fig1_r2low_palA/
    rm -f notes/figures/fig1_r2low_palA/sub-23_{areas_flat.png,area_labels.json,flatwindow_lateral.png,flatwindow_medial.png}
    PYTHONPATH=. $HOME/mambaforge/envs/pycortex2/bin/python retsupp/visualize/pycortex/render_fig3_flat.py 23 \
        --out-dir notes/figures/fig1_r2low_palA
fi

# Main figures
$PY -m $P.fig1_paradigm_behaviour_v5 --exclude-subjects 1 2 --suffix _n28_palA
$PY -m $P.fig1_paradigm_behaviour_v4 --exclude-subjects 1 2 --suffix _n28_palA
$EX $P.fig2_prf_fits --final-prfs --name fig2_prf_fits_n28_palA --map-dir notes/figures/fig1_r2low_palA
$EX $P.fig3_item_locations --name fig3_item_locations_n28_palA --deconv-folder $D/event_deconv_final2_2026-11 \
    --dct-folder $D/event_deconv_dct_final2_2026-11 --rel-folder $D/event_deconv_relative_final2_2026-11 \
    --rel-dct-folder $D/event_deconv_relative_dct_final2_2026-11
$PY -m $P.fig4_model --version blocks --name fig4_model_hp_blocks_palA
$EX $P.fig5_gain_fields_v6 --layout v11 --suffix _n28_palA --gain-table $D/gain_field_parameters_${TAG}.tsv \
    --conditional-table $D/conditional_estimates_${TAG}.tsv --slope-table $D/hierarchy_regression_${TAG}_n28.tsv \
    --widths-table $D/field_widths_reliable_${TAG}_n28_widths_thr2.tsv
$EX $P.fig6_decoding plot --v12 --name fig6_decoding_v13_n28_palA
$PY -m $P.fig7_compact --compact --meta $D/trialwise_meta_bayes_signal_se_${TAG} \
    --param $D/trialwise_signal_se_${TAG}/trialwise_af_signal_se_${TAG}_parametric.tsv --name fig7_compact_polished_n28_palA \
    --fit-label "each subject's joint gain-field fit with the pRFs refitted together with the gain fields (m7: DoG pRFs, one HRF per subject × area, each area's signal voxels)"

# Supplements
$EX $P.fig5_gain_fields --final --percent --suffix "_${TAG}_n28_palA" --gain-table $D/gain_field_parameters_${TAG}.tsv \
    --conditional-table $D/conditional_estimates_${TAG}.tsv --slope-table $D/hierarchy_regression_${TAG}_n28.tsv \
    --dt-table cond --outputs hierarchy pooled \
    --fit-label "gain-field model fits with DoG pRFs, the pRFs refitted together with the gain fields"
$PY -m $P.figS_field_widths_v2 --fits $D/field_widths_reliable_${TAG}_n28_fits.tsv --name figS_field_widths_v3_n28_palA \
    --legend auto --fit-label "joint gain-field model fits with DoG pRFs, the pRFs refitted together with the gain fields"
$EX $P.fig6_decoding plot --v10 --name fig6_decoding_v10_n28_palA_tmp --supp-suffix _v10_n28_palA \
    --ecc-name figS_decoding_eccentricity_n28_palA
rm -f notes/figures/paper_2026-11/fig6_decoding_v10_n28_palA_tmp.{pdf,png,md}
$EX $P.fig6_decoding plot_sweep_per_area --name figS_decoding_sweep_per_area_n28_palA
for st in plot_bar_crossing plot_all_rois plot_all_rois_examples; do $EX $P.fig6_decoding $st --supp-suffix _n28_palA; done
$EX $P.figS_decoded_bars plot_split --n-bins 11 --name figS_decoded_bars_rotated_n28_palA
$EX $P.figS_decoded_bars plot_per_roi --n-bins 11 --per-roi-name figS_decoded_bars_per_roi_n28_palA
$EX $P.compare_decoding_prf_models --figure --sources jointhrf gauss_jointhrf dn_jointhrf --suffix _n28_palA
$EX $P.figS_decode_simulation --suffix _n28_palA
$PY -m $P.figS_decoding_voxel_selection --exclude-subjects 1 2 --name figS_decoding_voxel_selection_n28_palA
$EX $P.figS_decoded_map_model_vs_measured --name figS_decoded_map_model_vs_measured_n28_palA
$EX $P.figS_dn_gain_locus --name figS_dn_gain_locus_n28_palA
$EX $P.figS_dn_params_by_hp --out notes/figures/paper_2026-11/figS_dn_params_by_hp_n28_palA.pdf --stats $SCR/dn_by_hp_stats.tsv
$PY -m $P.figS_dog_kappa --exclude-subjects 1 2 --loso $D/loso_kappa_n28_2026-11-11.tsv --name figS_dog_kappa_n28_palA
$EX $P.figS_baseline_vs_gain --name figS_baseline_vs_gain_n28_palA
$PY -c "import retsupp.visualize.paper.figS_rt_by_location as m; m.NAME = 'figS_rt_by_location_n28_palA'; m.main()"
$PY -c "
import retsupp.visualize.paper.fig3_location_per_area as m
t, v, md = m.main('$D/event_deconv_relative_final2_2026-11', '$SCR/a.tsv', '$SCR/b.tsv')
m.figure(v, t, name='figS_item_locations_per_area_n28_palA')"
$PY -c "
import retsupp.visualize.paper.fig3_voxel_selection as m
m.figure('$D/fig3_voxel_selection_n28_2026-11-12', '/data/ds-retsupp/derivatives/af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished', name='figS_item_locations_voxel_selection_n28_palA')"

# Supplements added later (inputs as read by the n = 28 renders, see notes/data/n28/logs/)
$EX $P.fig6_decoding plot --source canonical --rois V1 V2 V3 --name figS_decoding_canonical_n28_palA
F2="--final --gain-table $D/gain_field_parameters_final2_2026-11.tsv --conditional-table $D/conditional_estimates_dog_final2_2026-11.tsv \
    --slope-table $D/hierarchy_regression_final2_2026-11.tsv --robust-alone-table $D/conditional_exact_final2_allfam_2026-11.tsv --suffix _n28_palA"
$EX $P.fig5_gain_fields --family dn $F2          # figS_gain_fields_dn_n28_palA
$EX $P.fig5_gain_fields --family gauss $F2       # figS_gain_fields_gauss_n28_palA
$EX $P.fig5_gain_fields --outputs loso --loso-table $D/loso_per_roi_final2_2026-11.tsv $F2   # figS_loso_per_roi_n28_palA
rm -f notes/figures/paper_2026-11/table_parameters_2026-11_n28_palA.{md,tsv}   # side product of the three calls above
for r in 0.2 0.1; do
    T=ownvox10_relnull_r${r}_polished_2026-11-13
    $PY -m $P.fig7_compact --compact --meta $D/trialwise_meta_bayes_$T --param $D/trialwise_$T/trialwise_af_${T}_parametric.tsv \
        --name figS_fig7_relnull_r${r/./}_n28_palA --fit-label "each subject's joint gain-field fit on the looser voxel selection (cross-run reliability r > $r against a shift null; pRFs refitted together with the gain fields; m7: DoG pRFs, one HRF per subject × area, the fit's own voxels)"
done
$PY -m $P.figS_hierarchy_gain_terms --prefix $D/hierarchy_gain_terms_${TAG}_n28 --name figS_hierarchy_gain_terms_${TAG}_n28_palA \
    --fits-note "joint gain-field fits with DoG pRFs, the pRFs refitted together with the gain fields"
# Captions of figS_gain_fields_{dn,gauss}, figS_loso_per_roi, fig2, fig3, figS_baseline_vs_gain and figS_decode_simulation:
# the hand-edited n28 captions with their colour words replaced (the scripts' generated captions predate those edits).

# side tables written next to the inputs by some scripts: not part of this set
rm -f $D/*n28_palA*.tsv $D/decode_sim_summary_2026-11_n28_palA.tsv $D/decoding_prf_models_*_n28_palA.tsv
