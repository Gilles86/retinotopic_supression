"""Contact sheet of the _palA renders: contact_sheet.py [suffix] (run from the repo root)."""
import subprocess
import sys
from pathlib import Path

SUF = sys.argv[1] if len(sys.argv) > 1 else '_palA'
D = Path('notes/figures/paper_2026-11')
NAMES = ['fig1_paradigm_behaviour_v5_n28', 'fig1_paradigm_behaviour_v4_n28', 'fig2_prf_fits_n28', 'fig3_item_locations_n28', 'fig4_model_hp_blocks',
         'fig5_gain_fields_v11_n28', 'fig6_decoding_v13_n28', 'fig7_compact_polished_n28', 'figS_priming_rt_v4_n28',
         'figS_rt_by_location_n28', 'figS_item_locations_per_area_n28', 'figS_item_locations_voxel_selection_n28',
         'figS_field_widths_v3_n28', 'figS_gain_field_map_search_v3_n28', 'figS_gain_vs_prf_size_polished_2026-11-12_n28',
         'figS_pooled_dS_dD_polished_2026-11-12_n28', 'figS_dog_kappa_n28', 'figS_dn_gain_locus_n28',
         'figS_dn_params_by_hp_n28', 'figS_baseline_vs_gain_n28', 'figS_decoding_all_rois_n28',
         'figS_decoding_all_rois_examples_n28', 'figS_decoding_hp_per_area_v10_n28', 'figS_decoding_eccentricity_n28',
         'figS_decoding_sweep_per_area_n28', 'figS_decoded_bars_rotated_n28', 'figS_decoded_bars_per_roi_n28',
         'figS_decoding_prf_models_n28', 'figS_decode_simulation_n28', 'figS_decoded_map_model_vs_measured_n28',
         'figS_decoding_voxel_selection_n28', 'figS_bar_crossing_response_n28', 'figS_decoding_canonical_n28',
         'figS_loso_per_roi_n28', 'figS_gain_fields_dn_n28', 'figS_gain_fields_gauss_n28',
         'figS_hierarchy_gain_terms_polished_2026-11-12_n28', 'figS_fig7_relnull_r02_n28', 'figS_fig7_relnull_r01_n28']
files = [str(D / f'{n}{SUF}.png') for n in NAMES]
missing = [f for f in files if not Path(f).exists()]
print('missing:', missing)
files = [f for f in files if Path(f).exists()]
out = f'notes/figures/palette_proposal_2026-11-13/contact_sheet{SUF}.png'
subprocess.run(['magick', 'montage', '-font', '/System/Library/Fonts/Helvetica.ttc', '-pointsize', '18', '-label', '%t',
                *files, '-tile', '4x', '-geometry', '900x700+14+14', '-background', 'white', out], check=True)
print(out, len(files))
