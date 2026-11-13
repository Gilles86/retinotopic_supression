# Update 12, 13 Nov 2026: all figures in the new colour scheme (applies on top of updates 8–11)

This replaces every figure the earlier packages contained. The numbers are unchanged, apart from what updates 8–11
already changed. Each figure comes with its caption (`.md`), which differs from the earlier caption in its colour
words, plus the changes listed below.

## Colour scheme (use these words in captions)

| Meaning | Colour |
|---|---|
| HP location, sustained field, HP − LP effects | purple |
| Low-probability locations / next to HP | light purple; opposite HP is a lighter, dashed purple |
| Target, target field | orange |
| Distractor, distractor field | green |
| Neutral item, distractor absent, null, "no change" | grey |
| Bar stimulus | charcoal |
| pRF outline | charcoal outline |
| Decoding difference maps | purple (less) to gold (more) |
| Decoded frames and model input / effective drive images | black–red–yellow (magma) |
| Pathways | line style: early solid, dorsal dashed, occipitotemporal dotted |
| Visual areas | coloured only in Fig. 2 (blue, green and warm families); elsewhere identified by position |

The real display, as drawn in Fig. 1a and Fig. 4a, keeps its true colours: orange items, a green distractor on grey.

## Main figures (`main/`)

- **Fig. 1:** `fig1_paradigm_behaviour_v5_n28_palA`, new layout, condensed to 4.6 in.
  - Panel b is compact.
  - The within-participant difference strips sit beside c and d instead of underneath.
  - e is on the same ms scale with an inline key.
  - The explanatory text moved into the caption.
- **Fig. 2:** `fig2_prf_fits_n28_palA`. Vivid area colours, and viridis for the distance-from-item panel (e).
- **Fig. 3:** `fig3_item_locations_n28_palA`.
  - Panel a is tagged "All nine areas pooled".
  - Panel e prints the exact p for HP − next and HP − opposite in every pathway, with stars where p < .05 (values in
    update 9, section C).
- **Fig. 4:** `fig4_model_hp_blocks_palA`. New colours only.
- **Fig. 5:** two versions, as in update 11.
  - **Recommended:** `fig5_gain_fields_v11_abc_n28_palA`, panels a–c, with panel d as supplementary
    `figS_transient_hp_lp_n28_palA`.
  - **Alternative:** `fig5_gain_fields_v11_n28_palA`, the full a–d.
- **Fig. 6:** `fig6_decoding_v13_n28_palA`. Same layout as v13. HP / next / opposite use the purple shades, and the
  decoding maps are purple–gold.
- **Fig. 7:** `fig7_compact_polished_n28_palA`. The refitted-fit version from update 9, in the new colours.

## Supplementary figures (`supplement/`)

All current supplements are included in the new colours, plus the parameter table
(`table_parameters_2026-11_polished_2026-11-12_n28_palA`).

**Not yet re-rendered in the new colours.** Use the versions from the earlier packages until they follow:
- `figS_hierarchy_gain_terms` (update 10);
- `figS_fig7_relnull_r02` / `_r01` (update 9);
- `figS_decoding_canonical`;
- `figS_loso_per_roi`;
- `figS_gain_fields_dn` / `_gauss`.

## For the writing agent

- **Captions:** replace the figure files and captions with these; refer to colours with the words in the table above.
- **Panel letters:**
  - **Fig. 1:** letters are unchanged, but c, d and e changed layout; check text references to "the strips under c
    and d".
  - **Fig. 5:** if the a–c version is used, references to Fig. 5d go to the new supplementary figure.
