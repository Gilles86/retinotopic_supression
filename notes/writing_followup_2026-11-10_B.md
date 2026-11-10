# Writing follow-up, part B: figure changes (2026-11-10)

Part B of the follow-up request (figures); part A (lookups) is a separate file. No analysis, fit or number was changed:
every v3 figure is re-drawn from the same tables as its predecessor. Old versions are kept; the v3 files are in
`/Users/gdehol/git/retsupp/notes/figures/paper_2026-11/` and copied (added) to
`/Users/gdehol/git/retsupp/notes/writing_package_2026-11-10/figures/`.

## B1. Fig. 1c–f: within-subject differences visible

- **Files:** `fig1_paradigm_behaviour_v3.{pdf,png,md}`; code `retsupp/visualize/paper/fig1_paradigm_behaviour.py`
  (`--strips --name fig1_paradigm_behaviour_v3`).
- Raw RTs kept as they were (same y-range; 20% of subject values are off-scale triangles, as before). Under each panel,
  one horizontal strip per bracket with the within-subject difference: subject dots, the median and its bootstrapped 95%
  CI (5000 resamples), a zero line, and the count of subjects on the median's side. All strips share one scale
  (−105 to +105 ms) and one physical width. Chosen over widening the y-range because at 650–1200 ms all points fit but the
  20–40 ms effects disappear in the between-subject spread.
- **Annotations now medians** (the statistic of the Wilcoxon test). The part-A change had not reached
  `fig1_paradigm_behaviour.py` when I rendered (no commit, file unmodified), so I made it myself: the bracket labels use
  `diff.median()`.
- Strip values (median [95% CI], subjects on the median's side): HP − LP distractor −26.1 ms [−34.4, −20.9], 30/30
  faster; LP − absent +43.7 ms [+33.7, +52.7], 28/30 slower; target HP − LP +26.8 ms [+19.9, +37.5], 26/30 slower; target
  repeat − switch −38.4 ms [−47.2, −28.4], 28/30 faster; distractor repeat − switch −15.1 ms [−27.4, −5.9], 21/30 faster.
  Counts match key_numbers.md.
- Legend: the c–f sentence now says the brackets give the median difference, and describes the strips.

## B2. Fig. 3: short legend, statistics to a supplementary table

- **Files:** `fig3_item_locations_v3.{pdf,png,md}` (panels as `fig3_item_locations_final2`, re-rendered only for the
  6-pt minimum, B6); new `notes/writing_package_2026-11-10/figures/tableS_item_locations.md`.
- Legend keeps: what each panel shows, the b, c tests (values; all p < .001), the early-area ANOVA (location type
  F(2, 58) = 6.28, p = .003; × item type F(4, 116) = 0.82, p = .514), the early planned contrasts, and the all-areas
  contrasts that the brackets in e show (added: the earlier legend gave them only in the long list).
- Table S: (1) location type × item type and planned contrasts for each pathway and all areas; (2) per item type;
  (3) HP against all LP locations; (4) DCT basis. All numbers copied from `fig3_item_locations_final2.md`.
- Dropped the cross-reference "colours and order as Fig. 6d, g" (Fig. 6 is re-lettered, B4).

## B3. Fig. 5b: schematic and FWHM plot separated

- **Files:** `fig5_gain_fields_v3.{pdf,png,md}`, new `figS_field_widths_v3.{pdf,png,md}`; code
  `retsupp/visualize/paper/fig5_gain_fields.py` (`--widths-supplement`, with the final2 arguments and `--suffix _v3`).
- Chosen: the FWHM plot to the supplement, so the main-figure letters a–d stay as in key_numbers.md and the text. Panel b
  is now only the to-scale sustained field ("The sustained field, to scale"), larger, with its key (FWHM median 3.3°;
  median per area).
- In `figS_field_widths_v3`, fits at a σ bound are a count per area in the series colour (above the axis: upper bound;
  below: lower bound) instead of rows of open triangles; they are not drawn as dots. Medians and CIs still include them,
  as before (said in the legend). The FWHM and at-bound numbers moved from the Fig. 5 legend into this legend unchanged.

## B4. Fig. 6: panel d to the supplement, larger insets

- **Files:** `fig6_decoding_v3.{pdf,png,md}`, new `figS_decoding_hp_per_area_v3.{pdf,png,md}`; code
  `retsupp/visualize/paper/fig6_decoding.py` (`plot --source jointhrf --name fig6_decoding_v3 --v3`).
- Old d (HP quadrant − other three, per area, both decoders) is its own supplementary figure with the old d legend text.
  Main figure re-lettered: a, b, c unchanged; old e → **d** (quadrants); old f → **e** (by eccentricity); the e sentence
  now reads "the quadrant facing HP minus the three other quadrants, by eccentricity" instead of "the difference of d".
- The c insets are larger (1.12 in instead of 0.7 in) and their labels 7 pt (were 6 pt), at 180 mm width.
- The figure is now 180 mm wide (was 7.25 in = 184 mm).

## B5. Fig. 7: less empty space

- **Files:** `fig7_compact_v3.{pdf,png,md}`; code `retsupp/visualize/paper/fig7_compact.py` (`--compact`).
- Narrower panels (2.2 in instead of 2.75 in); the figure is 152 mm wide instead of 180 mm. Same content: the titles are
  shortened to fit (b: "Lower distractor and global gains on slower trials"), the keys say Target / Distractor / Global.
  A shared y-label column does not apply (the two panels have different units).
- The "Higher on repeats / slow trials" notes went from 5.8 to 6.5 pt; the per-area table in the legend now follows the
  paper's area order (hV4, VO, LO, TO).

## B6. All main figures at 180 mm: text ≥ 6 pt, ROI order and colours

Checked by rendering each figure with a hook that lists every visible text and its size (no outputs written), and the
PDF page sizes:

| Figure | Width | Smallest text before → after | Fixed |
|---|---|---|---|
| 1 (v3) | 180 mm | 5.8 → 6.0 pt | trial-phase labels in b |
| 2 (v3) | 180 mm | 5.5 → 6.0 pt | hemisphere labels of the angle key, n per area in b |
| 3 (v3) | 180 mm | 5.5 → 6.0 pt | "Time" and "pRF within 2° of an item" in the a inset |
| 4 (`fig4_model_hp_blocks`) | 180 mm | 6.5 pt | nothing needed, no v3 |
| 5 (v3) | 180 mm | 5.5 → 6.0 pt | HP / LP labels in the b sketch |
| 6 (v3) | 180 mm | 5.5 → 6.0 pt | aperture-edge label in e |
| 7 (v3) | 152 mm | 6.0 pt | (5.8-pt notes raised, B5) |
| figS_field_widths_v3, figS_decoding_hp_per_area_v3 | 180, 135 mm | 6.0 pt | key in the latter was 5.5 pt and clipped |

- ROI order: every per-area axis uses `paper_style.ROI_ORDER` (V1, V2, V3 | V3AB, IPS | hV4, VO, LO, TO) with the
  pathway brackets (Figs. 2b, 2d, 5, 6b, S decoding per area, S field widths). Fig. 7 lists areas only in its legend
  table, now in the same order.
- Colours: areas use `paper_style.ROI_COLORS` wherever an area is coloured (Fig. 2a–c); pathways in Fig. 3 use the
  colour of their first area (early = V1 blue, dorsal = V3AB purple, occipitotemporal = LO coral); HP red, target orange,
  distractor blue throughout. One possible confusion, not changed: in Fig. 2d the "DoG pRF" series is teal, the same
  teal as V2 in Fig. 2a–c.
- Font sizes were raised in the shared figure scripts, so the earlier versions would also change if re-rendered; the
  existing earlier files were not touched.

## What contradicts or changes references in key_numbers.md / README.md

No number changed. References that now differ:

1. Fig. 1: the figure now prints medians (−26, +44, +27, −38, −15 ms); key_numbers.md gives means for the first four
   rows (−27.8, +43.0, +28.4 ms; "HP vs absent" +15.2 ms is not in the figure) and medians for the repeat rows. Part A
   handles the statistic choice in key_numbers.
2. Fig. 5: "b, field widths (FWHM)" in key_numbers.md (line 89) now belongs to the supplementary figure
   `figS_field_widths_v3`; Fig. 5b is the to-scale sketch only.
3. Fig. 6 panel letters in key_numbers.md (section "Fig. 6 Decoding"): old d → supplementary figure
   `figS_decoding_hp_per_area_v3`; old e → d; old f → e. README figure table still names `fig6_decoding`.
4. README figure table names `fig1_paradigm_behaviour`, `fig2_prf_fits_final2`, `fig3_item_locations_final2`,
   `fig5_gain_fields_final2`, `fig6_decoding`, `fig7_compact_v2`; the v3 files are the ones to use now.
