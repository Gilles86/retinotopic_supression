# Paper colour palette: proposal under option A (2026-11-13)

Status: **proposal for approval. No paper figure has been changed or re-rendered.**

Files:
- Swatch sheet, normal vision plus deuteranopia, protanopia and tritanopia, and a ΔE table of the critical pairs (old vs new): `/Users/gdehol/git/retsupp/notes/figures/palette_proposal_2026-11-13/swatches.png` (`.pdf`)
- Mocks of key elements: `/Users/gdehol/git/retsupp/notes/figures/palette_proposal_2026-11-13/mocks.png` (`.pdf`). They show a Fig. 4a-like field row, Fig. 3d/e-like HP/next/opposite panels in both LP variants, a Fig. 1c-like behaviour row, a Fig. 6e/f-like decoding map and quadrant strip, Fig. 3b-like pathway line styles and the Fig. 2a area families.
- The same mocks under the three CVD simulations: `/Users/gdehol/git/retsupp/notes/figures/palette_proposal_2026-11-13/mocks_cvd.png`
- All pairwise ΔE values: `/Users/gdehol/git/retsupp/notes/figures/palette_proposal_2026-11-13/cvd_pairs.tsv`
- Code: `make_palette_proposal.py` and `cvd_check.py` (Machado et al. 2009 matrices at severity 1, CIEDE2000 via scikit-image) in the same folder.

## 1. Principles

1. **One colour, one meaning, in every figure.** A hue that stands for a condition (HP, target, distractor) is never reused for a grouping (pathway, area), a sign (more or less) or significance.
2. **Location probability is one hue family, and that hue is the sustained field.** HP is saturated purple. LP is a lighter tint of the same purple, because the sustained field sits at all four locations and is only weaker at LP. Distance from HP is shown by lightness: next to HP is the mid tint, opposite is the light tint.
3. **The transient fields keep their item colours:** target orange, distractor sky blue.
4. **Achromatic means "no model term".** The bar stimulus, model input, effective drive, decoded frames, the neutral item, the "absent" condition, the null and the generic model line are all black, grey or white. Magma goes, because its mid-tones are the sustained-field purple.
5. **Groupings do not take hues.** Pathways are shown by line style, or by the grey brackets that already sit under the ROI axis. Areas get colour only on the Fig. 2 anatomy, in low-chroma families that avoid the condition hues.
6. **Significance is never a hue.** Stars and brackets take the colour of the quantity they test, or FG for a comparison between conditions. In Fig. 7, solid versus pale already marks significance.
7. **Real stimulus colours appear only in depictions of the real display** (Fig. 1a and the "On screen" panel of Fig. 4a). Their labels are drawn in FG, not in the model-term colours (see conflict C4).

## 2. Inventory: colours that carry meaning now

| Colour (hex) | Constant / where defined | Meaning(s) now | Where | Conflict |
|---|---|---|---|---|
| Red `#D62828` | `paper_style.HP`; duplicated in `plot_fig1.HP`, `fig6_decoding.LOC_COLORS['hp']`, `figS_decode_simulation.COLS` | HP location; sustained field; HP − LP effects; HP quadrant in decoding; significance stars in Fig. 6c/e/f; FEF on the old area map (`render_fig1_prf.AREA_COLORS`) | Figs. 1–6, S | **C1**: close to target orange, and both turn olive/yellow in deuteranopia (ΔE 16, swatch table). Also used as a significance colour (Fig. 6 stars), so "red = significant" and "red = HP" mix. |
| Pale red `#F6D4D4`, `#F4A6A6`, `#E9A0A0`, `#7A1010` | `fig6_decoding` (HP quadrant fill), `fig5_gain_fields_v6.HP_CMAP`, `figS_dn_params_by_hp.REL_COLORS['Next']` | HP tint; sustained-field map; next-to-HP (one S figure only) | Fig. 5a, 6d–f, S | Next to HP is a red tint in one supplement but dark grey everywhere else (**two colours for one meaning**). |
| Orange `#F77F00` | `TARGET` (`TARGET_C` in `plot_fig1`) | Target, target field | Figs. 1, 3, 4, 5, 7 | C1 with HP. **C4**: the real items on screen are orange too (`ITEM_A #E39100`, all items), so in Fig. 1a the orange "Target" label points at a bar that has the same colour as every other item. |
| Blue `#0077B6` | `DISTRACTOR`; `ROI_COLORS['IPS']`; `AREA_COLORS['IPS']` | Distractor and distractor field; **also IPS** | Figs. 1, 3, 4, 5, 7; Fig. 2 (IPS) | **C2**: one hue for a condition and for an area. |
| Slate `#3D5A80` | `STIM`; `ROI_COLORS['V1']`; `PATHWAY_COL['Early']`; `figS_dn_gain_locus.INPUT` | Bar stimulus; decoded-vs-true points; **also V1 and the Early pathway** | Figs. 2, 3b/c, 6a/b | **C3**: stimulus = V1 = Early. In Fig. 6 the slate bar sits in the same blue family as the distractor. |
| Teal `#2A9D8F` | `PRF_TEAL` (fig2, fig3, fig4, fig6); `ROI_COLORS['V2']`; `fig5_gain_fields_v6.ZERO`; `figS_dn_gain_locus.BFIELD`; BrBG "more" end (Fig. 6e) | pRF outline; DoG pRF model step (Fig. 2d); **also V2**; "no change" line (Fig. 5 v6); the DN baseline field; more decoded intensity | Figs. 2, 3a, 4, 5 (v6), 6e, S | **C5**: four meanings. Teal vs grey item circles has ΔE 3 in deuteranopia, so the pRF ring disappears among the items in Fig. 4b icons. |
| Purple `#7209B7`, magenta `#B5179E`, `#5E3C99`, `#8A6FB0` | `ROI_COLORS['V3AB']`, `['TO']`; `PATHWAY_COL['Dorsal']`; `fig5_gain_fields.py` dD; `plot_fig1.HEMI['L']`; `figS_dog_kappa.SUR`; `figS_dn_gain_locus.DFIELD` | V3AB, Dorsal pathway, TO; distractor gain HP − LP (old Fig. 5); left hemisphere; DoG surround; DN divisive field | Figs. 2, 3b/c, S | **C6**: under option A these all collide with the sustained-field purple. |
| Orange-red / sand `#E76F51`, `#F4A261`, `#E9C46A`; `HEMI['R'] #E66101` | `ROI_COLORS['LO']`, `['VO']`, `['hV4']`; `PATHWAY_COL['Occipitotemporal']` | LO (and the Occ.-temporal pathway), VO, hV4; right hemisphere | Figs. 2, 3b/c | **C7**: the Occ.-temporal pathway curve in Fig. 3b (target panel) is orange-red next to target orange. |
| Green `#8AB17D` | `ROI_COLORS['V3']` | V3 | Fig. 2 | Minor; the green family would share a hue with the real distractor green. |
| Green `#00A900`, orange `#E39100`, grey `#808080` | `plot_fig1` (`ITEM_B`, `ITEM_A`, `SCREEN`) | Real display: colour-singleton distractor, items, screen | Fig. 1a, 4a "On screen" | **C4**: the real distractor is green while the model distractor is blue. The real items are orange, the same hue as the model target. |
| Dark grey `#52606D`, grey `#9AA5B1`/`#7B8794` | `LOC_COLORS['next'/'adjacent']`, `['opposite']`; `NEUTRAL`/`LP_GREY`; `MUTED` | Next to HP; opposite; LP (Fig. 1c uses FG); neutral item; distractor absent; frames shuffled | Figs. 1c/d, 3, 5c/d, 6 | **C8**: grey stands for LP locations and also for the neutral item and the null (Fig. 3a neutral item vs Fig. 3d next/opposite). Generic LP is FG black in Fig. 1c but grey in Figs. 3/6. |
| Magma | `fig4_model` (model input, effective drive), `fig5_gain_fields_v6` (older layouts, `figS_gain_field_map_search`), `fig6_decoding` (decoded frames, Fig. 6a) | Stimulus drive; combined gain; decoded intensity | Figs. 4a, 6a, S | **C9**: under option A the magma purples read as the sustained field. |
| BrBG (brown–teal); RdBu_r + `#2166AC`/`#B2182B` | `fig6_decoding.MAP_CMAP`; `figS_decoded_bars`, `figS_dn_params_by_hp`, `fig5_gain_fields` tables | Decoded intensity less/more; stronger/weaker | Fig. 6e, S | The teal end is the pRF colour, and RdBu's red end is the old HP. |
| viridis | `fig2_prf_fits` (Fig. 2e distance from nearest item) | Distance of the pRF from an item | Fig. 2e | Dark purple means "at the item", which reads as HP under option A. |
| hsv, nipy_spectral | `fig2_prf_fits`, flat maps | Polar angle, eccentricity | Fig. 2a | Retinotopy conventions with their own wheel and colourbar, so exempt. |
| Blues `#3D7EB8`, `#7DAAD6` | `figS_decoding_voxel_selection.COLORS` | Looser voxel selections | S | Near the distractor blue. |
| `#E07A5F` | `figS_dn_gain_locus` | Second observation series | S | Near target orange and old HP. |

## 3. Proposed palette (option A)

| Meaning | Colour | Hex | Old | Notes |
|---|---|---|---|---|
| **HP location; sustained field; HP − LP effects** | Saturated purple | `#7B3294` | `#D62828` | The ColorBrewer PRGn purple. ΔE vs target is 63 (deut) and 60 (prot), against 16 and 25 before. |
| **LP (generic), and next to HP** | Mid purple tint | `#9970AB` | FG / `#52606D` | Markers get a `#40004B` edge. |
| **LP opposite HP** | Light purple tint | `#D9C5E2` | `#9AA5B1` | Too light for text or a lone thin line: markers get a `#40004B` edge, lines are dashed, text uses `#8C6B9E`. |
| Edge / text on tints; dark end of the sustained-field map | Deep purple | `#40004B` | — | |
| **Target, target field** | Orange | `#F77F00` | same | Unchanged. |
| **Distractor, distractor field** | Sky blue | `#4AA3DF` | `#0077B6` | |
| Neutral item; distractor absent; frames shuffled / null; "no change" lines | Grey | `#9AA5B1` (`MUTED`) | `#7B8794`, `.6`, teal | One neutral grey. |
| **Bar stimulus** (Fig. 6a/b, decoded-vs-true) | Charcoal | `#3E4C59` | `#3D5A80` | Achromatic like the real checkerboard, and out of the blue family. |
| Images of stimulus drive (model input, effective drive, decoded frames) | `gray` colormap on black | — | magma | Stars in Fig. 6b in FG. |
| Combined gain field (several terms at once, e.g. `figS_gain_field_map_search`) | `Greys` | — | magma | It is not one term, so it gets no term hue. |
| **pRF** (outline, fill in Fig. 3a, DoG pRF step in Fig. 2d, DoG surround dashed) | Deep teal | `#00796B` | `#2A9D8F` | Darker so it separates from grey items under CVD: ΔE vs grey 22 (deut), against 3 before. |
| Text, axes, generic model line, global gain (Fig. 7) | FG | `#1F2933` | same | |
| **Model vs data** | Model = line + band in the condition colour (FG if generic); data = markers with bars | — | — | Unchanged convention. Key it with drawn glyphs. |
| **Decoding map** (signed, Fig. 6e and S) | Purple – white – charcoal ("less" = purple, "more" = `#4D5A66`) | custom | BrBG / RdBu_r | "Less" carries the suppression hue. No reused hue. Under CVD one end is chromatic and one is achromatic. |
| Distance-from-item ramp (Fig. 2e) | Greys, dark = at the item | `Greys` .3–.95 | viridis | An ordered, achromatic stimulus quantity. |
| **Pathways** (Fig. 3b/c curves, any curve per pathway) | Line style in the quantity's own colour: Early solid, Dorsal dashed, Occ.-temporal dotted | — | V1 slate / V3AB purple / LO orange-red | ROI x-axes keep the grey brackets. |
| **Areas** (Fig. 2a flat map, Fig. 2b/c only) | Families: Early steel `#2F4B5C` `#5A7A8C` `#93AEBB`; Dorsal umber/khaki `#7A6038` `#B8995E`; Occ.-temporal olives `#4F6B36` `#76924F` `#A3B97A` `#CCD9AE` | — | `ROI_COLORS` | Low chroma. Each area is labelled on the map, so CVD separation between families is not needed (Dorsal and Occ.-temporal merge in deuteranopia, see the mocks). Elsewhere areas are identified by x position only: `figS_dog_kappa` and `figS_hierarchy_gain_terms` dots go to FG/grey. |
| **Significance** | Stars and brackets in the tested quantity's colour (FG between conditions); Fig. 7 solid vs pale | — | red stars | Never a separate "significant" hue. |
| **Behaviour: distractor HP / LP / absent** (Fig. 1c) | `#7B3294` / `#9970AB` / `#9AA5B1` | | HP / FG / MUTED | The difference strips take the colour of the minuend: HP − LP purple, LP − absent mid tint. |
| **Behaviour: target HP / LP** (Fig. 1d) | `#7B3294` / `#9970AB` | | HP / FG | Colour = where the item is (location probability); the icon glyph shows which item. |
| **Intertrial priming** (Fig. 1e, Fig. 7a) | Target orange / distractor sky blue | | same | |
| Real display (Fig. 1a, Fig. 4a "On screen") | Screen `#808080`, items `#E39100`, distractor `#00A900` (as shown) | | same | Keep these as the truth. Draw the "Target" and "Distractor" labels in FG with leaders, not in orange and blue (C4). The HP ring stays purple dashed. |
| Hemisphere (if `plot_fig1` hemisphere legend is used) | FG solid / FG dashed | | `#5E3C99` / `#E66101` | Purple and orange are taken. |

### HP / LP recommendation

**Use purple shades: HP = saturated `#7B3294`, next to HP = `#9970AB`, opposite = `#D9C5E2`, and generic LP = the mid tint.**
- The model has one sustained field at all four item locations that is weaker at HP. Colouring all locations with that field's hue says this directly, and lightness then carries the design variable (distance from HP).
- The ordering HP < next < opposite in Fig. 3e and Fig. 6 becomes a visible lightness ordering instead of an arbitrary red/dark-grey/light-grey set.
- Grey is freed for "no model term": the neutral item, absent and the null. That removes C8.

Cost and guardrails, all checked in the mocks:
- **Next tint vs distractor blue is the tightest pair**: ΔE 10 (deut), 13 (prot). The two meet in Fig. 1c, where LP dots sit above an icon with a blue distractor. Lightness and the dark edge separate them, but they cannot be pushed further apart. A darker tint drifts towards HP (ΔE < 12) and a lighter one lands on the distractor's lightness (L* ≈ 64; e.g. `#A887BC` gives 7.7).
- **The opposite tint is faint as a line**, so draw it dashed (mock b). Subject dots in the light tint need an edge and opacity ≥ .6.
- **Text in a tint uses the darker step**: `#8C6B9E` for "Opposite", HP purple for "Next to HP".

Alternative (shown as b′ in the mocks): HP purple with LP in greys (next `#52606D`, opposite `#9AA5B1`), which is the current structure with red swapped for purple. It reads more robustly as thin lines and keeps every LP colour far from the distractor (ΔE ≥ 16). The cost is that grey again means both LP and neutral/null. Pick it if the light opposite line in Fig. 3d still looks too weak once rendered at print size.

## 4. Conflicts resolved

- **C1** HP vs target in deuteranopia: ΔE 16 → 63, and in tritanopia 14 → 32.
- **C2** distractor vs IPS, **C3** stimulus vs V1 vs Early, **C6** V3AB, Dorsal and TO vs the sustained purple, **C7** Occ.-temporal vs target. All four are removed because areas and pathways no longer take condition hues.
- **C5** teal: kept only for the pRF and darkened, so the pRF vs grey item ring goes from ΔE 3 to 22 in deuteranopia. "No change" lines become grey, the decoding "more" end becomes charcoal, and V2 moves to the steel family.
- **C8** grey for both LP and the neutral item: LP becomes purple tints.
- **C9** magma vs sustained purple: stimulus and drive images become achromatic.
- **Red as a significance colour**: gone. Stars follow the tested quantity.
- **Next to HP shown in two colours** (dark grey vs pale red in `figS_dn_params_by_hp`): one tint everywhere.
- **C4** real orange items / green distractor vs model target and distractor: kept as the truth, confined to depictions of the screen, with FG labels.

Remaining limitation: in deuteranopia and protanopia, purple and sky blue are both seen as blues and differ mainly in lightness (ΔE 26 and 34). That is adequate for marks and labels. Where the sustained and distractor maps sit side by side (Figs. 4a, 5a), their panel titles keep them apart.

## 5. CVD checks (CIEDE2000; N / deut / prot / trit)

| Pair | Old | New |
|---|---|---|
| HP – target | 27 / **16** / 25 / **14** | 58 / 63 / 60 / 32 |
| HP – distractor | 50 / 52 / 44 / 56 | 45 / 26 / 34 / 60 |
| Target – distractor | 53 / 61 / 53 / 66 | 50 / 56 / 51 / 58 |
| LP next – distractor | 16 / 16 / 17 / 16 | 33 / **10** / 13 / 50 |
| LP opposite – distractor | 24 / 26 / 21 / 20 | 33 / 20 / 15 / 46 |
| HP – LP next | 37 / 36 / 27 / 42 | 19 / 17 / 18 / 18 |
| LP next – LP opposite | 26 / 27 / 26 / 26 | 25 / 24 / 25 / 24 |
| LP opposite – neutral | **10 / 10 / 10 / 10** | 19 / 11 / 11 / 21 |
| LP next – neutral | 16 / 16 / 16 / 16 | 21 / 15 / 18 / 28 |
| pRF – distractor | 29 / 22 / 26 / **10** | 31 / 26 / 30 / 19 |
| pRF – neutral | 22 / **3** / 11 / 17 | 30 / 22 / 21 / 25 |
| Stim – distractor | 13 / 10 / 13 / 13 | 35 / 33 / 37 / 37 |
| HP – stim | 42 / 46 / 36 / 52 | 17 / 14 / 16 / 29 |

ΔE < 10 is confusable for small marks and 10–15 needs a second cue. Every new pair below 15 already has one: lightness, the dark edge, the icon or the line style. The whole-figure check is `mocks_cvd.png`.

## 6. What changes where (when approved)

`retsupp/visualize/paper/paper_style.py`: HP `#7B3294`; new LP_NEXT `#9970AB`, LP_OPP `#D9C5E2`, HP_DEEP `#40004B`, HP_TEXT_LIGHT `#8C6B9E`; DISTRACTOR `#4AA3DF`; STIM `#3E4C59`; new NEUTRAL = MUTED `#9AA5B1`; new PRF `#00796B`; new LOC_COLORS (hp/next/opposite), DIV_CMAP (purple–white–charcoal), FIELD_CMAPS (sustained/target/distractor), PATH_LS; ROI_COLORS → area families; drop PATHWAY_COL. `stimulus_icon`: the `'lp'` rings become mid-tint.

Figure scripts (all should import the constants from `paper_style` rather than redefining them):

| Script | Figure | Changes |
|---|---|---|
| `plot_fig1.py` | Fig. 1a, b (panel_display, timeline, HP-moves icons) | Drop the local FG/MUTED/HP/TARGET_C/DISTRACTOR_C copies and import them. HP label and HP icons → purple. "Target"/"Distractor" labels in Fig. 1a → FG. Keep SCREEN/ITEM_A/ITEM_B. AREAS/HEMI → area families / FG linestyles if still used. |
| `fig1_paradigm_behaviour_v4.py` (+ `fig1_paradigm_behaviour.py` `rt_panel`, `cond_icons`) | Fig. 1c–e, `figS_priming_rt_v4` | rt_panel colours `[HP, FG, MUTED]` → `[HP, LP_NEXT, NEUTRAL]`; `[HP, FG]` → `[HP, LP_NEXT]`; strip colours as above; distractor icons sky blue. |
| `fig2_prf_fits.py`, `pycortex/render_fig1_prf.py` (`AREA_COLORS`), `pycortex/render_fig3_flat.py` | Fig. 2 | Area families on the flat map (re-render the flat maps); PRF_TEAL → PRF; Fig. 2e viridis → Greys; PATHWAY_COL → line styles. |
| `fig3_item_locations.py` | Fig. 3, `tableS_item_locations` | LOC_COLORS → purple ramp; NEUTRAL/LP_GREY → NEUTRAL; PRF_TEAL → PRF; Fig. 3b/c pathway curves → panel colour (target/distractor) with PATH_LS, stars in the same colour; opposite line dashed; tint labels as in §3. |
| `fig4_model.py` | Fig. 4 | Model input and effective drive magma → `gray`; sustained map → purple field cmap; PRF_TEAL → PRF; PRF_COL (near HP/LP/fixation) → HP/LP_NEXT/NEUTRAL; sustained band, arrows and labels → purple; "On screen" labels FG. |
| `fig5_gain_fields_v6.py` (v11 layout and `figS_gain_field_map_search_v3`) | Fig. 5, S | HP_CMAP → white→`#C9A9D6`→HP→HP_DEEP; HP rings and 5c dots → purple; ZERO teal → grey; magma (search-display maps) → Greys; distractor cmap from new blue. |
| `fig5_gain_fields.py` | Tables, `figS_gain_vs_prf_size`, `figS_pooled_dS_dD`, older Fig. 5s | HP → purple; dD `#5E3C99` → DISTRACTOR; RdBu_r table shading → DIV_CMAP; TRANSIENT grey stays. |
| `fig6_decoding.py`, `fig6_collect.py` | Fig. 6, `figS_decoding_*` | LOC_COLORS → purple ramp; `#F6D4D4` quadrant fills → the location's own colour; MAP_CMAP BrBG → DIV_CMAP, Less/More text colours; decoded frames magma → `gray`; STIM charcoal; Fig. 6b stars FG; HP stars → purple. |
| `fig7_compact.py` | Fig. 7 | Distractor blue only (follows `paper_style`). |
| `figS_decoded_bars.py` | `figS_decoded_bars_*` | RdBu_r and `#B2182B`/`#2166AC` → DIV_CMAP; `#52606D` → LP_NEXT. |
| `figS_decode_simulation.py` | `figS_decode_simulation` | COLS → LOC_COLORS. |
| `figS_dn_params_by_hp.py` | `figS_dn_params_by_hp` | REL_COLORS → HP/LP_NEXT/LP_OPP; RdBu_r → DIV_CMAP. |
| `figS_dn_gain_locus.py` | `figS_dn_gain_locus` | INPUT → STIM; BFIELD teal / DFIELD purple → FG solid / FG dashed (BOTH grey); LOC → purple ramp; `#E07A5F` → grey. |
| `figS_dog_kappa.py` | `figS_dog_kappa` | SUR `#8A6FB0` → PRF dashed; ROI_COLORS dots → FG/grey. |
| `figS_hierarchy_gain_terms.py`, `figS_field_widths_v2.py` | S | ROI_COLORS dots → FG/grey; HP → purple (follows import). |
| `figS_decoding_voxel_selection.py` | `figS_decoding_voxel_selection` | Blues → FG / `#616E7C` / `#9AA5B1` (selection variants are not conditions). |
| `compare_decoding_prf_models.py`, `figS_bar_crossing_response`, `richter_followup.py`, `analyze_trialwise_af_gains.py`, `plot_gain_parameters.py`, `plot_trialwise_rt_bayes.py` | S / checks | Follow `paper_style` imports; check for local `#D62828` copies (`grep -n "D62828\|0077B6\|2A9D8F\|3D5A80" retsupp/visualize/paper/*.py`). |

Legends and captions that name colours must change in the same pass. `grep -n -i "red\|teal\|magma\|blue\|purple" notes/writing_package_*/figures/*.md` lists them, e.g. "magma for the sustained field, teal line…" in `fig5_gain_fields_v6.py` and "pRF (teal)" in `fig4_model.py`.
