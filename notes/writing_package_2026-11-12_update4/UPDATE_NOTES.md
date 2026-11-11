# Update 4 to writing package 2026-11-12 (n = 28): Fig. 5 restructured with the gain-field maps

Apply on top of update 2 (and update 3). No new fits: every value is from the joint fits in
`notes/data/gain_field_parameters_final2_2026-11.tsv` (m7_full, participants 3–30).

## What replaces what

| File | Status | What |
|---|---|---|
| `figures/fig5_gain_fields_v6_n28.{pdf,png,md}` | replaces `fig5_gain_fields_n28` | New panel a: gain-field maps, 3 × 3 (rows sustained, target, distractor field; columns V3, hV4, TO). b, c, d: the earlier a, c, d, unchanged in content. The earlier to-scale panel b is removed (widths stay in `figS_field_widths_n28`). |
| `figures/figS_gain_field_map_search_n28.{pdf,png,md}` | new | The two search rows (sum of the three fields, distractor at HP vs at the LP location next to HP, target opposite HP), V3, hV4, TO, with "At the distractor, HP − LP" per area. |
| `numbering_supplement_2026-11-12_update4.md` | replaces the A3 table of update 2 | New Fig. S4 after the field-widths figure; S4–S21 move to S5–S22 (the update-3 decoded map is now S21). |
| `key_numbers.md` | replaces | Fig. 5 section: panel letters, a new block "a, the fitted gain-field maps" (per-area g_HP, g_LP, sustained width, joint dS, target and distractor gains used), "a, joint fit" renamed "b"; field-width line updated. Nothing else changed. |
| `counts_k_of_n_n28.md` | replaces | "Fig. 5a" rows relabelled "Fig. 5b" (numbers unchanged). |
| `figures/figS_gain_vs_prf_size_n28.md` | replaces the legend only | "(as in Fig. 5a)" → "(as in Fig. 5b)"; PDF/PNG unchanged. |
| `fig5b_gain_field_map_*` prototypes (update 2/3) | superseded | Drop them from the figure list; Fig. 5a replaces them. |

## Panel a: how the maps are drawn

- Per area, the median over participants of each participant's joint-fit parameter. Sustained row: LP gain = median
  g_LP; HP gain = median g_LP + median joint dS, so the HP − LP printed on each map is the joint dS of the text.
  Distractor row: the distractor gain averaged over its locations, (g_HP_dyn + 3 g_LP_dyn) / 4, as in panel b.
- One number per map, as fractional gain: sustained, HP − LP; target, the gain at the target; distractor, the gain at
  the distractor.

| Area | g_LP | g_HP (median) | drawn at HP (g_LP + dS) | joint dS | σ sustained (FWHM) | target gain | distractor gain |
|---|---|---|---|---|---|---|---|
| V3 | −0.004 | −0.138 | −0.094 | −0.090 | 1.94° (4.6°) | +0.30 | +0.034 |
| hV4 | −0.058 | −0.160 | −0.115 | −0.057 | 1.29° (3.0°) | +0.56 | +0.22 |
| TO | −0.354 | −0.476 | −0.378 | −0.024 | 1.07° (2.5°) | +0.22 | +0.084 |

All agree with `key_numbers.md` and `table_parameters_2026-11_n28.md` (target, distractor, joint dS per area, FWHM).

- Colour scales: one shared scale per row for the target row (−0.15 to +0.60) and the distractor row (−0.08 to
  +0.30). For the sustained row a shared scale (−0.40 to 0, set by TO) hid the hV4 HP − LP difference (−0.06 is 14% of
  the range), so that row has a scale per area, with the range printed under each map (V3 −0.10 to 0, hV4 −0.12 to 0,
  TO −0.38 to 0); the legend says so.
- Transient width: fixed for display at FWHM 3.2° (σ 1.36°) in every area, not interpreted; stated in the legend.
- Row labels: "Sustained field" (lower at the HP location), "Target field" (transient gain at the target),
  "Distractor field" (smaller transient gain). The distractor is drawn at an LP location next to HP, because its gain is
  the average over locations.
- Legend: fitted parametric field, not data, from group medians; the sustained gain is negative at all four locations
  in 7 of 9 areas (not V1, V2).

## Supplementary Fig. S4 (search rows)

Same parameter rules. Distractor gain at LP = median g_LP_dyn, at HP = that + median joint dD (the same rule as the
sustained field). At the distractor, HP − LP: V3 +0.03, hV4 +0.05, TO −0.02. In V3 and hV4 the median joint transient
HP − LP (+0.11 each) outweighs the sustained HP − LP (−0.09, −0.06). The joint dD is not reliable in any area (smallest
q = .29), and the legend says the transient part of the drawn difference is not an effect. (The earlier v5 prototype gave
V3 −7%, because it used the medians of g_HP and g_LP separately instead of median g_LP + median difference.)

## Manuscript edits needed

1. Fig. 5 panel letters: the earlier a → b; c and d unchanged; the earlier b (to-scale field) is gone; new a = maps.
   Every text reference to "Fig. 5a" (target and distractor gain per area, gain against pRF size) becomes "Fig. 5b";
   references to "Fig. 5b" (field size, the field drawn to scale) become "Fig. 5a" (maps) or "Supplementary Fig. S3"
   (widths). "Fig. 5c" and "Fig. 5d" stay.
2. References that changed in the package files:
   - `key_numbers.md`: the Fig. 5 heading and panel note; "a, joint fit, median per area" → "b, …"; the field-width
     line ("Fig. 5b is only the to-scale sustained field" → "the maps of Fig. 5a use each area's median sustained
     width").
   - `counts_k_of_n_n28.md`: six "Fig. 5a" rows → "Fig. 5b".
   - `figures/figS_gain_vs_prf_size_n28.md`: "(as in Fig. 5a)" → "(as in Fig. 5b)".
   - Numbering table: S3 now supports the Fig. 5a maps; the old S4 (gain vs pRF size, Fig. 5a) is S5 and supports
     Fig. 5b.
   - Unchanged references: README "Fig. 5c" (sessions), "Fig. 5d" (title "target and distractor alike") in README,
     key_numbers and the prompt; `figS_gain_fields_gauss_n28` / `figS_gain_fields_dn_n28` ("as Fig. 5", "c, d … as in
     Fig. 5"). Their own panels are not restructured; if the robustness figures should get maps too, that is a
     follow-up.
3. Supplementary numbers: every citation of Fig. S4–S21 shifts by one (see the numbering table).
4. New Results/Discussion sentence options: the maps show what the numbers say (sustained dip at HP in V3 and hV4, all
   four locations suppressed in TO with HP slightly deeper; strong transient gain at the target; weaker at the
   distractor). Supplementary Fig. S4 shows that during a search display the transient fields dominate in V3 and hV4.
