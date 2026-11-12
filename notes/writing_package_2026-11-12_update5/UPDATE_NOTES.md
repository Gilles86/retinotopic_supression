# Update 5 to writing package 2026-11-12 (n = 28)

Apply on top of the consolidated update 4 (`update4_final`). Files here replace those with the same name; one figure is
new. No existing number changed. A baseline-vs-gain analysis will be added to this update later (its files and notes
will be appended below).

| File | Status | What changed |
|---|---|---|
| `key_numbers.md` | replaces | New block under Fig. 3: "Fig. 3d, e per area" (nine areas, location-type ANOVA and the three contrasts). Everything else as update 4. |
| `figures/tableS_item_locations_n28.md` | replaces | New section at the end: the same per-area table. |
| `figures/figS_item_locations_per_area_n28.{pdf,png,md}` | new | Supplementary figure: Fig. 3e per area (own y-scale per area), with legend. Suggested place: right after the item-locations table in the supplementary numbering. |

## Fig. 3d, e per area (n = 28; items pooled; % of each participant's mean response in that area)

Contrasts: mean ± SEM, Wilcoxon p, Benjamini-Hochberg q across the nine areas per contrast. ANOVA: one-way
repeated-measures on location type (HP, next, opposite), q across the nine areas.

| Area | n | ANOVA F, q | HP − opposite (p, q) | Next − opposite (p, q) | HP − next (p, q) |
|---|---|---|---|---|---|
| V1 | 28 | 5.79, .047 | −52.8 ± 16.4% (.002, .015) | −30.3 ± 14.6% (.090, .405) | −22.5 ± 15.6% (.150, .385) |
| V2 | 28 | 2.10, .538 | −20.6 ± 9.4% (.010, .031) | −9.2 ± 12.3% (.138, .413) | −11.4 ± 8.1% (.171, .385) |
| V3 | 28 | 0.70, .754 | −7.0 ± 10.5% (.327, .491) | −9.1 ± 5.3% (.210, .472) | +2.1 ± 7.6% (.884, .884) |
| V3AB | 25 | 0.26, .811 | −9.3 ± 13.5% (.275, .491) | −6.4 ± 16.2% (.771, .868) | −2.9 ± 8.5% (.339, .510) |
| IPS | 13 | 1.60, .538 | −11.7 ± 17.6% (.635, .817) | −35.7 ± 26.7% (.273, .492) | +24.0 ± 14.7% (.168, .385) |
| hV4 | 28 | 1.18, .566 | −12.4 ± 11.2% (.264, .491) | −13.4 ± 7.1% (.086, .405) | +1.0 ± 10.3% (.412, .529) |
| VO | 17 | 0.21, .811 | +16.1 ± 31.5% (.747, .840) | +11.8 ± 14.8% (.890, .890) | +4.3 ± 27.7% (.782, .880) |
| LO | 26 | 0.21, .811 | +1.7 ± 9.3% (.842, .842) | +5.0 ± 6.2% (.565, .727) | −3.3 ± 7.4% (.340, .510) |
| TO | 21 | 1.48, .538 | −111.5 ± 87.4% (.010, .031) | −55.4 ± 63.9% (.495, .727) | −56.1 ± 28.8% (.042, .379) |

Bootstrapped 95% CIs and k of n are in `key_numbers.md` / `tableS_item_locations_n28.md`.

- q < .05: **HP − opposite in V1, V2 and TO**; next − opposite in no area; HP − next in no area; location-type ANOVA in V1.
- Single-area percentages spread widely where a participant's mean response in that area is small (TO, IPS, VO; the TO
  mean is driven by a few participants, median HP − opposite −25%); n is lower where participants lack voxels at every
  item location or have a non-positive mean response (IPS 13, VO 17, TO 21).
- These per-area values match the earlier V1–V3 numbers (Richter follow-up D6, same data and measure).

Suggested sentence for Results (after the pathway-level location effect): "Per area, the lower response at HP than
opposite it was reliable in V1, V2 and TO (Benjamini-Hochberg across the nine areas; Fig. S…), whereas the lower
response next to HP than opposite it did not reach significance in any single area and is carried by the pooled early
areas."

Source: `notes/data/fig3_location_per_area_n28_2026-11-12.tsv` (per-participant values in
`notes/data/fig3_location_per_area_values_n28_2026-11-12.tsv`); code `retsupp/visualize/paper/fig3_location_per_area.py`.
