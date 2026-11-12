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

## Fig. 3 robustness and the ordered test (2026-11-12; n = 28)

Sources: `notes/fig3_robustness_2026-11-12.md`, `notes/data/fig3_robustness_n28_2026-11-12.tsv`,
`notes/data/fig3_location_ordered_tests_n28_2026-11-12.tsv`. New supplementary figure
`figures/figS_item_locations_per_area_abs_n28` (absolute units, one panel, shared axis) — prefer it over
`figS_item_locations_per_area_n28` (relative units) for the per-area supplement.

- **Ordered test (predicted gradient HP < next to HP < opposite; Page's L).** Early visual cortex p = .0016; per area
  q < .05 in V1 (.017), V2 (.015) and TO (.020) (BH over nine areas); all areas p = .008. The unordered location-type
  ANOVA was significant only in V1. Describe the order as the predicted gradient (a gain field centred on HP falls off
  with distance; Richter et al. found the same gradient); the test is carried mainly by HP vs opposite (only 9–12 of 28
  participants show the full order).
- **Absolute units (% signal change, no division by the participant's mean response, nobody excluded).** HP − opposite:
  early −0.11% [−0.13, −0.06], 20/28, p = .005 (Page p = .0016); V1 −0.08 (q = .030), V2 −0.06 (q = .030), TO −0.13
  (q = .041). No single participant decides the result (leave-one-out max p = .025).
- **Next − opposite** is weaker in absolute units (early p = .038, q = .11) and not reliable in any single area: present
  it as a trend supported by the ordered test, not as a finding on its own.
- **Few voxels in higher areas.** Voxels are selected by a pRF centre within 2° of an item location (and reliable
  across runs). Few pRFs are centred on the small item locations, especially in higher areas with large pRFs: the
  median count at a participant's sparsest location is 7–12 voxels in V1–V3 but 1–2 in IPS, hV4, VO, LO and TO. So
  per-area results in higher areas (incl. TO) rest on very few voxels; claim early visual cortex (and V1, V2), mention
  TO with this caveat. Early visual cortex survives a floor of ≥ 5 and ≥ 10 voxels per location (≥ 5: p = .001).

**Suggested wording.**
- Results (Fig. 3): "Tested for the predicted gradient (HP < next to HP < opposite; Page's L), the location effect was
  reliable in early visual cortex (p = .0016) and in V1 and V2 separately (q ≤ .02). In absolute units the effect was
  the same (early visual cortex, HP − opposite −0.11% signal change, p = .005)."
- Methods or Results caveat: "Because voxels were selected by a pRF centred within 2° of an item location, and few
  pRFs in higher visual areas are centred on these small locations, per-area estimates outside early visual cortex
  rest on few voxels (median 1–2 per location) and are reported for completeness only."

## Instruction: why item-specific responses are measurable only in early visual cortex (2026-11-12)

**Writing agent: say this explicitly** (Methods, voxel selection, and Results/Discussion where Fig. 3 per-area or
higher-area item effects come up). The item-location analysis needs voxels whose pRF sees one item location and not its
neighbours. That is possible in early visual cortex but not in higher areas, for two reasons, with numbers (n = 28):

1. **pRF size versus item spacing.** Near the item ring (3–5° eccentricity), median pRF σ (final DoG pRFs) is V1 0.64°,
   V2 0.81°, V3 0.99° (FWHM 1.5°, 1.9°, 2.3°), but V3AB 1.40°, hV4 1.07°, LO 1.10°, TO 1.67° (FWHM 3.3°, 2.5°, 2.6°,
   3.9°). The eight items sit 3.1° apart on the 4° ring (the four possible target/distractor locations 5.7° apart), and
   each item is 1.5° long. So a higher-area pRF spans more than one item position and cannot isolate the response to
   the item at one location; an early-visual pRF can. (IPS and VO have too few voxels near 4° eccentricity for a stable
   size estimate.) Source: `notes/data/fig1_prf_sigma_final2.tsv` (participants 3–30, voxel-weighted mean σ within
   3–5° per participant, median over participants).
2. **Few pRFs centred on the item locations.** Voxels count for a location if their pRF centre lies within 2° of it
   (and they are reliable across runs). At a participant's sparsest location the median number of such voxels is
   V1 8, V2 12, V3 7, but V3AB 3, IPS 1, hV4 2, VO 2, LO 2, TO 2. With a floor of 5 voxels per location, early visual
   cortex keeps 26 of 28 participants, TO 4, VO 2. Source: `notes/fig3_robustness_2026-11-12.md`.

Suggested wording (Methods or Results): "Item-specific responses require pRFs that respond to one item location and not
its neighbours. Near the item ring, pRFs in V1–V3 are small (median FWHM 1.5–2.3°) relative to the 3.1° spacing of the
items, but pRFs in higher areas are larger (FWHM 2.5–3.9° in hV4, LO, V3AB and TO), and few of them are centred on an
item location (median 1–3 voxels per location, against 7–12 in V1–V3). We therefore draw conclusions about
item-specific responses from early visual cortex and report the other areas for completeness."

Do NOT present the absence of reliable per-area item effects in higher areas as evidence that the HP effect is absent
there: the gain-field model (Fig. 5), which uses all of an area's voxels and does not require item-specific pRFs, finds
the sustained HP effect pooled over all areas and in TO.
