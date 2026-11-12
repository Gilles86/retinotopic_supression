# Update 4 (consolidated) to writing package 2026-11-12 (n = 28)

**This single update replaces updates 4, 5 and 6; ignore those if you have them.** Apply it on top of the base
package 2026-11-12 and updates 1–3. No new fits: every value comes from the existing n = 28 outputs named below.

## What replaces what

| File | Status | What |
|---|---|---|
| `figures/fig5_gain_fields_v10_n28.{pdf,png,md}` | **replaces** `fig5_gain_fields_n28` as Fig. 5 | New panel layout and units, see "Fig. 5" below. |
| `figures/figS_gain_field_map_search_v2_n28.{pdf,png,md}` | **new**, Fig. S4 | The fitted gain field during a search display, distractor at HP vs at the LP location next to HP (target opposite HP), V3, hV4, TO; in %; sustained field exactly as Fig. 5a. |
| `figures/figS_field_widths_n28.{pdf,png,md}` | unchanged, now Fig. S3 | The field widths; the widths behind the Fig. 5a maps (the old to-scale Fig. 5b is gone). Included for completeness. |
| `figures/figS_gain_vs_prf_size_n28.{pdf,png,md}` | legend patched, now Fig. S5 | "(as in Fig. 5a)" → "(as in Fig. 5b)". Its axis is still in fractional gain; give its numbers in % in the text (× 100). |
| `key_numbers.md` | **replaces** | All gain values in % (differences in percentage points); Fig. 5 section: panel letters, a block "a, the fitted gain-field maps", the v10 note, and the pooled lead statistic with the per-pathway numbers for Fig. 5c. |
| `counts_k_of_n_n28.md` | **replaces** | "Fig. 5a" rows relabelled "Fig. 5b" (counts unchanged; shares of participants, not gains). |
| `numbering_supplement_2026-11-12.md` | **replaces** the numbering table of update 2 | New Fig. S4 after the field-widths figure; S4–S21 move to S5–S22 (the update-3 decoded map, measured vs model, is now S21). Tables unchanged. |
| `fig5b_gain_field_map_*` prototypes (updates 2–3) | superseded | Drop them; Fig. 5a replaces them. |

## Fig. 5 (`fig5_gain_fields_v10_n28`, 180 mm × 5.8 in)

- **a (new): the fitted gain fields over the display**, drawn from group medians (the fitted parametric field, not
  data), for V3, hV4 and TO, in one row of six maps.
  - Left block, the sustained field as its **HP − LP difference**: the area's dS estimated alone (the value panel c
    plots) times the field's Gaussian at HP, one red scale for the three areas. Printed next to HP: V3 −3.0%,
    hV4 −1.5%, TO −1.9% — **equal to panel c's per-area estimates**. Under each map, the full sustained gain at LP / at
    HP (V3 −0.4% / −3.5%; hV4 −5.8% / −7.3%; TO −35% / −37%): the sustained field itself is a suppression at all four
    locations in most areas (7 of 9), deepest at HP.
  - Right block, the transient fields while a display is on: target (orange) V3 +30%, hV4 +56%, TO +22%; distractor
    (blue, averaged over its locations) +3%, +22%, +8%; one scale 0 to +60%. The transient width is set for display
    (FWHM 3.2°) and not interpreted.
- **b: target and distractor gain per area** (was a), y-axis in %.
- **c: sustained gain at HP − LP, dS estimated alone** (was c), now led by the **all-areas pooled estimate**
  (leftmost, larger), the nine areas to its right as detail; y-axis in %.
- **d: transient HP − LP, dD and dT** (was d), y-axis in %.
- The old to-scale panel b is removed (widths: Fig. S3).

## Units: gains are now in percent

Convert every gain value in the manuscript × 100; differences between gains (HP − LP, dS, dD, dT) are percentage
points, written as %. One decimal below 10% (−1.4%), none from 10% up (+30%); CIs the same way. This covers:
- Fig. 5 Results and Discussion: target gain (pooled +16% [+13%, +24%]; e.g. hV4 +56%), distractor gain (pooled +6.5%
  [+3.2%, +15%]), distractor gain on top of the target term (+3.1%), target-gain slope with pRF size (+33% per
  doubling), shared sustained gain S (−3.4%), g_LP / g_HP (−6.7% / −15%), dS estimated alone (pooled −1.4%
  [−1.8%, −0.6%]; V3 −3.0%, TO −1.9%), dS joint (−3.4%; V3 −9.0%, hV4 −5.7%, TO −2.4%), dD (−1.0%), dT (−3.4%),
  dD − dT (+0.9%), joint dD (+0.4%), the split models (HP − opposite −3.3%, HP − next −1.6%, next − opposite −0.6%;
  target / distractor HP − LP +4.7% / +4.6%; one-width model dS −2.3%).
- κ section: only the dS values (dS with κ −2.5%; change −0.01%); κ itself stays as is.
- Richter follow-up A3: dS per session (−0.6%, −3.5%; difference −2.2%; next − opposite +0.5%, −0.2%).
- Fig. 7: absolute target-repeat gain (−5.2%) and the RT modulators (distractor −2.4% per SD of log RT, global −2.1%,
  target +0.9%); the relative values (−19% of the typical gain) were already in %.
- Robustness tables (Gaussian, DN) and the LOSO table's "median δ" column (gain terms only).
- **Not converted:** RTs (ms), BOLD responses (% signal change), R² and ΔR², decoded intensities, κ, the DN baseline
  weights, correlations, the item weight (a ratio).
- Suggested Methods sentence: "Gains are expressed as the percentage change in the stimulus drive at the centre of a
  field (a gain of +30% means 30% more drive); differences between gains are in percentage points."

## Results: the lead statistic for the sustained HP effect

Lead with the pooled effect: **all areas pooled, dS estimated alone, −1.4% [−1.8%, −0.6%], negative in 22 of 28
participants, p = 1.1 × 10⁻⁴** (per participant the median dS over the nine areas; Wilcoxon). Then the two areas that
survive correction (V3 −3.0%, q = .046; TO −1.9%, q = .035), then the joint fit (−3.4%, p = .0027). The claim rests on
the pooled effect, consistent across participants, rather than on any single area. Optional clause: "present in early
and occipitotemporal areas, not reliable in dorsal areas" (early −0.5%, q = .040; dorsal −0.8%, q = .90;
occipitotemporal −1.1%, q = .040; BH across the three; numbers in the Fig. 5 legend and `key_numbers.md`).

## Other manuscript edits

1. Fig. 5 panel letters: every text reference to "Fig. 5a" (target and distractor gain, gain vs pRF size) becomes
   "Fig. 5b"; references to the to-scale field ("Fig. 5b") become "Fig. 5a" (maps) or "Fig. S3" (widths); "Fig. 5c",
   "Fig. 5d" stay.
2. Supplementary citations: Fig. S4 onward shift by one (table in `numbering_supplement_2026-11-12.md`).
3. Panel a in the text: "the sustained field is lower at HP than at the other locations (by 3.0% in V3, 1.9% in TO),
   on top of a suppression at all four locations in most areas; during a display, the target gets a strong transient
   gain and the distractor a weaker one". Fig. S4: during a search display the transient fields dominate in V3 and hV4;
   the transient HP − LP part drawn there is not reliable (legend).

Sources: `retsupp/visualize/paper/fig5_gain_fields_v6.py --layout v10`, run as
`python -m retsupp.visualize.paper.run_excluding --exclude 1 2 -- retsupp.visualize.paper.fig5_gain_fields_v6 --suffix _n28 --layout v10`;
map parameters `notes/data/fig5_gain_field_map_parameters_n28_2026-11-12_v10.tsv`; pooled numbers
`notes/data/dS_pooled_by_pathway_n28_2026-11-12.tsv`.
