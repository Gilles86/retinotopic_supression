# Update 5 to writing package 2026-11-12 (n = 28): Fig. 5 v9, gains in percent

Apply on top of update 4. No new fits; the same numbers, in new units, and a new drawing of Fig. 5a's sustained block.

| File | Status | What changed |
|---|---|---|
| `figures/fig5_gain_fields_v9_n28.{pdf,png,md}` | **replaces** Fig. 5 v8 (v6–v8 kept for reference) | All gains in %. Sustained block of panel a is now the HP − LP difference map. Panels b–d: same data and statistics, y-axes in % ("Gain change (%)", "HP − LP (%)"). |
| `figures/figS_gain_field_map_search_v2_n28.{pdf,png,md}` | **replaces** `figS_gain_field_map_search_n28` | In %, and the sustained HP gain now uses dS estimated alone, exactly as Fig. 5a (its legend's "as in Fig. 5a" is exact again). |
| `key_numbers.md` | **replaces** the update-4 version | Every gain value converted to % (differences in percentage points); new line at the top of the Fig. 5 section; a v9 line in the Fig. 5a block. |

## Fig. 5 v9 (180 mm × 5.8 in)

- **Units.** A gain is the change in drive at the centre of a field, now in %: the target gain in V3 is +30% (was +0.30). Differences between gains (HP − LP, dS, dD, dT) are percentage points of gain, written as % in the figure and explained in the legend.
- **Sustained block = HP − LP difference map.** Only the HP-specific part of the sustained field is drawn: the area's dS estimated alone (the value panel c plots) times the field's Gaussian at HP. Why: the HP-specific part is the paper's claim; on the full field the HP dip was a small extra on top of a suppression at all four locations (in TO −37% at HP vs −35% elsewhere) and hard to see. The full field is not hidden: under each map a grey line gives the sustained gain at LP and at HP, and the legend says the sustained field is a suppression at all four locations in 7 of 9 areas.
  - One red scale (white → HP red → dark red) shared by V3, hV4 and TO, ending at the deepest area's value (V3 −3.0%), so V3 saturates and hV4 (−1.5%) and TO (−1.9%) are drawn at about half and two-thirds strength.
  - Thin red line: the field's half-maximum (FWHM) contour around HP. The HP − LP value is printed in bold red next to HP; the red dashed HP ring stays.
  - Values on the maps: V3 −3.0%, hV4 −1.5%, TO −1.9%; under the maps, gain at LP / at HP: V3 −0.4% / −3.5%, hV4 −5.8% / −7.3%, TO −35% / −37%.
- **Transient block** unchanged except units: target V3 +30%, hV4 +56%, TO +22%; distractor (averaged over its locations) +3%, +22%, +8%; one shared scale 0 to +60%.

## What to do in the manuscript (writing agent)

1. **Convert every gain value to percent (× 100); differences of gains are percentage points.** Use one decimal below 10% (−1.4%, −3.0%), none from 10% up (+30%, −35%). Convert CIs the same way. This covers:
   - Results and Discussion on Fig. 5: target gain (pooled +16% [+13%, +24%]; per area, e.g. hV4 +56%), distractor gain (pooled +6.5% [+3.2%, +15%]), the distractor gain on top of the target term (+3.1%), the target-gain slope with pRF size (+33% per doubling), the shared sustained gain S (−3.4%), g_LP / g_HP (−6.7% / −15%), dS estimated alone (pooled −1.4% [−1.8%, −0.6%]; V3 −3.0%, TO −1.9%), dS joint (−3.4%; V3 −9.0%, hV4 −5.7%, TO −2.4%), dD (−1.0%), dT (−3.4%), dD − dT (+0.9%), joint dD (+0.4%), the split models (HP − opposite −3.3%, HP − next −1.6%, next − opposite −0.6%; target / distractor HP − LP +4.7% / +4.6%; one-width model dS −2.3%).
   - The κ section: only the dS values (dS with κ −2.5%; change −0.01%); κ itself is not a gain and stays as is.
   - Richter follow-up A3: dS per session (−0.6%, −3.5%; difference −2.2%; next − opposite +0.5%, −0.2%).
   - Fig. 7: the absolute target-repeat gain (−5.2%) and the RT modulators (distractor −2.4% per SD of log RT, global −2.1%, target +0.9%); the relative values (−19% of the typical gain) were already in %.
   - Robustness tables (Gaussian, DN) and the LOSO table's "median δ" column (gain terms only).
   - Not converted: RTs (ms), BOLD responses (% signal change), R² and ΔR², decoded intensities, κ, the DN baseline weights, correlations, the item weight (a ratio, ≈ 3).
2. **Methods sentence (suggested):** "Gains are expressed as the percentage change in the stimulus drive at the centre of a field (a gain of +30% means 30% more drive); differences between gains are in percentage points."
3. **Fig. 5 legend:** use `fig5_gain_fields_v9_n28.md`. Panel a's sustained maps now show HP − LP only; say so where the text refers to Fig. 5a ("the sustained field is lower at HP than at the other locations, by 3.0% in V3 …"), and mention that the sustained field itself is a suppression at all four locations in most areas.
4. **Supplementary search figure:** cite `figS_gain_field_map_search_v2_n28` (same S-number as in update 4).

Sources: `retsupp/visualize/paper/fig5_gain_fields_v6.py` (`--layout v9`), run as
`python -m retsupp.visualize.paper.run_excluding --exclude 1 2 -- retsupp.visualize.paper.fig5_gain_fields_v6 --suffix _n28 --layout v9`;
parameters `notes/data/fig5_gain_field_map_parameters_n28_2026-11-12_v9.tsv`.
