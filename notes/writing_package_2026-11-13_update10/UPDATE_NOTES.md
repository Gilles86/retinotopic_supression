# Update 10: do the gain terms grow along the visual hierarchy? (applies on top of update 8)

**Fits:** the refitted (polished) reference of update 8, i.e. DoG pRFs refitted together with the gain fields
(`af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished`, 241 of 252 participant × area cells),
with update 8's estimated-alone terms and field widths. The voxel selection is unchanged. A converged refit is in
progress and may change the size of the LP gain (dS is stable). Everything here reruns on another root with one
command per stage (below). Update 9 was not issued; this update covers what it was planned for.

## Files

| File | Status | What |
|---|---|---|
| `figures/figS_hierarchy_gain_terms_polished_2026-11-12_n28.{pdf,png,md}` | **new** supplementary figure | a, area medians of target, distractor, shared sustained gain and dS against area pRF size, each relative to its own pooled median; b, the slopes; c, within areas, gains by pRF-size tercile (in %); d, sustained field width against pRF size |
| `figures/table_hierarchy_gain_terms_polished_2026-11-12_n28.md` | **new** supplementary table | Slopes of every term for DoG, Gaussian and DN pRFs; all nine areas, rank predictor, both branches |
| `key_numbers_additions.md` | **append** | All new numbers with source tables |
| `suggested_text.md` | new | Results (section 5) and Discussion paragraph |

Details and all tests: `notes/hierarchy_gain_terms_2026-11-13.md`.

## What it shows (short)

1. The target gain grows along the hierarchy (+27 pp per doubling of pRF size, p = .002), in both streams and with
   all three pRF models. The shared sustained gain becomes more negative higher up. dS, dD and dT do not change
   (dS −0.7 pp [−3.7, +3.3], p = .61).
2. dS is flat in absolute terms: growth beyond ±3.7 pp per doubling is excluded (minimum detectable slope 4.6 pp),
   far below the target's +27 pp (paired p = .018). Relative to its own size (≈ 1%) the data are too noisy to say
   whether dS grows (p = .49 for target vs dS in relative units).
3. dS is closer to a constant amount than to a constant fraction of the shared sustained gain: the ratio falls from
   about 1 (V1, V3, hV4) to about 0.2 (higher areas).
4. Within areas, the target gain is larger in voxels with larger pRFs (+7 pp per doubling, p = .007), but within
   an area pRF size goes with eccentricity, so this cannot be attributed to pRF size; dS shows no within-area trend.
5. The sustained field width does not scale with pRF size (|gain|/SE > 3: −0.04° per doubling, within ±0.32°).

Under DN pRFs (not refitted), dS estimated alone is larger in higher areas (−2.4 pp per doubling, p = .023, q = .105).
Mention this in the robustness paragraph and do not build on it.

## What to change in the manuscript

- Results 5 (gain vs hierarchy): replace or extend with `suggested_text.md` §Results. If the current text says the
  target gain rises with pRF size (update 8: +27% per doubling), keep that sentence; the new part is the comparison
  with dS and the field width.
- Discussion: add the paragraph in `suggested_text.md` §Discussion (gain accumulated up to an area; dS not built up
  stage by stage).
- Supplement: add the figure and the table after Fig. S7 (pooled dS, dD), or where the numbering of update 7/8 puts
  the hierarchy material.

## Regenerating for another reference fit

```bash
# 1. regenerate_reference.sh (update 8) for that root first: it writes the joint, estimated-alone and width tables
# 2. cluster login node, ~/retsupp_plan after git pull:
REF_ROOT=<root> TAG=<tag> bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh cluster-submit   # 28 tasks, ~5 min each
# 3. laptop, repo root, when done:
REF_ROOT=<root> TAG=<tag> bash retsupp/visualize/paper/regenerate_hierarchy_gain_terms.sh local
```

Writes `notes/data/hierarchy_gain_terms_<TAG>_n28_*.tsv`, the table and the figure under
`notes/figures/paper_2026-11/` with `<TAG>` in their names; nothing earlier is overwritten.

## Not done / caveats

- Gaussian and DN columns use the fits without the pRF refit (as in update 8).
- The shared sustained gain estimated alone comes from a model that was not refitted (as in update 8).
- Within-area bins drop areas with fewer than 60 voxels (mostly IPS and VO; 170 of 241 fits remain).
