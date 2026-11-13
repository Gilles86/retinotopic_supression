# Item 4b: Fig. 5 without panel d, and panel d as its own supplementary figure

**What was done.** Fig. 5 v11 is split in two, with the same code, data and statistics. v11 itself is unchanged
(`notes/figures/paper_2026-11/fig5_gain_fields_v11_n28{,_palA}.*`, not rewritten). Both new figures use the new
colour scheme (palette A: purple HP and sustained field, green distractor). The only changes are the split itself and
panel c widened to the full width.

| File | What |
|---|---|
| `notes/writing_package_2026-11-13_update11/figures/fig5_gain_fields_v11_abc_n28_palA.{pdf,png,md}` | **Fig. 5, panels a–c.** a, maps; b, target and distractor gain; c, dS (all areas pooled, then per area), now over the full width. The legend is v11's without panel d: the d text and the dD, dT and dD − dT statistics are removed; "b–d" becomes "b, c"; the transient HP − LP difference is referred to the supplement |
| `notes/writing_package_2026-11-13_update11/figures/figS_transient_hp_lp_n28_palA.{pdf,png,md}` | **Supplementary figure, panel d alone.** dD and dT estimated alone per area, with the Fig. 1 icons. Its legend carries the d text and statistics (estimated alone: dD, dT, dD − dT; joint dD), and notes that the joint fit has no separate target term at HP |

The same files are also in `notes/figures/paper_2026-11/`.

**In plain language.** Both versions exist side by side, so the choice between "d in the main figure" (v11) and "d in
the supplement" (abc + figS) needs no further rendering.

**If panel d moves to the supplement:**
- The main text keeps one sentence on the transient terms, with the hedged wording of item 1b ("not reliably
  different from zero, but not shown to be absent").
- The panel's title "slightly less, target and distractor alike" should be read with item 1b in mind. dT estimated
  alone leans negative (p = .032, BF10 up to 6.5). "Alike" is supported only in the weak sense that dD − dT is not
  different from 0, which is itself bounded only at ±5 pp.

## How to regenerate

The split is a new layout option, `--split-d`, of the existing v11 code path (`fig5_gain_fields_v6.py`); no code was
copied. It needs `--layout v11`. It writes only the two files above. The full v11 figure, the search supplement and the
side tables are not written.

```bash
D=notes/data; TAG=polished_2026-11-12
~/mambaforge/envs/retsupp/bin/python -m retsupp.visualize.paper.run_excluding --exclude 1 2 -- \
    retsupp.visualize.paper.fig5_gain_fields_v6 --layout v11 --split-d --suffix _n28_palA \
    --gain-table $D/gain_field_parameters_${TAG}.tsv --conditional-table $D/conditional_estimates_${TAG}.tsv \
    --slope-table $D/hierarchy_regression_${TAG}_n28.tsv --widths-table $D/field_widths_reliable_${TAG}_n28_widths_thr2.tsv
```

These are the same inputs as the v11 line of `notes/figures/palette_proposal_2026-11-13/render_palA.sh` and
`retsupp/visualize/paper/regenerate_reference.sh`, plus `--split-d`. One run takes about 1 min.

## Not done (as asked: no polish)

- **Supplementary figure:** the "As Fig. 1c, d" note sits close to the significance star above LO.
- **Panel letters:** the supplementary figure has no panel letter.
- **Panel c:** keeps its v11 height.
