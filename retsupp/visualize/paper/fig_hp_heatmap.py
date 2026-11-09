"""HP vs LP per item type and pathway, in the measured responses: the main effect of location and its interaction.

Per subject and pathway (ROIs pooled weighted by voxel count): the deconvolved peak response (3.2-6.4 s) at an item
location (pRF centre within 2° of it) when the location is HP vs LP, for the target, the distractor and a neutral
item (``fig45_common.item_timecourses``). Each HP − LP difference is expressed in % of that subject's and pathway's mean
LP response over the three items, so pathways with larger responses do not dominate.

Per pathway, a repeated-measures ANOVA, location (HP, LP) × item (target, distractor, neutral), on the peak responses
(same normalisation) gives the two questions separately:

* main effect of location: are items at HP weaker overall? (the model's sustained gain, dS)
* location × item: does the HP effect differ between items? The planned contrast is the distractor against the
  mean of target and neutral item (the model's distractor-specific term, dD).

Usage::

    python -m retsupp.visualize.paper.fig_hp_heatmap
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from retsupp.visualize.paper.fig2_prf_fits import PATHWAY
from retsupp.visualize.paper.fig45_common import item_timecourses
from retsupp.visualize.paper.paper_style import DISTRACTOR, FG, HP, SINGLE, TARGET, save

ITEMS = [('target', 'Target', TARGET), ('distractor', 'Distractor', DISTRACTOR), ('neutral', 'Neutral', '#7B8794')]
WINDOW = (3.2, 6.4)
LEGEND = """
**Figure. Less response at the HP location, per item type and pathway.** Deconvolved response (nideconv) of voxels
whose pRF centre lies within 2° of an item location, mean over {w0:.1f}-{w1:.1f} s after search onset, ROIs pooled per
pathway weighted by voxel count; per subject, the response when the location is HP minus when it is an LP location,
in % of the subject's mean LP response over the three item types in that pathway. Cells: mean over subjects (n = {n}).
Stars in cells: Wilcoxon signed-rank test against 0, uncorrected. Right: per pathway, a repeated-measures ANOVA
location (HP, LP) × item type (target, distractor, neutral) on the same normalised responses; main effect of
location and the location × item interaction, and the planned contrast distractor vs the mean of target and neutral
item (Wilcoxon). {tests}
"""


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def table(groups, folder='notes/data/event_deconv_2026-11'):
    p = item_timecourses(folder, groups=groups)
    pk = p[p.t.between(*WINDOW)].groupby(['subject', 'grp', 'cls']).r.mean().unstack()
    rows = []
    for (s, g), x in pk.iterrows():
        norm = np.mean([x[f'{it}_lp'] for it, _, _ in ITEMS])
        for it, _, _ in ITEMS:
            for loc in ('hp', 'lp'):
                rows.append(dict(subject=s, grp=g, item=it, loc=loc, r=100 * x[f'{it}_{loc}'] / norm))
    return pd.DataFrame(rows)


def hp_tests(folder='notes/data/event_deconv_2026-11'):
    """Per pathway and all areas: per-subject HP − LP per item (% of LP), and the location × item ANOVA.

    Returns (diff, res): diff has one row per subject, group and item; res one row per group."""
    groups = {r: g for g, rs in PATHWAY.items() for r in rs}
    d = pd.concat([table(groups, folder), table(None, folder).assign(grp='All areas')])
    order = list(PATHWAY) + ['All areas']
    diff = d.pivot_table(index=['subject', 'grp', 'item'], columns='loc', values='r')
    diff = (diff.hp - diff.lp).rename('diff').reset_index()
    res = []
    for g in order:
        x = d[d.grp == g]
        a = pg.rm_anova(data=x, dv='r', within=['loc', 'item'], subject='subject')
        main_p = float(a.loc[a.Source == 'loc', 'p-unc'].iloc[0])
        main_f = float(a.loc[a.Source == 'loc', 'F'].iloc[0])
        int_p = float(a.loc[a.Source == 'loc * item', 'p-unc'].iloc[0])
        int_f = float(a.loc[a.Source == 'loc * item', 'F'].iloc[0])
        w = diff[diff.grp == g].pivot(index='subject', columns='item', values='diff')
        c = w.distractor - (w.target + w.neutral) / 2
        res.append(dict(grp=g, main_F=main_f, main_p=main_p, main=w.mean(1).mean(), int_F=int_f, int_p=int_p,
                        d_vs_others=c.mean(), d_vs_others_p=stats.wilcoxon(c).pvalue, n=len(w)))
    return diff, pd.DataFrame(res).set_index('grp')


def main(name='fig_hp_heatmap'):
    diff, res = hp_tests()
    order = list(PATHWAY) + ['All areas']
    cell = diff.groupby(['grp', 'item'])['diff'].mean().unstack().loc[order, [it for it, _, _ in ITEMS]]
    pcell = diff.groupby(['grp', 'item'])['diff'].apply(lambda v: stats.wilcoxon(v).pvalue).unstack().loc[order, cell.columns]
    print(cell.round(1).to_string(), '\n', res.round(4).to_string())

    fig = plt.figure(figsize=(SINGLE * 1.45, 2.3))
    ax = fig.add_axes([.2, .24, .42, .55])
    lim = 15
    cmap = mpl.colormaps['RdBu_r']
    ax.imshow(cell.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
    for i in range(len(order)):
        for j in range(len(ITEMS)):
            v, pv = cell.iloc[i, j], pcell.iloc[i, j]
            ax.text(j, i, (f'{v:+.0f}%\n{stars(pv)}' if stars(pv) else f'{v:+.0f}%').replace('-', '−'), ha='center', va='center', fontsize=6,
                    color='w' if abs(v) > .6 * lim else FG, linespacing=.9)
    ax.axhline(len(order) - 1.5, color='w', lw=2)
    ax.set_xticks(range(len(ITEMS)))
    ax.xaxis.tick_top()
    ax.set_xticklabels([lab for _, lab, _ in ITEMS], fontsize=6.5)
    for t, (_, _, col) in zip(ax.get_xticklabels(), ITEMS):
        t.set_color(col)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-.02, 1.3, 'Items at HP vs LP', transform=ax.transAxes, fontsize=7, color=HP, ha='right', va='center')
    # Right: the two tests per pathway.
    for k, (col, lab) in enumerate((('main', 'All items'), ('int', 'Item ×\nlocation'))):
        x0 = 1.15 + k * .38
        ax.text(x0, 1.03, lab, transform=ax.transAxes, fontsize=6.5, color=FG, ha='center', va='bottom', linespacing=.95)
        for i, g in enumerate(order):
            pv = res.loc[g, f'{col}_p']
            txt = ((f"{res.loc[g, 'main']:+.0f}% " if col == 'main' else '') + (stars(pv) or 'n.s.')).replace('-', '−')
            ax.text(x0, 1 - (i + .5) / len(order), txt, transform=ax.transAxes, fontsize=6, ha='center', va='center',
                    color=HP if (col == 'main' and pv < .05) else FG if pv < .05 else '.5')
    cax = ax.inset_axes([.15, -.12, .7, .05])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(-lim, lim), orientation='horizontal')
    cb.set_ticks([-lim, 0, lim])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)
    cax.set_xlabel('HP − LP (% of LP response)', fontsize=6, labelpad=1)
    tests = '; '.join(f"{g}: location F(1, {int(r.n) - 1}) = {r.main_F:.2f}, p = {r.main_p:.3g}; location × item "
                      f"F(2, {2 * (int(r.n) - 1)}) = {r.int_F:.2f}, p = {r.int_p:.3g}; distractor vs others "
                      f"{r.d_vs_others:+.1f}%, p = {r.d_vs_others_p:.3g}" for g, r in res.iterrows())
    save(fig, name, LEGEND.format(w0=WINDOW[0], w1=WINDOW[1], n=int(res.n.iloc[0]), tests=tests))
    res.to_csv('notes/data/hp_item_anova_2026-11.tsv', sep='\t')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    main()
