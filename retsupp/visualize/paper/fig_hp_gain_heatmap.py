"""Estimated gain at HP vs LP per item type and ROI: the location effect and the item-specific effects.

From the joint gain-field fits (DoG pRFs, all terms fitted together): the gain an item receives at the HP location
minus at an LP location, at the field centre, in % (gain 0.01 = 1% more drive).

* neutral item: dS = g_HP − g_LP (sustained gain; applies to every item at that location)
* distractor:   dS + dD, dD = g_HP_dyn − g_LP_dyn (m7_full)
* target:       dS + dT, dT = target gain at HP − at LP (m8_full_dT, ``joint_dT_dog_2026-11.tsv``)

The right-hand block tests the components separately: the location effect (dS, the main effect) and the
item-specific terms (dD, dT, the interactions). Wilcoxon signed-rank tests against 0, Benjamini-Hochberg across ROIs.

Usage::

    python -m retsupp.visualize.paper.fig_hp_gain_heatmap
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import joint
from retsupp.visualize.paper.paper_style import DISTRACTOR, FG, HP, MUTED, PROVISIONAL, ROI_ORDER, SINGLE, TARGET, save

GROUPS = [('Early', ('V1', 'V2', 'V3')), ('Dorsal', ('V3AB', 'IPS')), ('Occipitotemporal', ('hV4', 'VO', 'LO', 'TO'))]
LEGEND = """
**Figure. Estimated gain at the HP location relative to the LP locations, per item type and ROI.** Joint gain-field
fits (DoG pRFs, all terms fitted together); gain at the field centre, HP minus LP, in % (+1%: 1% more drive at HP).
Left: what each item receives, neutral item dS = g_HP − g_LP (sustained gain), distractor dS + dD (dD, its transient
gain at HP minus at LP; m7_full), target dS + dT (dT, its transient gain at HP minus at LP; m8_full_dT). Right: the
components, the location effect common to all items (dS) and the item-specific terms (dD, dT). Colour: median over
subjects (n = {n}); stars: Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs per column;
* q < .05, ** q < .01, *** q < .001. {tests} Provisional: current inputs.
"""


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def load(gain_table, dt_table):
    d = joint(gain_table, 'dog')[['subject', 'roi', 'dS', 'dD']]
    t = pd.read_csv(dt_table, sep='\t')[['subject', 'roi', 'dT']]
    d = d.merge(t, on=['subject', 'roi'], how='left')
    d[['dS', 'dD', 'dT']] *= 100
    d['Neutral'] = d.dS
    d['Distractor'] = d.dS + d.dD
    d['Target'] = d.dS + d.dT
    return d


def column_stats(d, col):
    med, q = [], []
    ps = []
    for roi in ROI_ORDER:
        v = d[d.roi == roi][col].dropna()
        med.append(v.median())
        ps.append(stats.wilcoxon(v).pvalue)
    return np.array(med), stats.false_discovery_control(ps), np.array(ps)


def main(gain_table, dt_table, name='fig_hp_gain_heatmap'):
    d = load(gain_table, dt_table)
    left = [('Target', TARGET), ('Distractor', DISTRACTOR), ('Neutral', '#7B8794')]
    right = [('dS', 'All items\n(location)', HP), ('dD', 'Distractor\nonly', DISTRACTOR), ('dT', 'Target\nonly', TARGET)]
    cols = [c for c, _ in left] + [c for c, _, _ in right]
    M = np.full((len(ROI_ORDER), len(cols)), np.nan)
    Q = np.ones_like(M)
    out = []
    for j, c in enumerate(cols):
        m, q, p = column_stats(d, c)
        M[:, j], Q[:, j] = m, q
        if c in ('dS', 'dD', 'dT'):
            out.append(f'{c}: ' + ', '.join(f'{r} {mm:+.1f}% (q = {qq:.2g})' for r, mm, qq in zip(ROI_ORDER, m, q)))
    print(pd.DataFrame(M, index=ROI_ORDER, columns=cols).round(2).to_string())
    print(pd.DataFrame(Q, index=ROI_ORDER, columns=cols).round(3).to_string())

    lim = max(3., np.ceil(np.nanmax(np.abs(M))))
    cmap = mpl.colormaps['RdBu_r']
    fig = plt.figure(figsize=(SINGLE * 1.25, 3.3))
    # Rows: ROIs with a gap between pathways.
    ypos, y = {}, 0.
    for g, rois in GROUPS:
        for r in rois:
            ypos[r] = y
            y += 1
        y += .45
    ax = fig.add_axes([.25, .15, .7, .6])
    xpos = [0, 1, 2, 3.45, 4.45, 5.45]
    for j, c in enumerate(cols):
        for i, r in enumerate(ROI_ORDER):
            v = M[i, j]
            ax.add_patch(plt.Rectangle((xpos[j] - .48, ypos[r] - .48), .96, .96, color=cmap(.5 + v / (2 * lim)), lw=0))
            ax.text(xpos[j], ypos[r], stars(Q[i, j]), ha='center', va='center', fontsize=6.5,
                    color='w' if abs(v) > .6 * lim else FG)
    for g, rois in GROUPS:
        y0 = np.mean([ypos[r] for r in rois])
        ax.text(-1.6, y0, g, ha='right', va='center', fontsize=6, color=MUTED, rotation=0)
    ax.set_yticks([ypos[r] for r in ROI_ORDER])
    ax.set_yticklabels(ROI_ORDER, fontsize=6.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([c for c, _ in left] + [lab for _, lab, _ in right], fontsize=6.5, rotation=90)
    for t, col in zip(ax.get_xticklabels(), [c for _, c in left] + [c for _, _, c in right]):
        t.set_color(col)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-.6, xpos[-1] + .6)
    ax.set_ylim(y - .45 + .1, -.6)
    for x0, lab in ((1, 'Gain per item'), (4.45, 'Components')):
        ax.annotate(lab, xy=(x0, 1), xycoords=('data', 'figure fraction'), xytext=(0, -14), textcoords='offset points',
                    ha='center', va='top', fontsize=7, color=FG)
    cax = fig.add_axes([.35, .085, .5, .018])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(-lim, lim), orientation='horizontal')
    cb.set_ticks([-lim, 0, lim])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)
    cax.set_xlabel('Gain at HP − LP (%)', fontsize=6, labelpad=1)
    fig.text(.01, .005, PROVISIONAL, fontsize=5.5, color=MUTED, ha='left', va='bottom')
    save(fig, name, LEGEND.format(n=d.subject.nunique(), tests='; '.join(out) + '.'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--dt-table', default='notes/data/joint_dT_dog_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.dt_table)
