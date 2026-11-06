"""Event-related mean time series at the voxel's ring location: data, m0 and m7 (candidate Figure 3 panel).

Reads ``notes/data/fig3_event_average/sub-XX_2026-11.tsv`` (``fig3_event_average.py``). Per subject, ROIs are
pooled into the three pathways, weighting each ROI by its number of voxels; each class is baselined on its
mean over the epochs' first three TRs (-3.2 to 0 s). Points and bars: mean ± SEM across subjects.

Usage::

    python -m retsupp.visualize.paper.fig3_event_timecourses
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_GROUPS, TARGET, letter, offset_spines,
                                                 provisional, save, title_icon)

NEUTRAL = '#7B8794'
PANELS = [('Target vs neutral item', ('target', 'neutral_lp'), (TARGET, NEUTRAL), ('Target', 'Neutral item'), ('target',)),
          ('Distractor: at HP vs elsewhere', ('distractor_hp', 'distractor_lp'), (HP, DISTRACTOR), ('At HP', 'Elsewhere'),
           ('distractor', 'hp')),
          ('Neutral item: at HP vs elsewhere', ('neutral_hp', 'neutral_lp'), (HP, NEUTRAL), ('At HP', 'Elsewhere'), ('hp', 'lp'))]
LEGEND = """
**Figure 3 (candidate panel). Responses to the search items at a pRF's ring location: data and models.** Voxels with
cross-run reliability > 0 and a pRF centre within 2° of a ring location (full-data nested fits; no R² criterion). Each
search onset is classed by what appears at the voxel's location: the target, the distractor (at the HP location or
elsewhere) or a neutral item (at HP or elsewhere). The response to the bar (m0's prediction for the bar alone) is
removed from the data and from both models; time series are averaged over trials and voxels per subject, ROIs pooled
per pathway weighted by voxel count, baselined on −3.2 to 0 s. Points and bars: data, mean ± SEM across subjects (n = {n});
dashed lines: m0 (pRFs and items, no gain fields); solid lines: m7 (with gain fields). Trials follow each other every few
seconds, so each epoch also contains neighbouring trials, equally for every class. Provisional: current inputs.
"""


def load(folder):
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    group = {r: g for g, rois in ROI_GROUPS for r in rois}
    d['group'] = d.roi.map(group)
    d = d.dropna(subset=['group'])
    cols = ['data', 'm0', 'm7']
    d[cols] = d[cols].mul(d.n_voxels, axis=0)
    g = d.groupby(['subject', 'group', 'cls', 'lag', 't'])
    p = g[cols].sum().div(g.n_voxels.sum(), axis=0).reset_index()
    base = p[p.lag <= 0].groupby(['subject', 'group', 'cls'])[['data', 'm0', 'm7']].mean()
    p = p.set_index(['subject', 'group', 'cls'])
    p[['data', 'm0', 'm7']] = p[['data', 'm0', 'm7']] - base.reindex(p.index).to_numpy()
    return p.reset_index()


def panel(ax, d, classes, colors):
    for cls, col in zip(classes, colors):
        x = d[d.cls == cls]
        g = x.groupby('t')
        m, se = g.data.mean(), g.data.sem()
        ax.errorbar(m.index, m, yerr=se, fmt='o', ms=3, color=col, mec='w', mew=.4, lw=.9, capsize=0, zorder=3)
        ax.plot(g.m7.mean().index, g.m7.mean(), color=col, lw=1.4, zorder=2)
        ax.plot(g.m0.mean().index, g.m0.mean(), color=col, lw=.9, ls=(0, (3, 1.5)), alpha=.8, zorder=2)
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    offset_spines(ax)


def key(ax, x=.04, y=.95, dy=.13):
    """Glyph key: data points, m7 solid, m0 dashed."""
    tf = ax.transAxes
    ax.errorbar([x + .035], [y], yerr=[[.04], [.04]], fmt='o', ms=3, color=FG, mec='w', mew=.4, lw=.9, capsize=0, transform=tf)
    ax.text(x + .09, y, 'Data', transform=tf, fontsize=6, va='center', color=FG)
    ax.plot([x, x + .07], [y - dy] * 2, color=FG, lw=1.4, transform=tf)
    ax.text(x + .09, y - dy, 'm7, gain fields', transform=tf, fontsize=6, va='center', color=FG)
    ax.plot([x, x + .07], [y - 2 * dy] * 2, color=FG, lw=.9, ls=(0, (3, 1.5)), transform=tf)
    ax.text(x + .09, y - 2 * dy, 'm0, no gain', transform=tf, fontsize=6, va='center', color=FG)


def main(folder, name='fig3_event_timecourses'):
    d = load(folder)
    n = d.subject.nunique()
    groups = [g for g, _ in ROI_GROUPS]
    fig, axes = plt.subplots(len(groups), len(PANELS), figsize=(DOUBLE, 5.0), sharex=True,
                             gridspec_kw=dict(left=.09, right=.99, top=.86, bottom=.09, hspace=.35, wspace=.28))
    for j, (title, classes, colors, labels, marks) in enumerate(PANELS):
        lims = []
        for i, grp in enumerate(groups):
            ax = axes[i, j]
            panel(ax, d[d.group == grp], classes, colors)
            lims.append(ax.get_ylim())
            if j == 0:
                ax.set_ylabel(f'{grp}\nBOLD (% signal)')
            if i == len(groups) - 1:
                ax.set_xlabel('Time from search onset (s)')
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        for i in range(len(groups)):
            axes[i, j].set_ylim(lo, hi)
        ax = axes[0, j]
        title_icon(ax, title, marks, FG, y=1.12, h=.32)
        for k, (lab, col) in enumerate(zip(labels, colors)):
            ax.text(.98, .95 - k * .13, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
        letter(ax, 'abc'[j], x=-.18, y=1.18)
    key(axes[0, 0], x=.04, y=.95)
    provisional(axes[0, 2], y=1.42)
    save(fig, name, LEGEND.format(n=n))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', default='notes/data/fig3_event_average')
    a = ap.parse_args()
    main(a.folder)
