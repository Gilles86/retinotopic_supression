"""Figure: what appears at a pRF's ring location changes its response (data only, no gain-field model).

a, What is measured: voxels whose pRF sits on one item location, sorted by what appears there.
b, Response when the target vs a neutral item appears at that location (all nine ROIs).
c, The same for the distractor at the HP location vs elsewhere.
d, Peak response difference (4.8-6.4 s) per pathway, one dot per subject.

Reads ``notes/data/fig3_event_average/sub-XX_2026-11.tsv`` (``fig3_event_average.py``); pooling as in
``fig3_event_timecourses.load``.

Usage::

    python -m retsupp.visualize.paper.fig_event_responses
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_GROUPS, ROI_ORDER, TARGET, letter,
                                                 offset_spines, save, title_icon)

NEUTRAL = '#7B8794'
PEAK = (4.7, 6.5)
LEGEND = """
**Figure. What appears at a pRF's ring location changes its response.** Voxels with cross-run reliability > 0 and a pRF
centre within 2° of one of the four ring locations (no fit-quality criterion, no gain-field model). Every search onset
is classed by what appears at the voxel's location. The response to the bar (the pRF prediction for the bar alone) is
removed; time series are averaged over trials and voxels per subject and baselined on −3.2 to 0 s. a, Schematic of the
measure: the response of voxels whose pRF (teal) covers one of the four item locations of the search display. b, The
target vs a neutral item at that location (outside HP), all nine ROIs pooled, weighted by voxel count. c, The distractor
at the HP location vs at an LP location. Lines and bands: mean ± SEM across subjects (n = {n}); grey strip, the search display (1.5 s); grey
column, the window used in d. Trials follow each other every few seconds, so each epoch also contains neighbouring trials,
equally for every class. d, Mean difference in the window 4.8-6.4 s per pathway (early V1-V3; dorsal V3AB, IPS;
occipitotemporal hV4, VO, LO, TO): dots are subjects, points and bars the mean and 95% CI (t); Wilcoxon signed-rank
test against 0, Benjamini-Hochberg over the six tests: * q < .05, ** q < .01, *** q < .001.
"""


def load(folder, groups):
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    d['group'] = d.roi.map(groups)
    d = d.dropna(subset=['group'])
    cols = ['data']
    d[cols] = d[cols].mul(d.n_voxels, axis=0)
    g = d.groupby(['subject', 'group', 'cls', 't'])
    p = g[cols].sum().div(g.n_voxels.sum(), axis=0).reset_index()
    base = p[p.t <= 0].groupby(['subject', 'group', 'cls']).data.mean()
    p = p.set_index(['subject', 'group', 'cls'])
    p['data'] = p['data'] - base.reindex(p.index).to_numpy()
    return p.reset_index()


PRF_TEAL = '#2A9D8F'


def schematic(ax):
    """The display with one voxel's pRF on an item location: what the time series are measured from."""
    from matplotlib.patches import Circle
    ring = 4. / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.6))
    for a in np.deg2rad(np.arange(8) * 45):
        ax.add_patch(plt.Rectangle((4 * np.cos(a) - .45, 4 * np.sin(a) - .12), .9, .24, color='.55', lw=0))
    ax.add_patch(Circle(ring[1], 1.2, fill=False, ec=PRF_TEAL, lw=1.4))
    ax.plot(0, 0, '+', color=FG, ms=4, mew=.6)
    ax.annotate('pRF on an item\nlocation', xy=ring[1] + [.75, .95], xytext=(-.6, 7.2), fontsize=6.5, color=PRF_TEAL, ha='left',
                va='center', arrowprops=dict(arrowstyle='-|>', color=PRF_TEAL, lw=1., mutation_scale=7, shrinkA=1, shrinkB=2,
                                             relpos=(0, .5)))
    ax.text(0, -5.6, 'Its response after each\nsearch onset, sorted by\nwhat appears there', ha='center', va='top', fontsize=6.5,
            color=FG, linespacing=1.25)
    ax.set_xlim(-5.6, 5.6)
    ax.set_ylim(-9.8, 8.6)
    ax.set_aspect('equal')
    ax.set_axis_off()


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def timecourse(ax, d, classes, colors, labels):
    for k, (lab, col) in enumerate(zip(labels, colors)):
        ax.text(.98, .97 - k * .12, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
    for cls, col, lab in zip(classes, colors, labels):
        m = d[d.cls == cls].groupby('t').data
        mu, se = m.mean(), m.sem()
        ax.fill_between(mu.index, mu - se, mu + se, color=col, alpha=.22, lw=0, zorder=1)
        ax.plot(mu.index, mu, color=col, lw=1.5, zorder=2)
    ax.axvspan(*PEAK, color='#ECEFF2', lw=0, zorder=0)
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    lo, hi = ax.get_ylim()
    ax.add_patch(plt.Rectangle((0, lo), 1.5, (hi - lo) * .04, color=MUTED, lw=0, zorder=3, clip_on=False))
    ax.text(1.8, lo + (hi - lo) * .02, 'Search', fontsize=6, color='.4', va='center', ha='left')
    ax.set_ylim(lo, hi)
    ax.set_xlim(-3.4, 19.4)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel('Time from search onset (s)')
    offset_spines(ax)


def summary(ax, d, effects, groups):
    rows, pos = [], []
    gap = len(groups) + 1
    for j, (lab, (a, b), col) in enumerate(effects):
        for i, grp in enumerate(groups):
            w = d[(d.group == grp) & d.t.between(*PEAK)].groupby(['subject', 'cls']).data.mean().unstack()
            v = (w[a] - w[b]).dropna().to_numpy()
            x = j * gap + i
            rng = np.random.default_rng(10 * j + i)
            ax.scatter(x + rng.uniform(-.16, .16, len(v)), v, s=6, color=col, alpha=.35, lw=0, zorder=1)
            m, h = v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.errorbar(x, m, yerr=h, fmt='o', ms=5, color=col, mec='w', mew=.7, lw=1.4, capsize=0, zorder=3)
            rows.append(stats.wilcoxon(v).pvalue)
            pos.append((x, col))
    for (x, col), q in zip(pos, stats.false_discovery_control(rows)):
        ax.text(x, 1.0, stars(q), ha='center', va='bottom', fontsize=7, color=col, transform=ax.get_xaxis_transform())
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    ax.set_xticks([j * gap + i for j in range(len(effects)) for i in range(len(groups))])
    ax.set_xticklabels([{'Occipitotemporal': 'Occ.-temp.'}.get(g, g) for g in groups] * len(effects), fontsize=6.5)
    ax.tick_params(axis='x', labelrotation=90)
    for j, (lab, _, col) in enumerate(effects):
        ax.text(j * gap + (len(groups) - 1) / 2, 1.13, lab, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=6.5, color=col)
    ax.set_xlim(-.6, (len(effects) - 1) * gap + len(groups) - .4)
    ax.set_ylabel('Peak difference (% signal)')
    offset_spines(ax)


def main(folder, name='fig_event_responses'):
    groups = {r: g for g, rois in ROI_GROUPS for r in rois}
    pooled = load(folder, {r: 'All' for r in ROI_ORDER})
    by_group = load(folder, groups)
    n = pooled.subject.nunique()
    fig = plt.figure(figsize=(DOUBLE, 2.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[.8, 1.1, 1.1, 1.05], left=.01, right=.985, top=.76, bottom=.3, wspace=.45)
    ax = fig.add_subplot(gs[0])
    schematic(ax)
    letter(ax, 'a', x=.08, y=1.08)
    ax = fig.add_subplot(gs[1])
    timecourse(ax, pooled, ('target', 'neutral_lp'), (TARGET, NEUTRAL), ('Target', 'Neutral item'))
    ax.set_ylabel('Response at the pRF (% signal)')
    title_icon(ax, 'Larger for the target', ('target',), TARGET, y=1.08, h=.3)
    letter(ax, 'b', x=-.36, y=1.15)
    lims = ax.get_ylim()
    ax = fig.add_subplot(gs[2])
    timecourse(ax, pooled, ('distractor_lp', 'distractor_hp'), (DISTRACTOR, HP), ('Distractor elsewhere', 'Distractor at HP'))
    title_icon(ax, 'Distractor: smaller at HP', ('distractor', 'hp'), HP, y=1.08, h=.3)
    letter(ax, 'c', x=-.12, y=1.15)
    ax.set_ylim(lims)
    ax = fig.add_subplot(gs[3])
    summary(ax, by_group, [('Target − neutral', ('target', 'neutral_lp'), TARGET),
                           ('HP − elsewhere', ('distractor_hp', 'distractor_lp'), HP)], [g for g, _ in ROI_GROUPS])
    letter(ax, 'd', x=-.38, y=1.15)
    save(fig, name, LEGEND.format(n=n))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', default='notes/data/fig3_event_average')
    a = ap.parse_args()
    main(a.folder)
