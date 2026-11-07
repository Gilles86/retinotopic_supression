"""Location gain (dS) or distractor-specific suppression (dD)? The response at the HP location, per item type, in the data.

A location gain at HP lowers the response to every item there by the same proportion; a distractor-specific
suppression lowers only the distractor's. Per subject: the deconvolved peak response (3.2-6.4 s) at HP relative to LP,
for the target, the distractor and a neutral item (``fig_event_responses_2x3.load_deconv``; all nine ROIs pooled).

Usage::

    python -m retsupp.visualize.paper.fig_hp_items
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig_event_responses_2x3 import NEUTRAL, load_deconv
from retsupp.visualize.paper.paper_style import DISTRACTOR, FG, HP, SINGLE, TARGET, offset_spines, save, title_icon

ITEMS = [('target', 'Target', TARGET), ('distractor', 'Distractor', DISTRACTOR), ('neutral', 'Neutral', NEUTRAL)]
PEAK = (3.2, 6.4)
LIM = (-40, 30)
LEGEND = """
**Figure. The response at the HP location, per item type.** Deconvolved response (nideconv) of voxels whose pRF centre
lies within 2° of an item location, mean over {p0:.1f}-{p1:.1f} s after search onset, all nine ROIs pooled weighted by voxel
count; per subject the response when the location is HP relative to when it is an LP location, in %. Dots are subjects
(triangles beyond the axis), points and bars the mean and 95% CI (t), n = {n}. Wilcoxon signed-rank tests against 0:
{tests}. The bracket compares the distractor with the mean of target and neutral item ({contrast}).
"""


def main(folder, name='fig_hp_items'):
    d = load_deconv(folder)
    pk = d[d.t.between(*PEAK)].groupby(['subject', 'item', 'loc']).data.mean().unstack(['item', 'loc'])
    rel = pd.DataFrame({it: 100 * (pk[(it, 'hp')] - pk[(it, 'lp')]) / pk[(it, 'lp')] for it, _, _ in ITEMS}).dropna()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3), gridspec_kw=dict(left=.2, right=.97, top=.72, bottom=.13))
    lo, hi = LIM
    tests = []
    for i, (it, lab, col) in enumerate(ITEMS):
        v = rel[it].to_numpy()
        rng = np.random.default_rng(i)
        xx = i + rng.uniform(-.15, .15, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=8, color=col, alpha=.4, lw=0)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=10, marker='v', color=col, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=10, marker='^', color=col, alpha=.6, lw=0, clip_on=False)
        m, h = v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(i + .3, m, yerr=h, fmt='o', ms=5.5, color=col, mec='w', mew=.7, lw=1.4, capsize=0, zorder=4)
        p = stats.wilcoxon(v).pvalue
        tests.append(f'{lab.lower()} {m:+.1f}% [{m - h:+.1f}, {m + h:+.1f}], p = {p:.2g}')
        ax.text(i + .1, hi + 2, f'p = {p:.2f}'.replace('0.', '.'), ha='center', va='bottom', fontsize=6, color=col, clip_on=False)
    x = rel.distractor - (rel.target + rel.neutral) / 2
    pc = stats.wilcoxon(x).pvalue
    contrast = f'{x.mean():+.1f}%, p = {pc:.2g}'
    y = hi + 10
    ax.plot([1.1, 1.1, .1, .1], [y - 2, y, y, y - 2], color=FG, lw=.6, clip_on=False)
    ax.plot([1.1, 1.1, 2.1, 2.1], [y - 2, y, y, y - 2], color=FG, lw=.6, clip_on=False)
    ax.text(1.1, y + 1, f'Distractor vs others: p = {pc:.2f}'.replace('0.', '.'), ha='center', va='bottom', fontsize=6, color=FG,
            clip_on=False)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xticks([i + .1 for i in range(len(ITEMS))])
    ax.set_xticklabels([lab for _, lab, _ in ITEMS])
    for t, (_, _, col) in zip(ax.get_xticklabels(), ITEMS):
        t.set_color(col)
    ax.set_xlim(-.45, len(ITEMS) - .4)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('At HP vs elsewhere (%)')
    offset_spines(ax)
    title_icon(ax, 'Response at the HP location', ('hp',), HP, y=1.3, h=.2)
    save(fig, name, LEGEND.format(p0=PEAK[0], p1=PEAK[1], n=len(rel), tests='; '.join(tests), contrast=contrast))
    print(tests, contrast)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', default='notes/data/event_deconv_2026-11')
    a = ap.parse_args()
    main(a.folder)
