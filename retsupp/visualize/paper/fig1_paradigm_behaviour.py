"""Figure 1: the paradigm and behaviour.

``--rt-table`` holds per-subject mean RTs; it is built from the events if missing.

Usage::

    python -m retsupp.visualize.paper.fig1_paradigm_behaviour
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, MUTED, letter, offset_spines, save
from retsupp.visualize.paper.plot_fig1 import panel_display, panel_trial

LEGEND = """
**Figure 1. Paradigm and behaviour.** a, Search display with the bar in its 3.17° aperture: 8 items on a ring at 4°, with target and singleton distractor at the four diagonal positions; percentages are over all trials. b, Trial timing to scale and the HP location across blocks of runs (example order). c, Response time with the singleton distractor at the HP location, at an LP location, or absent. d, Response time with the target at the HP or an LP location, on trials with a distractor. c, d, n = 30, correct trials. Dots are subjects, triangles are subjects beyond the axis limits; large points and bars are the mean ± within-subject SEM (Cousineau-Morey). Wilcoxon signed-rank tests: *** p < .001.
"""


def build_rt(path, bids):
    from retsupp.visualize.vss2026 import plot_rt_talk as rt
    df = rt.add_condition_labels(rt.collect_trials(bids))
    a = rt.per_subject_means(df, 'dist_cond', ['HP', 'LP', 'Absent']).rename(columns={'dist_cond': 'condition'}).assign(contrast='distractor')
    b = rt.per_subject_means(df[df.has_distractor], 'target_cond', ['HP', 'LP']).rename(columns={'target_cond': 'condition'}).assign(contrast='target')
    pd.concat([a, b]).to_csv(path, sep='\t', index=False)


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


def rt_panel(ax, d, conds, colors, pairs, ylim):
    """Actual subject RTs on an axis capped around the means; subjects beyond it are triangles on
    the cap, so none is dropped. Error bars are within-subject SEM (Cousineau-Morey)."""
    w = d.pivot(index='subject', columns='condition', values='rt')[conds]
    k = len(conds)
    m = w.mean()
    se = (w.sub(w.mean(1), axis=0) + w.values.mean()).sem() * np.sqrt(k / (k - 1))
    lo, hi = ylim
    rng = np.random.default_rng(3)
    for i, c in enumerate(conds):
        xx = i + rng.uniform(-.14, .14, len(w))
        v = w[c].to_numpy()
        ok, over, under = (v >= lo) & (v <= hi), v > hi, v < lo
        ax.scatter(xx[ok], v[ok], s=9, color=colors[i], alpha=.3, lw=0, zorder=1)
        ax.scatter(xx[over], np.full(over.sum(), hi), s=10, marker='^', color=colors[i], alpha=.55, lw=0, zorder=1, clip_on=False)
        ax.scatter(xx[under], np.full(under.sum(), lo), s=10, marker='v', color=colors[i], alpha=.55, lw=0, zorder=1, clip_on=False)
        ax.errorbar(i, m[c], yerr=se[c], color=colors[i], marker='o', ms=6, mfc=colors[i], mec='w', mew=.7, lw=1.4, capsize=0, zorder=4)
    ax.plot(range(k), m.values, color=FG, lw=.8, zorder=3)
    span = hi - lo
    for n, (a, b) in enumerate(pairs):
        diff = w[a] - w[b]
        i, j = conds.index(a), conds.index(b)
        y = hi + (.2 + n * .17) * span
        ax.plot([i, j], [y, y], color=FG, lw=.7, clip_on=False)
        ax.text((i + j) / 2, y + .01 * span, f'{diff.mean():+.0f} ms {stars(stats.wilcoxon(diff).pvalue)}'.replace('-', '−'),
                ha='center', va='bottom', fontsize=6.5, color=FG, clip_on=False)
    ax.set_xticks(range(k))
    ax.set_xticklabels(conds)
    ax.set_xlim(-.45, k - .55)
    ax.set_ylim(lo, hi)
    ax.set_yticks([800, 900, 1000])
    offset_spines(ax)


def main(rt_table, bids, name='fig1_paradigm_behaviour'):
    if not Path(rt_table).exists():
        build_rt(rt_table, bids)
    d = pd.read_csv(rt_table, sep='\t')
    fig = plt.figure(figsize=(DOUBLE, 3.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.62], height_ratios=[1.12, 1], left=.005, right=.995, top=.975, bottom=.13, hspace=.46, wspace=.17)
    ax = fig.add_subplot(gs[:, 0])
    panel_display(ax, margin=7.4)
    box = ax.get_position()
    ax.set_position([box.x0, .005, box.width, .985])   # down to the figure edge, the x labels of c and d need no room here
    ax.set_anchor('N')
    letter(ax, 'a', x=.05, y=.93)
    ax = fig.add_subplot(gs[0, 1])
    panel_trial(fig, ax)
    letter(ax, 'b', x=0, y=.9)
    low = gs[1, 1].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=.2)
    ylim = (770, 1020)
    ax = fig.add_subplot(low[0])
    rt_panel(ax, d[d.contrast == 'distractor'], ['HP', 'LP', 'Absent'], [HP, FG, MUTED], [('HP', 'LP'), ('LP', 'Absent')], ylim)
    ax.set_xlabel('Singleton distractor')
    ax.set_ylabel('Response time (ms)')
    letter(ax, 'c', x=-.2, y=1.2)
    ax = fig.add_subplot(low[1])
    rt_panel(ax, d[d.contrast == 'target'], ['HP', 'LP'], [HP, FG], [('HP', 'LP')], ylim)
    ax.set_xlabel('Target')
    ax.set_yticklabels([])
    letter(ax, 'd', x=-.1, y=1.2)
    ax.scatter([.1], [1.13], s=10, marker='^', color=MUTED, lw=0, transform=ax.transAxes, clip_on=False)
    ax.scatter([.16], [1.13], s=10, marker='v', color=MUTED, lw=0, transform=ax.transAxes, clip_on=False)
    ax.text(.21, 1.13, 'Off scale', transform=ax.transAxes, fontsize=6, color=FG, va='center')
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rt-table', default='notes/data/behaviour_rt_2026-11.tsv')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    a = ap.parse_args()
    main(a.rt_table, a.bids_folder)
