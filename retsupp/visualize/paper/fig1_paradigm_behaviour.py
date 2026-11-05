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
**Figure 1. Paradigm and behaviour.** a, Search display with the bar in its 3.17° aperture: 8 items on a ring at 4°, with target and singleton distractor at the four diagonal positions; percentages are over all trials. b, Trial timing to scale and the HP location across blocks of runs (example order). c, Response time with the singleton distractor at the HP location, at an LP location, or absent. d, Response time with the target at the HP or an LP location, on trials with a distractor. c, d, n = 30, lines are subjects, points and bars are mean ± SEM across subjects, correct trials, Wilcoxon signed-rank tests.
"""


def build_rt(path, bids):
    from retsupp.visualize.vss2026 import plot_rt_talk as rt
    df = rt.add_condition_labels(rt.collect_trials(bids))
    a = rt.per_subject_means(df, 'dist_cond', ['HP', 'LP', 'Absent']).rename(columns={'dist_cond': 'condition'}).assign(contrast='distractor')
    b = rt.per_subject_means(df[df.has_distractor], 'target_cond', ['HP', 'LP']).rename(columns={'target_cond': 'condition'}).assign(contrast='target')
    pd.concat([a, b]).to_csv(path, sep='\t', index=False)


def p_text(p):
    return 'p < .0001' if p < 1e-4 else f'p = {p:.3f}'.replace('0.', '.')


def rt_panel(ax, d, conds, colors, pairs):
    w = d.pivot(index='subject', columns='condition', values='rt')[conds]
    x = np.arange(len(conds))
    for _, row in w.iterrows():
        ax.plot(x, row.values, color=MUTED, lw=.4, alpha=.5, zorder=1)
    m, se = w.mean(), w.sem()
    for i, c in enumerate(conds):
        ax.errorbar(i, m[c], yerr=se[c], color=colors[i], marker='o', ms=4, lw=1.2, capsize=0, zorder=3)
    top = np.percentile(w.values, 97)
    for k, (a, b) in enumerate(pairs):
        diff = w[a] - w[b]
        y = top + k * 38
        i, j = conds.index(a), conds.index(b)
        ax.plot([i, j], [y, y], color=FG, lw=.7)
        ax.text((i + j) / 2, y + 4, f'{diff.mean():+.0f} ms, {p_text(stats.wilcoxon(diff).pvalue)}', ha='center', va='bottom', fontsize=6, color=FG)
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_xlim(-.4, len(conds) - .6)
    offset_spines(ax)


def main(rt_table, bids, name='fig1_paradigm_behaviour'):
    if not Path(rt_table).exists():
        build_rt(rt_table, bids)
    d = pd.read_csv(rt_table, sep='\t')
    fig = plt.figure(figsize=(DOUBLE, 3.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], height_ratios=[1, 1.25], left=.01, right=.99, top=.985, bottom=.115, hspace=.3, wspace=.2)
    ax = fig.add_subplot(gs[:, 0])
    panel_display(ax)
    letter(ax, 'a', x=.05, y=.9)
    ax = fig.add_subplot(gs[0, 1])
    panel_trial(fig, ax)
    letter(ax, 'b', x=0, y=.95)
    low = gs[1, 1].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=.28)
    ax = fig.add_subplot(low[0])
    rt_panel(ax, d[d.contrast == 'distractor'], ['HP', 'LP', 'Absent'], [HP, FG, MUTED], [('HP', 'LP'), ('LP', 'Absent')])
    ax.set_xlabel('Singleton distractor')
    ax.set_ylabel('Response time (ms)')
    letter(ax, 'c', x=-.2, y=.96)
    ylim = ax.get_ylim()
    ax = fig.add_subplot(low[1])
    rt_panel(ax, d[d.contrast == 'target'], ['HP', 'LP'], [HP, FG], [('HP', 'LP')])
    ax.set_xlabel('Target')
    ax.set_ylim(ylim)
    letter(ax, 'd', x=-.22, y=.96)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rt-table', default='notes/data/behaviour_rt_2026-11.tsv')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    a = ap.parse_args()
    main(a.rt_table, a.bids_folder)
