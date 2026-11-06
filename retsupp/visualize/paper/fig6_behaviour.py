"""Figure 6 (prototype): trial history and trial-by-trial behaviour in the gains.

a  Behaviour: repeating the target location speeds responses, repeating the distractor location less so.
b  Target gain on target-repeat minus target-switch trials, per ROI (joint fits with the target gain split by repeat).
c  Trial by trial, within subjects: the overall trial response vs RT, per ROI (linear mixed models).

b and c come from the October trial-wise analysis on the earlier attention-field fits (pSig0.5) and are
stamped as such until they are redone on the ladder fits. Null results (between-subject brain-behaviour
correlations, distractor-repeat gain) stay out of this main figure; they are reported in the text.

Usage::

    python -m retsupp.visualize.paper.fig6_behaviour
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import median_ci
from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, MUTED, ROI_ORDER, ROI_X, TARGET, letter, offset_spines,
                                                 roi_axis, save, title_icon)
from retsupp.visualize.paper.plot_trialwise_story import behaviour

LEGEND = """
**Figure 6 (prototype). Trial history and trial-by-trial behaviour.** a, Response time on trials where the target (or the distractor) appears at the same location as on the previous trial, minus trials where it switches; distractor repeats are compared within HP and within LP trials and averaged. Dots are subjects (n = 30), points and bars the median and bootstrapped 95% CI; Wilcoxon signed-rank test against 0. b, Target gain on target-repeat minus target-switch trials, from fits with the target gain split by repeat; dots are subjects, points and bars the median and bootstrapped 95% CI; Wilcoxon against 0, Benjamini-Hochberg across ROIs. The distractor gain did not differ between distractor repeats and switches (text). c, Within subjects, trial by trial: the change in RT (z within subject) per unit of the overall trial response (z), from linear mixed models per ROI with random intercepts and slopes per subject; bars are 95% CIs; Benjamini-Hochberg over the eleven ROIs fitted (two not shown). b and c use the earlier attention-field fits (October 2026) and will be redone on the final fits. * q < .05, ** q < .01, *** q < .001.
"""
EARLIER = 'Earlier fits (Oct 2026)'


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def panel_rt(ax, b, ylim=(-120, 40)):
    lo, hi = ylim
    for i, (kind, col) in enumerate((('Target', TARGET), ('Distractor', DISTRACTOR))):
        v = b[b.kind == kind].d.dropna().to_numpy()
        rng = np.random.default_rng(i)
        xx = i + rng.uniform(-.15, .15, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=8, color=col, alpha=.4, lw=0)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=10, marker='v', color=col, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=10, marker='^', color=col, alpha=.6, lw=0, clip_on=False)
        m, a, c = median_ci(v)
        ax.errorbar(i + .3, m, yerr=[[m - a], [c - m]], fmt='o', ms=6, color=col, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        ax.text(i + .1, hi + .03 * (hi - lo), stars(stats.wilcoxon(v).pvalue), ha='center', va='bottom', fontsize=6.5, color=col,
                clip_on=False)
    ax.hlines(0, -.4, 1.5, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_xticks([.1, 1.1])
    ax.set_xticklabels(['Target', 'Distractor'])
    ax.set_xlim(-.4, 1.5)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('RT, repeat − switch (ms)')
    offset_spines(ax)


def panel_repeat(ax, rep, ylim=(-1.5, 1.)):
    d = rep[rep.model == 'targetRepeat'].copy()
    d['diff'] = d.g_T_dyn_repeat - d.g_T_dyn_switch
    lo, hi = ylim
    ps = []
    for i, roi in enumerate(ROI_ORDER):
        v = d[d.roi == roi]['diff'].dropna().to_numpy()
        rng = np.random.default_rng(i)
        xx = ROI_X[roi] + rng.uniform(-.17, .17, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=7, color=TARGET, alpha=.35, lw=0, zorder=1)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=10, marker='^', color=TARGET, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=10, marker='v', color=TARGET, alpha=.6, lw=0, clip_on=False)
        m, a, b = median_ci(v)
        ax.errorbar(ROI_X[roi], m, yerr=[[m - a], [b - m]], fmt='o', ms=6, color=TARGET, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        ps.append(stats.wilcoxon(v).pvalue)
    for roi, q in zip(ROI_ORDER, stats.false_discovery_control(ps)):
        ax.text(ROI_X[roi], hi + .05 * (hi - lo), stars(q), ha='center', va='bottom', fontsize=6.5, color=TARGET, clip_on=False)
    ax.hlines(0, -.6, ROI_X[ROI_ORDER[-1]] + .6, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('Target gain, repeat − switch')
    offset_spines(ax)
    roi_axis(ax)


def panel_amplitude(ax, mixed):
    d = mixed[(mixed.level == 'roi') & (mixed.model == 'overall') & (mixed.term == 'A')].set_index('unit').loc[ROI_ORDER]
    x = [ROI_X[r] for r in ROI_ORDER]
    ax.errorbar(x, d.beta, yerr=1.96 * d.se, fmt='o', ms=5.5, color=FG, mec='w', mew=.8, lw=1.5, capsize=0)
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.text(xi, .13, stars(r.q), ha='center', va='bottom', fontsize=6.5, color=FG, clip_on=False)
    ax.hlines(0, -.6, x[-1] + .6, color='.7', lw=.7, ls='--', zorder=0)
    ax.text(x[-1] + .6, .12, 'Larger response: slower', ha='right', va='top', fontsize=6, color='.4')
    ax.text(x[-1] + .6, -.075, 'Larger response: faster', ha='right', va='bottom', fontsize=6, color='.4')
    ax.set_ylim(-.08, .125)
    ax.set_ylabel('RT per unit of\ntrial response (z)')
    offset_spines(ax)
    roi_axis(ax)


def stamp(ax):
    ax.text(1., 1.36, EARLIER, transform=ax.transAxes, fontsize=5.5, color=MUTED, ha='right', va='bottom')


def main(trial_table, mixed_table, repeat_table, name='fig6_behaviour'):
    tt = pd.read_csv(trial_table, sep='\t')
    fig = plt.figure(figsize=(DOUBLE, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 2], wspace=.42, left=.075, right=.99, top=.74, bottom=.22)
    ax = fig.add_subplot(gs[0])
    panel_rt(ax, behaviour(tt))
    ax.text(0, 1.22, 'Repeats speed search', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    letter(ax, 'a', x=-.42, y=1.26)
    ax = fig.add_subplot(gs[1])
    panel_repeat(ax, pd.read_csv(repeat_table, sep='\t'))
    title_icon(ax, 'Repeated target: less target gain', ('target',), TARGET, h=.24)
    stamp(ax)
    letter(ax, 'b', x=-.2, y=1.26)
    ax = fig.add_subplot(gs[2])
    panel_amplitude(ax, pd.read_csv(mixed_table, sep='\t'))
    ax.text(0, 1.22, 'Trial by trial: the sign flips along the hierarchy', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    stamp(ax)
    letter(ax, 'c', x=-.2, y=1.26)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial-table', default='notes/data/trialwise_af_trial_table.tsv.gz')
    ap.add_argument('--mixed-table', default='notes/data/trialwise_af_mixed_stats.tsv')
    ap.add_argument('--repeat-table', default='notes/data/repeat_af_fits_params.tsv')
    a = ap.parse_args()
    main(a.trial_table, a.mixed_table, a.repeat_table)
