"""Figure 7 (prototype): the three sources of gain and behaviour.

a  Illustration: gain at the target (each trial), at the distractor (each trial) and at the learned HP
   location (whole block; Fig. 5).
b  Target gain on target-repeat minus target-switch trials, per ROI (joint fits with the target gain split by repeat);
   the behavioural repeat effect is Fig. 1e.
c  Trial by trial: target, distractor-at-HP and global gain as predictors of log RT (one Bayesian hierarchical
   model per ROI); the global gain's sign flips along the hierarchy.

b and c come from the October trial-wise analysis on the earlier attention-field fits (pSig0.5) and are
stamped as such until they are redone on the ladder fits. Null results (between-subject brain-behaviour
correlations, distractor-repeat gain) stay out of this main figure; they are reported in the text.

Usage::

    python -m retsupp.visualize.paper.fig7_behaviour
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import median_ci
from retsupp.visualize.paper.paper_style import (HP, DISTRACTOR, DOUBLE, FG, MUTED, ROI_ORDER, ROI_X, TARGET, letter, offset_spines,
                                                 roi_axis, save, title_icon)
from retsupp.visualize.paper.plot_trialwise_story import behaviour

LEGEND = """
**Figure 7 (prototype). Three sources of gain, and behaviour.** a, The gain terms (illustrative): at the target and at the distractor on each trial, at the HP location over a whole block (Fig. 5), and a global gain over the whole display on each trial. b, Target gain (repeating the target location also speeds responses, Fig. 1e) on target-repeat minus target-switch trials, from fits with the target gain split by repeat; dots are subjects (n = 30), points and bars the median and bootstrapped 95% CI; Wilcoxon against 0, Benjamini-Hochberg across ROIs. The distractor gain did not differ between distractor repeats and switches (text). c, Within subjects, trial by trial: Bayesian hierarchical linear model of log RT (z within subject) per ROI on the per-trial target gain, distractor gain on HP and on LP trials, and global gain (each z-scored within run and trial type), with trial number, target repeat, distractor repeat and target-at-previous-distractor as covariates, and random intercepts and gain slopes per subject (bambi, 4 × 1000 draws, no divergences, R̂ ≤ 1.01). Points are posterior means, bars 95% HDIs; stars give the posterior probability that the effect has the sign shown: * > .975, ** > .995, *** > .9995 (not corrected across ROIs). The distractor gain on LP trials and the HP − LP difference are in the table. b, c use the earlier attention-field fits (October 2026) and will be redone on the final fits. * q < .05, ** q < .01, *** q < .001.
"""
EARLIER = 'b, c: earlier fits (Oct 2026)'


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
        ax.text(i + .1, hi + .03 * (hi - lo), stars(stats.wilcoxon(v).pvalue), ha='center', va='bottom', fontsize=8.5, color=col,
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
        ax.text(ROI_X[roi], hi + .05 * (hi - lo), stars(q), ha='center', va='bottom', fontsize=8.5, color=TARGET, clip_on=False)
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
        ax.text(xi, .13, stars(r.q), ha='center', va='bottom', fontsize=8.5, color=FG, clip_on=False)
    ax.hlines(0, -.6, x[-1] + .6, color='.7', lw=.7, ls='--', zorder=0)
    ax.text(x[-1] + .6, .12, 'Larger response: slower', ha='right', va='top', fontsize=6, color='.4')
    ax.text(x[-1] + .6, -.075, 'Larger response: faster', ha='right', va='bottom', fontsize=6, color='.4')
    ax.set_ylim(-.08, .125)
    ax.set_ylabel('RT per unit of\ntrial response (z)')
    offset_spines(ax)
    roi_axis(ax)


def pd_stars(row):
    """Stars from the posterior probability of the sign: * > .975 (95% HDI-like), ** > .995, *** > .9995."""
    pdir = max(row.p_gt0, row.p_lt0)
    return '***' if pdir > .9995 else '**' if pdir > .995 else '*' if pdir > .975 else ''


def panel_bayes(ax, bayes, terms, ylim, ylabel, notes, stack=True):
    d = bayes[bayes.roi.isin(ROI_ORDER)]
    off = np.linspace(-.22, .22, len(terms)) if len(terms) > 1 else [0]
    for k, (o, (term, col)) in enumerate(zip(off, terms)):
        r = d[d.term == term].set_index('roi').loc[ROI_ORDER]
        for roi, row in r.iterrows():
            ax.errorbar(ROI_X[roi] + o, row['mean'], yerr=[[row['mean'] - row.hdi_lo], [row.hdi_hi - row['mean']]], fmt='o', ms=5,
                        color=col, mec='w', mew=.7, lw=1.4, capsize=0)
            ax.text(ROI_X[roi] + o, ylim[1] + (.02 + .08 * k * stack) * (ylim[1] - ylim[0]), pd_stars(row), ha='center', va='bottom',
                    fontsize=8.5, color=col, clip_on=False)
    ax.hlines(0, -.6, ROI_X[ROI_ORDER[-1]] + .6, color='.7', lw=.7, ls='--', zorder=0)
    x1 = ROI_X[ROI_ORDER[-1]] + .6
    ax.text(x1, ylim[1], notes[0], ha='right', va='top', fontsize=6, color='.4')
    ax.text(x1, ylim[0], notes[1], ha='right', va='bottom', fontsize=6, color='.4')
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    roi_axis(ax)


def panel_sources(ax):
    """Four displays, each with one gain field: target, distractor, HP location, and the global (whole-display) gain."""
    from matplotlib.patches import Circle
    ring = 4. / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    g = np.linspace(-8.5, 8.5, 281)
    xx, yy = np.meshgrid(g, g)
    # title, where and when, colour, centre (None: everywhere), width, panel origin
    src = [('Target', 'At the target, each trial', TARGET, 1, 1.4, (0, 0)),
           ('Distractor', 'At the distractor, each trial', DISTRACTOR, 3, 1.4, (17, 0)),
           ('HP location', 'At HP, whole block (Fig. 5)', HP, 3, 2., (0, -19)),
           ('Global', 'Everywhere, each trial', FG, None, None, (17, -19))]
    for title, sub, col, ix, sig, (x0, y0) in src:
        if ix is None:
            f = .3 / (1 + np.exp((np.hypot(xx, yy) - 5.6) / .25))   # flat over the display, soft edge
        else:
            f = .75 * np.exp(-((xx - ring[ix, 0]) ** 2 + (yy - ring[ix, 1]) ** 2) / (2 * sig ** 2))
        rgba = np.zeros(f.shape + (4,))
        rgba[..., :3] = mpl.colors.to_rgb(col)
        rgba[..., 3] = f
        ax.imshow(rgba, extent=(x0 - 8.5, x0 + 8.5, y0 - 8.5, y0 + 8.5), origin='lower', zorder=1)
        ax.add_patch(Circle((x0, y0), 3.17, fill=False, ec=MUTED, lw=.6, zorder=2))
        for p in ring:
            ax.add_patch(Circle((x0 + p[0], y0 + p[1]), .55, facecolor='w', ec=MUTED, lw=.6, zorder=2))
        ax.add_patch(Circle((x0 + ring[3, 0], y0 + ring[3, 1]), 1.15, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2)), zorder=3))
        if title == 'Target':
            ax.scatter(x0 + ring[1, 0], y0 + ring[1, 1], marker='D', s=16, color=TARGET, lw=0, zorder=4)
        if title == 'Distractor':
            ax.scatter(x0 + ring[3, 0], y0 + ring[3, 1], marker='o', s=20, color=DISTRACTOR, lw=0, zorder=4)
        ax.plot(x0, y0, '+', color=FG, ms=4, mew=.6, zorder=4)
        ax.text(x0, y0 + 7.2, title, ha='center', va='bottom', fontsize=7, color=col)
        ax.text(x0, y0 - 7.2, sub, ha='center', va='top', fontsize=5.8, color='.35')
    ax.set_xlim(-8, 25)
    ax.set_ylim(-28.5, 9.5)
    ax.set_aspect('equal')
    ax.set_axis_off()


def stamp(ax):
    ax.text(1., 1.36, EARLIER, transform=ax.transAxes, fontsize=5.5, color=MUTED, ha='right', va='bottom')


def header(fig, y, text, x0=.02, x1=.99):
    """Row header across the figure: what the panels below have in common."""
    fig.text(x0, y, text, fontsize=7.5, fontweight='bold', color=FG, va='bottom', ha='left')
    fig.add_artist(plt.Line2D([x0, x1], [y - .006] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


def main(trial_table, mixed_table, repeat_table, bayes_table, name='fig7_behaviour'):
    tt = pd.read_csv(trial_table, sep='\t')
    bayes = pd.read_csv(bayes_table, sep='\t')
    fig = plt.figure(figsize=(DOUBLE, 5.1))
    ax = fig.add_axes([.0, .52, .3, .44])
    panel_sources(ax)
    letter(ax, 'a', x=.06, y=.98)
    header(fig, .95, 'Repeating the target location: faster search (Fig. 1e), less target gain', x0=.36)
    ax = fig.add_axes([.43, .63, .55, .23])
    panel_repeat(ax, pd.read_csv(repeat_table, sep='\t'))
    title_icon(ax, 'Target gain: lower on repeats', ('target',), TARGET, h=.24)
    letter(ax, 'b', x=-.12, y=1.26)
    header(fig, .485, 'Trial by trial: gain on a trial predicts its response time')
    ax = fig.add_axes([.075, .135, .9, .245])
    panel_bayes(ax, bayes, [('T', TARGET), ('D_HP', DISTRACTOR), ('G', FG)], (-.08, .13), 'Log RT per unit gain (z)',
                ('Slower', 'Faster'), stack=False)
    ax.set_yticks([-.05, 0, .05, .1])
    for k, (lab, col) in enumerate((('Target gain', TARGET), ('Distractor gain, HP trials', DISTRACTOR),
                                    ('Global gain: the sign flips', FG))):
        ax.text(-.4, .125 - .022 * k, lab, color=col, fontsize=6.5, ha='left', va='center')
    letter(ax, 'c', x=-.055, y=1.3)
    fig.text(.99, .998, EARLIER, fontsize=5.5, color=MUTED, ha='right', va='top')
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial-table', default='notes/data/trialwise_af_trial_table.tsv.gz')
    ap.add_argument('--mixed-table', default='notes/data/trialwise_af_mixed_stats.tsv')
    ap.add_argument('--repeat-table', default='notes/data/repeat_af_fits_params.tsv')
    ap.add_argument('--bayes-table', default='notes/data/trialwise_rt_bayes_summary.tsv')
    a = ap.parse_args()
    main(a.trial_table, a.mixed_table, a.repeat_table, a.bayes_table)
