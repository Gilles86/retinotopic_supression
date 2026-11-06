"""Figure 5: less sustained gain at the HP location. Provisional until the final fits.

Usage::

    python -m retsupp.visualize.paper.fig5_sustained_hp
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, joint, median_ci
from retsupp.visualize.paper.paper_style import SINGLE, DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_COLORS, ROI_ORDER, letter, ROI_X, title_icon, offset_spines, pending, provisional, roi_axis, roi_icon, save

LEGEND = """
**Figure 5. Less sustained gain at the location where the distractor is likely.** a, dS = g_HP − g_LP per ROI, each term estimated alone on m4 (conditional, DoG pRFs): dots are subjects (n = 30; triangles, beyond the axis), points and bars the median and bootstrapped 95% CI. Stars: Wilcoxon signed-rank test against 0, Benjamini-Hochberg across ROIs. b, Each subject's dS, median over the nine ROIs, as the conditional estimate and from the joint m7 fit (DoG pRFs); the share of negative subjects and the Wilcoxon p are given above. DN pRFs: supplementary figure. c, d, The same for dD = g_HP_dyn − g_LP_dyn, the transient gain when the distractor appears at the HP location minus when it appears elsewhere. The conditional estimate is negative, but the joint m7 estimate is not: once dS and the other terms are fitted together, dD is not distinguishable from 0, so dD is not a robust result. Provisional: current inputs.
"""
POWER_LEGEND = """
**Figure S. Within-subject cross-validation cannot test dS.** When a known dS is put into the data (x-axis), the share of subjects whose held-out prediction improves (y-axis) passes 50% only near −.05, about three times the estimate (dashed). Provisional: current inputs.
"""


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def panel_roi(ax, cond, term='dS', color=HP, ylim=(-.12, .06), ylabel='dS: HP − LP sustained gain', annot=('Less gain at HP', (2, -.016), (3.2, -.095))):
    lo, hi = ylim
    ps, med = [], []
    for i, roi in enumerate(ROI_ORDER):
        v = cond[cond.roi == roi][term].dropna().to_numpy()
        rng = np.random.default_rng(i)
        xx = ROI_X[roi] + rng.uniform(-.17, .17, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=7, color=color, alpha=.35, lw=0, zorder=1)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=10, marker='^', color=color, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=10, marker='v', color=color, alpha=.6, lw=0, clip_on=False)
        m, a, b = median_ci(v)
        ax.errorbar(ROI_X[roi], m, yerr=[[m - a], [b - m]], fmt='o', ms=6, color=color, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        ps.append(stats.wilcoxon(v).pvalue)
    for i, q in enumerate(stats.false_discovery_control(ps)):
        ax.text(ROI_X[ROI_ORDER[i]], hi + .03 * (hi - lo), stars(q), ha='center', va='bottom', fontsize=8, color=color, clip_on=False)
    ax.hlines(0, -.6, ROI_X[ROI_ORDER[-1]] + .6, color='.6', lw=.8, ls='--', zorder=0)
    ax.annotate(annot[0], xy=annot[1], xytext=annot[2], fontsize=7, color=color, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1., mutation_scale=7, shrinkA=2, shrinkB=4))
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    roi_axis(ax)


def panel_pooled(ax, cond, js, term='dS', color=HP, ylim=(-.12, .06)):
    sets = [('Conditional', cond.groupby('subject')[term].median()), ('Joint (m7)', js['dog'].groupby('subject')[term].median())]
    lo, hi = ylim
    for i, (lab, v) in enumerate(sets):
        v = v.dropna().to_numpy()
        rng = np.random.default_rng(10 + i)
        xx = i + rng.uniform(-.15, .15, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=10, color=color, alpha=.35, lw=0)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=12, marker='v', color=color, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=12, marker='^', color=color, alpha=.6, lw=0, clip_on=False)
        m, a, b = median_ci(v)
        ax.errorbar(i, m, yerr=[[m - a], [b - m]], fmt='o', ms=6, color=color, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        p = stats.wilcoxon(v).pvalue
        ptxt = 'p < .001' if p < .001 else f'p = {p:.3f}'.replace('0.', '.')
        ax.text(i, hi + .03 * (hi - lo), f'{(v < 0).mean() * 100:.0f}% < 0\n{ptxt}', ha='center', va='bottom', fontsize=6, color=FG,
                clip_on=False)
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s[0] for s in sets], fontsize=6.5)
    ax.set_xlim(-.5, len(sets) - .5)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(f'{term}, median over ROIs')
    offset_spines(ax)


def panel_power(ax, table, estimate=-.016):
    d = pd.read_csv(table, sep='\t')
    x = -d.planted_dS.to_numpy()
    ax.plot(x, d.pct_improved, 'o-', color=FG, ms=4, lw=1.3)
    ax.axhline(50, color='.6', lw=.8, ls='--', zorder=0)
    ax.text(.2, 52, 'Chance', fontsize=6, color='.45', ha='right', va='bottom')
    ax.axvline(-estimate, color=HP, lw=1, ls='--')
    ax.text(-estimate * 1.1, 97, 'Our\nestimate', fontsize=6.5, color=HP, va='top', ha='left')
    ax.annotate('Break-even\nnear .05', xy=(.05, 50), xytext=(.075, 20), fontsize=6.5, color=FG, ha='left',
                arrowprops=dict(arrowstyle='-|>', color=FG, lw=.9, mutation_scale=6, shrinkA=2, shrinkB=3))
    ax.set_xscale('symlog', linthresh=.01)
    ax.set_xticks([0, .01, .03, .1, .2])
    ax.set_xticklabels(['0', '.01', '.03', '.1', '.2'])
    ax.set_xlabel('Size of a planted dS (−)')
    ax.set_ylabel('Subjects improved (%)')
    ax.set_ylim(-3, 103)
    offset_spines(ax)


def main(gain_table, conditional_table, power_table, name='fig5_sustained_hp'):
    cond = conditional(conditional_table)
    js = {f: joint(gain_table, f) for f in ('dog', 'dn')}
    fig = plt.figure(figsize=(DOUBLE, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], hspace=1.05, wspace=.3, left=.08, right=.99, top=.86, bottom=.12)
    ax = fig.add_subplot(gs[0, 0])
    panel_roi(ax, cond)
    title_icon(ax, 'Sustained gain: less at the likely distractor location', ('hp', 'lp'), HP)
    letter(ax, 'a', x=-.06, y=1.12)
    provisional(ax, y=1.42)
    ax = fig.add_subplot(gs[0, 1])
    panel_pooled(ax, cond, js)
    letter(ax, 'b', x=-.3, y=1.12)
    dd_lim = (-.3, .2)
    ax = fig.add_subplot(gs[1, 0])
    panel_roi(ax, cond, 'dD', DISTRACTOR, dd_lim, 'dD: HP − LP transient gain',
              ('Less transient gain at HP\nwhen estimated alone', (ROI_X['IPS'], -.04), (1.2, -.24)))
    title_icon(ax, 'Transient gain at the distractor: HP vs LP', ('distractor', 'lp'), DISTRACTOR)
    letter(ax, 'c', x=-.06, y=1.12)
    ax = fig.add_subplot(gs[1, 1])
    panel_pooled(ax, cond, js, 'dD', DISTRACTOR, dd_lim)
    letter(ax, 'd', x=-.3, y=1.12)
    save(fig, name, LEGEND)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2), gridspec_kw=dict(left=.17, right=.97, top=.92, bottom=.2))
    panel_power(ax, power_table)
    save(fig, 'figS_cv_power', POWER_LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--power-table', default='notes/data/cv_power_simulation_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.power_table)
