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
from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, MUTED, ROI_COLORS, ROI_ORDER, letter, offset_spines, pending, provisional, save, stimulus_icon

LEGEND = """
**Figure 5. Less sustained gain at the location where the singleton is likely.** a, dS = g_HP − g_LP per ROI, each term estimated alone on m4 (conditional, DoG pRFs): dots are subjects (n = 30; triangles, beyond the axis), points and bars the median and bootstrapped 95% CI. Stars: Wilcoxon signed-rank test against 0, Benjamini-Hochberg across ROIs. b, Each subject's dS, median over the nine ROIs, as the conditional estimate and from the joint m7 fit (DoG pRFs); the share of negative subjects and the Wilcoxon p are given above. DN pRFs: supplementary figure. c, The observed group dS against a null from relabelling the HP location (pending). d, Why within-subject cross-validation cannot test dS: when a known dS is put into the data (x-axis), the share of subjects whose held-out prediction improves (y-axis) passes 50% only near −.05, about three times the estimate (dashed). Provisional: current inputs.
"""


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def panel_roi(ax, cond, ylim=(-.12, .06)):
    lo, hi = ylim
    ps, med = [], []
    for i, roi in enumerate(ROI_ORDER):
        v = cond[cond.roi == roi].dS.dropna().to_numpy()
        rng = np.random.default_rng(i)
        xx = i + rng.uniform(-.17, .17, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=7, color=HP, alpha=.35, lw=0, zorder=1)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=10, marker='^', color=HP, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=10, marker='v', color=HP, alpha=.6, lw=0, clip_on=False)
        m, a, b = median_ci(v)
        ax.errorbar(i, m, yerr=[[m - a], [b - m]], fmt='o', ms=6, color=HP, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        ps.append(stats.wilcoxon(v).pvalue)
    for i, q in enumerate(stats.false_discovery_control(ps)):
        ax.text(i, hi + .004, stars(q), ha='center', va='bottom', fontsize=8, color=HP, clip_on=False)
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.text(-.4, .002, 'No difference', fontsize=6, color='.45', ha='left', va='bottom')
    ax.annotate('Less gain at HP', xy=(2, -.016), xytext=(3.2, -.095), fontsize=7, color=HP, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=HP, lw=1., mutation_scale=7, shrinkA=2, shrinkB=4))
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, fontsize=6.5)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('dS: HP − LP sustained gain')
    offset_spines(ax)


def panel_pooled(ax, cond, js):
    sets = [('Conditional', cond.groupby('subject').dS.median()), ('Joint (m7)', js['dog'].groupby('subject').dS.median())]
    lo, hi = -.12, .06
    for i, (lab, v) in enumerate(sets):
        v = v.dropna().to_numpy()
        rng = np.random.default_rng(10 + i)
        xx = i + rng.uniform(-.15, .15, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=10, color=HP, alpha=.35, lw=0)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=12, marker='v', color=HP, alpha=.6, lw=0, clip_on=False)
        m, a, b = median_ci(v)
        ax.errorbar(i, m, yerr=[[m - a], [b - m]], fmt='o', ms=6, color=HP, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        p = stats.wilcoxon(v).pvalue
        ptxt = 'p < .001' if p < .001 else f'p = {p:.3f}'.replace('0.', '.')
        ax.text(i, hi + .004, f'{(v < 0).mean() * 100:.0f}% < 0\n{ptxt}', ha='center', va='bottom', fontsize=6, color=FG,
                clip_on=False)
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s[0] for s in sets], fontsize=6.5)
    ax.set_xlim(-.5, len(sets) - .5)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('dS, median over ROIs')
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
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1], height_ratios=[1, 1], hspace=.6, wspace=.45, left=.08, right=.99, top=.9, bottom=.1)
    ax = fig.add_subplot(gs[0, :2])
    panel_roi(ax, cond)
    stimulus_icon(ax, (.855, .0, .145, .36), ('hp', 'lp'))
    letter(ax, 'a', x=-.08, y=1.04)
    provisional(ax, y=1.1)
    ax = fig.add_subplot(gs[0, 2])
    panel_pooled(ax, cond, js)
    stimulus_icon(ax, (.0, .66, .3, .34), ('hp', 'lp'))
    letter(ax, 'b', x=-.25, y=1.04)
    ax = fig.add_subplot(gs[1, 0])
    pending(ax, 'observed dS against the HP-rotation null, DoG and DN')
    letter(ax, 'c', x=-.02, y=1.04)
    ax = fig.add_subplot(gs[1, 1:])
    panel_power(ax, power_table)
    letter(ax, 'd', x=-.1, y=1.04)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--power-table', default='notes/data/cv_power_simulation_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.power_table)
