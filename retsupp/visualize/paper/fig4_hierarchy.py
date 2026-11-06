"""Figure 4: gain terms across the visual hierarchy. Provisional until the final fits.

Usage::

    python -m retsupp.visualize.paper.fig4_hierarchy --gain-table notes/data/gain_field_parameters.tsv \\
        --conditional-table notes/data/conditional_estimates_dog_2026-11.tsv
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, joint, jzs_bf10, median_ci
from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_ORDER, TARGET, letter, offset_spines,
                                                 pending, provisional, save)

TERMS = [('T', 'Target', TARGET), ('D', 'Singleton', DISTRACTOR), ('dS', 'Sustained, HP − LP', HP), ('dD', 'Singleton, HP − LP', '#5E3C99')]
FAM = {'dog': ('DoG', '-'), 'dn': ('DN', '--')}
LEGEND = """
**Figure 4. Gain terms across the visual hierarchy.** a, Median across subjects (n = 30) of each term per ROI, each scaled to its largest absolute median, with bootstrapped 95% confidence intervals; joint estimates from m7_full under DoG and DN, and conditional estimates (each term alone on its parent model, DoG only). b, Slope of each term across the hierarchy per subject (ROI rank), median and 95% CI; a placeholder for the hierarchical term × ROI model. c, Bayes factor for dS ≠ 0 per ROI (JZS, r = .707), joint DoG and DN. d, Widths of the sustained and transient fields per ROI, median and 95% CI, with the share of fits at a bound (.5° or 8°). All values provisional: current inputs.
"""


def panel_profile(ax, js, cond):
    x = np.arange(len(ROI_ORDER))
    for k, (term, lab, col) in enumerate(TERMS):
        off = (k - 1.5) * .17
        for fam, (name, ls) in FAM.items():
            d = js[fam]
            m = np.array([median_ci(d[d.roi == r][term])[0] for r in ROI_ORDER])
            ci = np.array([median_ci(d[d.roi == r][term])[1:] for r in ROI_ORDER])
            scale = np.nanmax(np.abs(m))
            ax.errorbar(x + off + (.04 if fam == 'dn' else -.04), m / scale, yerr=np.abs(ci.T / scale - m / scale), fmt='o' if fam == 'dog' else 's',
                        ms=2.8, color=col, mfc=col if fam == 'dog' else 'w', mew=.7, lw=.7, capsize=0)
        if term in cond.columns:
            c = [cond[cond.roi == r][term].median() for r in ROI_ORDER]
            ax.scatter(x + off, np.array(c) / np.nanmax(np.abs(c)), marker='_', s=30, color=col, lw=1.2, zorder=4)
        ax.text(1.0, 1.0 - k * .09, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER, fontsize=6.5)
    ax.set_ylabel('Median, scaled to its maximum')
    ax.set_ylim(-1.6, 1.6)
    ax.text(0, 1.04, 'Circles: DoG joint · squares: DN joint · dashes: DoG conditional', transform=ax.transAxes, fontsize=6, color=FG)
    offset_spines(ax)


def panel_slopes(ax, js):
    rank = {r: i for i, r in enumerate(ROI_ORDER)}
    for k, (term, lab, col) in enumerate(TERMS):
        for j, (fam, (name, _)) in enumerate(FAM.items()):
            d = js[fam]
            s = [stats.linregress([rank[r] for r in g.roi], g[term]).slope for _, g in d.groupby('subject') if g[term].notna().sum() > 3]
            m, lo, hi = median_ci(s)
            ax.errorbar(k + (j - .5) * .3, m, yerr=[[m - lo], [hi - m]], fmt='o' if fam == 'dog' else 's', ms=3, color=col,
                        mfc=col if fam == 'dog' else 'w', mew=.7, lw=.8, capsize=0)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(TERMS)))
    ax.set_xticklabels(['T', 'D', 'dS', 'dD'])
    ax.set_ylabel('Slope per ROI step')
    ax.text(0, 1.04, 'Pending: hierarchical term × ROI model', transform=ax.transAxes, fontsize=6, color=MUTED)
    offset_spines(ax)


def panel_bf(ax, js):
    x = np.arange(len(ROI_ORDER))
    for j, (fam, (name, _)) in enumerate(FAM.items()):
        d = js[fam]
        bf = [jzs_bf10(d[d.roi == r].dS) for r in ROI_ORDER]
        ax.bar(x + (j - .5) * .38, np.log10(bf), width=.36, color=HP if fam == 'dog' else 'w', edgecolor=HP, lw=.7)
    for v, lab in ((np.log10(3), '3'), (-np.log10(3), '1/3')):
        ax.axhline(v, color='.7', lw=.6, ls='--', zorder=0)
        ax.text(len(ROI_ORDER) - .5, v, lab, fontsize=5.5, color='.5', va='bottom', ha='right')
    ax.axhline(0, color=FG, lw=.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER, fontsize=6.5)
    ax.set_ylabel('log₁₀ BF₁₀, dS ≠ 0')
    ax.text(0, 1.04, 'Filled: DoG · open: DN', transform=ax.transAxes, fontsize=6, color=FG)
    offset_spines(ax)


def panel_widths(ax, js):
    x = np.arange(len(ROI_ORDER))
    for j, (par, lab, col) in enumerate((('sigma_AF', 'Sustained σ', HP), ('sigma_dyn', 'Transient σ', DISTRACTOR))):
        d = js['dog']
        m = np.array([median_ci(d[d.roi == r][par]) for r in ROI_ORDER])
        ax.errorbar(x + (j - .5) * .3, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='o', ms=3, color=col, lw=.8, capsize=0)
        at_bound = [((d[d.roi == r][par] < .51) | (d[d.roi == r][par] > 7.9)).mean() * 100 for r in ROI_ORDER]
        for xi, b in zip(x, at_bound):
            ax.text(xi + (j - .5) * .3, -.25 - j * .55, f'{b:.0f}', fontsize=5, color=col, ha='center', va='top')
        ax.text(1.0, 1.0 - j * .09, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER, fontsize=6.5)
    ax.set_ylim(-1.4, 8.2)
    ax.set_yticks([0, 4, 8])
    ax.set_ylabel('Field width (°), DoG')
    ax.text(-.03, .02, '% at\nbound', transform=ax.transAxes, fontsize=5, color=MUTED, ha='right')
    offset_spines(ax)


def main(gain_table, conditional_table, name='fig4_hierarchy'):
    js = {f: joint(gain_table, f) for f in FAM}
    cond = conditional(conditional_table)
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=.55, wspace=.25, left=.075, right=.99, top=.93, bottom=.09)
    for (r, c), fn, lab in (((0, 0), lambda ax: panel_profile(ax, js, cond), 'a'), ((0, 1), lambda ax: panel_slopes(ax, js), 'b'),
                            ((1, 0), lambda ax: panel_bf(ax, js), 'c'), ((1, 1), lambda ax: panel_widths(ax, js), 'd')):
        ax = fig.add_subplot(gs[r, c])
        fn(ax)
        letter(ax, lab, x=-.08 if c == 0 else -.14, y=1.03)
        provisional(ax, y=1.1)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table)
