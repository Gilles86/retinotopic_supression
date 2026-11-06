"""Figure 5: sustained gain at the HP location. Provisional until the final fits.

Usage::

    python -m retsupp.visualize.paper.fig5_sustained_hp
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig45_common import conditional, joint, median_ci
from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, MUTED, ROI_COLORS, ROI_ORDER, letter, offset_spines, pending, provisional, save

LEGEND = """
**Figure 5. Sustained gain at the HP location.** a, dS = g_HP − g_LP per ROI: conditional estimates under DoG (dots: subjects, bar: median with bootstrapped 95% CI) and joint m7_full estimates under DN (open squares: median with 95% CI); n = 30. b, Observed group dS against the HP-rotation null, DoG and DN. c, Leave-one-subject-out test of dS by voxel sensitivity to the HP gain (percentile of the summed squared derivative), DoG, signal voxels: share of subjects improved (bars) and the subjects' own conditional dS (points). d, Within-subject cross-validation (leave one HP condition out) with a known dS put into the data: share of subjects whose held-out prediction improves; the dashed line marks the group estimate and the break-even point is where the curve crosses 50%. All values provisional: current inputs.
"""


def panel_dS(ax, cond, js_dn):
    x = np.arange(len(ROI_ORDER))
    for i, roi in enumerate(ROI_ORDER):
        v = cond[cond.roi == roi].dS.dropna().to_numpy()
        rng = np.random.default_rng(i)
        ax.scatter(i - .12 + rng.uniform(-.1, .1, len(v)), np.clip(v, -.15, .15), s=4, color=ROI_COLORS[roi], alpha=.5, lw=0)
        m, lo, hi = median_ci(v)
        ax.errorbar(i - .12, m, yerr=[[m - lo], [hi - m]], fmt='_', ms=9, mew=1.5, color=FG, lw=1., capsize=0)
        m, lo, hi = median_ci(js_dn[js_dn.roi == roi].dS)
        ax.errorbar(i + .2, m, yerr=[[m - lo], [hi - m]], fmt='s', ms=3, mfc='w', mec=FG, color=FG, lw=.8, capsize=0)
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER, fontsize=6.5)
    ax.set_ylim(-.15, .15)
    ax.set_yticks([-.1, 0, .1])
    ax.set_ylabel('dS (g_HP − g_LP)')
    ax.text(0, 1.04, 'Dots, bar: DoG conditional · open squares: DN joint', transform=ax.transAxes, fontsize=6, color=FG)
    offset_spines(ax)


def panel_sensitivity(ax, table):
    d = pd.read_csv(table, sep='\t')
    x = np.arange(len(d))
    ax.bar(x, d.pct_improved - 50, bottom=50, width=.6, color=HP, alpha=.75, lw=0)
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(d['bin'], fontsize=6.5)
    ax.set_xlabel('Sensitivity percentile')
    ax.set_ylabel('Subjects improved (%)')
    ax.set_ylim(40, 100)
    a2 = ax.twinx()
    a2.plot(x, d.own_dS, 'o-', color=FG, ms=3, lw=.9)
    a2.set_ylim(-.03, .003)
    a2.set_ylabel('Own dS', fontsize=7)
    a2.spines['right'].set_visible(True)
    a2.tick_params(labelsize=6)
    offset_spines(ax)


def panel_power(ax, table, estimate=-.016):
    d = pd.read_csv(table, sep='\t')
    ax.plot(-d.planted_dS, d.pct_improved, 'o-', color=FG, ms=3.5, lw=1.1)
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.axvline(-estimate, color=HP, lw=.8, ls='--')
    ax.text(-estimate + .004, 92, 'Our\nestimate', fontsize=6, color=HP, va='top')
    ax.set_xscale('symlog', linthresh=.01)
    ax.set_xticks([0, .01, .03, .1, .2])
    ax.set_xticklabels(['0', '.01', '.03', '.1', '.2'])
    ax.set_xlabel('Size of dS put in (−dS)')
    ax.set_ylabel('Subjects improved (%)')
    ax.set_ylim(-3, 103)
    offset_spines(ax)


def main(gain_table, conditional_table, sensitivity_table, power_table, name='fig5_sustained_hp'):
    cond = conditional(conditional_table)
    js_dn = joint(gain_table, 'dn')
    fig = plt.figure(figsize=(DOUBLE, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1], hspace=.6, wspace=.55, left=.07, right=.97, top=.93, bottom=.1)
    ax = fig.add_subplot(gs[0, :2])
    panel_dS(ax, cond, js_dn)
    letter(ax, 'a', x=-.06, y=1.03)
    provisional(ax, y=1.1)
    ax = fig.add_subplot(gs[0, 2])
    pending(ax, 'observed dS against the HP-rotation null, DoG and DN')
    letter(ax, 'b', x=-.04, y=1.03)
    ax = fig.add_subplot(gs[1, 0])
    panel_sensitivity(ax, sensitivity_table)
    letter(ax, 'c', x=-.15, y=1.03)
    provisional(ax, y=1.1)
    ax = fig.add_subplot(gs[1, 1])
    panel_power(ax, power_table)
    letter(ax, 'd', x=-.2, y=1.03)
    provisional(ax, y=1.1)
    ax = fig.add_subplot(gs[1, 2])
    pending(ax, 'gain-free decoding of the bar at HP vs LP (moved to Fig. 6)')
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--sensitivity-table', default='notes/data/dS_by_sensitivity_2026-11.tsv')
    ap.add_argument('--power-table', default='notes/data/cv_power_simulation_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.sensitivity_table, a.power_table)
