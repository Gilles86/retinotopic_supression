"""Supplementary figure: field widths among fits with a reliable gain, and how well each width is constrained.

a  Sustained field FWHM per area: all fits (pale) and fits with a reliable sustained gain (|gain| / SE > 2 at HP or at
   the LP locations; solid), median and bootstrapped 95% CI; dotted, the 5.7° spacing of neighbouring item locations.
b  The same for the transient field, kept when the target gain is reliable.
c, d  Width of each fit's 95% profile interval (FWHM degrees, leave-one-run-out jackknife of the profile minimum) against
   |gain| / SE: the width is pinned down only when the gain is clearly non-zero.

    python -m retsupp.visualize.paper.figS_field_widths_v2
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, MUTED, ROI_COLORS, ROI_ORDER, ROI_X, offset_spines, roi_axis, save
from retsupp.visualize.paper.field_widths_reliable import SPACING, boot_ci

TRANS = '#5B6770'
plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.linewidth': .8})


def width_panel(ax, d, col, rel, color, ylab, thr):
    rng = np.random.default_rng(1)
    for roi in ROI_ORDER:
        x0 = ROI_X[roi]
        for k, (keep, off, alpha, ms) in enumerate(((d.roi == roi, -.17, .35, 3.4), ((d.roi == roi) & rel, .17, 1., 4.2))):
            v = d.loc[keep, col].to_numpy()
            if not len(v):
                continue
            jit = rng.uniform(-.06, .06, len(v))
            ax.scatter(x0 + off + jit, np.clip(v, 1.0, 20), s=4, color=color, alpha=.18 if k == 0 else .45, lw=0, zorder=2)
            lo, hi = boot_ci(v)
            m = np.median(v)
            ax.plot([x0 + off] * 2, [lo, hi], color=color, alpha=alpha, lw=1.1, solid_capstyle='butt', zorder=3)
            ax.plot(x0 + off, m, 'o', ms=ms, color=color, alpha=alpha, mec='w', mew=.5, zorder=4)
            ax.text(x0 + off, 21.5, f'{len(v)}', ha='center', va='bottom', fontsize=6, color=color, alpha=max(alpha, .6))
    ax.axhline(SPACING, color=FG, lw=.6, ls=(0, (1, 2)), zorder=1)
    ax.set_yscale('log')
    ax.set_ylim(1.0, 21)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(['1', '2', '5', '10', '20'])
    ax.set_ylabel(ylab)
    offset_spines(ax)
    roi_axis(ax, rotate=False)


def profile_panel(ax, d, z, width, color, title):
    x = np.clip(d[z].abs().to_numpy(), .05, 60)
    y = d[width].to_numpy()
    open_ = ~np.isfinite(y)
    yc = np.clip(np.where(open_, 20, y), .1, 20)
    ax.scatter(x[~open_], yc[~open_], s=5, color=color, alpha=.45, lw=0)
    ax.scatter(x[open_], yc[open_], s=9, marker='^', facecolor='none', edgecolor=color, lw=.5, alpha=.7)
    ax.scatter([14], [.25], s=9, marker='^', facecolor='none', edgecolor=color, lw=.5)
    ax.text(17, .25, 'Open interval', fontsize=6, color='.4', va='center')
    ax.axvline(2, color=FG, lw=.6, ls=(0, (2, 2)))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.05, 60)
    ax.set_ylim(.1, 25)
    ax.set_xticks([.1, 1, 2, 10, 50])
    ax.set_xticklabels(['0.1', '1', '2', '10', '50'])
    ax.set_yticks([.1, 1, 10])
    ax.set_yticklabels(['0.1', '1', '10'])
    ax.set_xlabel('|Gain| / SE')
    ax.set_ylabel('95% interval of the\nwidth (FWHM, °)')
    ax.set_title(title, fontsize=7.5, loc='left', color=color, pad=6)
    offset_spines(ax)


def main(fits, name, thr=2.):
    d = pd.read_csv(fits, sep='\t')
    d = d[d.roi.isin(ROI_ORDER)]
    W, H = DOUBLE, 5.4
    fig = plt.figure(figsize=(W, H))
    axa = fig.add_axes([.08, .715, .90, .225])
    axb = fig.add_axes([.08, .375, .90, .225])
    axc = fig.add_axes([.10, .07, .36, .18])
    axd = fig.add_axes([.60, .07, .36, .18])
    width_panel(axa, d, 'fwhm_sus', d.sus_z > thr, HP, 'Sustained field\nFWHM (°)', thr)
    width_panel(axb, d, 'fwhm_dyn', d.g_T_z.abs() > thr, TRANS, 'Transient field\nFWHM (°)', thr)
    profile_panel(axc, d, 'sus_z', 'prof_width_sus', HP, 'Sustained field')
    profile_panel(axd, d, 'g_T_z', 'prof_width_dyn', TRANS, 'Transient field (target gain)')
    for ax, let in ((axa, 'a'), (axb, 'b'), (axc, 'c'), (axd, 'd')):
        ax.text(-.06 if ax in (axa, axb) else -.2, 1.08, let, transform=ax.transAxes, fontsize=8, weight='bold', va='bottom')
    # key (glyphs)
    k = fig.add_axes([.08, .955, .9, .04])
    k.set_axis_off()
    k.set_xlim(0, 10)
    k.set_ylim(-1, 1)
    k.plot(.1, 0, 'o', ms=3.4, color=FG, alpha=.35, mec='w', mew=.5)
    k.text(.25, 0, 'All fits', va='center', fontsize=6.5, color='.4')
    k.plot(1.3, 0, 'o', ms=4.2, color=FG, mec='w', mew=.5)
    k.text(1.45, 0, f'Fits with a reliable gain (|gain| / SE > {thr:g})', va='center', fontsize=6.5, color=FG)
    k.plot([5.2, 5.6], [0, 0], color=FG, lw=.6, ls=(0, (1, 2)))
    k.text(5.7, 0, 'Spacing of neighbouring item locations, 5.7°', va='center', fontsize=6.5, color='.4')
    k.text(9.95, 0, 'Numbers: fits per series', va='center', ha='right', fontsize=6, color='.4')
    return fig, d


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fits', default='notes/data/field_widths_reliable_n28_2026-11-12_fits.tsv')
    ap.add_argument('--name', default='figS_field_widths_v2_n28')
    ap.add_argument('--legend', default='notes/data/field_widths_reliable_n28_2026-11-12_legend.md')
    a = ap.parse_args()
    fig, _ = main(a.fits, a.name)
    from pathlib import Path
    save(fig, a.name, Path(a.legend).read_text() if Path(a.legend).exists() else 'Legend pending.')
