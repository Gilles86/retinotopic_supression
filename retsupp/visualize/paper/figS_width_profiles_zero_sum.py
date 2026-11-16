"""Supplementary figure: group width profiles of the sustained and transient fields (zero-sum main model).

Per area, the loss in fit (training SSE above its minimum) summed over participants at each field width, from
``width_profiles_zero_sum`` (sustained gain along the zero-sum direction; distractor tied HP = LP). Each row is scaled
to its own maximum. Marker: the group minimum; bar: the range of minima when leaving out one participant at a time.
Widths as FWHM (2.355 sigma).

    python -m retsupp.visualize.paper.figS_width_profiles_zero_sum \
        --profiles notes/data/width_profiles_zero_sum_2026-11-14_profiles.tsv --name figS_width_profiles_zs_n28_palA
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

FWHM = 2 * np.sqrt(2 * np.log(2))
AREAS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
PURPLE, CHARCOAL = '#7B3294', '#33393F'
OUT = Path('notes/figures/paper_2026-11')

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'], 'font.size': 7,
    'axes.labelsize': 7.5, 'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def group_profiles(pr, field):
    rows = []
    for a in AREAS:
        P = pr[(pr.field == field) & (pr.roi == a)].pivot_table(index='subject', columns='sigma', values='dsse')
        tot = P.sum()
        loo = np.array([P.drop(s).sum().idxmin() for s in P.index])
        rows.append(dict(roi=a, n=len(P), sigma=tot.index.values, rel=(tot / tot.max()).values,
                         best=tot.idxmin(), loo_lo=loo.min(), loo_hi=loo.max()))
    return rows


def panel(ax, rows, cmap, title, color):
    grid = rows[0]['sigma'] * FWHM
    img = np.stack([r['rel'] for r in rows])
    edges = np.geomspace(grid[0] / (grid[1] / grid[0]) ** .5, grid[-1] * (grid[1] / grid[0]) ** .5, len(grid) + 1)
    ax.pcolormesh(edges, np.arange(len(rows) + 1), img, cmap=cmap, vmin=0, vmax=1, rasterized=True)
    for i, r in enumerate(rows):
        ax.plot([r['loo_lo'] * FWHM, r['loo_hi'] * FWHM], [i + .5] * 2, color='white', lw=3.2, solid_capstyle='round')
        ax.plot([r['loo_lo'] * FWHM, r['loo_hi'] * FWHM], [i + .5] * 2, color=color, lw=1.6, solid_capstyle='round')
        ax.plot(r['best'] * FWHM, i + .5, 'o', ms=4.2, mfc=color, mec='white', mew=.8)
        ax.text(edges[-1] * 1.04, i + .5, f"{r['best'] * FWHM:.1f}°", va='center', fontsize=6.5, color=color)
    ax.set_xscale('log')
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(len(rows), 0)
    ax.set_yticks(np.arange(len(rows)) + .5)
    ax.set_yticklabels([r['roi'] for r in rows])
    ticks = [1.5, 3, 6, 12]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}°' for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel('Field width (FWHM)')
    ax.set_title(title, loc='left', color=color)
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)


def main(profiles, name):
    pr = pd.read_csv(profiles, sep='\t')
    n = pr.subject.nunique()
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.05), gridspec_kw=dict(width_ratios=[1, 1, .06], wspace=.42))
    cm_s = LinearSegmentedColormap.from_list('s', ['#FFFFFF', '#D1BBDC', '#9970AB', PURPLE])
    cm_t = LinearSegmentedColormap.from_list('t', ['#FFFFFF', '#C9CDD2', '#7D858D', CHARCOAL])
    sus, dyn = group_profiles(pr, 'sus'), group_profiles(pr, 'dyn')
    panel(axes[0], sus, cm_s, 'Sustained field: 4–7° in every area but IPS', PURPLE)
    panel(axes[1], dyn, cm_t, 'Transient field: as wide as possible early, narrow in hV4/V3AB/VO', CHARCOAL)
    axes[1].set_yticklabels([])
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('g', ['#FFFFFF', '#7D858D']),
                                            norm=mpl.colors.Normalize(0, 1)), cax=axes[2])
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Best', 'Worst'])
    cb.set_label('Loss in fit\n(per area)', fontsize=6.5)
    cb.outline.set_visible(False)
    for ax, lab in zip(axes[:2], 'ab'):
        ax.text(-.16, 1.08, lab, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=200)
    s = '; '.join(f"{r['roi']} {r['best'] * FWHM:.1f}° [{r['loo_lo'] * FWHM:.1f}–{r['loo_hi'] * FWHM:.1f}]" for r in sus)
    d = '; '.join(f"{r['roi']} {r['best'] * FWHM:.1f}° [{r['loo_lo'] * FWHM:.1f}–{r['loo_hi'] * FWHM:.1f}]" for r in dyn)
    legend = (
        f"**Supplementary Figure S. Widths of the fitted gain fields, group level.** Zero-sum main model (DoG pRFs, "
        f"pRFs and gain fields fitted jointly), n = {n}. For each participant and area, every parameter was held at "
        "its fit except the width of one field, which was set to each of 17 values (FWHM 1.2–18.8°), and that field's "
        "gains were re-solved by least squares: the sustained gain along the zero-sum direction (HP = ¾ dS, "
        "LP = −¼ dS), and the target and distractor gains (distractor equal at HP and LP, as in the model). Colour: "
        "the training error above its minimum, summed over participants, scaled per area (white, best width). "
        "Dot: the group minimum; bar: the range of minima when leaving out one participant at a time. "
        "(a) Sustained field. (b) Transient target and distractor field; at 18.8° it is nearly uniform over the "
        "display, at 1.2° it acts only at the item locations. Per participant the widths were not identified "
        "(no width excluded at 2 leave-one-run-out SE in 215 of 241 sustained and 144 of 241 transient profiles), "
        "and dS (pooled over the nine areas) stayed negative at every sustained width, −1.6 to −3.8 points, 17–22 of 28 participants below 0, p < .05 from FWHM 2.0 to 11.2° and p ≤ .066 elsewhere. "
        f"Sustained: {s}. Transient: {d}.")
    (OUT / f'{name}.md').write_text(legend + '\n')
    print(legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', default='notes/data/width_profiles_zero_sum_2026-11-14_profiles.tsv')
    ap.add_argument('--name', default='figS_width_profiles_zs_n28_palA')
    a = ap.parse_args()
    main(a.profiles, a.name)
