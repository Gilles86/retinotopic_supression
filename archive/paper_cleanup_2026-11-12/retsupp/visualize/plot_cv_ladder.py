"""Plot the nested gain-field CV ladder (``aggregate_cv_ladder`` output).

Three rows of per-subject ΔCV-R² (median over signal voxels) per ROI:
a) search items added to the pRF stimulus model, b) each gain-field
component added to the pRF, c) statistical-learning terms (HP ≠ LP) added
to the location-blind T+D+S model. Points are subjects; diamond + bar is
mean ± SEM; asterisks mark BH-FDR < 0.05 (one-sample t-test across ROIs).

Usage::

    ~/mambaforge/envs/retsupp/bin/python -m retsupp.visualize.plot_cv_ladder \\
        --ladder-dir /data/ds-retsupp/derivatives/af_prf_cv_ladder \\
        --out notes/figures/cv_ladder.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

# Talk palette (notes/figures/talk/make_figures.py): stim / target / AF-dynamic / HP-sustained.
PAL = {'fg': '#1F2933', 'muted': '#9AA5B1', 'stim': '#3D5A80',
       'hp': '#D62828', 'target': '#F77F00', 'af': '#0077B6'}

# Interpretable ROIs only: SPL1/FEF have ~0-1 signal voxels per subject.
ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']

YLIM_ITEMS = (-10, 60)
YLIM_GAIN = (-6, 6)

ROWS = [
    ('Search items in the stimulus model', [
        ('m0_prf', 'm00_bar', 'Items vs bar-only pRF', PAL['stim']),
    ]),
    ('Gain-field components vs pRF + items', [
        ('m1_T', 'm0_prf', 'Target (T)', PAL['target']),
        ('m2_D', 'm0_prf', 'Distractor (D)', PAL['af']),
        ('m3_S', 'm0_prf', 'Sustained (S)', PAL['hp']),
        ('m4_TDS', 'm0_prf', 'T + D + S', PAL['fg']),
    ]),
    ('Statistical learning (HP ≠ LP) vs T + D + S', [
        ('m5_TDS_dS', 'm4_TDS', 'Sustained HP ≠ LP', PAL['hp']),
        ('m6_TDS_dD', 'm4_TDS', 'Dynamic HP ≠ LP', PAL['af']),
        ('m7_full', 'm4_TDS', 'Both', PAL['fg']),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder-dir', type=Path,
                    default=Path('/data/ds-retsupp/derivatives/af_prf_cv_ladder'))
    ap.add_argument('--out', type=Path, default=Path('notes/figures/cv_ladder.pdf'))
    a = ap.parse_args()

    subj = pd.read_csv(a.ladder_dir / 'cv_ladder_subject.tsv', sep='\t')
    stats = pd.read_csv(a.ladder_dir / 'cv_ladder_stats.tsv', sep='\t')
    wide = subj.pivot_table(index=['subject', 'roi'], columns='model',
                            values='cv_r2_signal_median') * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(7.25, 6.2), sharex=True,
                             constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1.1, 1, 1]))
    rng = np.random.default_rng(0)
    xs = np.arange(len(ROIS))
    n_clip = dict(a=0, b=0, c=0)
    for ax, (title, contrasts), letter in zip(axes, ROWS, 'abc'):
        k = len(contrasts)
        width = 0.8 / k
        for j, (rich, simple, label, col) in enumerate(contrasts):
            off = (j - (k - 1) / 2) * width
            for i, roi in enumerate(ROIS):
                d = (wide.xs(roi, level='roi')[rich]
                     - wide.xs(roi, level='roi')[simple]).dropna()
                x = xs[i] + off
                lim = YLIM_ITEMS if letter == 'a' else YLIM_GAIN
                n_clip[letter] += int(((d < lim[0]) | (d > lim[1])).sum())
                ax.scatter(x + rng.uniform(-.28, .28, len(d)) * width, d,
                           s=3, color=col, alpha=.35, lw=0, zorder=2)
                m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
                ax.plot([x, x], [m - se, m + se], color=col, lw=1.4, zorder=3,
                        solid_capstyle='butt')
                ax.plot(x, m, 'D', ms=4.5, mfc=col, mec='k', mew=.6, zorder=4)
                r = stats[(stats.richer == rich) & (stats.simpler == simple)
                          & (stats.roi == roi)]
                if len(r) and r.p_t_fdr.iloc[0] < .05:
                    ax.text(x, m + se, '*', ha='center', va='bottom',
                            fontsize=8, color=col)
            # direct label: coloured text, stacked top-left
            ax.text(0.005 + j * 0.17, 1.0, label, transform=ax.transAxes,
                    color=col, fontsize=7, va='top', ha='left')
        ax.axhline(0, color='0.6', lw=.6, ls='--', zorder=0)
        ax.set_ylabel('ΔCV-R² (×10⁻³)')
        ax.set_title(title, loc='center')
        ax.text(-0.07, 1.04, letter, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='right')
        # Rows b and c share a zoomed range; the few subject points beyond
        # either range are clipped (counts printed for the caption).
        ax.set_ylim(*(YLIM_ITEMS if letter == 'a' else YLIM_GAIN))
    axes[-1].set_xticks(xs, ROIS)
    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=False)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix('.svg'))
    print(f'Clipped subject points per row: {n_clip}')
    print(f'Wrote {a.out}')


if __name__ == '__main__':
    main()
