"""One-panel summary of the CV ladder: how much does each model ingredient
improve held-out prediction, as a percentage of the simpler model's CV-R²?

Lines are ROIs along the x axis; filled markers are BH-FDR < 0.05 across
ROIs (one-sample t-test on per-subject ΔCV-R², from ``cv_ladder_stats.tsv``);
error bars are ±1 SEM across subjects, scaled by the same denominator.
Talk-sized fonts on purpose (readability over the 7 pt paper ceiling).

Usage::

    ~/mambaforge/envs/retsupp/bin/python -m retsupp.visualize.plot_cv_ladder_simple \\
        --out notes/figures/cv_ladder_simple.pdf
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
    'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.9, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

PAL = {'fg': '#1F2933', 'stim': '#3D5A80', 'hp': '#D62828',
       'target': '#F77F00', 'af': '#0077B6'}
ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']

# (richer, simpler, label, colour, linestyle)
STEPS = [
    ('m0_prf', 'm00_bar', 'Search items in the stimulus', PAL['stim'], '-'),
    ('m1_T', 'm0_prf', 'Target gain', PAL['target'], '-'),
    ('m2_D', 'm0_prf', 'Distractor gain', PAL['af'], '-'),
    ('m3_S', 'm0_prf', 'Sustained gain', PAL['hp'], '-'),
    ('m7_full', 'm4_TDS', 'HP-specific learning', PAL['fg'], '--'),
]


def summarize(subj: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    wide = subj.pivot_table(index=['subject', 'roi'], columns='model',
                            values='cv_r2_signal_median')
    rows = []
    for rich, simple, label, col, ls in STEPS:
        for roi in ROIS:
            w = wide.xs(roi, level='roi')[[rich, simple]].dropna()
            d = w[rich] - w[simple]
            base = w[simple].mean()
            st = stats[(stats.richer == rich) & (stats.simpler == simple)
                       & (stats.roi == roi)]
            rows.append(dict(step=label, roi=roi,
                             pct=100 * d.mean() / base,
                             sem=100 * d.std(ddof=1) / np.sqrt(len(d)) / base,
                             sig=bool(len(st) and st.p_t_fdr.iloc[0] < .05)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder-dir', type=Path,
                    default=Path('/data/ds-retsupp/derivatives/af_prf_cv_ladder'))
    ap.add_argument('--out', type=Path,
                    default=Path('notes/figures/cv_ladder_simple.pdf'))
    a = ap.parse_args()

    subj = pd.read_csv(a.ladder_dir / 'cv_ladder_subject.tsv', sep='\t')
    stats = pd.read_csv(a.ladder_dir / 'cv_ladder_stats.tsv', sep='\t')
    df = summarize(subj, stats)
    df.to_csv(a.out.with_suffix('.tsv'), sep='\t', index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.9), constrained_layout=True)
    xs = np.arange(len(ROIS))
    ax.axhline(0, color='0.6', lw=0.8, ls=':', zorder=0)
    n_steps = len(STEPS)
    for j, (_r, _s, label, col, ls) in enumerate(STEPS):
        g = df[df.step == label].set_index('roi').loc[ROIS]
        x = xs + (j - (n_steps - 1) / 2) * 0.06
        ax.plot(x, g.pct, color=col, ls=ls, lw=1.8, zorder=2)
        ax.vlines(x, g.pct - g['sem'], g.pct + g['sem'], color=col, lw=1.1, zorder=2)
        for xi, yi, s in zip(x, g.pct, g.sig):
            ax.plot(xi, yi, 'o', ms=6, mec=col, mew=1.3,
                    mfc=col if s else 'white', zorder=3)

    # Direct label at the line's right end for the one ingredient that helps.
    items = df[df.step == STEPS[0][2]].set_index('roi')
    ax.text(xs[-1] + 0.25, items.loc['IPS', 'pct'], 'Search items\nin the stimulus',
            color=PAL['stim'], fontsize=10.5, va='center', ha='left')

    # Key for the attention ingredients, drawn as glyphs, in the empty mid band.
    kx, ky, dy = 0.03, 0.80, 0.07
    ax.text(kx, ky + 0.08, 'Attention gain fields (all ≈ 0)',
            transform=ax.transAxes, fontsize=10.5, color=PAL['fg'],
            fontweight='bold', va='center')
    for i, (_r, _s, label, col, ls) in enumerate(STEPS[1:]):
        yy = ky - i * dy
        ax.plot([kx, kx + 0.06], [yy, yy], transform=ax.transAxes, color=col,
                ls=ls, lw=1.8, clip_on=False)
        ax.text(kx + 0.08, yy, label, transform=ax.transAxes, color=col,
                fontsize=10, va='center')
    # Filled vs open marker key.
    yy = ky - 4 * dy - 0.03
    ax.plot(kx + 0.015, yy, 'o', transform=ax.transAxes, ms=6, mfc='0.3',
            mec='0.3', clip_on=False)
    ax.plot(kx + 0.045, yy, 'o', transform=ax.transAxes, ms=6, mfc='white',
            mec='0.3', mew=1.3, clip_on=False)
    ax.text(kx + 0.08, yy, 'Significant / not (FDR < .05)',
            transform=ax.transAxes, color='0.3', fontsize=9.5, va='center')

    ax.set_xticks(xs, ROIS)
    ax.set_xlim(-0.4, len(ROIS) + 0.6)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylim(-4, 42)
    ax.set_ylabel('Better held-out prediction (%)')
    ax.set_title('What improves cross-validated pRF predictions?',
                 fontsize=11.5, loc='left')
    sns.despine(ax=ax, offset=5, trim=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix('.svg'))
    print(df.round(2).to_string(index=False))
    print(f'Wrote {a.out}')


if __name__ == '__main__':
    main()
