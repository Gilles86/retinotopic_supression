"""Two-panel summary of the CV ladder: how much does each model ingredient
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
# Vertical nudges (in %) so right-end labels of panel b don't collide.
LABEL_NUDGE = {'Target gain': 0.25, 'Distractor gain': 0.15,
               'Sustained gain': -0.1, 'HP-specific learning': -0.35}
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

    fig, (ax_i, ax_a) = plt.subplots(
        2, 1, figsize=(6.8, 5.6), sharex=True, constrained_layout=True,
        gridspec_kw=dict(height_ratios=[1, 1.25]))
    xs = np.arange(len(ROIS))

    def draw(ax, steps):
        ax.axhline(0, color='0.6', lw=0.8, ls=':', zorder=0)
        for j, (_r, _s, label, col, ls) in enumerate(steps):
            g = df[df.step == label].set_index('roi').loc[ROIS]
            x = xs + (j - (len(steps) - 1) / 2) * 0.08
            ax.plot(x, g.pct, color=col, ls=ls, lw=1.8, zorder=2)
            ax.vlines(x, g.pct - g['sem'], g.pct + g['sem'], color=col,
                      lw=1.1, zorder=2)
            for xi, yi, s in zip(x, g.pct, g.sig):
                ax.plot(xi, yi, 'o', ms=6, mec=col, mew=1.3,
                        mfc=col if s else 'white', zorder=3)
            # Direct label at the line's right end.
            ax.text(x[-1] + 0.2, g.pct.iloc[-1] + LABEL_NUDGE.get(label, 0),
                    label, color=col, fontsize=10, va='center', ha='left')

    # a: the stimulus-model ingredient, on its own (large) scale.
    draw(ax_i, STEPS[:1])
    ax_i.set_yticks([0, 10, 20, 30, 40])
    ax_i.set_ylim(-3, 42)
    ax_i.set_title('Adding the search items to the stimulus model helps',
                   fontsize=11, loc='left')
    # Filled vs open marker key.
    kx, ky = 0.03, 0.85
    ax_i.plot(kx + 0.01, ky, 'o', transform=ax_i.transAxes, ms=6, mfc='0.3',
              mec='0.3', clip_on=False)
    ax_i.plot(kx + 0.04, ky, 'o', transform=ax_i.transAxes, ms=6, mfc='white',
              mec='0.3', mew=1.3, clip_on=False)
    ax_i.text(kx + 0.07, ky, 'Significant / not (FDR < .05)',
              transform=ax_i.transAxes, color='0.3', fontsize=9.5, va='center')

    # b: attention ingredients, on a zoomed y-axis.
    draw(ax_a, STEPS[1:])
    ax_a.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax_a.set_ylim(-3, 3)
    ax_a.set_title('Adding attention gain fields does not (zoomed y-axis)',
                   fontsize=11, loc='left')

    for ax, letter in ((ax_i, 'a'), (ax_a, 'b')):
        ax.set_ylabel('Better held-out\nprediction (%)')
        ax.text(-0.09, 1.02, letter, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='bottom', ha='right')
        ax.set_xlim(-0.4, len(ROIS) + 1.2)
        sns.despine(ax=ax, offset=5)
        ax.spines['bottom'].set_bounds(xs[0], xs[-1])
    ax_a.set_xticks(xs, ROIS)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix('.svg'))
    print(df.round(2).to_string(index=False))
    print(f'Wrote {a.out}')


if __name__ == '__main__':
    main()
