"""Held-out R² of the PRF model ladder (m1-m6) in reliable voxels.

Reads ``notes/data/prf_cv_roi_summary.tsv`` (aggregate_prf_cv_roi); one value per
subject is the mean over 3 folds. Prints paired tests for every selection.

Usage::

    python -m retsupp.visualize.paper.plot_prf_cv_roi \\
        --tsv notes/data/prf_cv_roi_summary.tsv --out notes/figures/prf_cv_roi_reliable.pdf
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3,
    'ytick.major.size': 3, 'xtick.major.width': .8, 'ytick.major.width': .8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})
FG, MUTED, FLEX = '#1F2933', '#9AA5B1', '#2A9D8F'
NAMES = {2: 'DoG', 5: 'DN', 3: 'Gaussian', 4: 'DoG', 6: 'DN'}
ORDER = [2, 5, 3, 4, 6]
ROIS = ['V1', 'V2', 'V3', 'hV4', 'LO', 'TO', 'V3AB', 'VO', 'IPS']
CONTRASTS = [(2, 1, 'DoG vs Gaussian, fixed HRF'), (4, 3, 'DoG vs Gaussian, flexible HRF'),
             (5, 2, 'DN vs DoG, fixed HRF'), (6, 4, 'DN vs DoG, flexible HRF'),
             (3, 1, 'Flexible HRF, Gaussian'), (4, 2, 'Flexible HRF, DoG'), (6, 5, 'Flexible HRF, DN')]


def subject_table(d, selection, metric, roi='all'):
    x = d[(d.selection == selection) & (d.roi == roi)]
    return x.groupby(['subject', 'model'])[metric].mean().unstack()


def report(d):
    for sel, met in [('reliable', 'median_r2'), ('reliable', 'pooled_r2'),
                     ('reliable_sd20', 'pooled_r2'), ('sd20', 'pooled_r2'), ('all', 'median_r2')]:
        s = subject_table(d, sel, met)
        print(f'{sel}, {met}: ' + ' '.join(f'm{m} {s[m].median():.4f}' for m in s))
        for a, b, lab in CONTRASTS:
            e = s[a] - s[b]
            print(f'  {lab:30s} {e.median():+.4f}  {(e > 0).mean() * 100:3.0f}%  '
                  f'p={wilcoxon(e).pvalue:.2g}')


def main(tsv, out):
    d = pd.read_csv(tsv, sep='\t')
    report(d)
    s = subject_table(d, 'reliable', 'median_r2')
    delta = s.sub(s[1], axis=0)[ORDER]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.25, 2.6), width_ratios=[1, 1.25],
                                 constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, m in enumerate(ORDER):
        col = FLEX if m in (3, 4, 6) else MUTED
        a1.scatter(i + rng.uniform(-.13, .13, len(delta)), delta[m], s=7, color=col,
                   alpha=.55, lw=0, clip_on=False)
        a1.plot([i - .28, i + .28], [delta[m].median()] * 2, color=FG, lw=1.4)
    a1.axhline(0, color=FG, lw=.8)
    a1.set_xticks(range(5), [NAMES[m] for m in ORDER])
    a1.set_xlim(-.6, 4.6)
    top = delta.max().max()
    a1.set_ylim(top=top * 1.14)
    for x0, x1, lab, col in [(0, 1, 'Fixed HRF', MUTED), (2, 4, 'Flexible HRF', FLEX)]:
        a1.text((x0 + x1) / 2, top * 1.1, lab, color=col, ha='center', va='center', fontsize=7)
    a1.set_ylabel('Held-out R² gain over\nGaussian with fixed HRF')
    a1.set_title('Flexible HRF adds the most', loc='left', pad=8)

    rows = []
    for m in ORDER:
        for roi in ROIS:
            x = d[(d.selection == 'reliable') & (d.roi == roi) & (d.n_voxels >= 20)]
            e = x.groupby(['subject', 'model']).median_r2.mean().unstack().dropna()
            rows.append(dict(model=m, roi=roi, gain=(e[m] - e[1]).median()))
    g = pd.DataFrame(rows)
    styles = {2: (MUTED, '--'), 5: (MUTED, '-'), 3: (FLEX, ':'), 4: (FLEX, '--'), 6: (FLEX, '-')}
    for m in ORDER:
        y = g[g.model == m].set_index('roi').gain.reindex(ROIS)
        a2.plot(range(len(ROIS)), y, color=styles[m][0], ls=styles[m][1], lw=1.3)
        lab = f'{NAMES[m]}, {"flexible" if m in (3, 4, 6) else "fixed"}'
        a2.text(-.2, y.iloc[0], lab, color=styles[m][0], va='center', ha='right', fontsize=6.5)
    a2.plot([0, len(ROIS) - 1], [0, 0], color=FG, lw=.8)
    a2.set_xticks(range(len(ROIS)), ROIS)
    a2.set_xlim(-2.6, len(ROIS) - .7)
    a2.spines['bottom'].set_bounds(0, len(ROIS) - 1)
    a2.set_ylabel('Median R² gain')
    a2.set_title('Across the visual hierarchy', loc='left', pad=8)
    sns.despine(fig=fig, offset=5, trim=False)
    fig.savefig(out)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=200)
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='notes/data/prf_cv_roi_summary.tsv')
    ap.add_argument('--out', default='notes/figures/prf_cv_roi_reliable.pdf')
    a = ap.parse_args()
    main(a.tsv, a.out)
