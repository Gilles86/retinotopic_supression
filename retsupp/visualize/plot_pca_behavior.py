"""PCA of per-(subject, ROI) AF parameters → behavioral correlations.

Per-ROI behavioral correlations have 8 weak tests with N=30 each.
Instead, treat the 8 ROIs as 8 features per parameter and do PCA across
subjects. The first principal component (PC1) captures the dominant
shared variance across ROIs — i.e. a single subject-level "trait" that
summarizes whatever covaries across the visual hierarchy.

For each parameter (σ_AF, g_HP, g_LP, g_diff, g_avg, log_ratio):
  • Build a (n_subjects, n_rois) matrix.
  • Optionally rank-transform per column (robust to outliers).
  • Standardize per column (mean 0, var 1).
  • Run PCA, retain PC1 + PC2.
  • Plot the PC1 loadings (which ROIs contribute and with what sign).
  • Correlate per-subject PC1 with each behavioral RT contrast
    (RT_HP − RT_LP, RT_dist − RT_no_dist).

Uses Spearman by default after rank-transforming the inputs.

Usage:
    python -m retsupp.visualize.plot_pca_behavior \\
        --out notes/pca_behavior.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.decomposition import PCA


ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


def build_wide(df: pd.DataFrame, metric: str, rank_transform: bool = True):
    """Return (n_subjects × n_rois) numpy matrix and metadata."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    wide = (df.pivot_table(index='subject', columns='roi',
                              values=metric, aggfunc='first')
              [rois])
    # Drop subjects with any missing ROI.
    wide = wide.dropna()
    if rank_transform:
        # Per-column ranks — robust to outliers, scale-free.
        wide = wide.rank(axis=0)
    # Standardize per column.
    Z = (wide.values - wide.values.mean(axis=0)) / wide.values.std(axis=0, ddof=1)
    return Z, wide.index.values, rois


def per_subject_PC(Z: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components, svd_solver='full')
    PCs = pca.fit_transform(Z)
    return pca, PCs


def page_for_metric(pdf, df, metric: str, label: str, beh: pd.DataFrame,
                     rank_transform: bool = True):
    """One page: PC loadings + PC1/PC2 vs each behavioral contrast."""
    Z, subjects, rois = build_wide(df, metric, rank_transform=rank_transform)
    pca, PCs = per_subject_PC(Z, n_components=2)
    loadings = pca.components_              # (2, n_rois)
    var = pca.explained_variance_ratio_

    # Merge PC scores with behavior.
    pc_df = pd.DataFrame({'subject': subjects,
                            'PC1': PCs[:, 0],
                            'PC2': PCs[:, 1]})
    merged = pc_df.merge(beh, on='subject', how='inner')

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f'{label}  —  PCA across ROIs '
        f'({"rank-transformed, " if rank_transform else ""}'
        f'PC1 var={var[0]*100:.0f}%, PC2 var={var[1]*100:.0f}%)',
        fontsize=12,
    )

    # (a) PC1 + PC2 loadings.
    ax = fig.add_subplot(2, 3, 1)
    width = 0.4
    x = np.arange(len(rois))
    ax.bar(x - width/2, loadings[0], width=width, label='PC1', color='C0')
    ax.bar(x + width/2, loadings[1], width=width, label='PC2', color='C1')
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(rois, rotation=30)
    ax.set_ylabel('PC loading')
    ax.set_title('Loadings — which ROIs drive each PC?', fontsize=10)
    ax.legend()
    ax.grid(alpha=0.2)

    # (b) Subject scores on PC1 vs PC2.
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(PCs[:, 0], PCs[:, 1], s=50, alpha=0.85,
                color='C0', edgecolor='k', linewidth=0.4)
    for i, s in enumerate(subjects):
        ax.annotate(f'{int(s):02d}', (PCs[i, 0], PCs[i, 1]),
                     fontsize=6, color='0.4',
                     xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.set_xlabel('PC1 score'); ax.set_ylabel('PC2 score')
    ax.set_title('Subject scores in PC space', fontsize=10)
    ax.grid(alpha=0.2)

    # (c) Variance explained bar.
    ax = fig.add_subplot(2, 3, 3)
    var_full = pca.explained_variance_ratio_
    ax.bar(np.arange(len(var_full)) + 1, var_full,
            color='C0', edgecolor='k')
    ax.set_xlabel('PC index')
    ax.set_ylabel('variance explained')
    ax.set_title(f'PC1 explains {var[0]*100:.0f}% of variance', fontsize=10)
    ax.set_xticks(np.arange(len(var_full)) + 1)

    # (d-f) PC1 vs each of three behavior contrasts.
    contrasts = [
        ('RT_HP_minus_LP',   'RT_HP − RT_LP  (s)'),
        ('RT_dist_minus_no', 'RT_dist − RT_no_dist  (s)'),
        ('RT_HP_minus_no',   'RT_HP − RT_no_dist  (s)'),
    ]
    for i, (col, x_label) in enumerate(contrasts):
        ax = fig.add_subplot(2, 3, 4 + i)
        x = merged[col].values; y = merged['PC1'].values
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) >= 4:
            r, p = stats.spearmanr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
        else:
            r = p = slope = intercept = np.nan
        ax.scatter(x, y, s=50, alpha=0.85, color='C0',
                    edgecolor='k', linewidth=0.4)
        if np.isfinite(slope):
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color='0.4', ls='--', lw=1)
        ax.axhline(0, color='gray', lw=0.4, ls=':')
        ax.axvline(0, color='gray', lw=0.4, ls=':')
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel('PC1 score', fontsize=9)
        sig = '*' if (np.isfinite(p) and p < 0.05) else ''
        ax.set_title(
            f'PC1 × {x_label}  r_s={r:+.2f}, p={p:.3f}{sig}',
            fontsize=9, color='C3' if sig else 'k',
        )
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--af-tsv', type=Path,
                        default=Path('notes/af_parameters.tsv'))
    parser.add_argument('--rt-tsv', type=Path,
                        default=Path('notes/rt_summary.tsv'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/pca_behavior.pdf'))
    parser.add_argument('--no-rank', action='store_true',
                        help='Skip per-column rank transform '
                             '(default rank-transforms before PCA).')
    args = parser.parse_args()

    df = pd.read_csv(args.af_tsv, sep='\t')
    beh = pd.read_csv(args.rt_tsv, sep='\t')
    beh['subject'] = beh['subject'].astype(int)
    print(f'Loaded {len(df)} fits, {df.subject.nunique()} subjects')
    print(f'Loaded {len(beh)} behavior rows')

    metrics = [
        ('sigma_AF',  'σ_AF'),
        ('g_HP',      'g_HP'),
        ('g_LP',      'g_LP'),
        ('g_diff',    'g_HP − g_LP'),
        ('g_avg',     '½(g_HP + g_LP)'),
        ('log_ratio', 'log((1+g_HP)/(1+g_LP))'),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        # Cover.
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.96,
                 'Per-parameter PCA across ROIs  ×  behavior',
                 ha='center', va='top', fontsize=15, weight='bold',
                 transform=ax.transAxes)
        body = (
            f'  • For each AF parameter, build a (subject × ROI) matrix\n'
            f'    {"(rank-transformed per ROI, then standardized)" if not args.no_rank else "(standardized only)"}.\n'
            f'  • Run PCA. PC1 captures the largest shared subject-level\n'
            f'    variance across the visual hierarchy.\n'
            f'  • Correlate per-subject PC1 with each behavioral contrast.\n'
            f'\n'
            f'Why this is more powerful than per-ROI:\n'
            f'    Per-ROI behavior correlations test 8 weak signals\n'
            f'    independently. PC1 pools the SIGNAL across ROIs and\n'
            f'    rejects ROI-specific noise — one strong test per\n'
            f'    parameter instead of 8 weak ones.\n'
        )
        ax.text(0.04, 0.86, body, ha='left', va='top',
                family='monospace', fontsize=10, transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        # One page per metric.
        rank = not args.no_rank
        for col, label in metrics:
            try:
                page_for_metric(pdf, df, col, label, beh,
                                  rank_transform=rank)
                print(f'  {label}: page rendered.')
            except Exception as e:
                print(f'  {label}: skipped — {e}')

    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
