"""Per-ROI correlations between AF model parameters and behavior.

For each ROI, we compute Spearman correlations across the 30 subjects
between THREE model parameters and TWO behavioral RT contrasts:

  Model parameters
    log_ratio = log((1+g_HP) / (1+g_LP))   — HP-vs-LP suppression
                                               strength (scale-invariant).
    g_avg     = ½(g_HP + g_LP)             — overall AF engagement.
    sigma_AF                               — spatial extent of the AF.

  Behavioral contrasts
    RT_HP − RT_LP        — behavioral HP-suppression
                            (negative ⇒ HP attended/suppressed faster).
    RT_dist − RT_no_dist — behavioral cost of having any distractor
                            (positive ⇒ distractors slow you down).

Spearman is rank-based, robust to the σ_AF outlier subjects
(σ_AF ≈ 9°) without arbitrarily dropping them.

Inputs:
    notes/af_parameters.tsv   (per-fit shared params)
    notes/rt_summary.tsv      (per-subject RT contrasts)

Usage:
    python -m retsupp.visualize.plot_behavior_correlation \\
        --out notes/behavior_correlation.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


def load_data(af_tsv: Path, rt_tsv: Path):
    af = pd.read_csv(af_tsv, sep='\t')
    rt = pd.read_csv(rt_tsv, sep='\t')
    rt['subject'] = rt['subject'].astype(int)
    print(f'Loaded {len(af)} fits across {af.subject.nunique()} subjects')
    print(f'Loaded RT summary for {len(rt)} subjects')
    merged = af.merge(rt, on='subject', how='inner')
    print(f'Merged: {len(merged)} (subject, ROI) rows')
    return merged


def cover(pdf, args):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.96, 'AF model parameters × behavioral RT contrasts',
            ha='center', va='top', fontsize=15, weight='bold',
            transform=ax.transAxes)
    body = (
        '  Model parameters per (subject, ROI):\n'
        '    log_ratio = log((1+g_HP)/(1+g_LP))\n'
        '    g_avg     = ½(g_HP + g_LP)\n'
        '    sigma_AF\n'
        '\n'
        '  Behavioral contrasts per subject:\n'
        '    RT_HP − RT_LP        (HP-vs-LP cost; usually NEGATIVE)\n'
        '    RT_dist − RT_no_dist (any distractor cost; positive)\n'
        '\n'
        '  All correlations are SPEARMAN (rank-based, robust to fit outliers).\n'
        '\n'
        '  Expected directions if the model is recovering real attention:\n'
        '    • log_ratio  vs RT_HP−LP  ↔ POSITIVE (both negative)\n'
        '    • g_avg      vs RT_dist−no_dist ↔ POSITIVE\n'
        '      (more attentive engagement ⇒ stronger AF gain ⇒ more cost)\n'
        '    • sigma_AF   vs RT_dist−no_dist ↔ either sign — narrower AF\n'
        '      could mean stronger but more localized engagement.\n'
    )
    ax.text(0.04, 0.86, body, ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def correlation_grid(merged: pd.DataFrame, model_metric: str,
                       behavior_metric: str, ylabel: str, xlabel: str,
                       y_clip: float | None = None):
    """4×2 panel of per-ROI scatter; returns the per-ROI Spearman summary."""
    rois = [r for r in ROI_ORDER if r in merged['roi'].unique()]
    n = len(rois)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))
    axes = np.atleast_2d(axes)
    rec = []
    for i, roi in enumerate(rois):
        ax = axes[i // ncol, i % ncol]
        sub = merged[merged['roi'] == roi].copy()
        x = sub[behavior_metric].values
        y = sub[model_metric].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        # Spearman on UNCLIPPED values.
        if len(x) >= 4:
            r, p = stats.spearmanr(x, y)
        else:
            r, p = np.nan, np.nan
        rec.append(dict(roi=roi, n=len(x), r=r, p=p))
        n_clip = 0
        if y_clip is not None:
            n_clip = int((np.abs(y) > y_clip).sum())
            y_plot = np.clip(y, -y_clip, +y_clip)
        else:
            y_plot = y
        ax.scatter(x, y_plot, s=40, alpha=0.85, color='C0',
                    edgecolor='k', linewidth=0.4)
        # Trendline.
        if len(x) >= 4 and np.std(x) > 0 and np.std(y_plot) > 0:
            order = np.argsort(x)
            slope, intercept = np.polyfit(x, y_plot, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept,
                     color='0.4', lw=1.0, ls='--')
        ax.axhline(0, color='gray', lw=0.4, ls=':')
        ax.axvline(0, color='gray', lw=0.4, ls=':')
        if y_clip is not None:
            ax.set_ylim(-y_clip, +y_clip)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        sig = '*' if (np.isfinite(p) and p < 0.05) else ''
        ax.set_title(
            f'{roi}  r_s={r:+.2f}, p={p:.3f}{sig}'
            + (f'  (↕={n_clip})' if n_clip else ''),
            fontsize=9, color='C3' if sig else 'k',
        )
        ax.grid(alpha=0.2)
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis('off')
    fig.suptitle(f'{ylabel}  vs  {xlabel}',
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, pd.DataFrame(rec)


def summary_page(pdf, summaries: dict):
    """One-page table of per-ROI Spearman r across all 6 correlations."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis('off')
    ax.text(0.5, 0.96, 'Per-ROI Spearman r summary across all model × behavior contrasts',
            ha='center', va='top', fontsize=12, weight='bold',
            transform=ax.transAxes)
    rois = [r for r in ROI_ORDER]
    cols = list(summaries)
    table = pd.DataFrame(index=rois)
    for c, df in summaries.items():
        d = df.set_index('roi')
        table[c + ' r'] = d['r'].apply(lambda v: f'{v:+.2f}' if np.isfinite(v) else '—')
        table[c + ' p'] = d['p'].apply(lambda v: f'{v:.3f}' if np.isfinite(v) else '—')
    ax.text(0.02, 0.88, table.to_string(),
            family='monospace', fontsize=9, va='top', ha='left',
            transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--af-tsv', type=Path,
                        default=Path('notes/af_parameters.tsv'))
    parser.add_argument('--rt-tsv', type=Path,
                        default=Path('notes/rt_summary.tsv'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/behavior_correlation.pdf'))
    args = parser.parse_args()
    merged = load_data(args.af_tsv, args.rt_tsv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 6 correlations: 3 model metrics × 2 behavioral contrasts.
    contrasts = [
        # (model_metric, behavior_metric, ylabel, xlabel, y_clip)
        ('log_ratio',
            'RT_HP_minus_LP',
            'log((1+g_HP)/(1+g_LP))',
            'RT_HP − RT_LP  (s)',
            1.5),
        ('g_avg',
            'RT_dist_minus_no',
            '½(g_HP + g_LP)',
            'RT_dist − RT_no_dist  (s)',
            2.0),
        ('sigma_AF',
            'RT_dist_minus_no',
            'σ_AF  (deg)',
            'RT_dist − RT_no_dist  (s)',
            5.0),
        ('log_ratio',
            'RT_dist_minus_no',
            'log((1+g_HP)/(1+g_LP))',
            'RT_dist − RT_no_dist  (s)',
            1.5),
        ('g_avg',
            'RT_HP_minus_LP',
            '½(g_HP + g_LP)',
            'RT_HP − RT_LP  (s)',
            2.0),
        ('sigma_AF',
            'RT_HP_minus_LP',
            'σ_AF  (deg)',
            'RT_HP − RT_LP  (s)',
            5.0),
    ]

    summaries = {}
    with PdfPages(args.out) as pdf:
        cover(pdf, args)
        for model_m, beh_m, ylab, xlab, yclip in contrasts:
            fig, summary = correlation_grid(
                merged, model_m, beh_m, ylab, xlab, y_clip=yclip,
            )
            pdf.savefig(fig); plt.close(fig)
            summaries[f'{ylab} × {xlab}'] = summary
            sig_count = (summary['p'] < 0.05).sum()
            print(f'{ylab} × {xlab}: {sig_count}/{len(summary)} '
                  f'ROIs significant (p<0.05)')
        summary_page(pdf, summaries)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
