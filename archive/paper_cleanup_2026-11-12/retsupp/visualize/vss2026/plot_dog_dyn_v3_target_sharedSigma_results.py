"""Figures for the DoG-dyn-v3 + target-capture model results.

Three panels:
  (1) g_T_dyn per ROI — the target-onset gain. Wilcoxon two-sided vs 0
      per ROI. Predicted positive (capture) in higher visual areas.
  (2) g_HP_dyn vs g_LP_dyn per ROI in this richer model. Sanity-check:
      do the distractor-suppression numbers survive once the target
      term is included, or do they collapse?
  (3) σ_T_dyn vs σ_dyn per ROI. The target-onset spatial extent vs
      the (existing) distractor-onset spatial extent.

Usage:
  ~/mambaforge/envs/retsupp/bin/python \
    retsupp/visualize/plot_dog_dyn_v3_target_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'notes' / 'data' / 'af_dog_v3_target_sharedSigma_parameters.tsv'
OUT = REPO / 'notes' / 'figures' / 'af_dyn_v3_dog_target_sharedSigma_sigma2_results.pdf'

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.titlesize': 13,
})


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def panel_one_value(ax, df, col, ylim, title, ylabel,
                    direction_caption, alternative='two-sided'):
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    for i, roi in enumerate(rois):
        sub = df[df['roi'] == roi]
        if len(sub) < 2:
            continue
        v = np.clip(sub[col].values, *ylim)
        xs = np.full(len(sub), i)
        ax.scatter(xs, v, color='C0', s=24, alpha=0.6,
                   edgecolor='k', linewidth=0.3)
        med = float(np.median(v))
        ax.scatter([i], [med],
                   marker='_', s=400, color='k', zorder=10, lw=3)
        try:
            _, p = stats.wilcoxon(sub[col],
                                  alternative=alternative,
                                  zero_method='pratt')
        except ValueError:
            p = np.nan
        sig = stars(p)
        if np.isfinite(p) and p < 0.05:
            c = 'C2' if med < 0 else 'C3'
        else:
            c = '0.5'
        ax.text(i, ylim[1] + 0.04 * (ylim[1] - ylim[0]),
                f'{med:+.2f}\np={p:.3f}\n{sig}'
                if np.isfinite(p) else 'n/a',
                ha='center', va='bottom', fontsize=8.5,
                color=c, weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([f'{r}\nn={len(df[df["roi"] == r])}' for r in rois],
                       fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1] + 0.30 * (ylim[1] - ylim[0]))
    ax.set_title(title)
    ax.text(0.5, -0.16, direction_caption,
            ha='center', va='top', fontsize=9, color='0.3',
            transform=ax.transAxes, style='italic')
    ax.grid(alpha=0.2, axis='y')


def panel_two_raw(axes, df, cols, labels, ylim, suptitle):
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    for ax, col, lab in zip(axes, cols, labels):
        for i, roi in enumerate(rois):
            sub = df[df['roi'] == roi]
            if len(sub) < 2:
                continue
            v = np.clip(sub[col].values, *ylim)
            xs = np.full(len(sub), i)
            ax.scatter(xs, v, color='C0', s=24, alpha=0.6,
                       edgecolor='k', linewidth=0.3)
            med = float(np.median(v))
            ax.scatter([i], [med], marker='_', s=350, color='k',
                       zorder=10, lw=2.5)
            try:
                _, p = stats.wilcoxon(sub[col], alternative='two-sided',
                                      zero_method='pratt')
            except ValueError:
                p = np.nan
            sig = stars(p)
            c = ('C2' if (np.isfinite(p) and p < 0.05 and med < 0)
                 else 'C3' if (np.isfinite(p) and p < 0.05 and med > 0)
                 else '0.5')
            ax.text(i, ylim[1] + 0.04 * (ylim[1] - ylim[0]),
                    f'{med:+.2f}\np={p:.3f}\n{sig}',
                    ha='center', va='bottom', fontsize=8,
                    color=c, weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_ylabel(lab)
        ax.set_ylim(ylim[0], ylim[1] + 0.30 * (ylim[1] - ylim[0]))
        ax.set_title(lab)
        ax.grid(alpha=0.2, axis='y')


def main(tsv_path: Path = TSV, out_path: Path = OUT):
    df = pd.read_csv(tsv_path, sep='\t')
    n = df['subject'].nunique()
    print(f'n subjects: {n}, n rows: {len(df)}')
    df['g_diff_sus'] = df['g_HP'] - df['g_LP']
    df['g_diff_dyn'] = df['g_HP_dyn'] - df['g_LP_dyn']
    df['g_dyn_avg'] = 0.5 * (df['g_HP_dyn'] + df['g_LP_dyn'])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        # Page 1: g_T_dyn per ROI — THE headline figure for target capture
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_one_value(
            ax, df, 'g_T_dyn',
            ylim=(-8.0, 8.0),
            title='Target-onset gain  (g_T_dyn)',
            ylabel='g_T_dyn',
            direction_caption=(
                '> 0  ⇒  attentional CAPTURE at target location  '
                '(the predicted positive control)'),
            alternative='two-sided',
        )
        fig.suptitle(
            f'DoG dyn-v3 + target sharedSigma  —  PHASIC TARGET-CAPTURE gain  '
            f'(n varies per ROI; annotated below)',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Page 2: distractor gains — do they survive the model expansion?
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
        panel_two_raw(
            axes, df,
            cols=['g_HP_dyn', 'g_LP_dyn'],
            labels=['g_HP_dyn  (phasic gain at HP-distractor)',
                    'g_LP_dyn  (phasic gain at LP-distractor)'],
            ylim=(-8.0, 8.0),
            suptitle='',
        )
        fig.suptitle(
            f'DoG dyn-v3 + target sharedSigma  —  RAW dynamic-distractor gains  '
            f'(survives target expansion?)  n varies per ROI',
            fontsize=12, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

        # Page 2b: Sustained HP-vs-LP differential
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_one_value(
            ax, df, 'g_diff_sus',
            ylim=(-3.0, 3.0),
            title='SUSTAINED HP-vs-LP differential  (g_HP − g_LP)',
            ylabel='g_HP − g_LP',
            direction_caption='< 0  ⇒  HP suppressed more than LP (one-sided)',
            alternative='less',
        )
        fig.suptitle(
            f'DoG dyn-v3 + target sharedSigma  —  sustained HP-specificity  '
            f'(n varies per ROI; annotated below)',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Page 2c: Dynamic HP-vs-LP differential
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_one_value(
            ax, df, 'g_diff_dyn',
            ylim=(-4.0, 4.0),
            title='DYNAMIC HP-vs-LP differential  (g_HP_dyn − g_LP_dyn)',
            ylabel='g_HP_dyn − g_LP_dyn',
            direction_caption='< 0  ⇒  HP-distractor suppressed more than LP-distractor (one-sided)',
            alternative='less',
        )
        fig.suptitle(
            f'DoG dyn-v3 + target sharedSigma  —  dynamic HP-specificity  '
            f'(n varies per ROI; annotated below)',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Page 2d: Raw sustained gains side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
        panel_two_raw(
            axes, df,
            cols=['g_HP', 'g_LP'],
            labels=['g_HP  (sustained gain at HP location)',
                    'g_LP  (sustained gain at LP locations)'],
            ylim=(-3.0, 3.0),
            suptitle='',
        )
        fig.suptitle(
            f'DoG dyn-v3 + target sharedSigma  —  RAW sustained gains  '
            f'(green=suppression / red=capture)  n varies per ROI',
            fontsize=12, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

        # Page 2e: Phasic average effect
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_one_value(
            ax, df, 'g_dyn_avg',
            ylim=(-6.0, 6.0),
            title='Phasic distractor effect  (½(g_HP_dyn + g_LP_dyn))',
            ylabel='g_dyn_avg',
            direction_caption='< 0  ⇒  net phasic SUPPRESSION   |   > 0  ⇒  net phasic CAPTURE',
            alternative='two-sided',
        )
        fig.suptitle(
            f'DoG dyn-v3 + target sharedSigma  —  net phasic distractor effect  '
            f'(n varies per ROI; annotated below)',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Page 3: σ_T_dyn vs σ_dyn per ROI (paired)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
        ylim = (0.0, 10.0)
        for i, roi in enumerate(rois):
            sub = df[df['roi'] == roi]
            if len(sub) < 2:
                continue
            a = np.clip(sub['sigma_dyn'].values, *ylim)
            b = np.clip(sub['sigma_T_dyn'].values, *ylim)
            xs_a = np.full(len(sub), i - 0.16)
            xs_b = np.full(len(sub), i + 0.16)
            ax.plot(np.vstack([xs_a, xs_b]), np.vstack([a, b]),
                    color='0.6', lw=0.4, alpha=0.4)
            ax.scatter(xs_a, a, color='C3', s=24, alpha=0.7,
                       edgecolor='k', linewidth=0.3)
            ax.scatter(xs_b, b, color='C5', s=24, alpha=0.7,
                       edgecolor='k', linewidth=0.3)
            ax.scatter([i - 0.16], [np.median(a)],
                       marker='_', s=350, color='k', zorder=10, lw=2.5)
            ax.scatter([i + 0.16], [np.median(b)],
                       marker='_', s=350, color='k', zorder=10, lw=2.5)
            try:
                _, p = stats.wilcoxon(sub['sigma_dyn'],
                                      sub['sigma_T_dyn'],
                                      alternative='two-sided',
                                      zero_method='pratt')
            except ValueError:
                p = np.nan
            sig = stars(p)
            c = 'C2' if (np.isfinite(p) and p < 0.05) else '0.5'
            ax.text(i, ylim[1] + 0.05 * (ylim[1] - ylim[0]),
                    f'p={p:.3f}\n{sig}',
                    ha='center', va='bottom', fontsize=9,
                    color=c, weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_ylabel('σ (deg)')
        ax.set_ylim(ylim[0], ylim[1] + 0.20 * (ylim[1] - ylim[0]))
        ax.set_title('σ_dyn (distractor) vs σ_T_dyn (target) per ROI')
        ax.grid(alpha=0.2, axis='y')
        ax.scatter([], [], color='C3', s=30, label='σ_dyn (distractor)',
                   edgecolor='k', linewidth=0.3)
        ax.scatter([], [], color='C5', s=30, label='σ_T_dyn (target)',
                   edgecolor='k', linewidth=0.3)
        ax.legend(loc='upper right', fontsize=10)
        fig.suptitle(
            f'DoG dyn-v3 + target sharedSigma  —  spatial extents  n={n}',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig); plt.close(fig)

    print(f'wrote {out_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tsv', type=Path, default=TSV,
                   help='Per-(subject, ROI) AF params TSV. Default: '
                        'notes/data/af_dog_v3_target_sharedSigma_parameters.tsv')
    p.add_argument('--out', type=Path, default=OUT,
                   help='Output PDF path.')
    a = p.parse_args()
    main(tsv_path=a.tsv, out_path=a.out)
