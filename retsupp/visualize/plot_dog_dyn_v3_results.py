"""Figures corresponding to notes/models/af_dynamic_v3_dog.md.

Four panels of the DoG-AF dyn-v3 results:
  (1) sigma_AF vs sigma_dyn paired strip plot per ROI
  (2) Sustained HP-vs-LP differential (g_HP - g_LP) per ROI
  (3) Dynamic   HP-vs-LP differential (g_HP_dyn - g_LP_dyn) per ROI
  (4) Phasic effect (g_dyn_avg = mean(g_HP_dyn, g_LP_dyn)) per ROI

Each panel shows per-subject points + median line + paired Wilcoxon
test annotation.

Usage:
  ~/mambaforge/envs/retsupp/bin/python \
    retsupp/visualize/plot_dog_dyn_v3_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'notes' / 'af_dog_v3_parameters.tsv'
OUT = REPO / 'notes' / 'figures' / 'af_dyn_v3_dog_results.pdf'

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.titlesize': 14,
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


def panel_paired(ax, df, col_a, col_b, label_a, label_b, ylim,
                 title, ylabel, alternative='two-sided'):
    """Two paired columns (e.g. sigma_AF vs sigma_dyn) per ROI, with
    Wilcoxon paired test annotation."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    for i, roi in enumerate(rois):
        sub = df[df['roi'] == roi]
        if len(sub) < 2:
            continue
        a = np.clip(sub[col_a].values, *ylim)
        b = np.clip(sub[col_b].values, *ylim)
        xs_a = np.full(len(sub), i - 0.16)
        xs_b = np.full(len(sub), i + 0.16)
        ax.plot(np.vstack([xs_a, xs_b]), np.vstack([a, b]),
                color='0.6', lw=0.4, alpha=0.4)
        ax.scatter(xs_a, a, color='C0', s=24, alpha=0.7,
                   edgecolor='k', linewidth=0.3)
        ax.scatter(xs_b, b, color='C3', s=24, alpha=0.7,
                   edgecolor='k', linewidth=0.3)
        ax.scatter([i - 0.16], [np.median(a)],
                   marker='_', s=350, color='k', zorder=10, lw=2.5)
        ax.scatter([i + 0.16], [np.median(b)],
                   marker='_', s=350, color='k', zorder=10, lw=2.5)
        try:
            _, p = stats.wilcoxon(sub[col_a], sub[col_b],
                                  alternative=alternative,
                                  zero_method='pratt')
        except ValueError:
            p = np.nan
        sig = stars(p)
        c = 'C2' if (np.isfinite(p) and p < 0.05) else '0.5'
        ax.text(i, ylim[1] + 0.05 * (ylim[1] - ylim[0]),
                f'p={p:.3f}\n{sig}' if np.isfinite(p) else 'n/a',
                ha='center', va='bottom', fontsize=9,
                color=c, weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1] + 0.18 * (ylim[1] - ylim[0]))
    ax.set_title(title)
    ax.grid(alpha=0.2, axis='y')
    ax.scatter([], [], color='C0', s=30, label=label_a,
               edgecolor='k', linewidth=0.3)
    ax.scatter([], [], color='C3', s=30, label=label_b,
               edgecolor='k', linewidth=0.3)
    ax.legend(loc='upper right', fontsize=9, frameon=True)


def panel_two_raw_gains(axes, df, cols, labels, ylim, suptitle):
    """Two raw gains (e.g. g_HP, g_LP) — one subplot each, with
    Wilcoxon two-sided vs 0 per ROI. Shows whether each gain is
    positive, negative, or null."""
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
            ax.scatter([i], [med],
                       marker='_', s=350, color='k', zorder=10, lw=2.5)
            try:
                _, p = stats.wilcoxon(sub[col],
                                      alternative='two-sided',
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
                    ha='center', va='bottom', fontsize=8,
                    color=c, weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_ylabel(lab)
        ax.set_ylim(ylim[0], ylim[1] + 0.30 * (ylim[1] - ylim[0]))
        ax.set_title(lab)
        ax.grid(alpha=0.2, axis='y')


def panel_one_value(ax, df, col, ylim, title, ylabel, alternative,
                    direction_caption):
    """One value per (subject, ROI), test if != or < 0."""
    rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
    for i, roi in enumerate(rois):
        sub = df[df['roi'] == roi]
        if len(sub) < 2:
            continue
        v = np.clip(sub[col].values, *ylim)
        xs = np.full(len(sub), i)
        ax.scatter(xs, v, color='C0', s=28, alpha=0.65,
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
        ax.text(i, ylim[1] + 0.05 * (ylim[1] - ylim[0]),
                f'med={med:.2f}\np={p:.3f}\n{sig}'
                if np.isfinite(p) else 'n/a',
                ha='center', va='bottom', fontsize=8.5,
                color=c, weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1] + 0.30 * (ylim[1] - ylim[0]))
    ax.set_title(title)
    ax.text(0.5, -0.16, direction_caption,
            ha='center', va='top', fontsize=9, color='0.3',
            transform=ax.transAxes, style='italic')
    ax.grid(alpha=0.2, axis='y')


def main():
    df = pd.read_csv(TSV, sep='\t')
    n_sub = df['subject'].nunique()
    df['g_diff_sus'] = df['g_HP'] - df['g_LP']
    df['g_diff_dyn'] = df['g_HP_dyn'] - df['g_LP_dyn']
    df['g_dyn_avg'] = 0.5 * (df['g_HP_dyn'] + df['g_LP_dyn'])

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUT) as pdf:
        # Panel 1 — sigma comparison
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_paired(
            ax, df, 'sigma_AF', 'sigma_dyn',
            label_a='σ_AF (sustained)',
            label_b='σ_dyn (phasic)',
            ylim=(0.0, 8.0),
            title='σ_AF vs σ_dyn (DoG dyn-v3)',
            ylabel='σ (deg)',
            alternative='two-sided',
        )
        fig.suptitle(
            f'DoG dyn-v3 — sustained vs phasic AF size  '
            f'(n={n_sub})',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Panel 2 — sustained HP vs LP differential
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_one_value(
            ax, df, 'g_diff_sus',
            ylim=(-3.0, 3.0),
            title='Sustained HP-vs-LP differential  (g_HP − g_LP)',
            ylabel='g_HP − g_LP',
            alternative='less',
            direction_caption='< 0  ⇒  HP suppressed more than LP (one-sided)',
        )
        fig.suptitle(
            f'DoG dyn-v3 — sustained HP-specificity  (n={n_sub})',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Panel 3 — dynamic HP vs LP differential
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_one_value(
            ax, df, 'g_diff_dyn',
            ylim=(-4.0, 4.0),
            title='Dynamic HP-vs-LP differential  (g_HP_dyn − g_LP_dyn)',
            ylabel='g_HP_dyn − g_LP_dyn',
            alternative='less',
            direction_caption='< 0  ⇒  HP-distractor suppressed more than LP-distractor (one-sided)',
        )
        fig.suptitle(
            f'DoG dyn-v3 — dynamic HP-specificity  (n={n_sub})',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Panel 5 — raw sustained gains (g_HP, g_LP) per ROI
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
        panel_two_raw_gains(
            axes, df,
            cols=['g_HP', 'g_LP'],
            labels=['g_HP  (sustained gain at HP location)',
                    'g_LP  (sustained gain at LP locations)'],
            ylim=(-3.0, 3.0),
            suptitle='',
        )
        fig.suptitle(
            f'DoG dyn-v3 — RAW sustained gains per ROI  '
            f'(Wilcoxon two-sided vs 0,  n={n_sub})\n'
            'green = significantly negative (suppression);  '
            'red = significantly positive (capture);  grey = n.s.',
            fontsize=12, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig); plt.close(fig)

        # Panel 6 — raw dynamic gains (g_HP_dyn, g_LP_dyn) per ROI
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
        panel_two_raw_gains(
            axes, df,
            cols=['g_HP_dyn', 'g_LP_dyn'],
            labels=['g_HP_dyn  (phasic gain when distractor at HP)',
                    'g_LP_dyn  (phasic gain when distractor at LP)'],
            ylim=(-8.0, 8.0),
            suptitle='',
        )
        fig.suptitle(
            f'DoG dyn-v3 — RAW dynamic gains per ROI  '
            f'(Wilcoxon two-sided vs 0,  n={n_sub})\n'
            'green = significantly negative (suppression);  '
            'red = significantly positive (capture);  grey = n.s.',
            fontsize=12, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig); plt.close(fig)

        # Panel 4 — phasic average effect
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel_one_value(
            ax, df, 'g_dyn_avg',
            ylim=(-6.0, 6.0),
            title='Phasic distractor effect  (½(g_HP_dyn + g_LP_dyn))',
            ylabel='g_dyn_avg',
            alternative='two-sided',
            direction_caption='< 0  ⇒  net phasic SUPPRESSION   |   > 0  ⇒  net phasic CAPTURE',
        )
        fig.suptitle(
            f'DoG dyn-v3 — net phasic effect of any distractor  (n={n_sub})',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
