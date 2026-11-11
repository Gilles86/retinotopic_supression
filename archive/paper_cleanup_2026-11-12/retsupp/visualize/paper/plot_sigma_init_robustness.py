"""σ-init robustness: compare DoG dyn-v3 fits with original (σ_AF=2.0,
σ_dyn=0.5) vs neutral (σ_AF=σ_dyn=2.0) init. Same model, same 30
subjects × 8 ROIs, only the σ_dyn init differs.

If parameter estimates stay similar, the σ_AF >> σ_dyn finding is
robust to init bias. If they shift, the original was init-driven.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
TSV_ORIG = REPO / 'notes' / 'data' / 'af_dog_v3_parameters.tsv'
TSV_NEUTRAL = REPO / 'notes' / 'data' / 'af_dog_v3_neutralsigma_parameters.tsv'
OUT = REPO / 'notes' / 'figures' / 'sigma_init_robustness.pdf'

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.titlesize': 13,
})


def stars(p):
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def panel(ax, df_orig, df_neut, col, ylim, title, init_orig_label,
          init_neut_label):
    rois = [r for r in ROI_ORDER if r in df_orig['roi'].unique()]
    for i, roi in enumerate(rois):
        sub_o = df_orig[df_orig['roi'] == roi].set_index('subject')[col]
        sub_n = df_neut[df_neut['roi'] == roi].set_index('subject')[col]
        # Align on subject
        common = sub_o.index.intersection(sub_n.index)
        a = np.clip(sub_o.loc[common].values, *ylim)
        b = np.clip(sub_n.loc[common].values, *ylim)
        if len(a) < 2:
            continue
        xs_a = np.full(len(a), i - 0.16)
        xs_b = np.full(len(a), i + 0.16)
        ax.plot(np.vstack([xs_a, xs_b]), np.vstack([a, b]),
                color='0.6', lw=0.4, alpha=0.4)
        ax.scatter(xs_a, a, color='C0', s=22, alpha=0.7,
                   edgecolor='k', linewidth=0.3)
        ax.scatter(xs_b, b, color='C3', s=22, alpha=0.7,
                   edgecolor='k', linewidth=0.3)
        ax.scatter([i - 0.16], [np.median(a)],
                   marker='_', s=350, color='k', zorder=10, lw=2.5)
        ax.scatter([i + 0.16], [np.median(b)],
                   marker='_', s=350, color='k', zorder=10, lw=2.5)
        try:
            _, p = stats.wilcoxon(sub_o.loc[common], sub_n.loc[common],
                                  alternative='two-sided',
                                  zero_method='pratt')
        except ValueError:
            p = np.nan
        sig = stars(p)
        c = 'C2' if (np.isfinite(p) and p < 0.05) else '0.5'
        med_diff = float(np.median(a) - np.median(b))
        ax.text(i, ylim[1] + 0.04 * (ylim[1] - ylim[0]),
                f'Δmed={med_diff:+.2f}\np={p:.3f}\n{sig}',
                ha='center', va='bottom', fontsize=8.5,
                color=c, weight='bold' if (np.isfinite(p) and p < 0.05) else 'normal')
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_ylabel('σ (deg)')
    ax.set_ylim(ylim[0], ylim[1] + 0.3 * (ylim[1] - ylim[0]))
    ax.set_title(title)
    ax.scatter([], [], color='C0', s=30, label=init_orig_label,
               edgecolor='k', linewidth=0.3)
    ax.scatter([], [], color='C3', s=30, label=init_neut_label,
               edgecolor='k', linewidth=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2, axis='y')


def main():
    df_o = pd.read_csv(TSV_ORIG, sep='\t')
    df_n = pd.read_csv(TSV_NEUTRAL, sep='\t')
    print(f'orig n_fits: {len(df_o)}, neutral n_fits: {len(df_n)}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    init_o = 'init σ_AF=2.0, σ_dyn=0.5'
    init_n = 'init σ_AF=2.0, σ_dyn=2.0 (neutral)'

    with PdfPages(OUT) as pdf:
        # Page 1: σ_AF
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel(ax, df_o, df_n, 'sigma_AF', (0.0, 8.0),
              'σ_AF (sustained AF spatial extent)',
              init_o, init_n)
        fig.suptitle(
            f'σ-init robustness — sustained σ_AF  (paired Wilcoxon, n=30)',
            fontsize=12, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Page 2: σ_dyn
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel(ax, df_o, df_n, 'sigma_dyn', (0.0, 12.0),
              'σ_dyn (phasic AF spatial extent)',
              init_o, init_n)
        fig.suptitle(
            f'σ-init robustness — phasic σ_dyn  (paired Wilcoxon, n=30)',
            fontsize=12, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # Page 3: σ_AF − σ_dyn (the headline differential)
        df_o['sigma_diff'] = df_o['sigma_AF'] - df_o['sigma_dyn']
        df_n['sigma_diff'] = df_n['sigma_AF'] - df_n['sigma_dyn']
        fig, ax = plt.subplots(figsize=(11, 5.5))
        panel(ax, df_o, df_n, 'sigma_diff', (-8.0, 8.0),
              'σ_AF − σ_dyn  (positive = σ_AF wider, the original finding)',
              init_o, init_n)
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        fig.suptitle(
            f'σ-init robustness — σ_AF − σ_dyn differential  '
            f'(positive = sustained wider, n=30)',
            fontsize=12, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
