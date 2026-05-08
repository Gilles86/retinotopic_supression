"""Quick peek at CV factorial results so far (3508/4080 fits done).

Per ROI: median CV-R² across subjects for each of the 17 classes,
ranked. Highlights which gain pattern best predicts held-out data.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'notes' / 'data' / 'cv_factorial_summary.tsv'
OUT = REPO / 'notes' / 'figures' / 'cv_factorial_peek.pdf'
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11})


def short_label(cls):
    if cls == 'signed-control':
        return 'signed (free)'
    # sus-X-Y_dyn-Z-W
    parts = cls.replace('sus-', 's:').replace('_dyn-', ' d:')
    parts = parts.replace('plus-plus', '++').replace('minus-zero', '−0')
    parts = parts.replace('minus-minus', '−−').replace('zero-zero', '00')
    return parts


def main():
    df = pd.read_csv(TSV, sep='\t')
    print(f'rows: {len(df)}, classes: {df.cls.nunique()}, '
          f'subjects: {df.subject.nunique()}')

    # Median CV-R² per (roi, cls), and per-class subject count.
    summary = df.groupby(['roi', 'cls']).agg(
        median_cv=('cv_r2_median', 'median'),
        n=('cv_r2_median', 'size'),
    ).reset_index()
    summary['short'] = summary['cls'].map(short_label)

    # Per-subject median CV-R² is already in the TSV (cv_r2_median).
    # Aggregate across subjects: mean and SEM per (ROI, class).
    persub_summary = df.groupby(['roi', 'cls']).agg(
        mean_cv=('cv_r2_median', 'mean'),
        sem_cv=('cv_r2_median', 'sem'),
        n=('cv_r2_median', 'size'),
    ).reset_index()
    persub_summary['short'] = persub_summary['cls'].map(short_label)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        # Page 1: per-ROI ranked classes
        rois = [r for r in ROI_ORDER if r in summary['roi'].unique()]
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=False)
        for ax, roi in zip(axes.flat, rois):
            sub = summary[summary['roi'] == roi].sort_values(
                'median_cv', ascending=False).reset_index(drop=True)
            colors = ['C2' if c == sub.iloc[0]['cls']
                      else 'C7' if 'plus-plus' in c
                      else 'C0' for c in sub['cls']]
            ax.barh(range(len(sub)), sub['median_cv'].values,
                    color=colors, alpha=0.8)
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(row['median_cv'] + 0.001, i,
                        f'{row["short"]}  (n={row["n"]})',
                        va='center', fontsize=7)
            ax.invert_yaxis()
            ax.set_yticks([])
            ax.set_xlabel('median CV-R² across subjects')
            ax.set_title(f'{roi}  (winning: {sub.iloc[0]["short"]})')
            ax.grid(alpha=0.2, axis='x')
            ax.set_xlim(0, summary['median_cv'].max() * 1.4)
        fig.suptitle(
            f'CV factorial peek (n_fits={len(df)}, partial)  —  '
            f'per-ROI ranked classes by median CV-R²',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # Page 1b: mean ± SEM across subjects, per ROI (ranked).
        rois = [r for r in ROI_ORDER if r in persub_summary['roi'].unique()]
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for ax, roi in zip(axes.flat, rois):
            sub = persub_summary[persub_summary['roi'] == roi].sort_values(
                'mean_cv', ascending=False).reset_index(drop=True)
            colors = ['C2' if c == sub.iloc[0]['cls']
                      else 'C7' if 'plus-plus' in c
                      else 'C0' for c in sub['cls']]
            y = np.arange(len(sub))
            ax.barh(y, sub['mean_cv'].values, xerr=sub['sem_cv'].values,
                    color=colors, alpha=0.8, ecolor='k', capsize=2)
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(max(row['mean_cv'] + row['sem_cv'] + 0.001, 0.005), i,
                        f' {row["short"]}  (n={row["n"]})',
                        va='center', fontsize=7)
            ax.invert_yaxis()
            ax.set_yticks([])
            ax.set_xlabel('mean ± SEM CV-R² across subjects')
            ax.set_title(f'{roi}  (winning: {sub.iloc[0]["short"]})')
            ax.grid(alpha=0.2, axis='x')
            xmax = (persub_summary['mean_cv'] +
                    persub_summary['sem_cv'].fillna(0)).max()
            ax.set_xlim(min(0, sub['mean_cv'].min() - 0.005),
                        xmax * 1.45)
        fig.suptitle(
            f'CV factorial — mean ± SEM of per-subject median CV-R²  '
            f'(n_fits={len(df)}, partial)',
            fontsize=13, weight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # Page 1c: per-ROI winner vs null PAIRED test.
        rois = [r for r in ROI_ORDER if r in df['roi'].unique()]
        null_cls = 'sus-zero-zero_dyn-zero-zero'
        fig, ax = plt.subplots(figsize=(13, 6))
        results = []
        for i, roi in enumerate(rois):
            sub = persub_summary[persub_summary['roi'] == roi].sort_values(
                'mean_cv', ascending=False)
            winner_cls = sub.iloc[0]['cls']
            # Per-subject paired data
            wide = df[df['roi'] == roi].pivot_table(
                index='subject', columns='cls',
                values='cv_r2_median')
            if winner_cls not in wide.columns or null_cls not in wide.columns:
                continue
            paired = wide[[winner_cls, null_cls]].dropna()
            delta = paired[winner_cls] - paired[null_cls]
            n = len(delta)
            try:
                _, p_w = stats.wilcoxon(delta, alternative='greater',
                                         zero_method='pratt')
            except ValueError:
                p_w = np.nan
            try:
                _, p_t = stats.ttest_rel(paired[winner_cls],
                                          paired[null_cls])
                p_t = p_t / 2 if (paired[winner_cls].mean()
                                   > paired[null_cls].mean()) else 1 - p_t / 2
            except Exception:
                p_t = np.nan
            d_med = float(np.median(delta))
            d_mean = float(np.mean(delta))
            d_sem = float(np.std(delta, ddof=1) / np.sqrt(n))
            # Plot: each subject's Δ as a point, overlay mean ± SEM
            xs = np.full(len(delta), i)
            ax.scatter(xs, delta.values, color='C0', alpha=0.5, s=22,
                       edgecolor='k', linewidth=0.3)
            ax.errorbar([i], [d_mean], yerr=[d_sem], fmt='_',
                        color='k', markersize=20, lw=2.5, capsize=6)
            sig = ('***' if p_w < 0.001 else '**' if p_w < 0.01
                    else '*' if p_w < 0.05 else 'n.s.')
            c = 'C2' if (np.isfinite(p_w) and p_w < 0.05) else '0.5'
            label = (f'{short_label(winner_cls)}\n'
                     f'n={n}, Δmed={d_med:+.3f}\n'
                     f'p_w={p_w:.4f}\n{sig}')
            ax.text(i, max(delta.max(), 0.05) * 1.05, label,
                    ha='center', va='bottom', fontsize=8,
                    color=c, weight='bold' if (np.isfinite(p_w) and p_w < 0.05) else 'normal')
            results.append({'roi': roi, 'winner_cls': winner_cls,
                            'n': n, 'd_med': d_med, 'd_mean': d_mean,
                            'p_wilcoxon': p_w, 'p_ttest_1sided': p_t})
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_ylabel('Δ CV-R²  (winner − null, paired per subject)')
        ax.set_title('Per-ROI WINNER vs null model — paired Wilcoxon (one-sided > 0)')
        ax.grid(alpha=0.2, axis='y')
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)
        # also stash for printing
        for r in results:
            print(f"  {r['roi']:5} winner={short_label(r['winner_cls']):15} "
                  f"n={r['n']:2} Δmean={r['d_mean']:+.4f} p_w={r['p_wilcoxon']:.4f}")

        # Page 2: heatmap (class × ROI) of median CV-R²
        wide = summary.pivot(index='cls', columns='roi',
                             values='median_cv')
        wide = wide.reindex(columns=ROI_ORDER)
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(wide.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(wide.columns)))
        ax.set_xticklabels(wide.columns)
        ax.set_yticks(range(len(wide.index)))
        ax.set_yticklabels([short_label(c) for c in wide.index],
                           fontsize=8)
        for i in range(wide.shape[0]):
            for j in range(wide.shape[1]):
                v = wide.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.3f}', ha='center',
                            va='center', fontsize=7,
                            color='white' if v < 0.05 else 'black')
        plt.colorbar(im, label='median CV-R²')
        ax.set_title(
            f'CV-R² heatmap: class × ROI  '
            f'(median across subjects, n_fits={len(df)})')
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
