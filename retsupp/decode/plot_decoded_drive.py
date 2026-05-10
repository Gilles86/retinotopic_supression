"""Plot the HP-vs-LP decoded drive figure from a single TSV.

Reads the long-format ``decoded_drive.tsv`` produced by
:mod:`retsupp.decode.decode_drive_fast`, aggregates per (subject, ROI,
hp_role), and draws one panel per ROI with paired t-tests for HP vs
opposite and HP vs orth.

Output: ``notes/figures/decoded_drive_HP_vs_LP.pdf``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']
ROLE_ORDER = ['HP', 'orth', 'opposite']
ROLE_COLOR = {'HP': '#d62728', 'orth': '#9467bd', 'opposite': '#1f77b4'}


def aggregate_per_subject_role(df: pd.DataFrame,
                                drive_col: str = 'drive_weighted') -> pd.DataFrame:
    """Mean drive per (subject, roi, hp_role), averaging over runs and rings."""
    return (df.groupby(['subject', 'roi', 'rel'])
              [drive_col]
              .mean().rename('drive').reset_index()
              .rename(columns={'rel': 'hp_role'}))


def make_figure(per_subj: pd.DataFrame, out_pdf: Path,
                rois=ROI_ORDER, drive_col: str = 'drive') -> None:
    from scipy import stats as scistats

    rois_pres = [r for r in rois if r in per_subj['roi'].unique()]
    n = len(rois_pres)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.0 * nrow),
                              sharey=False, squeeze=False)
    axes = axes.flatten()
    rng = np.random.default_rng(0)

    for i, roi in enumerate(rois_pres):
        ax = axes[i]
        d = per_subj[per_subj['roi'] == roi]
        pv = d.pivot_table(index='subject', columns='hp_role',
                            values=drive_col)

        for ci, role in enumerate(ROLE_ORDER):
            if role not in pv.columns:
                continue
            vals = pv[role].dropna().values
            if len(vals) == 0:
                continue
            xc = ci
            jit = rng.normal(0, 0.07, len(vals))
            ax.scatter([xc] * len(vals) + jit, vals,
                       color=ROLE_COLOR[role], alpha=0.5, s=28,
                       edgecolors='white', linewidths=0.4, zorder=2)
            m = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))
                        if len(vals) > 1 else 0.0)
            ax.errorbar([xc], [m], yerr=[sem], fmt='o',
                        color='k', mec='k', mfc='k', markersize=8,
                        capsize=5, capthick=1.8, elinewidth=1.8, zorder=3)

        # Per-subject paired lines (HP-orth-opp), faint.
        if all(r in pv.columns for r in ROLE_ORDER):
            for _, row in pv.dropna(subset=ROLE_ORDER).iterrows():
                ys = [row[r] for r in ROLE_ORDER]
                ax.plot([0, 1, 2], ys, color='0.7', alpha=0.25, lw=0.6,
                        zorder=1)

        ax.set_xticks(range(len(ROLE_ORDER)))
        ax.set_xticklabels(ROLE_ORDER, fontsize=11)
        ax.set_title(f'{roi}  (n={pv.shape[0]})', fontsize=13, weight='bold')
        ax.axhline(0, color='k', lw=0.6, alpha=0.4)
        if i % ncol == 0:
            ax.set_ylabel('decoded drive (a.u.)', fontsize=11)

        # Paired stats: HP vs opposite, HP vs orth.
        contrasts = [('HP', 'opposite'), ('HP', 'orth')]
        ymax = ax.get_ylim()[1]
        ymin = ax.get_ylim()[0]
        span = ymax - ymin
        level_y = ymax + 0.04 * span
        for a, b in contrasts:
            if a not in pv.columns or b not in pv.columns:
                continue
            paired = pv[[a, b]].dropna()
            if len(paired) < 5:
                continue
            t, p = scistats.ttest_rel(paired[a], paired[b])
            mean_diff = (paired[a] - paired[b]).mean()
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'
            xa = ROLE_ORDER.index(a)
            xb = ROLE_ORDER.index(b)
            bh = 0.025 * span
            ax.plot([xa, xa, xb, xb],
                    [level_y, level_y + bh, level_y + bh, level_y],
                    lw=1.0, color='k', clip_on=False)
            ax.text((xa + xb) / 2, level_y + bh,
                    f'{sig}',
                    ha='center', va='bottom', fontsize=10)
            print(f'  {roi}  {a} vs {b}: t={t:.2f}, p={p:.3g}, '
                  f'mean_diff={mean_diff:+.4f}, n={len(paired)}')
            level_y += 0.07 * span
        ax.set_ylim(ymin, level_y + 0.05 * span)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    n_subs = per_subj['subject'].nunique()
    fig.suptitle(f'Decoded stimulus drive at HP vs LP ring positions '
                  f'(n={n_subs} subjects, time-weighted by HRF-convolved indicator)',
                  fontsize=14, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f'Wrote: {out_pdf}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tsv', type=Path,
                   default=Path('/data/ds-retsupp/derivatives/decode/decoded_drive.tsv'))
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/decoded_drive_HP_vs_LP.pdf'))
    p.add_argument('--rois', nargs='+', default=ROI_ORDER)
    p.add_argument('--drive-col', default='drive_weighted',
                   choices=['drive_weighted', 'drive_unweighted'])
    p.add_argument('--repo-root', type=Path,
                   default=Path(__file__).resolve().parents[2])
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = args.repo_root / out

    df = pd.read_csv(args.tsv, sep='\t')
    print(f'Loaded {len(df)} rows from {args.tsv}')
    print(f'Subjects: {sorted(df["subject"].unique().tolist())}')
    print(f'ROIs:     {sorted(df["roi"].unique().tolist())}')

    # Drop rows where 'rel' is unknown (no HP for that run).
    df = df[df['rel'].isin(['HP', 'orth', 'opposite'])]

    per_subj = aggregate_per_subject_role(df, drive_col=args.drive_col)
    print('\nPer-condition group means:')
    print(per_subj.groupby(['roi', 'hp_role'])['drive']
                  .agg(['mean', 'std', 'count']).round(4))
    print()

    make_figure(per_subj, out, rois=args.rois)


if __name__ == '__main__':
    main()
