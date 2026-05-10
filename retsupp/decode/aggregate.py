"""Aggregate per-(subject, ROI) decoded-drive TSVs and draw the figure.

Inputs:
    derivatives/decode/sub-XX/sub-XX_roi-{ROI}_decoded_drive.tsv
    (one per (subject, ROI), produced by run_decode.py)

Outputs:
    derivatives/decode/decoded_drive.tsv  (concatenated long-format)
    notes/figures/decoded_drive_HP_vs_LP.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']
ROLE_ORDER = ['HP', 'orth', 'opposite']
ROLE_COLOR = {'HP': '#d62728', 'orth': '#9467bd', 'opposite': '#1f77b4'}


def collect_tsvs(bids_folder: Path, subjects, rois) -> pd.DataFrame:
    """Read all per-(subject, ROI) TSVs and return concatenated long frame."""
    base = bids_folder / 'derivatives' / 'decode'
    parts = []
    missing = []
    for s in subjects:
        for roi in rois:
            fn = (base / f'sub-{s:02d}'
                  / f'sub-{s:02d}_roi-{roi}_decoded_drive.tsv')
            if not fn.exists():
                missing.append((s, roi))
                continue
            df = pd.read_csv(fn, sep='\t')
            parts.append(df)
    if missing:
        print(f'  missing {len(missing)} (subject, roi) cells; '
              f'sample: {missing[:5]}', flush=True)
    if not parts:
        raise RuntimeError('No TSVs found.')
    return pd.concat(parts, axis=0, ignore_index=True)


def aggregate_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(subject, roi, session, run, ring_location), mean over the run.
    Tracks hp_role for each ring within the run.
    """
    g = (df.groupby(['subject', 'roi', 'session', 'run',
                     'ring_location', 'hp_role', 'hp_location'],
                    as_index=False)['decoded'].mean())
    return g


def aggregate_per_subject_role(per_run: pd.DataFrame) -> pd.DataFrame:
    """Per-(subject, roi, hp_role): mean decoded drive averaged over the
    relevant rings × runs."""
    return (per_run.groupby(['subject', 'roi', 'hp_role'],
                             as_index=False)['decoded'].mean())


def make_figure(per_subj: pd.DataFrame, out_pdf: Path,
                rois=ROI_ORDER) -> None:
    """One panel per ROI: mean decoded drive HP vs orth vs opposite,
    one point per subject, group mean ± SEM overlaid, paired t-tests."""
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
        # Pivot for paired tests + per-subject lines.
        pv = d.pivot_table(index='subject', columns='hp_role',
                            values='decoded')

        means_sems = {}
        for ci, role in enumerate(ROLE_ORDER):
            if role not in pv.columns:
                continue
            vals = pv[role].dropna().values
            xc = ci
            jit = rng.normal(0, 0.07, len(vals))
            ax.scatter([xc] * len(vals) + jit, vals,
                       color=ROLE_COLOR[role], alpha=0.5, s=28,
                       edgecolors='white', linewidths=0.4, zorder=2)
            m = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))
                        if len(vals) > 1 else 0.0)
            means_sems[role] = (m, sem, len(vals))
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
        ax.set_title(roi, fontsize=14, weight='bold')
        ax.axhline(0, color='k', lw=0.6, alpha=0.4)
        if i % ncol == 0:
            ax.set_ylabel('mean decoded drive', fontsize=12)

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
            ax.text((xa + xb) / 2, level_y + bh, sig,
                    ha='center', va='bottom', fontsize=11)
            level_y += 0.07 * span
        ax.set_ylim(ymin, level_y + 0.05 * span)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    n_subs = per_subj['subject'].nunique()
    fig.suptitle(f'Decoded stimulus drive at HP vs LP ring positions '
                  f'(n={n_subs} subjects)',
                  fontsize=15, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f'Wrote: {out_pdf}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subjects', type=int, nargs='+',
                   default=[s for s in range(1, 31) if s not in (6, 8)])
    p.add_argument('--rois', nargs='+', default=ROI_ORDER)
    p.add_argument('--out-tsv', type=Path,
                   default=None,
                   help='Default: <bids>/derivatives/decode/decoded_drive.tsv')
    p.add_argument('--out-fig', type=Path,
                   default=None,
                   help='Default: notes/figures/decoded_drive_HP_vs_LP.pdf')
    p.add_argument('--repo-root', type=Path,
                   default=Path(__file__).resolve().parents[2])
    args = p.parse_args()

    bids = Path(args.bids_folder)
    out_tsv = args.out_tsv or bids / 'derivatives' / 'decode' / 'decoded_drive.tsv'
    out_fig = args.out_fig or args.repo_root / 'notes' / 'figures' / 'decoded_drive_HP_vs_LP.pdf'

    print(f'Subjects: {args.subjects}', flush=True)
    print(f'ROIs: {args.rois}', flush=True)
    print(f'Out TSV: {out_tsv}', flush=True)
    print(f'Out fig: {out_fig}', flush=True)

    df = collect_tsvs(bids, args.subjects, args.rois)
    print(f'Collected {len(df)} rows.', flush=True)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'Wrote: {out_tsv}', flush=True)

    per_run = aggregate_per_run(df)
    per_subj = aggregate_per_subject_role(per_run)
    make_figure(per_subj, out_fig, rois=args.rois)

    # Tiny console summary.
    print('\nGroup-mean drive per (ROI, hp_role):')
    print(per_subj.groupby(['roi', 'hp_role'])['decoded']
                  .agg(['mean', 'std', 'count']).round(4))


if __name__ == '__main__':
    main()
