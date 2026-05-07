"""Per-ROI predicted-vs-observed projection-vs-distance from HP.

The most direct test of the joint AF + PRF model against the
model-free shift data: take every (subject × condition × voxel) row in
``predict_shifts_cluster.tsv``, compute its rotated distance from HP,
and plot

    observed projection on away-from-HP axis  vs  distance from HP

with the AF model's analytic prediction overlaid.

Per ROI we plot two curves:
  • OBSERVED — median projection across (subject × condition × voxel)
    rows in each distance bin, with the per-subject-median IQR as a
    shaded band.  This is the model-free signature.
  • PREDICTED — median predicted projection per bin (same binning,
    same data, but using the model's predicted shifts).  Smooth.

If the predicted curve tracks the observed median, the joint
BOLD-level AF + PRF fit is recovering the empirical voxelwise shift
pattern — exactly what we want to see.

Usage
-----
    python -m retsupp.visualize.plot_proj_distance_curves \\
        --tsv notes/predict_shifts_cluster.tsv \\
        --out notes/proj_distance_curves.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from retsupp.utils.data import distractor_locations


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']
APERTURE = 3.17
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def add_distance_from_hp(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Euclidean distance of base PRF center from HP location."""
    df = df.copy()
    ring = get_ring_positions()
    cond_to_idx = {c: i for i, c in enumerate(CONDITIONS)}
    hp_x = np.zeros(len(df), dtype=np.float32)
    hp_y = np.zeros(len(df), dtype=np.float32)
    for c in CONDITIONS:
        m = (df['condition'] == c).values
        if not m.any():
            continue
        hp_x[m] = ring[cond_to_idx[c], 0]
        hp_y[m] = ring[cond_to_idx[c], 1]
    df['dist_HP'] = np.sqrt(
        (df['base_x'].values - hp_x) ** 2
        + (df['base_y'].values - hp_y) ** 2
    )
    return df


def aperture_filter(df: pd.DataFrame, aperture=APERTURE) -> pd.DataFrame:
    """Drop rows whose base PRF is outside the bar aperture."""
    inside = np.sqrt(df['base_x'].values ** 2
                      + df['base_y'].values ** 2) <= aperture
    return df.loc[inside].copy()


def per_subject_bin_mean(df_roi: pd.DataFrame, bin_edges: np.ndarray):
    """Per-(subject, distance-bin) mean of proj_obs and proj_pred.

    Returns a long-format DataFrame with columns
    `subject, distance, proj_obs, proj_pred, n`.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df = df_roi.copy()
    df['d_bin'] = pd.cut(df['dist_HP'], bin_edges, labels=centers,
                          include_lowest=True)
    df = df.dropna(subset=['d_bin', 'proj_obs', 'proj_pred'])
    df['d_bin'] = df['d_bin'].astype(float)
    return (df.groupby(['subject', 'd_bin'], observed=True)
              [['proj_obs', 'proj_pred']]
              .mean()
              .reset_index()
              .rename(columns={'d_bin': 'distance'}))


def plot_one_roi(ax, agg: pd.DataFrame, roi: str, n_subj: int):
    """Two curves on one ax: observed (blue) + predicted (red)."""
    summary = (agg.groupby('distance', observed=True)
                .agg(obs_med=('proj_obs', 'median'),
                     obs_q1=('proj_obs', lambda v: np.quantile(v, 0.25)),
                     obs_q3=('proj_obs', lambda v: np.quantile(v, 0.75)),
                     obs_sem=('proj_obs',
                               lambda v: v.std(ddof=1) / np.sqrt(max(len(v), 1))),
                     pred_med=('proj_pred', 'median'),
                     pred_q1=('proj_pred', lambda v: np.quantile(v, 0.25)),
                     pred_q3=('proj_pred', lambda v: np.quantile(v, 0.75)),
                     n=('proj_obs', 'count'))
                .reset_index()
                .sort_values('distance'))

    # Observed: median per bin, +/- 1 SEM (across subjects).
    ax.fill_between(
        summary['distance'],
        summary['obs_med'] - summary['obs_sem'],
        summary['obs_med'] + summary['obs_sem'],
        color='C0', alpha=0.20, label='observed ± SEM',
    )
    ax.plot(summary['distance'], summary['obs_med'],
             color='C0', lw=2.0, marker='o', markersize=5,
             label='observed (median)')

    # Predicted: median per bin, smooth-looking curve.
    ax.plot(summary['distance'], summary['pred_med'],
             color='C3', lw=2.5, ls='--', marker='s', markersize=5,
             label='AF model prediction (median)')
    ax.fill_between(
        summary['distance'],
        summary['pred_q1'], summary['pred_q3'],
        color='C3', alpha=0.10,
    )
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('distance of base PRF from HP (deg)')
    ax.set_ylabel('projection on away-from-HP axis (deg)\n(+ = away, − = toward)')
    ax.set_title(f'{roi}   |   n_subjects = {n_subj}', fontsize=11)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)


def plot_all_rois_grid(rois_data, out_pdf: Path, args):
    """One page per ROI plus an overview page with all ROIs in a grid."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # ---- Cover.
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.95,
                 'Predicted vs observed projection-vs-distance from HP',
                 ha='center', va='top', fontsize=16, weight='bold',
                 transform=ax.transAxes)
        body = (
            f'TSV  : {args.tsv}\n'
            f'Bins : {args.bin_step}° wide, {args.bin_min}° to {args.bin_max}°\n'
            f'\n'
            f'Per ROI:\n'
            f'  • For every (subject × condition × voxel) row inside the\n'
            f'    bar aperture (radius {APERTURE}°), compute distance from HP.\n'
            f'  • Per-subject mean of proj_obs and proj_pred per bin.\n'
            f'  • Plot the across-subject MEDIAN for both, with ±1 SEM band\n'
            f'    on the observed curve.\n'
            f'\n'
            f'If the predicted curve (red dashed) tracks the observed\n'
            f'curve (blue), the joint AF+PRF fit recovers the empirical\n'
            f'voxelwise shift pattern.\n'
        )
        ax.text(0.05, 0.85, body, ha='left', va='top',
                 family='monospace', fontsize=10, transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        # ---- Per-ROI pages (full width).
        for roi, (agg, n_subj) in rois_data.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_one_roi(ax, agg, roi, n_subj)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ---- Overview grid: 4×2 with all ROIs.
        n_rois = len(rois_data)
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
        for ax, (roi, (agg, n_subj)) in zip(axes.ravel(), rois_data.items()):
            plot_one_roi(ax, agg, roi, n_subj)
            ax.legend(fontsize=7, loc='best')
            ax.set_xlabel(''); ax.set_ylabel('')
        fig.suptitle('Per-ROI overview: model prediction (red) vs '
                     'observed (blue) projection-vs-distance',
                     fontsize=13)
        fig.text(0.5, 0.02, 'distance of base PRF from HP (deg)',
                  ha='center', fontsize=10)
        fig.text(0.005, 0.5, 'projection on away-from-HP axis (deg)',
                  ha='center', va='center', rotation='vertical', fontsize=10)
        fig.tight_layout(rect=[0.02, 0.04, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

    print(f'Wrote {out_pdf}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tsv', type=Path,
                        default=Path('notes/predict_shifts_cluster.tsv'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/proj_distance_curves.pdf'))
    parser.add_argument('--bin-step', type=float, default=0.5)
    parser.add_argument('--bin-min', type=float, default=0.0)
    parser.add_argument('--bin-max', type=float, default=8.0)
    parser.add_argument('--rois', nargs='+', default=None)
    parser.add_argument('--no-aperture', action='store_true',
                        help='Skip the inside-aperture filter.')
    args = parser.parse_args()

    df = pd.read_csv(args.tsv, sep='\t')
    print(f'Loaded {len(df):,} rows from {args.tsv}')
    df = df.dropna(subset=['base_x', 'base_y', 'proj_obs', 'proj_pred',
                            'condition'])
    print(f'After dropna: {len(df):,} rows')

    if not args.no_aperture:
        df = aperture_filter(df, APERTURE)
        print(f'After aperture filter ({APERTURE}°): {len(df):,} rows')

    df = add_distance_from_hp(df)
    bin_edges = np.arange(args.bin_min,
                            args.bin_max + args.bin_step / 2,
                            args.bin_step)

    rois = (args.rois if args.rois
            else [r for r in ROI_ORDER if r in df['roi'].unique()] +
                 sorted(set(df['roi']) - set(ROI_ORDER)))
    rois_data = {}
    for roi in rois:
        df_roi = df[df['roi'] == roi]
        if len(df_roi) < 100:
            print(f'  {roi}: only {len(df_roi)} rows, skipping.')
            continue
        agg = per_subject_bin_mean(df_roi, bin_edges)
        rois_data[roi] = (agg, df_roi['subject'].nunique())
        print(f'  {roi}: {len(df_roi):,} rows, {len(agg):,} (subject, bin) cells.')

    plot_all_rois_grid(rois_data, args.out, args)


if __name__ == '__main__':
    main()
