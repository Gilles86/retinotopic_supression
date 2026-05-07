"""Population-level predicted vs observed shift vector fields per ROI.

Reads the long-format TSV from `predict_shifts_from_af.py` (one row per
subject × ROI × HP-condition × voxel with base, predicted, observed
positions) and produces a per-ROI 3-panel page:

  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ PREDICTED    │ │ OBSERVED     │ │ OBS − PRED   │
  │ shift field  │ │ shift field  │ │ residual     │
  └──────────────┘ └──────────────┘ └──────────────┘

Each cell shows a 2D vector field across the rotated visual field where
HP is always at canonical (0, +4°). All voxels (across subjects and
conditions) are rotated into this frame, binned into a 2D grid, and the
shift vectors averaged per bin.

This collapses the 4-condition rotational symmetry, accumulating
4× more data per bin than any single-condition view, and shows
population-level patterns at a glance.

Usage
-----
    python -m retsupp.visualize.plot_rotated_shift_fields \\
        --tsv notes/predict_shifts_cluster.tsv \\
        --out notes/rotated_shift_fields.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

from retsupp.utils.data import distractor_locations


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']
APERTURE = 3.17  # bar aperture radius (deg)
RING_R = 4.0
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def rotation_to_canonical(condition: str):
    """Return 2x2 rotation that maps the condition's HP location to (0, +4°).

    HP_canonical = R · HP_actual, with HP_canonical = (0, +4).
    """
    ring_idx = CONDITIONS.index(condition)
    ring = get_ring_positions()
    hp_actual = ring[ring_idx]
    target = np.array([0.0, RING_R])
    # Rotation angle: from hp_actual direction to target direction.
    a_actual = np.arctan2(hp_actual[1], hp_actual[0])
    a_target = np.arctan2(target[1], target[0])
    theta = a_target - a_actual
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def rotate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add base_xy_rot, pred_xy_rot, obs_xy_rot columns (rotated by per-row condition)."""
    df = df.copy()
    # Pre-compute one rotation matrix per condition.
    R = {c: rotation_to_canonical(c) for c in CONDITIONS}
    bx = df['base_x'].values; by = df['base_y'].values
    px = df['pred_x'].values; py = df['pred_y'].values
    ox = df['obs_x'].values; oy = df['obs_y'].values
    rb = np.empty_like(bx); rby = np.empty_like(by)
    rp = np.empty_like(px); rpy = np.empty_like(py)
    ro = np.empty_like(ox); roy = np.empty_like(oy)
    for c in CONDITIONS:
        m = (df['condition'] == c).values
        if not m.any():
            continue
        Rc = R[c]
        rb[m]  = Rc[0, 0] * bx[m]  + Rc[0, 1] * by[m]
        rby[m] = Rc[1, 0] * bx[m]  + Rc[1, 1] * by[m]
        rp[m]  = Rc[0, 0] * px[m]  + Rc[0, 1] * py[m]
        rpy[m] = Rc[1, 0] * px[m]  + Rc[1, 1] * py[m]
        ro[m]  = Rc[0, 0] * ox[m]  + Rc[0, 1] * oy[m]
        roy[m] = Rc[1, 0] * ox[m]  + Rc[1, 1] * oy[m]
    df['base_x_rot']  = rb;  df['base_y_rot']  = rby
    df['pred_x_rot']  = rp;  df['pred_y_rot']  = rpy
    df['obs_x_rot']   = ro;  df['obs_y_rot']   = roy
    df['dx_pred_rot'] = rp - rb;  df['dy_pred_rot'] = rpy - rby
    df['dx_obs_rot']  = ro - rb;  df['dy_obs_rot']  = roy - rby
    return df


def bin_shifts(df_roi: pd.DataFrame,
                grid_extent: float = 4.0,
                grid_n: int = 13,
                min_n_per_bin: int = 5):
    """Bin voxels into a 2D grid; return mean shift vectors and counts.

    Each row in df_roi is a single (subject × condition × voxel)
    observation in the rotated frame.
    """
    edges = np.linspace(-grid_extent, grid_extent, grid_n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bx = df_roi['base_x_rot'].values
    by = df_roi['base_y_rot'].values

    ix = np.digitize(bx, edges) - 1
    iy = np.digitize(by, edges) - 1
    valid = (ix >= 0) & (ix < grid_n) & (iy >= 0) & (iy < grid_n)

    pred = np.zeros((grid_n, grid_n, 2), dtype=np.float64)
    obs  = np.zeros((grid_n, grid_n, 2), dtype=np.float64)
    cnt  = np.zeros((grid_n, grid_n), dtype=np.int32)

    sx_p = df_roi['dx_pred_rot'].values
    sy_p = df_roi['dy_pred_rot'].values
    sx_o = df_roi['dx_obs_rot'].values
    sy_o = df_roi['dy_obs_rot'].values

    for k in np.where(valid)[0]:
        i, j = ix[k], iy[k]
        cnt[i, j] += 1
        pred[i, j, 0] += sx_p[k]; pred[i, j, 1] += sy_p[k]
        obs[i, j, 0]  += sx_o[k]; obs[i, j, 1]  += sy_o[k]

    mask = cnt >= min_n_per_bin
    pred[mask] = pred[mask] / cnt[mask, None]
    obs[mask]  = obs[mask]  / cnt[mask, None]
    pred[~mask] = np.nan
    obs[~mask] = np.nan
    return centers, pred, obs, cnt, mask


def plot_one_field(ax, centers, vec, mask, scale, color, title,
                    aperture=APERTURE, ring=None, hp=(0.0, RING_R)):
    n = len(centers)
    cx, cy = np.meshgrid(centers, centers, indexing='ij')
    dx = vec[:, :, 0]
    dy = vec[:, :, 1]
    # Aperture + ring positions.
    ax.add_patch(Circle((0, 0), aperture, fill=False, ec='0.5',
                          ls='--', lw=0.6))
    if ring is not None:
        for r in ring:
            ax.plot(r[0], r[1], 'o', color='gray', mec='k',
                    markersize=6, alpha=0.7)
    ax.plot(hp[0], hp[1], '*', color='C3', mec='k', markersize=18,
            zorder=5)
    # Quiver.
    Q = ax.quiver(cx[mask], cy[mask], dx[mask], dy[mask],
                  angles='xy', scale_units='xy',
                  scale=1.0 / scale, color=color,
                  width=0.005, alpha=0.9)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def cover_page(pdf, df, args):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Rotated-frame shift fields per ROI',
            ha='center', va='top', fontsize=18, weight='bold',
            transform=ax.transAxes)
    body = (
        f'Source TSV : {args.tsv}\n'
        f'Total rows : {len(df):,}\n'
        f'Subjects   : {df["subject"].nunique()}\n'
        f'ROIs       : {df["roi"].nunique()}\n'
        f'\n'
        f'How to read each per-ROI page:\n'
        f'  • Every (subject × condition × voxel) row is rotated so the\n'
        f'    HP location is moved to canonical (0, +4°). The 4-condition\n'
        f'    rotational symmetry is collapsed — population pattern with\n'
        f'    4× more data per bin.\n'
        f'  • Voxels are binned into a {args.grid_n}×{args.grid_n} grid covering\n'
        f'    ±{args.grid_extent}°. Bins with < {args.min_n_per_bin} observations are dropped.\n'
        f'  • Three side-by-side panels:\n'
        f'      PREDICTED  – mean predicted shift vector per bin\n'
        f'                   (computed from the joint AF+PRF fit).\n'
        f'      OBSERVED   – mean observed shift vector per bin\n'
        f'                   (from the independent prf_conditionfit/model4 fits).\n'
        f'      OBS − PRED – residual after subtracting the model.\n'
        f'  • Arrows scaled ×{args.scale} for visibility.\n'
        f'  • Red star = HP location;  gray dots = LP locations;\n'
        f'    dashed circle = bar aperture (~{APERTURE}°).\n'
    )
    ax.text(0.04, 0.86, body, ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tsv', type=Path,
                        default=Path('notes/predict_shifts_cluster.tsv'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/rotated_shift_fields.pdf'))
    parser.add_argument('--grid-extent', type=float, default=4.0,
                        help='Half-width of the binning grid (deg).')
    parser.add_argument('--grid-n', type=int, default=13,
                        help='Number of bins per side.')
    parser.add_argument('--min-n-per-bin', type=int, default=10,
                        help='Drop bins with fewer than this many '
                             'observations.')
    parser.add_argument('--scale', type=float, default=8.0,
                        help='Arrow length scaling for visibility.')
    parser.add_argument('--rois', nargs='+', default=None,
                        help='Subset of ROIs (default: all in TSV).')
    args = parser.parse_args()

    df = pd.read_csv(args.tsv, sep='\t')
    print(f'Loaded {len(df):,} rows from {args.tsv}')

    needed = ['subject', 'roi', 'condition',
              'base_x', 'base_y', 'pred_x', 'pred_y',
              'obs_x', 'obs_y']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f'TSV is missing columns: {missing}')

    df = df.dropna(subset=needed)
    print(f'After dropna: {len(df):,} rows')

    df = rotate_table(df)
    print('Rotated all rows to canonical (HP at (0, +4°)) frame.')

    rois = (args.rois if args.rois
            else [r for r in ROI_ORDER if r in df['roi'].unique()] +
                 sorted(set(df['roi']) - set(ROI_ORDER)))
    ring = get_ring_positions()
    # Rotated ring positions are not needed: by construction the canonical
    # frame already has HP at (0,+4), and the other 3 LP positions are at
    # the other 3 corners — same in every condition.
    canon_ring = np.array([
        [0.0, +RING_R],                       # HP at top
        [-RING_R * np.sqrt(2)/2, RING_R * np.sqrt(2)/2 - 0],  # left of HP
        [0.0, -RING_R],                       # opposite HP
        [+RING_R * np.sqrt(2)/2, RING_R * np.sqrt(2)/2 - 0],  # right of HP
    ])
    # Actually the ring is at 45° intervals so canonical positions are:
    canon_ring = np.array([
        [0.0,           +RING_R],
        [-RING_R, 0.0],
        [0.0,           -RING_R],
        [+RING_R, 0.0],
    ], dtype=np.float32)
    # Note: original positions are 4° at 45°/135°/etc. After rotation putting
    # one HP→(0,+4), the other three are at 4° on a SQUARE (90° apart).

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        cover_page(pdf, df, args)

        for roi in rois:
            df_roi = df[df['roi'] == roi]
            if len(df_roi) < 100:
                print(f'  {roi}: only {len(df_roi)} rows, skipping.')
                continue
            centers, pred, obs, cnt, mask = bin_shifts(
                df_roi,
                grid_extent=args.grid_extent,
                grid_n=args.grid_n,
                min_n_per_bin=args.min_n_per_bin,
            )
            resid = np.full_like(pred, np.nan)
            resid[mask] = obs[mask] - pred[mask]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
            plot_one_field(axes[0], centers, pred, mask,
                            scale=args.scale, color='C0',
                            title='PREDICTED  (joint AF+PRF fit)',
                            ring=canon_ring)
            plot_one_field(axes[1], centers, obs, mask,
                            scale=args.scale, color='C2',
                            title='OBSERVED  (prf_conditionfit/model4)',
                            ring=canon_ring)
            plot_one_field(axes[2], centers, resid, mask,
                            scale=args.scale, color='C3',
                            title='OBS − PRED   (residual)',
                            ring=canon_ring)
            n_obs = int(cnt[mask].sum())
            n_subj = df_roi['subject'].nunique()
            fig.suptitle(
                f'{roi}   |   {n_subj} subjects, {n_obs:,} '
                f'voxel-condition obs in valid bins  |  '
                f'arrows ×{args.scale}',
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig); plt.close(fig)

            # Optional: a small fourth page with the bin-count heatmap.
            fig, ax = plt.subplots(figsize=(6, 5))
            cx, cy = np.meshgrid(centers, centers, indexing='ij')
            im = ax.pcolormesh(centers, centers, cnt.T,
                                cmap='viridis', shading='auto')
            ax.add_patch(Circle((0, 0), APERTURE, fill=False,
                                  ec='w', ls='--', lw=0.8))
            ax.plot(0, RING_R, '*', color='C3', mec='w', markersize=14)
            ax.set_aspect('equal')
            ax.set_title(f'{roi}  bin counts', fontsize=10)
            ax.set_xlabel('x_rot (°)'); ax.set_ylabel('y_rot (°)')
            plt.colorbar(im, ax=ax, label='n observations')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
