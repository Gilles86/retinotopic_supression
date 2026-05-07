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
                min_n_per_bin: int = 5,
                statistic: str = 'median'):
    """Bin voxels into a 2D grid; return per-bin shift summaries.

    Returns a dict of (grid_n, grid_n) arrays:
      pred_dx, pred_dy, pred_proj, pred_norm,
      obs_dx,  obs_dy,  obs_proj,  obs_norm,
      cnt, mask, centers.
    All scalar fields are MEDIAN (default) across the bin's
    observations — robust to a handful of outlier voxels with crazy
    shifts that would otherwise dominate the mean. Set statistic='mean'
    to recover the previous behavior.

    `proj` is the projection of the shift vector onto the away-from-HP
    unit vector (= +y in the canonical rotated frame, since HP is
    canonical (0, +4°)). Positive = shift AWAY from HP; negative =
    toward HP.
    """
    edges = np.linspace(-grid_extent, grid_extent, grid_n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bx = df_roi['base_x_rot'].values
    by = df_roi['base_y_rot'].values
    ix = np.digitize(bx, edges) - 1
    iy = np.digitize(by, edges) - 1
    valid = (ix >= 0) & (ix < grid_n) & (iy >= 0) & (iy < grid_n)

    sx_p = df_roi['dx_pred_rot'].values
    sy_p = df_roi['dy_pred_rot'].values
    sx_o = df_roi['dx_obs_rot'].values
    sy_o = df_roi['dy_obs_rot'].values
    # Away-from-HP unit vector in canonical frame: HP at (0, +4) → away-from-HP is -y.
    proj_p = -sy_p
    proj_o = -sy_o
    norm_p = np.sqrt(sx_p ** 2 + sy_p ** 2)
    norm_o = np.sqrt(sx_o ** 2 + sy_o ** 2)

    cnt = np.zeros((grid_n, grid_n), dtype=np.int32)
    out = {k: np.full((grid_n, grid_n), np.nan)
           for k in ['pred_dx', 'pred_dy', 'pred_proj', 'pred_norm',
                       'obs_dx',  'obs_dy',  'obs_proj',  'obs_norm']}

    # First pass: per-bin counts.
    for k in np.where(valid)[0]:
        cnt[ix[k], iy[k]] += 1

    # Second pass: per-bin statistics (median or mean).
    agg_func = np.median if statistic == 'median' else np.mean
    for i in range(grid_n):
        for j in range(grid_n):
            if cnt[i, j] < min_n_per_bin:
                continue
            sel = valid & (ix == i) & (iy == j)
            out['pred_dx'][i, j]   = agg_func(sx_p[sel])
            out['pred_dy'][i, j]   = agg_func(sy_p[sel])
            out['pred_proj'][i, j] = agg_func(proj_p[sel])
            out['pred_norm'][i, j] = agg_func(norm_p[sel])
            out['obs_dx'][i, j]    = agg_func(sx_o[sel])
            out['obs_dy'][i, j]    = agg_func(sy_o[sel])
            out['obs_proj'][i, j]  = agg_func(proj_o[sel])
            out['obs_norm'][i, j]  = agg_func(norm_o[sel])
    mask = cnt >= min_n_per_bin
    out.update({'cnt': cnt, 'mask': mask, 'centers': centers})
    return out


def plot_scalar_field(ax, centers, scalar, title, vmax,
                        cmap='RdBu_r', symmetric=True,
                        aperture=APERTURE, ring=None, hp=(0.0, RING_R),
                        cbar_label=''):
    """Heatmap of a scalar field on the rotated visual-field grid."""
    n = len(centers)
    edge_step = (centers[1] - centers[0]) / 2.0
    extent = (centers[0] - edge_step, centers[-1] + edge_step,
              centers[0] - edge_step, centers[-1] + edge_step)
    if symmetric:
        vmin, vmax_ = -vmax, +vmax
    else:
        vmin, vmax_ = 0, vmax
    # imshow expects (rows, cols) where rows=y, cols=x. We stored shape
    # (i_x, j_y) — transpose so rows=y.
    im = ax.imshow(scalar.T, extent=extent, origin='lower',
                    cmap=cmap, vmin=vmin, vmax=vmax_,
                    interpolation='nearest')
    ax.add_patch(Circle((0, 0), aperture, fill=False, ec='0.4',
                          ls='--', lw=0.7))
    if ring is not None:
        for r in ring:
            ax.plot(r[0], r[1], 'o', color='0.3', mec='k',
                    markersize=6, alpha=0.85)
    ax.plot(hp[0], hp[1], '*', color='C3', mec='k', markersize=18,
            zorder=5)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)


def plot_quiver_field(ax, centers, dx, dy, mask, scale, color, title,
                        aperture=APERTURE, ring=None, hp=(0.0, RING_R),
                        bin_w=None):
    """Quiver vector-field with dynamically-scaled arrows."""
    cx, cy = np.meshgrid(centers, centers, indexing='ij')
    ax.add_patch(Circle((0, 0), aperture, fill=False, ec='0.5',
                          ls='--', lw=0.6))
    if ring is not None:
        for r in ring:
            ax.plot(r[0], r[1], 'o', color='gray', mec='k',
                    markersize=6, alpha=0.7)
    ax.plot(hp[0], hp[1], '*', color='C3', mec='k', markersize=18,
            zorder=5)
    ax.quiver(cx[mask], cy[mask], dx[mask], dy[mask],
              angles='xy', scale_units='xy',
              scale=1.0 / scale, color=color, width=0.006, alpha=0.9)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_aspect('equal')
    title_suffix = f'  ×{scale:.0f}' if scale > 1.5 else ''
    ax.set_title(title + title_suffix, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def auto_arrow_scale(*shift_arrays, mask=None, bin_width=1.0,
                      target_fraction=0.6, percentile=85):
    """Pick a quiver scale so the (percentile)th-percentile arrow
    occupies `target_fraction` of `bin_width`.

    Returns scale (≥1; how much to multiply the data vectors when
    drawing). Robust to the few outlier bins — uses a percentile
    rather than the max.
    """
    mags = []
    for arr in shift_arrays:
        if arr is None:
            continue
        if mask is not None:
            mags.append(np.linalg.norm(arr[mask], axis=-1)
                          if arr.ndim == 3
                          else np.abs(arr[mask]))
        else:
            mags.append(np.linalg.norm(arr.reshape(-1, 2), axis=-1)
                          if arr.ndim == 3
                          else np.abs(arr.ravel()))
    if not mags:
        return 1.0
    all_mags = np.concatenate([m[np.isfinite(m)] for m in mags])
    if len(all_mags) == 0 or np.all(all_mags == 0):
        return 1.0
    p = float(np.percentile(all_mags, percentile))
    if p <= 0:
        return 1.0
    return max(1.0, target_fraction * bin_width / p)


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
        f'  • Per-bin aggregation : {args.statistic}\n'
        f'  • Arrow scale         : auto-fit per ROI so the 85th-pctile\n'
        f'    arrow occupies ~70% of a bin width.\n'
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
    parser.add_argument('--grid-n', type=int, default=9,
                        help='Number of bins per side. 9 keeps ~30+ '
                             'observations per bin in most ROIs.')
    parser.add_argument('--min-n-per-bin', type=int, default=20,
                        help='Drop bins with fewer than this many '
                             'observations.')
    parser.add_argument('--statistic', choices=['median', 'mean'],
                        default='median',
                        help='Per-bin aggregation. Median is robust '
                             'to outlier voxels (default).')
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

        bin_w = 2.0 * args.grid_extent / args.grid_n
        for roi in rois:
            df_roi = df[df['roi'] == roi]
            if len(df_roi) < 100:
                print(f'  {roi}: only {len(df_roi)} rows, skipping.')
                continue
            stats = bin_shifts(
                df_roi,
                grid_extent=args.grid_extent,
                grid_n=args.grid_n,
                min_n_per_bin=args.min_n_per_bin,
                statistic=args.statistic,
            )
            centers = stats['centers']
            mask = stats['mask']
            # Restrict to bins inside the aperture (the experiment's
            # actually-probed region).
            cx, cy = np.meshgrid(centers, centers, indexing='ij')
            inside_aperture = np.sqrt(cx ** 2 + cy ** 2) <= APERTURE
            mask = mask & inside_aperture

            pred_dx = stats['pred_dx']; pred_dy = stats['pred_dy']
            obs_dx  = stats['obs_dx'];  obs_dy  = stats['obs_dy']
            resid_dx = obs_dx - pred_dx
            resid_dy = obs_dy - pred_dy

            # Per-fig auto-scale: use 85th percentile of bin |shift|
            # across PRED and OBS so both panels share the same scale.
            scale = auto_arrow_scale(
                np.stack([pred_dx, pred_dy], axis=-1),
                np.stack([obs_dx, obs_dy], axis=-1),
                mask=mask, bin_width=bin_w,
                target_fraction=0.7, percentile=85,
            )
            scale_resid = auto_arrow_scale(
                np.stack([resid_dx, resid_dy], axis=-1),
                mask=mask, bin_width=bin_w,
                target_fraction=0.7, percentile=85,
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
            plot_quiver_field(axes[0], centers, pred_dx, pred_dy, mask,
                                scale=scale, color='C0',
                                title='PREDICTED  (AF+PRF fit)',
                                ring=canon_ring, bin_w=bin_w)
            plot_quiver_field(axes[1], centers, obs_dx, obs_dy, mask,
                                scale=scale, color='C2',
                                title='OBSERVED  (conditionwise PRF fit)',
                                ring=canon_ring, bin_w=bin_w)
            plot_quiver_field(axes[2], centers, resid_dx, resid_dy, mask,
                                scale=scale_resid, color='C3',
                                title='OBS − PRED  (residual)',
                                ring=canon_ring, bin_w=bin_w)
            n_obs = int(stats['cnt'][mask].sum())
            n_subj = df_roi['subject'].nunique()
            fig.suptitle(
                f'{roi}   |   {n_subj} subjects, {n_obs:,} obs in valid '
                f'(inside-aperture, n≥{args.min_n_per_bin}) bins  |  '
                f'per-bin {args.statistic}, arrow scale auto-fit '
                f'({bin_w:.2f}° bin)',
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig); plt.close(fig)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
