"""Per-ROI predicted-vs-observed shift HEATMAPS in the rotated frame.

A 2D analog of `proj_distance_curves.pdf`: instead of collapsing position
to 1D distance from HP, we show the full 2D map of shift behavior in the
canonical rotated frame (HP at (0, +4°)). Per ROI, three heatmaps
side-by-side:

  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ PREDICTED    │ │ OBSERVED     │ │ OBS − PRED   │
  │ (model)      │ │ (data)       │ │ residual     │
  └──────────────┘ └──────────────┘ └──────────────┘

Each cell colored by the **median projection of the shift onto the
away-from-HP axis** (= +y direction in the canonical frame, since HP is
at top). Diverging colormap: red = away from HP (suppression),
blue = toward HP (attraction).

PRED/OBS share a colorscale (so they're directly comparable). RESIDUAL
gets its own scale.

Why heatmaps and not arrows: arrows blow up when bins have outlier
voxels; the projection scalar field reads cleanly even when individual
shifts are noisy.

Median per bin and aperture-only ⇒ outlier-resistant.

Usage:
    python -m retsupp.visualize.plot_rotated_shift_heatmaps \\
        --tsv notes/predict_shifts_cluster.tsv \\
        --out notes/rotated_shift_heatmaps.pdf
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
APERTURE = 3.17
RING_R = 4.0
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def rotation_to_canonical(condition: str):
    ring_idx = CONDITIONS.index(condition)
    ring = get_ring_positions()
    hp_actual = ring[ring_idx]
    target = np.array([0.0, RING_R])
    a_actual = np.arctan2(hp_actual[1], hp_actual[0])
    a_target = np.arctan2(target[1], target[0])
    theta = a_target - a_actual
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def rotate_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    R = {c: rotation_to_canonical(c) for c in CONDITIONS}
    bx = df['base_x'].values; by = df['base_y'].values
    px = df['pred_x'].values; py = df['pred_y'].values
    ox = df['obs_x'].values;  oy = df['obs_y'].values
    rb  = np.empty_like(bx); rby = np.empty_like(by)
    rp  = np.empty_like(px); rpy = np.empty_like(py)
    ro  = np.empty_like(ox); roy = np.empty_like(oy)
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
    df['dx_pred_rot'] = rp - rb;  df['dy_pred_rot'] = rpy - rby
    df['dx_obs_rot']  = ro - rb;  df['dy_obs_rot']  = roy - rby
    # Projection on AWAY-FROM-HP axis (= -y in canonical frame, since
    # HP is canonical (0, +4°) and away-from-HP points in -y).
    df['proj_pred_rot'] = -df['dy_pred_rot']
    df['proj_obs_rot']  = -df['dy_obs_rot']
    return df


def bin_scalar(df_roi: pd.DataFrame, value_col: str,
                grid_extent: float, grid_n: int,
                min_n_per_bin: int, statistic: str = 'median'):
    """Return (grid_n × grid_n) array of per-bin statistic of `value_col`."""
    edges = np.linspace(-grid_extent, grid_extent, grid_n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bx = df_roi['base_x_rot'].values
    by = df_roi['base_y_rot'].values
    val = df_roi[value_col].values
    ix = np.digitize(bx, edges) - 1
    iy = np.digitize(by, edges) - 1
    valid = (ix >= 0) & (ix < grid_n) & (iy >= 0) & (iy < grid_n)
    out = np.full((grid_n, grid_n), np.nan, dtype=np.float64)
    cnt = np.zeros((grid_n, grid_n), dtype=np.int32)
    agg = np.median if statistic == 'median' else np.mean
    for i in range(grid_n):
        for j in range(grid_n):
            sel = valid & (ix == i) & (iy == j) & np.isfinite(val)
            n = sel.sum()
            cnt[i, j] = n
            if n >= min_n_per_bin:
                out[i, j] = agg(val[sel])
    return centers, out, cnt


def aperture_mask_grid(centers: np.ndarray, aperture: float) -> np.ndarray:
    """(grid_n × grid_n) boolean mask of bins whose CENTER is inside aperture."""
    cx, cy = np.meshgrid(centers, centers, indexing='ij')
    return np.sqrt(cx ** 2 + cy ** 2) <= aperture


def plot_heatmap(ax, centers, scalar, vmax, title,
                  ring=None, hp=(0.0, RING_R), aperture=APERTURE):
    edge_step = (centers[1] - centers[0]) / 2.0
    extent = (centers[0] - edge_step, centers[-1] + edge_step,
              centers[0] - edge_step, centers[-1] + edge_step)
    # imshow expects (rows=y, cols=x) → transpose so y is the row axis.
    im = ax.imshow(scalar.T, extent=extent, origin='lower',
                    cmap='RdBu_r', vmin=-vmax, vmax=+vmax,
                    interpolation='nearest')
    ax.add_patch(Circle((0, 0), aperture, fill=False, ec='0.3',
                          ls='--', lw=0.7))
    if ring is not None:
        for r in ring:
            ax.plot(r[0], r[1], 'o', color='0.3', mec='k',
                    markersize=6, alpha=0.85)
    ax.plot(hp[0], hp[1], '*', color='gold', mec='k',
            markersize=18, zorder=5)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def cover_page(pdf, df, args):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.96, 'Rotated-frame projection heatmaps per ROI',
            ha='center', va='top', fontsize=15, weight='bold',
            transform=ax.transAxes)
    body = (
        f'Source TSV : {args.tsv}\n'
        f'Total rows : {len(df):,}\n'
        f'Subjects   : {df["subject"].nunique()}\n'
        f'\n'
        f'How to read each ROI page:\n'
        f'  • Every (subject × condition × voxel) row rotated so HP is\n'
        f'    at canonical (0, +4°) — ★ in each panel.\n'
        f'  • Per-bin MEDIAN projection of the shift on the AWAY-FROM-HP\n'
        f'    axis (= −y in the canonical frame).\n'
        f'      red  = away from HP  (suppression direction)\n'
        f'      blue = toward HP     (attraction direction)\n'
        f'  • Bins with < {args.min_n_per_bin} observations dropped (white).\n'
        f'  • PRED and OBS share a colorscale; residual has its own.\n'
        f'  • Outside the bar aperture (radius {APERTURE}°) is not\n'
        f'    constrained by the experiment — bins outside the dashed\n'
        f'    circle are still shown when they have data, but read with\n'
        f'    caution.\n'
        f'\n'
        f'Why this is the right plot:\n'
        f'  • 2D analog of proj_distance_curves.pdf (which collapses to\n'
        f'    1D distance and so cannot show the spatial structure of\n'
        f'    where attention is acting).\n'
        f'  • Heatmap not arrows — robust to outlier voxels that blow\n'
        f'    up bin medians of vector magnitude.\n'
        f'\n'
        f'Caveat — Jensen-style noise bias:\n'
        f'  • base_xy and obs_xy come from independent fits, so noise on\n'
        f'    base partially leaks into the binning. Voxels that happen\n'
        f'    to land in the small-distance bin tend to have base_xy\n'
        f'    fluctuated TOWARD HP; the obs_xy estimate (independent\n'
        f'    noise) regresses toward truth → AWAY from HP. So the\n'
        f'    OBSERVED panel can show a near-HP "away" stripe even with\n'
        f'    zero true effect. The PRED panel is deterministic given\n'
        f'    base_xy and has NO such bias.\n'
        f'  • Read the PATTERN match (peak location, decay shape), not\n'
        f'    the magnitude difference, as the test of whether the model\n'
        f'    captures the data.\n'
    )
    ax.text(0.04, 0.86, body, ha='left', va='top',
            family='monospace', fontsize=9, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def render_one_section(pdf, df: pd.DataFrame, rois: list, canon_ring,
                         args, section_label: str):
    """Render one ROI page per ROI: 3 rows (proj-on-HP-axis, dx, dy)
    × 3 columns (PRED | OBS | RESIDUAL)."""
    components = [
        # (pred_col,         obs_col,         label,                 cbar_label)
        ('proj_pred_rot',    'proj_obs_rot',
         'projection on away-from-HP axis',
         'proj. away-from-HP (deg)'),
        ('dx_pred_rot',      'dx_obs_rot',
         'Δx (rotated)  — left/right component',
         'Δx (deg)'),
        ('dy_pred_rot',      'dy_obs_rot',
         'Δy (rotated)  — south/north component',
         'Δy (deg)'),
    ]

    for roi in rois:
        df_roi = df[df['roi'] == roi]
        if len(df_roi) < args.min_subject_rows:
            print(f'  [{section_label}] {roi}: only {len(df_roi)} rows, '
                  f'skipping.')
            continue

        # Per-component aggregations.
        comp_data = []
        for pred_col, obs_col, _, _ in components:
            centers, pred_grid, cnt = bin_scalar(
                df_roi, pred_col,
                args.grid_extent, args.grid_n,
                args.min_n_per_bin, args.statistic,
            )
            _, obs_grid, _ = bin_scalar(
                df_roi, obs_col,
                args.grid_extent, args.grid_n,
                args.min_n_per_bin, args.statistic,
            )
            ap_mask = aperture_mask_grid(centers, APERTURE)
            pred_in = np.where(ap_mask, pred_grid, np.nan)
            obs_in  = np.where(ap_mask, obs_grid, np.nan)
            resid_in = obs_in - pred_in
            both = np.r_[pred_in[np.isfinite(pred_in)].ravel(),
                         obs_in[np.isfinite(obs_in)].ravel()]
            vmax_main = float(np.nanpercentile(np.abs(both), 95)) \
                if len(both) else 0.05
            vmax_main = max(vmax_main, 0.02)
            r_arr = resid_in[np.isfinite(resid_in)]
            vmax_res = float(np.nanpercentile(np.abs(r_arr), 95)) \
                if len(r_arr) else 0.05
            vmax_res = max(vmax_res, 0.02)
            comp_data.append((centers, pred_in, obs_in, resid_in,
                                vmax_main, vmax_res, ap_mask, cnt))

        n_subj = df_roi['subject'].nunique()
        sub_str = (f'{n_subj} subjects' if n_subj > 1
                    else f"sub-{int(df_roi['subject'].iloc[0]):02d}")
        n_obs = int(comp_data[0][7][comp_data[0][6]
                                       & np.isfinite(comp_data[0][1])].sum())

        fig, axes = plt.subplots(3, 3, figsize=(15, 14))
        for r_idx, ((pred_col, obs_col, row_lbl, cbar_lbl),
                     (centers, pred_in, obs_in, resid_in,
                      vmax_main, vmax_res, ap_mask, cnt)) in enumerate(
                zip(components, comp_data)):
            im0 = plot_heatmap(axes[r_idx, 0], centers, pred_in,
                                 vmax=vmax_main,
                                 title=(f'PRED  —  {row_lbl}'
                                          if r_idx == 0
                                          else f'PRED  —  {row_lbl}'),
                                 ring=canon_ring)
            plt.colorbar(im0, ax=axes[r_idx, 0], fraction=0.046,
                          label=cbar_lbl)
            im1 = plot_heatmap(axes[r_idx, 1], centers, obs_in,
                                 vmax=vmax_main,
                                 title=f'OBS   —  {row_lbl}',
                                 ring=canon_ring)
            plt.colorbar(im1, ax=axes[r_idx, 1], fraction=0.046,
                          label=cbar_lbl)
            im2 = plot_heatmap(axes[r_idx, 2], centers, resid_in,
                                 vmax=vmax_res,
                                 title=f'OBS−PRED  —  {row_lbl}',
                                 ring=canon_ring)
            plt.colorbar(im2, ax=axes[r_idx, 2], fraction=0.046,
                          label=cbar_lbl)

        fig.suptitle(
            f'{section_label}  —  {roi}   |   {sub_str}, '
            f'{n_obs:,} obs in inside-aperture bins  |  '
            f'{args.statistic} per bin',
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tsv', type=Path,
                        default=Path('notes/predict_shifts_cluster.tsv'))
    parser.add_argument('--out', type=Path,
                        default=Path('notes/rotated_shift_heatmaps.pdf'))
    parser.add_argument('--grid-extent', type=float, default=4.5)
    parser.add_argument('--grid-n', type=int, default=15)
    parser.add_argument('--min-n-per-bin', type=int, default=20)
    parser.add_argument('--min-subject-rows', type=int, default=50,
                        help='Skip ROIs with fewer than this many rows '
                             '(needed for single-subject sections).')
    parser.add_argument('--statistic', choices=['median', 'mean'],
                        default='median')
    parser.add_argument('--rois', nargs='+', default=None)
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='Subject IDs to render individually after '
                             'the group section. Default: top-3 by '
                             'predicted-vs-observed mean Pearson r '
                             '(computed from the TSV).')
    args = parser.parse_args()

    df = pd.read_csv(args.tsv, sep='\t')
    print(f'Loaded {len(df):,} rows from {args.tsv}')
    df = df.dropna(subset=['base_x', 'base_y', 'pred_x', 'pred_y',
                            'obs_x', 'obs_y', 'condition'])
    print(f'After dropna: {len(df):,} rows')
    df = rotate_table(df)
    print('Rotated to canonical frame.')

    # Default: pick top-3 well-behaved subjects by mean predicted-vs-
    # observed r across ROIs (high r = model captures voxel-wise shift
    # pattern, regardless of absolute parameter values).
    if args.subjects is None:
        rec = []
        for (s, roi), g in df.groupby(['subject', 'roi']):
            m = g[['proj_pred', 'proj_obs']].dropna()
            if len(m) < 50:
                continue
            r = float(np.corrcoef(m['proj_pred'], m['proj_obs'])[0, 1])
            rec.append(dict(subject=int(s), roi=roi, r=r))
        rdf = pd.DataFrame(rec)
        if len(rdf):
            sub_score = (rdf.groupby('subject')['r'].mean()
                            .sort_values(ascending=False))
            args.subjects = sub_score.head(3).index.tolist()
            print(f'Top-3 by mean predicted-vs-observed r: {args.subjects}')
            print(sub_score.head(5).round(3).to_string())

    rois = (args.rois if args.rois
            else [r for r in ROI_ORDER if r in df['roi'].unique()])
    canon_ring = np.array([
        [0.0, +RING_R], [-RING_R, 0.0],
        [0.0, -RING_R], [+RING_R, 0.0],
    ], dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        cover_page(pdf, df, args)

        # ---- GROUP page set ----
        # Section divider.
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.5,
                 f'GROUP  (all {df["subject"].nunique()} subjects)',
                 ha='center', va='center', fontsize=24, weight='bold',
                 transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)
        render_one_section(pdf, df, rois, canon_ring, args,
                            section_label='GROUP')

        # ---- Individual subject sections ----
        for sub_id in (args.subjects or []):
            df_sub = df[df['subject'] == sub_id]
            if len(df_sub) == 0:
                print(f'No rows for sub-{sub_id:02d}, skipping.')
                continue
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.text(0.5, 0.5,
                     f'sub-{int(sub_id):02d}',
                     ha='center', va='center', fontsize=24, weight='bold',
                     transform=ax.transAxes)
            pdf.savefig(fig); plt.close(fig)
            render_one_section(pdf, df_sub, rois, canon_ring, args,
                                section_label=f'sub-{int(sub_id):02d}')

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
