"""Fit a Klein-style 4-AF static-shift model on observed conditionwise CoM shifts.

Per (subject, ROI):

    M_C(g) = 1 + sign·( A_HP(g; σ_HP) + Σ_{ℓ ≠ HP} A_LP(g; σ_LP) )

Four unit-amplitude AF Gaussians at the 4 ring positions, but with
two σ values: σ_HP (the per-condition HP location) and σ_LP (the 3
LP locations, shared). No gain — HP-vs-LP differences come purely
from the AF size.

Companion to ``predict_shifts_from_af.py`` for the talk-figure
comparison between our gain-based AF model (already fitted on BOLD)
and the size-based Klein-style parameterisation. Fitted on the OBSERVED
shifts (not BOLD) since that's what's analytically tractable and
already loaded by ``predict_shifts_from_af.py``.

Reads the long-format TSV ``predict_shifts_*.tsv`` (which contains
base_x/y, obs_x/y per (sub, ROI, voxel, condition)) and writes a new
TSV with the same rows + columns ``pred_klein_x``, ``pred_klein_y``,
``klein_sigma_HP``, ``klein_sigma_LP`` per (sub, ROI).

Run locally — small per-ROI 2-parameter fit, no cluster needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from retsupp.utils.data import distractor_locations

CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def klein_predict_centers(base_xy: np.ndarray, sd_v: np.ndarray,
                           sigma_HP: float, sigma_LP: float,
                           sign: int = -1,
                           resolution: int = 81,
                           grid_radius: float = 5.0):
    """Numerical CoM of PRF × Klein modulation for each (voxel, condition).

    Parameters
    ----------
    base_xy   : (V, 2) base PRF (x, y)
    sd_v      : (V,)   PRF σ
    sigma_HP  : Klein σ at HP ring (HP for each condition)
    sigma_LP  : Klein σ at the 3 LP rings (shared)
    sign      : -1 suppression, +1 attraction. Matches the AF formulation.

    Returns
    -------
    pred : (V, C, 2) — predicted (x, y) per voxel × condition
    """
    g1d = np.linspace(-grid_radius, grid_radius, resolution).astype(np.float32)
    GX, GY = np.meshgrid(g1d, g1d)
    G = np.stack([GX.ravel(), GY.ravel()], axis=1)            # (R^2, 2)

    ring = get_ring_positions()                                # (4, 2)
    # Per-ring distance squared on the grid.
    d2 = np.sum((G[:, None, :] - ring[None, :, :]) ** 2, axis=-1)  # (R^2, 4)

    # Per-condition modulation: HP ring uses σ_HP, others use σ_LP.
    M = np.empty((G.shape[0], len(CONDITIONS)), dtype=np.float32)
    for ci in range(len(CONDITIONS)):
        a_hp = np.exp(-d2[:, ci] / (2.0 * sigma_HP ** 2))
        a_lp_total = np.sum(
            np.exp(-d2[:, [j for j in range(4) if j != ci]]
                    / (2.0 * sigma_LP ** 2)),
            axis=1
        )
        M[:, ci] = 1.0 + sign * (a_hp + a_lp_total)

    x = base_xy[:, 0].astype(np.float32)
    y = base_xy[:, 1].astype(np.float32)
    sd = sd_v.astype(np.float32)
    V = len(x)
    dprf2 = (
        (G[:, 0:1] - x[None, :]) ** 2
        + (G[:, 1:2] - y[None, :]) ** 2
    )
    S = np.exp(-dprf2 / (2.0 * sd[None, :] ** 2))                # (R^2, V)

    pred = np.empty((V, len(CONDITIONS), 2), dtype=np.float32)
    for ci in range(len(CONDITIONS)):
        rho = S * M[:, ci:ci + 1]
        rho = np.clip(rho, 0.0, None)
        Z = rho.sum(axis=0) + 1e-12
        pred[:, ci, 0] = (G[:, 0:1] * rho).sum(axis=0) / Z
        pred[:, ci, 1] = (G[:, 1:2] * rho).sum(axis=0) / Z

    return pred


def fit_klein_one_roi(df_roi: pd.DataFrame, sign: int = -1,
                       initial=(1.0, 2.0), bounds=((0.3, 5.0), (0.3, 5.0))):
    """Fit (σ_HP, σ_LP) by minimizing per-voxel ||obs_shift - pred_shift||²."""
    voxel_ids = df_roi['voxel_idx'].unique()
    n_v = len(voxel_ids)
    base_df = (df_roi.groupby('voxel_idx')[['base_x', 'base_y', 'base_sd']]
                      .first().loc[voxel_ids])
    base = base_df[['base_x', 'base_y']].values
    sd_v = base_df['base_sd'].values.astype(np.float32)

    # Build observed shifts: (V, C, 2)
    obs = np.full((n_v, len(CONDITIONS), 2), np.nan, dtype=np.float32)
    for ci, cond in enumerate(CONDITIONS):
        d_cond = (df_roi[df_roi['condition'] == cond]
                  .set_index('voxel_idx')
                  .loc[voxel_ids, ['obs_x', 'obs_y']]
                  .values)
        obs[:, ci, :] = d_cond
    obs_shift = obs - base[:, None, :]   # (V, C, 2)

    def loss(params):
        sH, sL = params
        if sH <= 0 or sL <= 0:
            return 1e9
        pred = klein_predict_centers(base, sd_v, sH, sL, sign=sign)
        pred_shift = pred - base[:, None, :]
        diff = pred_shift - obs_shift
        valid = np.isfinite(diff)
        return float(np.sum(diff[valid] ** 2))

    res = minimize(loss, x0=list(initial), method='L-BFGS-B', bounds=bounds)
    return res.x[0], res.x[1], float(res.fun), res.success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-tsv', default=Path(
        '/Users/gdehol/git/retsupp/notes/data/predict_shifts_sharedSigma_8subs.tsv'))
    ap.add_argument('--out-tsv', default=Path(
        '/Users/gdehol/git/retsupp/notes/data/predict_shifts_with_klein_8subs.tsv'))
    ap.add_argument('--sign', type=int, default=-1,
                    help='-1 for suppression (default, matches retsupp), +1 attraction')
    args = ap.parse_args()

    print(f'loading {args.in_tsv}...')
    df = pd.read_csv(args.in_tsv, sep='\t')

    fits = []   # rows: subject, roi, sigma_HP, sigma_LP, sse
    pred_rows = []
    for (sub, roi), df_roi in df.groupby(['subject', 'roi']):
        if len(df_roi) < 50:
            continue
        sH, sL, sse, ok = fit_klein_one_roi(df_roi, sign=args.sign)
        fits.append(dict(subject=sub, roi=roi,
                         klein_sigma_HP=sH, klein_sigma_LP=sL,
                         klein_sse=sse, ok=ok))
        print(f'  sub-{sub:02d} {roi:5s}: σ_HP={sH:.2f}° σ_LP={sL:.2f}° '
              f'sse={sse:.2f}')

        # Predict centers with fitted params and write per-voxel rows.
        voxel_ids = df_roi['voxel_idx'].unique()
        n_v = len(voxel_ids)
        base_df = (df_roi.groupby('voxel_idx')[['base_x', 'base_y', 'base_sd']]
                          .first().loc[voxel_ids])
        base = base_df[['base_x', 'base_y']].values
        sd_v = base_df['base_sd'].values.astype(np.float32)
        pred = klein_predict_centers(base, sd_v, sH, sL, sign=args.sign)
        for ci, cond in enumerate(CONDITIONS):
            for vi, vid in enumerate(voxel_ids):
                pred_rows.append(dict(
                    subject=sub, roi=roi, condition=cond,
                    voxel_idx=int(vid),
                    pred_klein_x=float(pred[vi, ci, 0]),
                    pred_klein_y=float(pred[vi, ci, 1]),
                ))

    pred_df = pd.DataFrame(pred_rows)
    fits_df = pd.DataFrame(fits)

    # Merge Klein predictions into the original long-format TSV.
    merged = df.merge(pred_df, on=['subject', 'roi', 'condition',
                                    'voxel_idx'], how='left')
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_tsv, sep='\t', index=False)
    fits_df.to_csv(args.out_tsv.with_suffix('.fits.tsv'), sep='\t', index=False)
    print(f'wrote {args.out_tsv}  ({len(merged):,} rows)')
    print(f'wrote {args.out_tsv.with_suffix(".fits.tsv")}  ({len(fits_df)} fits)')


if __name__ == '__main__':
    main()
