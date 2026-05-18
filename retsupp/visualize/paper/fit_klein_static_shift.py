"""Fit a Klein-style 4-AF static-shift model on observed conditionwise PRF shifts.

Self-contained: reads m4 conditionwise PRFs directly via Subject. Does
NOT use any AF-fit output — Klein is the "no-AF" comparison model, so
all inputs come from AF-free PRF fits.

For each (subject, ROI):

  - Baseline per voxel: mean of (x, y, sd) across the 4 conditionwise fits.
  - Observed shift per (voxel, condition): cond_xy - mean_xy.
  - Klein 4-AF model:
        M_C(g) = 1 + sign * ( A_HP(g; σ_HP) + Σ_{ℓ ≠ HP} A_LP(g; σ_LP) )
    where A_ℓ are unit-amplitude Gaussians at the 4 ring positions, σ
    differs HP vs LP (no gain). sign = -1 for retsupp (suppression).
  - Numerical CoM of (baseline-Gaussian-PRF × M_C) per (voxel, condition).
  - Fit (σ_HP, σ_LP) by minimising Σ ||klein_predicted_shift - obs_shift||².

Outputs a long-format TSV: one row per (subject, ROI, voxel, condition)
with columns matching plot_rotated_shift_fields.py's expected shape +
Klein-specific columns.

Run locally (no cluster). ~few seconds per (sub, ROI), single CPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers, masking
import yaml
from scipy.optimize import minimize

from retsupp.utils.data import Subject, distractor_locations


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']
ROI_DEFAULT = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO',
               'IPS', 'SPL1', 'FEF']


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


# ---------------------------------------------------------------------
# Numerical CoM under Klein modulation
# ---------------------------------------------------------------------

def klein_predict_centers(base_xy: np.ndarray, sd_v: np.ndarray,
                          sigma_HP: float, sigma_LP: float,
                          sign: int = -1,
                          resolution: int = 81,
                          grid_radius: float = 5.0):
    g1d = np.linspace(-grid_radius, grid_radius, resolution).astype(np.float32)
    GX, GY = np.meshgrid(g1d, g1d)
    G = np.stack([GX.ravel(), GY.ravel()], axis=1)        # (R^2, 2)

    ring = get_ring_positions()                            # (4, 2)
    d2 = np.sum((G[:, None, :] - ring[None, :, :]) ** 2, axis=-1)  # (R^2, 4)

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
    S = np.exp(-dprf2 / (2.0 * sd[None, :] ** 2))           # (R^2, V)

    pred = np.empty((V, len(CONDITIONS), 2), dtype=np.float32)
    for ci in range(len(CONDITIONS)):
        rho = S * M[:, ci:ci + 1]
        rho = np.clip(rho, 0.0, None)
        Z = rho.sum(axis=0) + 1e-12
        pred[:, ci, 0] = (G[:, 0:1] * rho).sum(axis=0) / Z
        pred[:, ci, 1] = (G[:, 1:2] * rho).sum(axis=0) / Z
    return pred


# ---------------------------------------------------------------------
# Load m4 conditionwise PRFs per (subject, ROI)
# ---------------------------------------------------------------------

def load_cond_prfs(sub: Subject, roi: str, conditionfit_model: int = 4):
    """Returns (voxel_indices, base_xy (V,2), base_sd (V,), cond_xy (V,4,2)).

    Baseline = MEAN over the 4 conditionwise PRFs per voxel
    (no AF involvement). Only inside-ROI voxels with finite values
    across all 4 conditions are returned.
    """
    bold_mask = sub.get_bold_mask()
    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    df = sub.get_prf_parameters_volume(model=conditionfit_model,
                                        type='conditionwise')

    # Build (V_total, C, 3) array: x, y, sd per voxel per condition.
    n_v_total = int(np.prod(bold_mask.get_fdata().shape))
    n_v_mask = int(bold_mask.get_fdata().astype(bool).sum())

    pars = np.empty((n_v_mask, len(CONDITIONS), 3), dtype=np.float32)
    for ci, cond in enumerate(CONDITIONS):
        for pi, par in enumerate(['x', 'y', 'sd']):
            img = df.loc[cond, par]
            arr = masking.apply_mask(img, bold_mask, ensure_finite=False)
            pars[:, ci, pi] = arr

    # ROI mask
    roi_img = sub.get_retinotopic_roi(roi, bold_space=True)
    roi_mask_1d = masking.apply_mask(roi_img, bold_mask).astype(bool)

    # Drop voxels with any NaN in any condition × parameter
    valid = np.all(np.isfinite(pars).all(axis=-1), axis=-1)
    keep = valid & roi_mask_1d

    voxel_indices = np.where(keep)[0]
    cond_xy = pars[keep, :, :2]    # (V, 4, 2)
    cond_sd = pars[keep, :, 2]     # (V, 4)
    base_xy = cond_xy.mean(axis=1)     # (V, 2)  — mean of 4 conds
    base_sd = cond_sd.mean(axis=1)     # (V,)    — mean σ across conds
    return voxel_indices, base_xy, base_sd, cond_xy


def fit_klein_one_roi(base_xy, base_sd, cond_xy,
                      sign: int = -1,
                      initial=(1.0, 2.0),
                      bounds=((0.3, 5.0), (0.3, 5.0))):
    obs_shift = cond_xy - base_xy[:, None, :]   # (V, C, 2)
    def loss(params):
        sH, sL = params
        if sH <= 0 or sL <= 0:
            return 1e9
        pred = klein_predict_centers(base_xy, base_sd, sH, sL, sign=sign)
        pred_shift = pred - base_xy[:, None, :]
        diff = pred_shift - obs_shift
        return float(np.sum(diff ** 2))
    res = minimize(loss, x0=list(initial), method='L-BFGS-B', bounds=bounds)
    return res.x[0], res.x[1], float(res.fun), res.success


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', nargs='+', type=int, required=True,
                   help='Subject ids, e.g. --subjects 3 5 11 13 17 18 28 30')
    p.add_argument('--rois', nargs='+', default=ROI_DEFAULT)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--sign', type=int, default=-1,
                   help='-1 suppression (retsupp default), +1 attraction')
    p.add_argument('--out-tsv', type=Path,
                   default=Path('notes/data/klein_shifts.tsv'))
    args = p.parse_args()

    all_rows = []
    fits = []
    for sub_id in args.subjects:
        try:
            sub = Subject(sub_id, args.bids_folder)
        except Exception as e:
            print(f'  sub-{sub_id:02d}: skip ({e})')
            continue
        for roi in args.rois:
            try:
                vox, base_xy, base_sd, cond_xy = load_cond_prfs(sub, roi)
            except Exception as e:
                print(f'  sub-{sub_id:02d} {roi}: skip ({e})')
                continue
            if len(vox) < 20:
                print(f'  sub-{sub_id:02d} {roi}: only {len(vox)} voxels, skip')
                continue
            sH, sL, sse, ok = fit_klein_one_roi(base_xy, base_sd, cond_xy,
                                                sign=args.sign)
            fits.append(dict(subject=sub_id, roi=roi,
                             klein_sigma_HP=sH, klein_sigma_LP=sL,
                             klein_sse=sse, n_voxels=len(vox)))
            print(f'  sub-{sub_id:02d} {roi:5s}: σ_HP={sH:.2f}° '
                  f'σ_LP={sL:.2f}° sse={sse:.2f} n={len(vox)}')

            pred = klein_predict_centers(base_xy, base_sd, sH, sL,
                                          sign=args.sign)
            for ci, cond in enumerate(CONDITIONS):
                for vi, voxidx in enumerate(vox):
                    all_rows.append(dict(
                        subject=sub_id, roi=roi, condition=cond,
                        voxel_idx=int(voxidx),
                        base_x=float(base_xy[vi, 0]),
                        base_y=float(base_xy[vi, 1]),
                        base_sd=float(base_sd[vi]),
                        obs_x=float(cond_xy[vi, ci, 0]),
                        obs_y=float(cond_xy[vi, ci, 1]),
                        pred_klein_x=float(pred[vi, ci, 0]),
                        pred_klein_y=float(pred[vi, ci, 1]),
                        klein_sigma_HP=sH,
                        klein_sigma_LP=sL,
                    ))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_tsv, sep='\t', index=False)
    pd.DataFrame(fits).to_csv(
        args.out_tsv.with_suffix('.fits.tsv'), sep='\t', index=False)
    print(f'wrote {args.out_tsv}  ({len(df):,} rows)')
    print(f'wrote {args.out_tsv.with_suffix(".fits.tsv")}  ({len(fits)} fits)')


if __name__ == '__main__':
    main()
