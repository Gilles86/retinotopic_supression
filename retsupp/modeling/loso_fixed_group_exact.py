"""Leave-one-subject-out test from the ΔSSE(δ) curves of ``fit_fixed_group_mechanisms --exact``.

Same test as ``loso_fixed_group_compare``, without the linear approximation. A subject's
own δ minimises its ΔSSE over all AF voxels of the ROI; δ is the median of the other
subjects' own values, and it is scored as ΔR² on signal voxels (null cvR² > 0). The
per-subject value is the median over ROIs.

Usage::

    python -m retsupp.modeling.loso_fixed_group_exact af_fixed_group_exact_dog af_fixed_group_exact_dn
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon

MECH = ['items', 'T', 'D', 'S', 'dS', 'dD']


def own_delta(grid, curve):
    """Minimum of the spline through the curve; bounded by the neighbouring grid points."""
    i = int(curve.argmin())
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(CubicSpline(grid, curve), bounds=(lo, hi), method='bounded')
    return res.x, i in (0, len(grid) - 1)


def load(deriv, folder, null):
    rows = []
    for f in sorted(glob.glob(f'{deriv}/{folder}/sub-*/*_curves.npz')):
        z = np.load(f)
        subject, roi = int(z['subject']), str(z['roi'])
        signal = null.get((subject, roi), set())
        for m in MECH:
            grid, dsse = z[f'{m}_grid'], z[f'{m}_dsse'].astype(np.float64)
            keep = np.isin(z[f'{m}_voxel_ids'], list(signal))
            own, at_edge = own_delta(grid, dsse.sum(0))
            rows.append(dict(subject=subject, roi=roi, mech=m, own=own, at_edge=at_edge,
                             nvox=keep.sum(), sst=z[f'{m}_sst'][keep].sum(),
                             grid=grid, curve=dsse[keep].sum(0)))
    return pd.DataFrame(rows)


def loso(d):
    d = d.copy()
    d['delta'] = [d[(d.roi == r) & (d.mech == m) & (d.subject != s)].own.median()
                  for s, r, m in zip(d.subject, d.roi, d.mech)]
    d = d[d.nvox >= 20].copy()
    d['dr2'] = [-float(CubicSpline(g, c)(x)) / sst * 1e5
                for g, c, x, sst in zip(d.grid, d.curve, d.delta, d.sst)]
    return d


def main(bids_folder, folders):
    deriv = f'{bids_folder}/derivatives'
    null = pd.read_csv(f'{deriv}/af_prf_cv_ladder_nested/cv_ladder_per_voxel.tsv', sep='\t',
                       usecols=['subject', 'roi', 'voxel_id', 'cv_r2_null'])
    null = null.groupby(['subject', 'roi', 'voxel_id']).cv_r2_null.mean().reset_index()
    null = null[null.cv_r2_null > 0].groupby(['subject', 'roi']).voxel_id.apply(set).to_dict()
    for folder in folders:
        d = load(deriv, folder, null)
        a = loso(d)
        print(folder, f'({d.subject.nunique()} subjects, {len(d) // len(MECH)} subject x ROI)')
        for m in MECH:
            am = a[a.mech == m]
            x = am.pivot(index='subject', columns='roi', values='dr2').median(axis=1).dropna()
            print(f'  {m:6s}{(x > 0).mean() * 100:4.0f}%  p={wilcoxon(x).pvalue:.4f}  '
                  f'median ΔR² {x.median():+.2f}e-5  n={len(x)}  median δ {am.delta.median():+.3f}  '
                  f'own at grid edge {d[d.mech == m].at_edge.mean() * 100:.0f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    args = ap.parse_args()
    main(args.bids_folder, args.folders)
