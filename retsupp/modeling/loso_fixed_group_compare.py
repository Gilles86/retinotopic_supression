"""Leave-one-subject-out test from the fixed-group sufficient statistics.

For each mechanism: δ is the median of the other subjects' own estimates (all
AF voxels), scored as ΔR² on signal voxels (null cvR² > 0). The per-subject
value is the median over ROIs. Prints the test for each stats folder given.

Usage::

    python -m retsupp.modeling.loso_fixed_group_compare \\
        af_fixed_group_mechanisms af_fixed_group_mechanisms_maxsd20
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MECH = ['items', 'T', 'D', 'S', 'dS', 'dD', 'DonT', 'dT']
KEYS = ['subject', 'roi', 'mech']
COLS = ['subject', 'roi', 'mech', 'voxel_id', 'hp', 'sst', 'a', 'b', 'lin_err']


def load(deriv, folder, null):
    files = sorted(glob.glob(f'{deriv}/{folder}/sub-*/*_stats.tsv.gz'))
    st = pd.concat([pd.read_csv(f, sep='\t', usecols=COLS) for f in files])
    v = st.groupby(KEYS + ['voxel_id']).agg(
        a=('a', 'sum'), b=('b', 'sum'), sst=('sst', 'sum'),
        lin_err=('lin_err', 'first')).reset_index()
    return v.merge(null, on=['subject', 'roi', 'voxel_id'], how='left')


def loso(v):
    own = v.groupby(KEYS).agg(a=('a', 'sum'), b=('b', 'sum'),
                              lin_err=('lin_err', 'first')).reset_index()
    own['own'] = own.b / own.a
    own.loc[own.lin_err > 1e-3, 'own'] = np.nan
    own['delta'] = [own[(own.roi == r) & (own.mech == m) & (own.subject != s)].own.median()
                    for s, r, m in zip(own.subject, own.roi, own.mech)]
    sig = v[v.cv_r2_null > 0]
    a = sig.groupby(KEYS).agg(a=('a', 'sum'), b=('b', 'sum'), sst=('sst', 'sum'),
                              nvox=('voxel_id', 'nunique')).reset_index()
    a = a.merge(own[KEYS + ['delta', 'own', 'lin_err']], on=KEYS)
    a = a[(a.lin_err <= 1e-3) & (a.nvox >= 20)]
    a['dr2'] = -(a.delta ** 2 * a.a - 2 * a.delta * a.b) / a.sst * 1e5
    return a


def main(bids_folder, folders, out=None):
    deriv = f'{bids_folder}/derivatives'
    null = pd.read_csv(f'{deriv}/af_prf_cv_ladder_nested/cv_ladder_per_voxel.tsv', sep='\t',
                       usecols=['subject', 'roi', 'voxel_id', 'cv_r2_null'])
    null = null.groupby(['subject', 'roi', 'voxel_id']).cv_r2_null.mean().reset_index()
    tables = []
    for folder in folders:
        a = loso(load(deriv, folder, null))
        tables.append(a.assign(folder=folder))
        print(folder)
        for m in [m for m in MECH if m in set(a.mech)]:
            x = a[a.mech == m].pivot(index='subject', columns='roi', values='dr2').median(axis=1).dropna()
            print(f'  {m:6s}{(x > 0).mean() * 100:4.0f}%  p={wilcoxon(x).pvalue:.4f}  '
                  f'median ΔR² {x.median():+.2f}e-5  n={len(x)}')
    if out:
        pd.concat(tables).to_csv(out, sep='\t', index=False)
        print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    ap.add_argument('--out', default=None, help='TSV of the per subject x ROI x mechanism LOSO values')
    args = ap.parse_args()
    main(args.bids_folder, args.folders, args.out)
