"""LOSO per ladder step as a table, from the exact curves of ``fit_fixed_group_mechanisms --exact``.

Usage::

    python -m retsupp.visualize.paper.fig3_collect_loso \\
        --folder dog=af_fixed_group_exact_dog --folder dn=af_fixed_group_exact_dn
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from retsupp.modeling.loso_fixed_group_exact import MECH, load, loso


def main(bids, folders, out):
    deriv = f'{bids}/derivatives'
    null = pd.read_csv(f'{deriv}/af_prf_cv_ladder_nested/cv_ladder_per_voxel.tsv', sep='\t',
                       usecols=['subject', 'roi', 'voxel_id', 'cv_r2_null'])
    null = null.groupby(['subject', 'roi', 'voxel_id']).cv_r2_null.mean().reset_index()
    null = null[null.cv_r2_null > 0].groupby(['subject', 'roi']).voxel_id.apply(set).to_dict()
    rows, per_roi = [], []
    for spec in folders:
        family, folder = spec.split('=')
        a = loso(load(deriv, folder, null))
        per_roi.append(a[['subject', 'roi', 'mech', 'dr2']].assign(family=family))
        for m in MECH:
            x = a[a.mech == m].pivot(index='subject', columns='roi', values='dr2').median(axis=1).dropna()
            rows.append(dict(family=family, term=m, n=len(x), pct_improved=(x > 0).mean() * 100, p=wilcoxon(x).pvalue, median_dr2_e5=x.median()))
    d = pd.DataFrame(rows)
    d.to_csv(out, sep='\t', index=False)
    pd.concat(per_roi).to_csv(out.replace('per_step', 'per_roi'), sep='\t', index=False)
    print(d.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    ap.add_argument('--folder', action='append', default=['dog=af_fixed_group_exact_dog', 'dn=af_fixed_group_exact_dn'])
    ap.add_argument('--out', default='notes/data/loso_per_step_2026-11.tsv')
    a = ap.parse_args()
    main(a.bids_folder, a.folder, a.out)
