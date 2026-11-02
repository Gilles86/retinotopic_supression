"""Shared parameters of the nested full-data fits under different voxel treatments.

For each (subject, ROI, model) in the reference root and in the other roots:
one row per shared parameter with its value per root and the voxel counts.
Run on the cluster; plot or read the TSV locally::

    python -m retsupp.modeling.compare_shared_fits --out notes/data/shared_fit_comparison.tsv
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = {'sse': 'af_prf_cv_ladder_nested_full_bounded',
         'maxsd20': 'af_prf_cv_ladder_nested_full_bounded_maxsd20',
         'nll': 'af_prf_cv_ladder_nested_full_bounded_nll'}
MODELS = ['m1_T', 'm2_D', 'm3_S', 'm3b_TD', 'm4_TDS', 'm5_TDS_dS', 'm6_TDS_dD', 'm7_full']
SHARED = ['sigma_AF', 'sigma_dyn', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale']
ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']


def load(deriv, root, model, subject, roi):
    fn = (deriv / root / model / f'sub-{subject:02d}'
          / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl')
    if not fn.exists():
        return None
    with open(fn, 'rb') as f:
        fit = pickle.load(f)
    pars = fit['fit_pars_per_fold'][0]
    return pars.reindex(columns=SHARED).iloc[0], len(pars)


def main(bids_folder, out):
    deriv = Path(bids_folder) / 'derivatives'
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for subject in range(1, 31):
        for roi in ROIS:
            for model in MODELS:
                found = {k: load(deriv, r, model, subject, roi) for k, r in ROOTS.items()}
                found = {k: v for k, v in found.items() if v is not None}
                if 'sse' not in found or len(found) < 2:
                    continue
                for par in SHARED:
                    row = dict(subject=subject, roi=roi, model=model, par=par)
                    for k, (vals, n) in found.items():
                        row[k], row[f'n_{k}'] = vals[par], n
                    if np.isfinite(row['sse']):
                        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print(f'Saved {len(rows)} rows to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.out)
