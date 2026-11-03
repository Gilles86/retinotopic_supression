"""Per-voxel DN parameters of the held-out ROI fits, for the hierarchy analysis.

Reads m5 and m6 of ``prf_cv_roi`` (fitted on the training runs of each fold) with
their held-out R² and the training-run reliability, and keeps voxels that are
reliable or have training R² above .3. Run on the cluster; analyse the table
locally. Refuses to overwrite.

Usage::

    python -m retsupp.modeling.cv.collect_dn_params --out notes/data/prf_cv_roi_dn_params.tsv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODELS = (5, 6)
COLS = ['x', 'y', 'sd', 'rf_amplitude', 'srf_amplitude', 'srf_size',
        'neural_baseline', 'surround_baseline', 'bold_baseline']


def main(bids_folder, out, fit_subdir='prf_cv_roi', rel_subdir='prf_cv_roi_reliability'):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    deriv = Path(bids_folder) / 'derivatives'
    rows, missing = [], []
    for s in range(1, 31):
        for fold in range(3):
            stem = f'sub-{s:02d}_fold-{fold}'
            rel_fn = deriv / rel_subdir / f'sub-{s:02d}' / f'{stem}_reliability.tsv.gz'
            for m in MODELS:
                d = deriv / fit_subdir / f'sub-{s:02d}'
                tsv, pkl = d / f'{stem}_model-{m}_cvr2.tsv.gz', d / f'{stem}_model-{m}_pars.pkl.gz'
                if not (tsv.exists() and pkl.exists() and rel_fn.exists()):
                    missing.append(f'{stem} m{m}')
                    continue
                r2 = pd.read_csv(tsv, sep='\t')
                pars = pd.read_pickle(pkl).reset_index(drop=True)
                assert len(r2) == len(pars), f'{stem} m{m}: {len(r2)} vs {len(pars)}'
                rel = pd.read_csv(rel_fn, sep='\t')
                assert (rel.voxel_index.values == r2.voxel_index.values).all(), stem
                t = pd.concat([r2[['voxel_index', 'roi', 'r2_test', 'r2_train']],
                               pars[COLS + [c for c in ('hrf_delay', 'hrf_dispersion')
                                            if c in pars]]], axis=1)
                t['reliability'] = rel.reliability.values
                t['sd_train'] = rel.sd_train.values
                t.insert(0, 'model', m)
                t.insert(0, 'fold', fold)
                t.insert(0, 'subject', s)
                rows.append(t[(t.reliability > 0) | (t.r2_train > .3)])
    if missing:
        print('Missing:', missing)
    d = pd.concat(rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep='\t', index=False, float_format='%.5g')
    print(f'Saved {len(d)} rows to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.out)
