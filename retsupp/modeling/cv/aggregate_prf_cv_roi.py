"""Summarise the m0..m6 ROI cross-validation (fit_prf_cv_roi + prf_cv_reliability).

Per subject x fold x ROI (and over all ROIs, ``roi='all'``) x voxel selection x
model: median held-out R², pooled held-out R², voxel count and the largest
voxel's share of the variance. Every selection uses the fold's training runs only:

- ``reliable``: training-run reliability > 0 (the main selection);
- ``reliable_sd20``: also training SD <= 20% (percent signal);
- ``sd20``: training SD <= 20%;
- ``all``: no selection.

Only subject x fold cells with every requested model are used; the cells
left out are printed. Run on the cluster, then plot the TSV locally::

    python -m retsupp.modeling.cv.aggregate_prf_cv_roi \\
        --out notes/data/prf_cv_roi_summary.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = [0, 1, 2, 3, 4, 5, 6]
SUBJECTS = range(1, 31)
FOLDS = range(3)
SD_LIMIT = 20.0


def selections(rel):
    reliable = rel.reliability.to_numpy() > 0
    low_sd = rel.sd_train.to_numpy() <= SD_LIMIT
    return {'reliable': reliable, 'reliable_sd20': reliable & low_sd,
            'sd20': low_sd, 'all': np.ones(len(rel), bool)}


def summarise(tables, rel):
    """tables: {model: per-voxel DataFrame}, rel: reliability DataFrame."""
    rows = []
    roi = rel.roi.to_numpy()
    for sel, keep in selections(rel).items():
        for name in ['all', *pd.unique(roi)]:
            k = keep if name == 'all' else keep & (roi == name)
            for m, t in tables.items():
                sst = t.sst_test.to_numpy()[k]
                sse = t.sse_test.to_numpy()[k]
                n = int(k.sum())
                rows.append(dict(
                    roi=name, selection=sel, model=m, n_voxels=n,
                    median_r2=np.nanmedian(t.r2_test.to_numpy()[k]) if n else np.nan,
                    pooled_r2=1 - np.nansum(sse) / np.nansum(sst) if n else np.nan,
                    top1_sst_share=np.nanmax(sst) / np.nansum(sst) if n else np.nan))
    return pd.DataFrame(rows)


def main(bids_folder, out, fit_subdir='prf_cv_roi',
         rel_subdir='prf_cv_roi_reliability', subjects=SUBJECTS):
    deriv = Path(bids_folder) / 'derivatives'
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    all_rows, missing = [], []
    for s in subjects:
        for f in FOLDS:
            stem = f'sub-{s:02d}_fold-{f}'
            rel_fn = deriv / rel_subdir / f'sub-{s:02d}' / f'{stem}_reliability.tsv.gz'
            fns = {m: deriv / fit_subdir / f'sub-{s:02d}' / f'{stem}_model-{m}_cvr2.tsv.gz'
                   for m in MODELS}
            lost = [m for m, fn in fns.items() if not fn.exists()]
            if lost or not rel_fn.exists():
                missing.append(f'{stem}: models {lost}, reliability {rel_fn.exists()}')
                continue
            rel = pd.read_csv(rel_fn, sep='\t')
            tables = {m: pd.read_csv(fn, sep='\t') for m, fn in fns.items()}
            for m, t in tables.items():
                if not np.array_equal(t.voxel_index, rel.voxel_index):
                    raise ValueError(f'{stem} model {m}: voxels differ from reliability')
            all_rows.append(summarise(tables, rel).assign(subject=s, fold=f))
    print(f'{len(all_rows)} subject x fold cells used, {len(missing)} left out')
    print('\n'.join(missing))
    if not all_rows:
        raise RuntimeError('nothing to aggregate')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows).to_csv(out, sep='\t', index=False)
    print(f'Saved {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fit-subdir', default='prf_cv_roi')
    ap.add_argument('--rel-subdir', default='prf_cv_roi_reliability')
    ap.add_argument('--subjects', type=int, nargs='+', default=list(SUBJECTS))
    a = ap.parse_args()
    main(a.bids_folder, a.out, a.fit_subdir, a.rel_subdir, a.subjects)
