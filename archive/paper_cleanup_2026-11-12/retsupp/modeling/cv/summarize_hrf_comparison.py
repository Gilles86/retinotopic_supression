"""Held-out R² per subject × ROI × fold for the HRF variants: canonical, per-voxel and per subject × ROI.

Reads the per-voxel ``*_cvr2.tsv.gz`` of ``fit_prf_cv_roi`` (models 2 DoG and 5 DN with the canonical HRF, 4 and 6 with
a flexible HRF per voxel) and of ``fit_prf_cv_roi_hrf`` (14 and 16: one HRF per subject × ROI, the median of the training
fits), and the fold's cross-run reliability. Median held-out R² over the reliable voxels (reliability > 0) of each ROI.

Writes ``<out>`` (subject, fold, roi, model, n_voxels, median_r2). Run on the cluster (reads ~360 files).

Usage::

    python -m retsupp.modeling.cv.summarize_hrf_comparison --out /shares/.../paper_2026-11/hrf_comparison.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
SOURCES = {2: 'prf_cv_roi', 4: 'prf_cv_roi', 5: 'prf_cv_roi', 6: 'prf_cv_roi', 14: 'prf_cv_roi_sharedhrf', 16: 'prf_cv_roi_sharedhrf'}


def main(bids, out, n_subjects=30, n_folds=3):
    deriv = Path(bids) / 'derivatives'
    rows = []
    for s in range(1, n_subjects + 1):
        for f in range(n_folds):
            stem = f'sub-{s:02d}_fold-{f}'
            rel_f = deriv / 'prf_cv_roi_reliability' / f'sub-{s:02d}' / f'{stem}_reliability.tsv.gz'
            if not rel_f.exists():
                continue
            rel = pd.read_csv(rel_f, sep='\t')
            keep = set(rel.loc[rel.reliability > 0, 'voxel_index'])
            for m, src in SOURCES.items():
                p = deriv / src / f'sub-{s:02d}' / f'{stem}_model-{m}_cvr2.tsv.gz'
                if not p.exists():
                    continue
                d = pd.read_csv(p, sep='\t')
                d = d[d.voxel_index.isin(keep) & d.roi.isin(ROIS)]
                for roi, g in d.groupby('roi'):
                    rows.append(dict(subject=s, fold=f, roi=roi, model=m, n_voxels=len(g), median_r2=float(np.nanmedian(g.r2_test))))
        print('sub', s, flush=True)
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.out)
