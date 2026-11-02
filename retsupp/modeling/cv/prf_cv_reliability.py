"""Model-free voxel reliability for the m0..m6 comparison, on training runs only.

For every fold of ``fit_prf_cv_roi.py`` and every voxel, within that fold's
training runs: predict each run by the mean timecourse of the other training
runs (the bar sequence is the same in every run) and compare with that run's
own mean. No encoding model is involved, so selecting on this cannot favour
any of the models, and the fold's held-out runs play no part in it.

Outputs (never overwritten), per fold, in the voxel order of the fits:
``derivatives/prf_cv_roi_reliability/sub-XX/sub-XX_fold-K_reliability.tsv.gz``
with ``voxel_index``, ``roi``, ``reliability`` (pooled over the inner runs) and
``sd_train`` (SD of the BOLD in the training runs, percent signal).

Usage::

    python -m retsupp.modeling.cv.prf_cv_reliability 3 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

from retsupp.modeling.cv.fit_prf_cv import make_fold_assignments
from retsupp.modeling.cv.fit_prf_cv_roi import roi_voxels
from retsupp.utils.data import Subject

T_RUN = 258


def reliability(runs: np.ndarray) -> np.ndarray:
    """(R, T, V) runs -> (V,) leave-one-run-out R² of the other runs' mean."""
    n = runs.shape[0]
    total = runs.sum(0)
    sse = np.zeros(runs.shape[2])
    sst = np.zeros(runs.shape[2])
    for r in range(n):
        pred = (total - runs[r]) / (n - 1)
        sse += ((runs[r] - pred) ** 2).sum(0)
        sst += ((runs[r] - runs[r].mean(0, keepdims=True)) ** 2).sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sst > 0, 1 - sse / sst, np.nan)


def main(subject, bids_folder, n_folds=3,
         output_subdir='prf_cv_roi_reliability'):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    out_dir = bids_folder / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    path = lambda k: out_dir / f'sub-{subject:02d}_fold-{k}_reliability.tsv.gz'
    todo = [k for k in range(n_folds) if not path(k).exists()]
    if not todo:
        print('All outputs exist, nothing to do')
        return

    folds = make_fold_assignments(sub, n_folds=n_folds)
    masker = maskers.NiftiMasker(
        mask_img=sub.get_bold_mask(session=1, run=sub.get_runs(1)[0]))
    masker.fit()
    vox, roi = roi_voxels(sub, masker)

    bold = {}
    for ses, run in sorted(r for f in folds for r in f):
        fn = (bids_folder / 'derivatives' / 'cleaned' / f'sub-{subject:02d}'
              / f'ses-{ses}' / 'func' / f'sub-{subject:02d}_ses-{ses}_task-search_'
              f'desc-cleaned_run-{run}_bold.nii.gz')
        data = masker.transform(fn).astype(np.float32)[:T_RUN, vox]
        if data.shape[0] != T_RUN:
            raise ValueError(f'ses-{ses} run-{run}: {data.shape[0]} volumes')
        bold[ses, run] = data

    out_dir.mkdir(parents=True, exist_ok=True)
    for k in todo:
        train = sorted(r for j, f in enumerate(folds) if j != k for r in f)
        runs = np.stack([bold[r] for r in train]).astype(np.float64)
        rel = reliability(runs)
        pd.DataFrame(dict(
            voxel_index=vox, roi=roi, reliability=rel,
            sd_train=runs.reshape(-1, runs.shape[2]).std(0),
        )).to_csv(path(k), sep='\t', index=False)
        print(f'fold {k}: {len(train)} training runs, {len(vox)} voxels, '
              f'{np.nanmean(rel > 0):.3f} reliable', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--output-subdir', default='prf_cv_roi_reliability')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.n_folds, a.output_subdir)
