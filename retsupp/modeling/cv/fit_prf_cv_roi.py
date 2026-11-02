"""Cross-validated comparison of the pRF models m1..m6 inside the ROIs.

One call fits one (subject, fold): the whole warm-start chain of the
canonical pipeline (``MODEL_CFG`` in ``fit_prf.py``) is re-run *within the
fold*, on the training runs only, so no model sees held-out data through
its initialisation:

    m1 grid + GD -> m2 (DoG) -> m4 (DoG + flex HRF)
                 -> m3 (Gaussian + flex HRF)
          m2 -> m5 (DN) -> m6 (DN + flex HRF)

plus m0, the per-voxel training mean. Folds as in ``fit_prf_cv.py``: fold
k holds out the k-th run of every HP condition.

Voxels: the union of the analysis ROIs (Benson + Wang atlases), all
voxels, no signal selection; each voxel keeps its ROI label.

Outputs, never overwritten, under ``derivatives/<output_subdir>/sub-XX/``
(default ``prf_cv_roi``):

- ``sub-XX_fold-K_model-N_cvr2.tsv.gz``: per voxel ``voxel_index`` (into
  the BOLD mask), ``roi``, ``sse_test``, ``sst_test``, ``r2_test``,
  ``r2_train``.
- ``sub-XX_fold-K_model-N_pars.pkl.gz``: fitted parameters (seed for the
  next model in the chain; a rerun resumes from what exists).

Usage::

    python -m retsupp.modeling.cv.fit_prf_cv_roi 3 --fold 0 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

from retsupp.modeling.cv.fit_prf_cv import load_runs, make_fold_assignments
from retsupp.modeling.fit_prf import (
    MODEL_CFG,
    build_model_factory,
    gd_fit_scheduled,
    grid_fit,
)
from retsupp.utils.data import Subject
from retsupp.utils.sentinels import assert_gpu_available_if_expected

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
CHAIN = [1, 2, 3, 4, 5, 6]           # each model's init_from comes earlier
DERIVED = ['r2', 'theta', 'ecc']


def roi_voxels(sub, masker):
    """Indices (into the BOLD mask) and labels of the ROI-union voxels."""
    label = np.full(masker.mask_img_.get_fdata().astype(bool).sum(), '', dtype=object)
    for roi in ROIS:
        m = masker.transform(sub.get_retinotopic_roi(roi, bold_space=True)).ravel() > 0
        label[m & (label == '')] = roi
    idx = np.flatnonzero(label != '')
    return idx, label[idx]


def sse_sst(obs, pred):
    sse = ((obs - pred) ** 2).sum(0)
    sst = ((obs - obs.mean(0, keepdims=True)) ** 2).sum(0)
    return sse, sst


def main(subject, fold, bids_folder, n_folds=3, resolution=50,
         voxel_chunk_size=10000, sd_min=0.2, models=None,
         output_subdir='prf_cv_roi'):
    assert_gpu_available_if_expected()
    models = models or [0] + CHAIN
    sub = Subject(subject, Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    stem = f'sub-{subject:02d}_fold-{fold}'
    tsv = lambda m: out_dir / f'{stem}_model-{m}_cvr2.tsv.gz'
    pkl = lambda m: out_dir / f'{stem}_model-{m}_pars.pkl.gz'
    todo = [m for m in models if not tsv(m).exists()]
    if not todo:
        print('All outputs exist, nothing to do')
        return
    t0 = time.time()

    folds = make_fold_assignments(sub, n_folds=n_folds)
    test_runs = sorted(folds[fold])
    train_runs = sorted(r for k, f in enumerate(folds) if k != fold for r in f)
    print(f'fold {fold}: train {train_runs}\n        test  {test_runs}')

    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask(session=1, run=sub.get_runs(1)[0]))
    masker.fit()
    vox, roi = roi_voxels(sub, masker)
    print(f'{len(vox)} ROI voxels: ' + ', '.join(f'{r} {np.sum(roi == r)}' for r in ROIS))

    train_bold, train_par, grid = load_runs(sub, masker, resolution, 'full', train_runs)
    test_bold, test_par, _ = load_runs(sub, masker, resolution, 'full', test_runs)
    train_bold, test_bold = train_bold[:, vox], test_bold[:, vox]
    print(f'loaded: train T={train_bold.shape[0]}, test T={test_bold.shape[0]} '
          f'({time.time() - t0:.0f} s)')

    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = {}

    def write(m, pred_test, r2_train, pars=None):
        sse, sst = sse_sst(test_bold, pred_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = np.where(sst > 0, 1 - sse / sst, np.nan)
        pd.DataFrame(dict(voxel_index=vox, roi=roi, sse_test=sse, sst_test=sst,
                          r2_test=r2, r2_train=r2_train)).to_csv(tsv(m), sep='\t', index=False)
        if pars is not None:
            pars.to_pickle(pkl(m))
        print(f'  m{m}: median test R² {np.nanmedian(r2):.4f}, pooled '
              f'{1 - np.nansum(sse) / np.nansum(sst):.4f} ({time.time() - t0:.0f} s)')

    if 0 in todo:
        mean = train_bold.mean(0, keepdims=True)
        write(0, np.broadcast_to(mean, test_bold.shape), np.zeros(len(vox)))

    for m in CHAIN:
        if m not in models:
            continue
        if tsv(m).exists() and pkl(m).exists():
            fitted[m] = pd.read_pickle(pkl(m))
            print(f'  m{m}: loaded existing fit')
            continue
        cfg = MODEL_CFG[m]
        factory = build_model_factory(cfg, grid, sd_min)
        if cfg['init_from'] is None:
            init = grid_fit(factory(train_bold, train_par), train_bold, train_par,
                            voxel_chunk_size, sd_min=sd_min)
        else:
            if cfg['init_from'] not in fitted:
                raise RuntimeError(f'm{m} needs m{cfg["init_from"]} from the same fold')
            init = fitted[cfg['init_from']].drop(columns=DERIVED, errors='ignore')
            if cfg.get('adapt') is not None:
                init = cfg['adapt'](init)
        pars = gd_fit_scheduled(factory, train_bold, train_par, init,
                                voxel_chunk_size, cfg['schedule'])
        pars = pars.drop(columns=DERIVED, errors='ignore').reset_index(drop=True)
        fitted[m] = pars
        pred_train = np.asarray(factory(train_bold, train_par).predict(parameters=pars))
        sse_tr, sst_tr = sse_sst(train_bold, pred_train)
        pred_test = np.asarray(factory(test_bold, test_par).predict(parameters=pars))
        with np.errstate(divide='ignore', invalid='ignore'):
            write(m, pred_test, np.where(sst_tr > 0, 1 - sse_tr / sst_tr, np.nan), pars)
    print(f'done ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--resolution', type=int, default=50)
    ap.add_argument('--voxel-chunk-size', type=int, default=10000)
    ap.add_argument('--sd-min', type=float, default=0.2)
    ap.add_argument('--models', type=int, nargs='+', default=None)
    ap.add_argument('--output-subdir', default='prf_cv_roi')
    a = ap.parse_args()
    main(a.subject, a.fold, a.bids_folder, a.n_folds, a.resolution,
         a.voxel_chunk_size, a.sd_min, a.models, a.output_subdir)
