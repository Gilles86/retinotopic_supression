"""Held-out test of one HRF per subject x ROI, against the fits of ``fit_prf_cv_roi``.

The model is the flexible-HRF one, but every voxel of a ROI gets the same delay and dispersion,
held fixed. Those are the medians over the ROI's voxels with training R² >= ``MIN_R2`` in the
fold's own flexible fit, so held-out data play no part. A ROI with too few such voxels takes the
median over all ROIs. The pRF parameters then continue from that fit, on the training runs.

Models are numbered 10 above their flexible parent: 13 (Gaussian), 14 (DoG), 16 (DN).
Outputs as in ``fit_prf_cv_roi``, under ``derivatives/<output_subdir>/sub-XX/``, plus
``sub-XX_fold-K_model-N_hrf.tsv`` with the HRF used per ROI. Existing outputs are not overwritten.

Usage::

    python -m retsupp.modeling.cv.fit_prf_cv_roi_hrf 3 --fold 0 \\
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
from retsupp.modeling.cv.fit_prf_cv_roi import DERIVED, roi_voxels, sse_sst
from retsupp.modeling.fit_prf import HRF, MODEL_CFG, build_model_factory, gd_fit_scheduled
from retsupp.utils.data import Subject
from retsupp.utils.sentinels import assert_gpu_available_if_expected

PARENT = {13: 3, 14: 4, 16: 6}
MIN_R2 = .1
MIN_VOXELS = 20
STAGE = [(HRF, 0.001, 1500, 1e-5)]


def shared_hrf(pars, r2_train, roi):
    """Per-ROI medians of the HRF parameters, over voxels fitted well in training."""
    good = (r2_train >= MIN_R2) & pars[list(HRF)].notna().all(axis=1).to_numpy()
    if good.sum() < MIN_VOXELS:
        raise ValueError(f'only {good.sum()} voxels with training R² >= {MIN_R2}')
    overall = pars.loc[good, list(HRF)].median()
    rows = []
    out = pars.copy()
    for r in np.unique(roi):
        keep = good & (roi == r)
        value = pars.loc[keep, list(HRF)].median() if keep.sum() >= MIN_VOXELS else overall
        valid = (roi == r) & pars[list(HRF)].notna().all(axis=1).to_numpy()
        out.loc[valid, list(HRF)] = value.to_numpy()
        rows.append(dict(roi=r, n_voxels=int(keep.sum()), from_roi=bool(keep.sum() >= MIN_VOXELS), **value.to_dict()))
    return out, pd.DataFrame(rows)


def main(subject, fold, bids_folder, n_folds=3, resolution=50, voxel_chunk_size=10000, sd_min=0.2,
         models=None, base_subdir='prf_cv_roi', output_subdir='prf_cv_roi_sharedhrf'):
    assert_gpu_available_if_expected()
    models = models or list(PARENT)
    sub = Subject(subject, Path(bids_folder))
    deriv = Path(bids_folder) / 'derivatives'
    out_dir = deriv / output_subdir / f'sub-{subject:02d}'
    base_dir = deriv / base_subdir / f'sub-{subject:02d}'
    stem = f'sub-{subject:02d}_fold-{fold}'
    todo = [m for m in models if not (out_dir / f'{stem}_model-{m}_cvr2.tsv.gz').exists()]
    if not todo:
        print('All outputs exist, nothing to do')
        return
    t0 = time.time()

    folds = make_fold_assignments(sub, n_folds=n_folds)
    test_runs = sorted(folds[fold])
    train_runs = sorted(r for k, f in enumerate(folds) if k != fold for r in f)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask(session=1, run=sub.get_runs(1)[0]))
    masker.fit()
    vox, roi = roi_voxels(sub, masker)
    train_bold, train_par, grid = load_runs(sub, masker, resolution, 'full', train_runs)
    test_bold, test_par, _ = load_runs(sub, masker, resolution, 'full', test_runs)
    train_bold, test_bold = train_bold[:, vox], test_bold[:, vox]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{len(vox)} voxels, loaded in {time.time() - t0:.0f} s')

    for m in todo:
        parent = PARENT[m]
        base = pd.read_csv(base_dir / f'{stem}_model-{parent}_cvr2.tsv.gz', sep='\t')
        if not np.array_equal(base.voxel_index.to_numpy(), vox):
            raise ValueError(f'm{parent}: voxels differ from this ROI set')
        pars = pd.read_pickle(base_dir / f'{stem}_model-{parent}_pars.pkl.gz')
        pars = pars.drop(columns=DERIVED, errors='ignore').reset_index(drop=True)
        init, table = shared_hrf(pars, base.r2_train.to_numpy(), roi)
        table.to_csv(out_dir / f'{stem}_model-{m}_hrf.tsv', sep='\t', index=False)
        print(f'm{m}: HRF per ROI\n{table.round(3).to_string(index=False)}')

        factory = build_model_factory(MODEL_CFG[parent], grid, sd_min)
        fit = gd_fit_scheduled(factory, train_bold, train_par, init, voxel_chunk_size, STAGE)
        fit = fit.drop(columns=DERIVED, errors='ignore').reset_index(drop=True)
        sse_tr, sst_tr = sse_sst(train_bold, np.asarray(factory(train_bold, train_par).predict(parameters=fit)))
        sse, sst = sse_sst(test_bold, np.asarray(factory(test_bold, test_par).predict(parameters=fit)))
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = np.where(sst > 0, 1 - sse / sst, np.nan)
            r2_train = np.where(sst_tr > 0, 1 - sse_tr / sst_tr, np.nan)
        pd.DataFrame(dict(voxel_index=vox, roi=roi, sse_test=sse, sst_test=sst, r2_test=r2, r2_train=r2_train)).to_csv(
            out_dir / f'{stem}_model-{m}_cvr2.tsv.gz', sep='\t', index=False)
        fit.to_pickle(out_dir / f'{stem}_model-{m}_pars.pkl.gz')
        print(f'm{m}: median test R² {np.nanmedian(r2):.4f}, parent {np.nanmedian(base.r2_test):.4f} '
              f'({time.time() - t0:.0f} s)')
    print(f'done ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subject', type=int)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--resolution', type=int, default=50)
    ap.add_argument('--voxel-chunk-size', type=int, default=10000)
    ap.add_argument('--sd-min', type=float, default=0.2)
    ap.add_argument('--models', type=int, nargs='+', default=None)
    ap.add_argument('--base-subdir', default='prf_cv_roi')
    ap.add_argument('--output-subdir', default='prf_cv_roi_sharedhrf')
    a = ap.parse_args()
    main(a.subject, a.fold, a.bids_folder, a.n_folds, a.resolution, a.voxel_chunk_size, a.sd_min,
         a.models, a.base_subdir, a.output_subdir)
