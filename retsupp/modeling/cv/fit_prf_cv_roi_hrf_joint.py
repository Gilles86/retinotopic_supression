"""One HRF per subject × ROI, fitted jointly with the pRFs in three stages; held-out test per stage.

Stages, per ROI, on the training runs of one fold:

1. All pRF parameters with the canonical HRF: the fold's canonical fit from ``fit_prf_cv_roi``
   (model 2 for DoG, 5 for DN), carried into the flexible-HRF model at the canonical HRF values
   (delay 4.5 s, dispersion 0.75). Checked: same predictions as the canonical model.
2. Only the HRF free: one delay and one dispersion shared by all voxels of the ROI; every pRF
   parameter (including amplitude and baseline) fixed.
3. Everything free: all pRF parameters per voxel and the shared HRF of the ROI together.

After every stage: training and held-out R² per voxel, the ROI's HRF, and how far the pRFs moved from
stage 1 (centre shift and change in σ, reliable voxels). Stage 3 frees spatial and HRF parameters
together, which ``fit_prf`` never does for per-voxel HRFs; the shift columns are there to show whether
the shared HRF disturbs the retinotopy.

Outputs under ``derivatives/<output_subdir>/sub-XX/`` (tagged with ``--tag``):
``sub-XX_fold-K_<family>_<tag>_summary.tsv`` (one row per ROI × stage),
``..._stage-S_cvr2.tsv.gz`` and ``..._stage-S_pars.pkl.gz``.

Usage::

    python -m retsupp.modeling.cv.fit_prf_cv_roi_hrf_joint 3 --fold 0 --family dog --tag A
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
from retsupp.modeling.fit_prf import HRF, MODEL_CFG, build_model_factory
from retsupp.utils.data import Subject
from retsupp.utils.sentinels import assert_gpu_available_if_expected

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
FAMILY = {'dog': (2, 4), 'dn': (5, 6)}           # (canonical parent, flexible-HRF model class)
CANONICAL = {'hrf_delay': 4.5, 'hrf_dispersion': 0.75}


def r2(obs, pred):
    sse, sst = sse_sst(obs, pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sst > 0, 1 - sse / sst, np.nan)


def fit_roi(factory, train, par, init, fixed, lr, n_iter, min_iter, atol):
    from braincoder.optimize import ParameterFitter
    model = factory(train, par)
    labels = list(model.parameter_labels)
    fitter = ParameterFitter(model, train, par)
    pars = fitter.fit(init_pars=init[labels].reset_index(drop=True), max_n_iterations=n_iter, min_n_iterations=min_iter,
                      learning_rate=lr, fixed_pars=fixed or None, shared_pars=list(HRF), r2_atol=atol, progressbar=False)
    return pars[labels]


def main(subject, fold, bids_folder, family, tag, lr2, n2, lr3, n3, min_iter=200, atol=1e-7, n_folds=3, resolution=50,
         sd_min=0.2, base_subdir='prf_cv_roi', output_subdir='prf_cv_roi_jointhrf', rois=ROIS):
    assert_gpu_available_if_expected()
    parent, flex = FAMILY[family]
    sub = Subject(subject, Path(bids_folder))
    deriv = Path(bids_folder) / 'derivatives'
    base_dir = deriv / base_subdir / f'sub-{subject:02d}'
    out_dir = deriv / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'sub-{subject:02d}_fold-{fold}'
    out_stem = f'{stem}_{family}_{tag}'
    if (out_dir / f'{out_stem}_summary.tsv').exists():
        print('Output exists, nothing to do')
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
    print(f'{len(vox)} voxels, loaded in {time.time() - t0:.0f} s', flush=True)

    base = pd.read_csv(base_dir / f'{stem}_model-{parent}_cvr2.tsv.gz', sep='\t')
    if not np.array_equal(base.voxel_index.to_numpy(), vox):
        raise ValueError(f'm{parent}: voxels differ from this ROI set')
    canon = pd.read_pickle(base_dir / f'{stem}_model-{parent}_pars.pkl.gz').drop(columns=DERIVED, errors='ignore')
    canon = canon.reset_index(drop=True)
    rel = pd.read_csv(deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}' / f'{stem}_reliability.tsv.gz', sep='\t')
    reliable = np.isin(vox, rel.loc[rel.reliability > 0, 'voxel_index'].to_numpy())

    f_canon = build_model_factory(MODEL_CFG[parent], grid, sd_min)
    f_flex = build_model_factory(MODEL_CFG[flex], grid, sd_min)
    stage1 = canon.assign(**CANONICAL)
    stages = {1: stage1.copy(), 2: stage1.copy(), 3: stage1.copy()}
    rows = []
    for r in rois:
        ix = np.where((roi == r) & canon.notna().all(axis=1).to_numpy())[0]
        if len(ix) < 20:
            print(f'{r}: only {len(ix)} voxels, skipped', flush=True)
            continue
        tr, te = train_bold[:, ix], test_bold[:, ix]
        # Stage 1 check: the flexible model at the canonical HRF reproduces the canonical model.
        labels = list(f_flex(tr, train_par).parameter_labels)
        p_can = np.asarray(f_canon(tr, train_par).predict(parameters=canon.iloc[ix][list(f_canon(tr, train_par).parameter_labels)]))
        p_flex = np.asarray(f_flex(tr, train_par).predict(parameters=stage1.iloc[ix][labels]))
        mismatch = float(np.nanmax(np.abs(p_can - p_flex)) / max(np.nanmax(np.abs(p_can)), 1e-12))
        pars = {1: stage1.iloc[ix][labels].reset_index(drop=True)}
        ts = time.time()
        pars[2] = fit_roi(f_flex, tr, train_par, pars[1], [p for p in labels if p not in HRF], lr2, n2, min_iter, atol)
        t2 = time.time() - ts
        ts = time.time()
        pars[3] = fit_roi(f_flex, tr, train_par, pars[2], [], lr3, n3, min_iter, atol)
        t3 = time.time() - ts
        rv = reliable[ix]
        for s in (1, 2, 3):
            p = pars[s]
            stages[s].loc[ix, labels] = p.to_numpy()
            r2_tr = r2(tr, np.asarray(f_flex(tr, train_par).predict(parameters=p)))
            r2_te = r2(te, np.asarray(f_flex(te, test_par).predict(parameters=p)))
            shift = np.hypot(p.x - pars[1].x, p.y - pars[1].y).to_numpy()
            dsd = (p.sd - pars[1].sd).to_numpy()
            rows.append(dict(subject=subject, fold=fold, family=family, tag=tag, roi=r, stage=s, n_voxels=len(ix),
                             n_reliable=int(rv.sum()), hrf_delay=float(p.hrf_delay.median()),
                             hrf_dispersion=float(p.hrf_dispersion.median()),
                             hrf_unique=int(p[list(HRF)].round(6).drop_duplicates().shape[0]),
                             r2_train_reliable=np.nanmedian(r2_tr[rv]), r2_test_reliable=np.nanmedian(r2_te[rv]),
                             r2_train_all=np.nanmedian(r2_tr), r2_test_all=np.nanmedian(r2_te),
                             shift_median=np.nanmedian(shift[rv]), shift_p90=np.nanpercentile(shift[rv], 90) if rv.any() else np.nan,
                             dsd_median=np.nanmedian(dsd[rv]), seconds=(0 if s == 1 else t2 if s == 2 else t3),
                             stage1_mismatch=mismatch, lr=(np.nan if s == 1 else lr2 if s == 2 else lr3),
                             max_iter=(0 if s == 1 else n2 if s == 2 else n3)))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()
                   if k in ('roi', 'stage', 'hrf_delay', 'hrf_dispersion', 'r2_train_reliable', 'r2_test_reliable',
                            'shift_median', 'dsd_median', 'seconds', 'stage1_mismatch')}, flush=True)
    summary = pd.DataFrame(rows)
    for s, p in stages.items():
        labels = list(f_flex(train_bold[:, :2], train_par).parameter_labels)
        p.to_pickle(out_dir / f'{out_stem}_stage-{s}_pars.pkl.gz')
        keep = np.isin(roi, summary.roi.unique()) & p[labels].notna().all(axis=1).to_numpy()
        r2_te = np.full(len(vox), np.nan)
        r2_tr = np.full(len(vox), np.nan)
        ix = np.where(keep)[0]
        r2_te[ix] = r2(test_bold[:, ix], np.asarray(f_flex(test_bold[:, ix], test_par).predict(parameters=p.iloc[ix][labels])))
        r2_tr[ix] = r2(train_bold[:, ix], np.asarray(f_flex(train_bold[:, ix], train_par).predict(parameters=p.iloc[ix][labels])))
        pd.DataFrame(dict(voxel_index=vox, roi=roi, r2_test=r2_te, r2_train=r2_tr)).to_csv(
            out_dir / f'{out_stem}_stage-{s}_cvr2.tsv.gz', sep='\t', index=False)
    summary.to_csv(out_dir / f'{out_stem}_summary.tsv', sep='\t', index=False)
    print(summary.round(4).to_string(index=False))
    print(f'done ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--family', choices=list(FAMILY), default='dog')
    ap.add_argument('--tag', default='A')
    ap.add_argument('--lr2', type=float, default=.005)
    ap.add_argument('--n2', type=int, default=2000)
    ap.add_argument('--lr3', type=float, default=.001)
    ap.add_argument('--n3', type=int, default=2000)
    ap.add_argument('--min-iter', type=int, default=200)
    ap.add_argument('--atol', type=float, default=1e-7)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--output-subdir', default='prf_cv_roi_jointhrf')
    a = ap.parse_args()
    main(a.subject, a.fold, a.bids_folder, a.family, a.tag, a.lr2, a.n2, a.lr3, a.n3, a.min_iter, a.atol,
         rois=a.rois, output_subdir=a.output_subdir)
