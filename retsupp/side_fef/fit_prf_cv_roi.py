"""Side project (FEF / sPCS): cross-validated pRF fits in one frontal ROI, with a spatial-scramble null.

The fitting is that of ``side_hippocampus.fit_prf_cv_region`` (paper folds: fold k holds out the k-th run of every HP
condition; per fold on the training runs a Gaussian pRF, a DoG pRF and a DoG pRF with one HRF per region fitted
jointly; scored on the held-out runs), on the voxels of one ROI of ``side_fef_region_ts`` (``extract_rois.py``).
``--variant scramble`` permutes the stimulus pixels by one fixed permutation per participant (seed 1000 + subject, as
in the hippocampus project). ROIs with more than ``--max-voxels`` voxels are subsampled at random (seed = subject).

Output (never overwritten): ``derivatives/side_fef_prf_cv/sub-XX/sub-XX_region-R_variant-V_voxels.tsv.gz`` (one row
per voxel x fold x model), with ``voxel_index`` (BOLD-mask index) added.

Usage::

    python -m retsupp.side_fef.fit_prf_cv_roi 3 sPCS --variant true
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.modeling.cv.fit_prf_cv import make_fold_assignments
from retsupp.modeling.cv.fit_prf_cv_roi_hrf_joint import CANONICAL, fit_roi, flex_factory
from retsupp.modeling.fit_prf import HRF, MODEL_CFG, build_model_factory, gd_fit_scheduled, grid_fit
from retsupp.side_fef.extract_rois import TS_SUBDIR
from retsupp.side_hippocampus.fit_prf_cv_region import KEEP, sse_sst
from retsupp.utils.data import Subject

OUT_SUBDIR = 'side_fef_prf_cv'


def main(subject, region, variant, bids_folder, max_voxels=1500, n_joint=3000, chunk=2000, sd_min=0.2, n_folds=3):
    deriv = Path(bids_folder) / 'derivatives'
    s = f'sub-{subject:02d}'
    out = deriv / OUT_SUBDIR / s / f'{s}_region-{region}_variant-{variant}_voxels.tsv.gz'
    if out.exists():
        print('exists', out)
        return
    t0 = time.time()
    desc = 'cleanedloso' if region.endswith('loso') else 'cleaned'   # leave-one-out ROIs: loso_rois.py
    z = np.load(deriv / TS_SUBDIR / s / f'{s}_desc-{desc}_region_ts.npz', allow_pickle=False)
    rois = list(z['rois'])
    vox = np.flatnonzero(z['member'][:, rois.index(region)])
    if len(vox) < 3:
        print(f'{region}: only {len(vox)} voxels, nothing to fit')
        return
    bold = z['bold'][:, :, vox]
    ok = (bold.std(1) > 0).all(0) & np.isfinite(bold).all((0, 1))
    vox, bold = vox[ok], bold[:, :, ok]
    if len(vox) > max_voxels:
        keep = np.sort(np.random.default_rng(subject).choice(len(vox), max_voxels, replace=False))
        vox, bold = vox[keep], bold[:, :, keep]
    paradigm, grid = z['paradigm'], z['grid']
    assert paradigm.shape[1] == bold.shape[1] == 258, (paradigm.shape, bold.shape)
    if variant == 'scramble':
        paradigm = paradigm[:, :, np.random.default_rng(1000 + subject).permutation(paradigm.shape[2])]
    elif variant != 'true':
        raise ValueError(variant)
    runs = [tuple(r) for r in z['runs'].tolist()]
    folds = make_fold_assignments(Subject(subject, Path(bids_folder)), n_folds=n_folds)
    print(f'{region} {variant}: {len(vox)} voxels, {len(runs)} runs ({time.time() - t0:.0f} s)', flush=True)

    f1 = build_model_factory(MODEL_CFG[1], grid, sd_min)
    f2 = build_model_factory(MODEL_CFG[2], grid, sd_min)
    fj = flex_factory(MODEL_CFG[4], grid, sd_min, 2., .1)
    factories = {'gauss': f1, 'dog': f2, 'dog_roihrf': fj}
    rows = []
    for fold in range(n_folds):
        test = [runs.index(tuple(r)) for r in folds[fold]]
        train = [i for i in range(len(runs)) if i not in test]
        tr, te = bold[train].reshape(-1, len(vox)), bold[test].reshape(-1, len(vox))
        ptr = paradigm[train].reshape(-1, paradigm.shape[2])
        pte = paradigm[test].reshape(-1, paradigm.shape[2])
        fits = {}
        init = grid_fit(f1(tr, ptr), tr, ptr, chunk, sd_min=sd_min)
        fits['gauss'] = gd_fit_scheduled(f1, tr, ptr, init.reset_index(drop=True), chunk,
                                         MODEL_CFG[1]['schedule']).drop(columns='r2').reset_index(drop=True)
        fits['dog'] = gd_fit_scheduled(f2, tr, ptr, MODEL_CFG[2]['adapt'](fits['gauss']), chunk,
                                       MODEL_CFG[2]['schedule']).drop(columns='r2').reset_index(drop=True)
        labels = list(fj(tr, ptr).parameter_labels)
        p1 = fits['dog'].assign(**CANONICAL)[labels]
        p2 = fit_roi(fj, tr, ptr, p1, [p for p in labels if p not in HRF and p != 'baseline'], .005, n_joint, 200, 1e-6)
        fits['dog_roihrf'] = fit_roi(fj, tr, ptr, p2, [], .001, n_joint, 200, 1e-6)
        print(f'fold {fold} fitted ({time.time() - t0:.0f} s)', flush=True)
        for name, p in fits.items():
            f = factories[name]
            lab = list(f(tr, ptr).parameter_labels)
            pr = np.asarray(f(tr, ptr).predict(parameters=p[lab]))
            pe = np.asarray(f(te, pte).predict(parameters=p[lab]))
            sse_tr, sst_tr = sse_sst(tr, pr)
            sse, sst = sse_sst(te, pe)
            d = p[[c for c in KEEP if c in p.columns]].copy()
            d.insert(0, 'voxel', vox)
            d.insert(1, 'voxel_index', z['voxel_index'][vox])
            d.insert(2, 'hemi', z['hemi'][vox])
            d.insert(3, 'fold', fold)
            d.insert(4, 'model', name)
            with np.errstate(divide='ignore', invalid='ignore'):
                d['r2_train'] = 1 - sse_tr / sst_tr
            d['sse_test'], d['sst_test'] = sse, sst
            rows.append(d)
            print(f'  {name}: median test R2 {np.nanmedian(1 - sse / sst):.4f}, frac > 0 {np.mean(1 - sse / sst > 0):.3f}',
                  flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.insert(0, 'subject', subject)
    res.insert(1, 'region', region)
    res.insert(2, 'variant', variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, sep='\t', index=False)
    print(f'wrote {out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('region')
    ap.add_argument('--variant', choices=['true', 'scramble'], default='true')
    ap.add_argument('--max-voxels', type=int, default=1500)
    ap.add_argument('--n-joint', type=int, default=3000)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    a = ap.parse_args()
    main(a.subject, a.region, a.variant, a.bids_folder, a.max_voxels, a.n_joint)
