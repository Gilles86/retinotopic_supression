"""Refit the V1 pRFs of one (subject, fold) under one fitting condition of ``sim_prf_size_floor``.

Same chain and folds as the paper's final pRFs (``fit_prf_cv_roi`` m1 → m2, then ``fit_prf_cv_roi_hrf_joint``
final2: one HRF per subject × ROI, stage 2 and 3 with 3000 iterations), restricted to V1, with the fitting resolution,
the lowest σ in the m1 grid and ``sd_min`` of the condition (``CONDITIONS`` in ``sim_prf_size_floor``; 'paper50' is
the paper's recipe and should reproduce the final fits).

Output (never overwritten): ``derivatives/<output_subdir>/sub-XX/sub-XX_fold-K_<condition>_v1.tsv.gz``: per V1 voxel
the m1, m2 and joint-HRF (s3) parameters, training and held-out R² of s3.

Usage::

    python -m retsupp.modeling.cv.refit_v1_prf_resolution 23 --fold 0 --condition free100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers


def main(subject, fold, condition, bids_folder, output_subdir, n2=3000, n3=3000):
    from braincoder.optimize import ParameterFitter

    from retsupp.modeling.cv.fit_prf_cv import load_runs, make_fold_assignments
    from retsupp.modeling.cv.fit_prf_cv_roi import roi_voxels, sse_sst
    from retsupp.modeling.cv.fit_prf_cv_roi_hrf_joint import CANONICAL, fit_roi, flex_factory
    from retsupp.modeling.cv.sim_prf_size_floor import CONDITIONS
    from retsupp.modeling.fit_prf import HRF, MODEL_CFG, _adapt_m2_from_m1, build_model_factory, gd_fit_scheduled
    from retsupp.utils.data import Subject

    res, sd_low, sd_min = CONDITIONS[condition]
    sub = Subject(subject, Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'sub-{subject:02d}_fold-{fold}_{condition}_v1.tsv.gz'
    if out.exists():
        print(f'{out} exists, nothing to do')
        return
    t0 = time.time()
    folds = make_fold_assignments(sub)
    test_runs = sorted(folds[fold])
    train_runs = sorted(r for k, f in enumerate(folds) if k != fold for r in f)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask(session=1, run=sub.get_runs(1)[0]))
    masker.fit()
    vox, roi = roi_voxels(sub, masker)
    vox = vox[roi == 'V1']
    train, par, grid = load_runs(sub, masker, res, 'full', train_runs)
    test, tpar, _ = load_runs(sub, masker, res, 'full', test_runs)
    train, test = train[:, vox], test[:, vox]
    ok = (train.std(0) > 0) & np.isfinite(train).all(0) & np.isfinite(test).all(0)
    vox, train, test = vox[ok], train[:, ok], test[:, ok]
    print(f'{len(vox)} V1 voxels, loaded in {time.time() - t0:.0f} s', flush=True)

    f1 = build_model_factory(MODEL_CFG[1], grid, sd_min)
    fitter = ParameterFitter(f1(train, par), train, par)
    g = fitter.fit_grid(np.linspace(-3, 3, 12), np.linspace(-3, 3, 12), np.linspace(max(sd_low, sd_min + 1e-3), 4., 8),
                        [0], [1], use_correlation_cost=True)
    g = fitter.refine_baseline_and_amplitude(g, l2_alpha=0.001)
    sched = lambda m: MODEL_CFG[m]['schedule']
    m1 = gd_fit_scheduled(f1, train, par, g, 10000, sched(1)).drop(columns=['r2', 'theta', 'ecc'], errors='ignore')
    print(f'm1 done ({time.time() - t0:.0f} s)', flush=True)
    f2 = build_model_factory(MODEL_CFG[2], grid, sd_min)
    m2 = gd_fit_scheduled(f2, train, par, _adapt_m2_from_m1(m1), 10000, sched(2))
    m2 = m2.drop(columns=['r2', 'theta', 'ecc'], errors='ignore').reset_index(drop=True)
    print(f'm2 done ({time.time() - t0:.0f} s)', flush=True)
    ff = flex_factory(MODEL_CFG[4], grid, sd_min, 2., .1)
    flab = list(ff(train, par).parameter_labels)
    s1 = m2.assign(**CANONICAL)[flab]
    s2 = fit_roi(ff, train, par, s1, [p for p in flab if p not in HRF and p != 'baseline'], .005, n2, 200, 1e-6)
    s3 = fit_roi(ff, train, par, s2, [], .001, n3, 200, 1e-6)
    print(f'joint HRF done ({time.time() - t0:.0f} s)', flush=True)

    def r2(obs, pred):
        sse, sst = sse_sst(obs, pred)
        return 1 - sse / sst

    df = pd.DataFrame(dict(voxel_index=vox))
    for name, p in (('m1', m1), ('m2', m2), ('s3', s3)):
        for c in p.columns:
            df[f'{name}_{c}'] = p[c].to_numpy()
    df['r2_train'] = r2(train, np.asarray(ff(train, par).predict(parameters=s3)))
    df['r2_test'] = r2(test, np.asarray(ff(test, tpar).predict(parameters=s3)))
    df = df.assign(subject=subject, fold=fold, condition=condition, resolution=res, sd_grid_low=sd_low, sd_min=sd_min,
                   seconds=time.time() - t0)
    df.to_csv(out, sep='\t', index=False)
    print(f'median held-out R² {np.nanmedian(df.r2_test):.4f}; done ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--condition', default='paper50')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--output-subdir', default='prf_v1_resolution_2026-11-12')
    a = ap.parse_args()
    main(a.subject, a.fold, a.condition, a.bids_folder, a.output_subdir)
