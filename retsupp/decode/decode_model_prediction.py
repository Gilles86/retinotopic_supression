"""Decode the noise-free prediction of the joint gain-field fit with the gain-free decoder of Fig. 6.

For one subject: the joint gain-field fit (``m7_full`` of the final DoG ladder, one HRF per subject × ROI, the
8-item paradigm with all its gain fields) predicts every run's BOLD in its voxels. Per run and ROI (V1–V3), the voxels of
``decode_final.py`` (the run's held-out fold pRFs, reliability > 0, BOLD SD ≤ 20%, at most 2000) that are also in the
joint fit are decoded twice with the same decoder (``free``: bar-only paradigm, ResidualFitter noise model fitted on the
real run's residuals under the fold pRF model, StimulusFitter L2 1.0, lr 0.5):

* ``measured``: the real BOLD of these voxels (the Fig. 6 decode restricted to the voxels the joint fit has);
* ``model``: the joint fit's prediction for the run, shifted per voxel to the real run's mean (noise-free).

Per decode, the map of Fig. 6c is computed as in ``fig6_collect.one_run``: per pixel, the mean decoded intensity over
the frames the true bar covers it, with the field rotated in 90° steps so HP sits upper right. Output (one file per
subject)::

    <bids>/derivatives/decoded_model_pred_2026-11/sub-XX_maps.npz
        drive (kind, roi, run, 50, 50), kinds = (measured, model), rois = (V1, V2, V3), runs as listed,
        n_voxels (roi, run), n_decoder_voxels (roi, run)

Usage::

    python -m retsupp.decode.decode_model_prediction 23 --ladder-root <final DoG root> --hrf-table <tsv>
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.decode.decode_final import (decode, fit_noise, fold_of, load_prfs, make_model, select_voxels, slice_run)
from retsupp.utils.data import Subject

KINDS = ('measured', 'model')


def rotated_drive(D, paradigm, ri):
    T = D.shape[0]
    D = D.reshape(T, -1)
    bar = paradigm.reshape(T, -1) > 0
    cnt = bar.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        drive = np.where(cnt > 0, (D * bar).sum(0) / cnt, np.nan)
    return drive[ri].reshape(50, 50)


def main(subject, rois, bids, ladder_root, hrf_table, source, joint_tag, max_voxels, max_bold_sd, l2_norm, learning_rate,
         max_iter, min_iter, resid_max_iter, out_root=None, runs=None, resolution=50, grid_radius=5.0):
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import sign_pattern_of
    from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table
    from retsupp.visualize.paper.fig6_collect import rotation_index

    t0 = time.time()
    sub = Subject(subject, bids_folder=bids)
    deriv = Path(bids) / 'derivatives'
    out = Path(out_root or deriv / 'decoded_model_pred_2026-11')
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f'sub-{subject:02d}_maps.npz'
    if fn.exists():
        print('exists', fn)
        return

    # The joint fit's noise-free prediction of every run (all runs concatenated in ``meta`` order).
    _bold, par, ci, di, ti, grid_items, _, meta = build_data_and_paradigm(sub, resolution=resolution,
                                                                          grid_radius=grid_radius, item_shape='rectangle')
    del _bold
    table = load_table(hrf_table)
    pred, ids_m7 = {}, {}
    for roi in rois:
        pkl = Path(ladder_root) / 'm7_full' / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
        if not pkl.exists():
            print('no joint fit for', roi)
            continue
        with open(pkl, 'rb') as f:
            fit = pickle.load(f)
        model = build_model(grid_coords=grid_items, paradigm=par, condition_indicator=ci, dynamic_indicator=di,
                            target_indicator=ti, ring_positions=get_ring_positions(),
                            hrf_model=roi_hrf_model(sub, roi, table), sign_pattern=sign_pattern_of('m7_full'),
                            item_scale='free')
        p7 = fit['fit_pars_per_fold'][0].reindex(columns=list(model.parameter_labels))
        assert not p7.isna().any().any()
        pred[roi] = np.asarray(model.predict(paradigm=par, parameters=p7), dtype=np.float32)
        ids_m7[roi] = np.asarray(fit['voxel_ids'])
        print(f'{roi}: joint-fit prediction {pred[roi].shape} ({time.time() - t0:.0f} s)', flush=True)
    span = {(int(m.session), int(m.run)): (int(m.start), int(m.stop)) for m in meta}

    with np.load(deriv / 'cleaned_bold_cache' / f'sub-{subject:02d}' / f'sub-{subject:02d}_kind-full_res-{resolution}.npz') as f:
        bold_all = f['bold']
    all_runs = [(s, r) for s in (1, 2) for r in sub.get_runs(s)]
    runs = [x for x in all_runs if runs is None or x in runs]
    drive = np.full((len(KINDS), len(rois), len(runs), 50, 50), np.nan, np.float32)
    n_vox = np.zeros((len(rois), len(runs)), int)
    n_dec = np.zeros((len(rois), len(runs)), int)
    for k, (session, run) in enumerate(runs):
        fold = fold_of(sub, session, run)
        hp = sub.get_hpd_locations().get((session, run))
        pars_all, roi_of = load_prfs(deriv, subject, fold, source, joint_tag)
        rel = pd.read_csv(deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}' /
                          f'sub-{subject:02d}_fold-{fold}_reliability.tsv.gz', sep='\t').set_index('voxel_index')
        bold_run = slice_run(bold_all, sub, session, run).astype(np.float32)
        paradigm = sub.get_bar_stimulus(session=session, run=run, resolution=resolution,
                                        grid_radius=grid_radius).astype(np.float32)
        T = paradigm.shape[0]
        gx, gy = sub.get_extended_grid_coordinates(resolution=resolution, session=session, run=run, grid_radius=grid_radius)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        ri = rotation_index(grid, hp)
        par_flat = paradigm.reshape(T, -1)
        par_df = pd.DataFrame(par_flat, index=pd.Index(np.arange(T), name='frame'))
        a, b = span[(session, run)]
        assert b - a >= T, (b - a, T)
        for j, roi in enumerate(rois):
            if roi not in pred:
                continue
            sel, _ = select_voxels(pars_all, roi_of, rel, roi, max_voxels, max_bold_sd)
            n_dec[j, k] = len(sel)
            col = {v: i for i, v in enumerate(ids_m7[roi])}
            sel = [v for v in sel if v in col]
            n_vox[j, k] = len(sel)
            if len(sel) < 20:
                print(f'ses-{session} run-{run} {roi}: {len(sel)} voxels, skipped', flush=True)
                continue
            real = bold_run[:, np.asarray(sel)]
            mod = pred[roi][a:a + T, [col[v] for v in sel]]
            mod = mod - mod.mean(0) + real.mean(0)          # noise-free, at the real run's mean
            pars = pars_all.loc[sel].reset_index(drop=True).astype(np.float32)
            m = make_model(source, grid, par_flat, None, pars)
            pars = pars[list(m.parameter_labels)]
            real_df = pd.DataFrame(real, index=par_df.index)
            omega, dof = fit_noise(make_model(source, grid, par_flat, real_df, pars), real_df, par_df, pars, resid_max_iter)
            for i, (kind, data) in enumerate((('measured', real), ('model', mod))):
                dfd = pd.DataFrame(data, index=par_df.index)
                D = decode(make_model(source, grid, par_flat, dfd, pars), dfd, pars, omega, dof, None, l2_norm,
                           learning_rate, max_iter, min_iter, (T, resolution, resolution))
                drive[i, j, k] = rotated_drive(D, paradigm, ri)
            print(f'ses-{session} run-{run} {roi}: {len(sel)} of {n_dec[j, k]} decoder voxels ({time.time() - t0:.0f} s)',
                  flush=True)
    np.savez_compressed(fn, drive=drive, kinds=np.array(KINDS), rois=np.array(rois), runs=np.array(runs),
                        n_voxels=n_vox, n_decoder_voxels=n_dec)
    print('Wrote', fn, f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--ladder-root', required=True)
    ap.add_argument('--hrf-table', required=True)
    ap.add_argument('--prf-source', default='jointhrf')
    ap.add_argument('--joint-tag', default='final2')
    ap.add_argument('--runs', nargs='+', default=None, help='session:run pairs (canary), e.g. 1:1')
    ap.add_argument('--max-voxels', type=int, default=2000)
    ap.add_argument('--max-bold-sd', type=float, default=20.)
    ap.add_argument('--l2-norm', type=float, default=1.0)
    ap.add_argument('--learning-rate', type=float, default=0.5)
    ap.add_argument('--max-n-iterations', type=int, default=1000)
    ap.add_argument('--min-n-iterations', type=int, default=200)
    ap.add_argument('--resid-max-iter', type=int, default=2000)
    ap.add_argument('--out-root', default=None)
    a = ap.parse_args()
    runs = [tuple(int(x) for x in s.split(':')) for s in a.runs] if a.runs else None
    main(a.subject, a.rois, a.bids_folder, a.ladder_root, a.hrf_table, a.prf_source, a.joint_tag, a.max_voxels,
         a.max_bold_sd, a.l2_norm, a.learning_rate, a.max_n_iterations, a.min_n_iterations, a.resid_max_iter,
         out_root=a.out_root, runs=runs)
