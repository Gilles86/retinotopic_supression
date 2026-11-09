"""Gain-free stimulus decoding for the paper (pre-specified test, ``notes/primary_test_2026-11.md``).

One task = one subject × held-out run; loops over ROIs (default V1, V2, V3) and two decoders:

* ``free``: every pixel of every frame is a free parameter (the validated ``decode.py`` setup).
* ``outline``: the true bar footprint of each frame is the mask (``StimulusFitter.fit(mask=...)``, 3-D
  per-frame mask): only pixels the bar covers are free, everything else is fixed at 0. The outline is
  given, the intensities are decoded.
* ``free_items`` / ``outline_items``: the same, but the search items are part of the stimulus as well
  (bar + all 8 items, ``Subject.get_stimulus_with_distractors``, the paradigm the pRFs were fitted on). The
  noise model is fitted on the residuals of that paradigm, and ``outline_items`` frees the bar and the item
  footprints. Item-evoked responses then have pixels of their own instead of being pushed into the bar;
  the readout stays inside the bar aperture, so it measures bar drive only.

pRFs come from the cross-validation fold that held this run out, so the decoded run never entered the
pRF fit:

* ``--prf-source canonical``: ``derivatives/prf_cv_roi`` fold fits of model 2 (DoG, canonical HRF).
* ``--prf-source jointhrf``: ``derivatives/prf_cv_roi_jointhrf`` stage-3 fits (DoG, one HRF per
  subject × ROI fitted jointly with the pRFs; ``fit_prf_cv_roi_hrf_joint.py``, tag ``--joint-tag``),
  flexible-HRF model with lower bounds delay 2 s and dispersion 0.1.

Voxels: the ROI's voxels with cross-run reliability > 0 on the fold's training runs and BOLD SD ≤
``--max-bold-sd`` (the outlier rule), at most ``--max-voxels`` (highest reliability first). Decoder
settings are the calibrated ones of ``decode.py``: bar-only paradigm, L2 = 1.0, learning rate 0.5,
ResidualFitter 2000 iterations, sd_min 0.2.

Readout (written next to the NPZ as a small TSV): per quadrant of the visual field, inside the bar
aperture (radius from the run's settings), the mean decoded intensity over the pixels the true bar covers
(``drive_bar``; the pre-specified "bar drive") and over all pixels (``mean_all``), over all frames. The
quadrant facing the run's HP location is labelled ``hp``.

Outputs::

    <bids>/derivatives/decoded_final_2026-11/<prf-source>/<decoder>/sub-XX/
        decoded_ses-S_run-R_roi-ROI.npz   decoded (T, R, R) float16, paradigm, grid
        readout_ses-S_run-R_roi-ROI.tsv   quadrant readout

Usage::

    python -m retsupp.decode.decode_final 23 --session 1 --run 1 --rois V1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject

N_TR = 258
QUADRANTS = {'upper_right': (1, 1), 'upper_left': (-1, 1), 'lower_left': (-1, -1), 'lower_right': (1, -1)}
SOURCES = {'canonical': ('prf_cv_roi', 'model-2'), 'jointhrf': ('prf_cv_roi_jointhrf', None)}


def fold_of(sub, session, run, n_folds=3):
    from retsupp.modeling.cv.fit_prf_cv import make_fold_assignments
    for k, f in enumerate(make_fold_assignments(sub, n_folds=n_folds)):
        if (session, run) in f:
            return k
    raise KeyError(f'(ses-{session}, run-{run}) in no fold')


def load_prfs(deriv, subject, fold, source, joint_tag):
    """Per-voxel pRF parameters of the fold (rows = voxel_index of the ROI union) and their model kind."""
    stem = f'sub-{subject:02d}_fold-{fold}'
    if source == 'canonical':
        d = deriv / 'prf_cv_roi' / f'sub-{subject:02d}'
        cv = pd.read_csv(d / f'{stem}_model-2_cvr2.tsv.gz', sep='\t')
        pars = pd.read_pickle(d / f'{stem}_model-2_pars.pkl.gz').reset_index(drop=True)
    else:
        d = deriv / 'prf_cv_roi_jointhrf' / f'sub-{subject:02d}'
        cv = pd.read_csv(d / f'{stem}_dog_{joint_tag}_stage-3_cvr2.tsv.gz', sep='\t')
        pars = pd.read_pickle(d / f'{stem}_dog_{joint_tag}_stage-3_pars.pkl.gz').reset_index(drop=True)
    pars = pars.drop(columns=['r2', 'theta', 'ecc'], errors='ignore')
    pars.index = cv.voxel_index.to_numpy()
    return pars, cv.set_index('voxel_index').roi


def make_model(source, grid, paradigm, data, parameters, sd_min=.2):
    from braincoder.hrf import SPMHRFModel
    from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
    if source == 'canonical':
        hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
        flex = False
    else:
        hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75, min_hrf_delay=2., min_dispersion=.1)
        flex = True
    return DifferenceOfGaussiansPRF2DWithHRF(grid_coordinates=grid, paradigm=paradigm, hrf_model=hrf, data=data,
                                             parameters=parameters, flexible_hrf_parameters=flex, sd_min=sd_min)


def slice_run(bold_all, sub, session, run):
    row = 0
    for s in (1, 2):
        for r in sub.get_runs(s):
            if (s, r) == (session, run):
                return bold_all[row * N_TR:(row + 1) * N_TR]
            row += 1
    raise KeyError((session, run))


def readout(decoded, paradigm, grid, aperture, hp):
    """Mean decoded intensity per quadrant inside the aperture: over true-bar pixels and over all pixels."""
    gx, gy = grid[:, 0], grid[:, 1]
    inside = np.hypot(gx, gy) <= aperture
    T = decoded.shape[0]
    dec = decoded.reshape(T, -1)
    bar = paradigm.reshape(T, -1) > 0
    rows = []
    for name, (sx, sy) in QUADRANTS.items():
        q = inside & (np.sign(gx) == sx) & (np.sign(gy) == sy)
        on = bar[:, q]
        rows.append(dict(quadrant=name, is_hp=name == hp, n_pixels=int(q.sum()), n_bar_pixel_frames=int(on.sum()),
                         drive_bar=float(dec[:, q][on].mean()) if on.any() else np.nan, mean_all=float(dec[:, q].mean())))
    return pd.DataFrame(rows)


def main(subject, session, run, rois, bids, source, decoders, joint_tag, max_voxels, max_bold_sd, l2_norm, learning_rate,
         max_iter, min_iter, resid_max_iter, resolution=50, grid_radius=5.0, out_root=None):
    from braincoder.optimize import ResidualFitter, StimulusFitter
    sub = Subject(subject, bids_folder=bids)
    if run not in sub.get_runs(session):
        print(f'sub-{subject:02d} has no ses-{session} run-{run}; nothing to do')
        return
    deriv = Path(bids) / 'derivatives'
    out_root = Path(out_root or deriv / 'decoded_final_2026-11') / source
    fold = fold_of(sub, session, run)
    hp = sub.get_hpd_locations().get((session, run))
    aperture = float(sub.get_experimental_settings(session, run)['radius_bar_aperture'])
    print(f'sub-{subject:02d} ses-{session} run-{run}: fold {fold}, HP {hp}, aperture {aperture}°', flush=True)

    todo = [(roi, dec) for roi in rois for dec in decoders
            if not (out_root / dec / f'sub-{subject:02d}' / f'readout_ses-{session}_run-{run}_roi-{roi}.tsv').exists()]
    if not todo:
        print('All outputs exist')
        return
    pars_all, roi_of = load_prfs(deriv, subject, fold, source, joint_tag)
    rel = pd.read_csv(deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}' / f'sub-{subject:02d}_fold-{fold}_reliability.tsv.gz',
                      sep='\t').set_index('voxel_index')
    cache = deriv / 'cleaned_bold_cache' / f'sub-{subject:02d}' / f'sub-{subject:02d}_kind-full_res-{resolution}.npz'
    t0 = time.time()
    with np.load(cache) as f:
        bold_all = f['bold']
    bold_run = slice_run(bold_all, sub, session, run).astype(np.float32)
    del bold_all
    print(f'BOLD run {bold_run.shape} ({time.time() - t0:.0f} s)', flush=True)

    paradigm = sub.get_bar_stimulus(session=session, run=run, resolution=resolution, grid_radius=grid_radius).astype(np.float32)
    T = paradigm.shape[0]
    gx, gy = sub.get_extended_grid_coordinates(resolution=resolution, session=session, run=run, grid_radius=grid_radius)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    pars_of = {'bar': paradigm}
    if any(d.endswith('_items') for d in decoders):
        items = sub.get_stimulus_with_distractors(
            session=session, run=run, resolution=resolution, grid_radius=grid_radius, distractor_shape='rectangle',
            distractor_long_side=1.5, distractor_short_side=0.375).astype(np.float32)
        pars_of['items'] = np.maximum(items, paradigm)

    for roi in rois:
        ids = roi_of.index[roi_of.to_numpy() == roi]
        p = pars_all.loc[ids]
        r = rel.reindex(ids)
        keep = p.notna().all(axis=1).to_numpy() & (r.reliability.to_numpy() > 0)
        if max_bold_sd is not None and 'sd_train' in r:
            keep &= r.sd_train.to_numpy() <= max_bold_sd
        sel = ids[keep]
        if len(sel) > max_voxels:
            sel = r.loc[sel].reliability.sort_values(ascending=False).index[:max_voxels]
        if len(sel) < 20:
            print(f'{roi}: only {len(sel)} voxels, skipped', flush=True)
            continue
        print(f'{roi}: {len(sel)} voxels (reliability > 0 of {len(ids)})', flush=True)
        pars = pars_all.loc[sel].reset_index(drop=True).astype(np.float32)
        data = pd.DataFrame(bold_run[:, np.asarray(sel)], index=pd.Index(np.arange(T), name='frame'))

        def mk(kind):
            par_flat = pars_of[kind].reshape(T, -1)
            return make_model(source, grid, par_flat, data, pars), pd.DataFrame(
                par_flat, index=pd.Index(np.arange(T), name='frame'))
        pars = pars[list(mk('bar')[0].parameter_labels)]
        noise = {}
        for kind in {('items' if d.endswith('_items') else 'bar') for d in decoders}:
            t1 = time.time()
            m, par_df = mk(kind)
            rf = ResidualFitter(model=m, data=data, paradigm=par_df, parameters=pars)
            noise[kind] = rf.fit(max_n_iterations=resid_max_iter, progressbar=False, method='gauss')
            print(f'  residual fit ({kind} paradigm) {time.time() - t1:.0f} s', flush=True)
        for dec in decoders:
            out = out_root / dec / f'sub-{subject:02d}'
            if (out / f'readout_ses-{session}_run-{run}_roi-{roi}.tsv').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            kind = 'items' if dec.endswith('_items') else 'bar'
            omega, dof = noise[kind]
            sf = StimulusFitter(model=mk(kind)[0], data=data, omega=omega, parameters=pars, dof=dof)
            mask = (pars_of[kind] > 0) if dec.startswith('outline') else None
            fit = sf.fit(l2_norm=l2_norm, learning_rate=learning_rate, max_n_iterations=max_iter, min_n_iterations=min_iter,
                         progressbar=False, mask=mask)
            decoded = np.asarray(fit.values if hasattr(fit, 'values') else fit, dtype=np.float32).reshape(T, resolution, resolution)
            ro = readout(decoded, paradigm, grid, aperture, hp).assign(subject=subject, session=session, run=run, roi=roi,
                                                                       decoder=dec, source=source, fold=fold, hp_location=hp,
                                                                       n_voxels=len(sel))
            np.savez_compressed(out / f'decoded_ses-{session}_run-{run}_roi-{roi}.npz', decoded=decoded.astype(np.float16),
                                paradigm=paradigm.astype(np.uint8), grid=grid, hp_location=np.array(str(hp)),
                                voxel_index=np.asarray(sel), dof=np.float32(dof if dof is not None else np.nan),
                                **({'paradigm_items': pars_of['items'].astype(np.float16)} if kind == 'items' else {}))
            ro.to_csv(out / f'readout_ses-{session}_run-{run}_roi-{roi}.tsv', sep='\t', index=False)
            hp_row = ro[ro.is_hp].iloc[0]
            print(f'  {dec}: {time.time() - t1:.0f} s; drive_bar HP {hp_row.drive_bar:.3f}, LP mean '
                  f'{ro[~ro.is_hp].drive_bar.mean():.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--session', type=int, required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--prf-source', choices=list(SOURCES), default='canonical')
    ap.add_argument('--joint-tag', default='final2')
    ap.add_argument('--decoders', nargs='+', choices=['free', 'outline', 'free_items', 'outline_items'], default=['free', 'outline'])
    ap.add_argument('--max-voxels', type=int, default=2000)
    ap.add_argument('--max-bold-sd', type=float, default=20.)
    ap.add_argument('--l2-norm', type=float, default=1.0)
    ap.add_argument('--learning-rate', type=float, default=0.5)
    ap.add_argument('--max-n-iterations', type=int, default=1000)
    ap.add_argument('--min-n-iterations', type=int, default=200)
    ap.add_argument('--resid-max-iter', type=int, default=2000)
    ap.add_argument('--out-root', default=None)
    a = ap.parse_args()
    main(a.subject, a.session, a.run, a.rois, a.bids_folder, a.prf_source, a.decoders, a.joint_tag, a.max_voxels, a.max_bold_sd,
         a.l2_norm, a.learning_rate, a.max_n_iterations, a.min_n_iterations, a.resid_max_iter, out_root=a.out_root)
