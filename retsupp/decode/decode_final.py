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
* ``--prf-source dog_roihrf`` / ``dn_roihrf``: ``derivatives/prf_cv_roi_sharedhrf`` models 14 (DoG) and 16
  (divisive normalization, DN) of ``fit_prf_cv_roi_hrf.py``: one HRF per subject × ROI (the median of the
  fold's flexible-HRF training fits, m4 resp. m6), held fixed while the pRF was refit on the training runs.
* ``--prf-source gauss_roihrf``: the same for model 13 (Gaussian pRF; HRF from the m3 training fits).
* ``--prf-source gauss_jointhrf`` / ``dn_jointhrf``: as ``jointhrf`` for the Gaussian and the DN pRF (their own
  joint per-ROI HRF fits, same recipe and tag).
* ``--prf-source dn_canonical``: ``derivatives/prf_cv_roi`` model 5 (DN, canonical HRF).

All fits used ``sd_min = 0.2`` and ``SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)`` (jointhrf adds the
lower bounds), and the same model construction is used here (``SOURCES``). ``--check-prfs`` verifies that
for one subject × fold: it recomputes the held-out R² of the fold fit from this module's model and
compares it with the R² the fit wrote.

Voxels: the ROI's voxels with cross-run reliability > 0 on the fold's training runs and BOLD SD ≤
``--max-bold-sd`` (the outlier rule), at most ``--max-voxels`` (highest reliability first). With
``--selection relnull --min-r R``: instead, the ROI's voxels with cross-run reliability r > R and circular-shift
p <= ``--max-p`` (.01) on the fold's training runs (``modeling/cv/voxel_reliability_null.py --folds``) and mean
training-run BOLD SD <= ``--max-bold-sd``, at most ``--max-voxels`` (highest r first); outputs then go to
``<out-root>/<prf-source>_relnull_r<R>``. ``--rois`` also takes the pooled sets of ``POOLED`` (with
``--out-suffix _pooled`` they go to their own root). Decoder
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
# Pooled area sets (--rois mid high dorsal occtemp): the union of the members' voxels, each with its own fold pRF
# (and its area's HRF); selection per voxel as for single areas, the cap applies to the set.
POOLED = {'mid': ['V3AB', 'hV4'], 'high': ['LO', 'TO', 'VO', 'IPS'], 'dorsal': ['V3AB', 'IPS'],
          'occtemp': ['hV4', 'VO', 'LO', 'TO']}
QUADRANTS = {'upper_right': (1, 1), 'upper_left': (-1, 1), 'lower_left': (-1, -1), 'lower_right': (1, -1)}
# prf-source -> (derivatives folder, file tag, model class, flexible HRF?, HRF lower bounds?)
SOURCES = {
    'canonical': ('prf_cv_roi', 'model-2', 'dog', False, False),
    'jointhrf': ('prf_cv_roi_jointhrf', 'dog_{joint_tag}_stage-3', 'dog', True, True),
    'dog_roihrf': ('prf_cv_roi_sharedhrf', 'model-14', 'dog', True, False),
    'dn_roihrf': ('prf_cv_roi_sharedhrf', 'model-16', 'dn', True, False),
    'dn_canonical': ('prf_cv_roi', 'model-5', 'dn', False, False),
    'gauss_roihrf': ('prf_cv_roi_sharedhrf', 'model-13', 'gauss', True, False),
    'gauss_jointhrf': ('prf_cv_roi_jointhrf', 'gauss_{joint_tag}_stage-3', 'gauss', True, True),
    'dn_jointhrf': ('prf_cv_roi_jointhrf', 'dn_{joint_tag}_stage-3', 'dn', True, True),
}


def fold_of(sub, session, run, n_folds=3):
    from retsupp.modeling.cv.fit_prf_cv import make_fold_assignments
    for k, f in enumerate(make_fold_assignments(sub, n_folds=n_folds)):
        if (session, run) in f:
            return k
    raise KeyError(f'(ses-{session}, run-{run}) in no fold')


def load_prfs(deriv, subject, fold, source, joint_tag):
    """Per-voxel pRF parameters of the fold (rows = voxel_index of the ROI union) and their ROI labels."""
    folder, tag = SOURCES[source][:2]
    d = deriv / folder / f'sub-{subject:02d}'
    stem = f'sub-{subject:02d}_fold-{fold}_{tag.format(joint_tag=joint_tag)}'
    cv = pd.read_csv(d / f'{stem}_cvr2.tsv.gz', sep='\t')
    pars = pd.read_pickle(d / f'{stem}_pars.pkl.gz').reset_index(drop=True)
    pars = pars.drop(columns=['r2', 'theta', 'ecc'], errors='ignore')
    pars.index = cv.voxel_index.to_numpy()
    return pars, cv.set_index('voxel_index').roi


def make_model(source, grid, paradigm, data, parameters, sd_min=.2):
    from braincoder.hrf import SPMHRFModel
    from braincoder.models import (DifferenceOfGaussiansPRF2DWithHRF, DivisiveNormalizationGaussianPRF2DWithHRF,
                                   GaussianPRF2DWithHRF)
    _, _, kind, flex, bounded = SOURCES[source]
    hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75, **(dict(min_hrf_delay=2., min_dispersion=.1) if bounded else {}))
    cls = {'dog': DifferenceOfGaussiansPRF2DWithHRF, 'dn': DivisiveNormalizationGaussianPRF2DWithHRF,
           'gauss': GaussianPRF2DWithHRF}[kind]
    return cls(grid_coordinates=grid, paradigm=paradigm, hrf_model=hrf, data=data, parameters=parameters,
               flexible_hrf_parameters=flex, sd_min=sd_min)


def select_voxels(pars_all, roi_of, rel, roi, max_voxels, max_bold_sd):
    """The ROI's voxels with a complete fit, training reliability > 0 and BOLD SD <= max_bold_sd, at most
    ``max_voxels`` (highest reliability first). Returns (selected voxel_index, number of ROI voxels)."""
    ids = roi_of.index[roi_of.isin(POOLED.get(roi, [roi])).to_numpy()]
    p = pars_all.loc[ids]
    r = rel.reindex(ids)
    keep = p.notna().all(axis=1).to_numpy() & (r.reliability.to_numpy() > 0)
    if max_bold_sd is not None and 'sd_train' in r:
        keep &= r.sd_train.to_numpy() <= max_bold_sd
    sel = ids[keep]
    if len(sel) > max_voxels:
        sel = r.loc[sel].reliability.sort_values(ascending=False).index[:max_voxels]
    return sel, len(ids)


def select_voxels_relnull(pars_all, roi_of, tab, roi, max_voxels, max_bold_sd, min_r, max_p):
    """As ``select_voxels`` with the fold-wise reliability-null table: r > min_r, p <= max_p, sd <= max_bold_sd."""
    ids = roi_of.index[roi_of.isin(POOLED.get(roi, [roi])).to_numpy()]
    p = pars_all.loc[ids]
    t = tab.reindex(ids)
    keep = p.notna().all(axis=1).to_numpy() & (t.r.to_numpy() > min_r) & (t.p.to_numpy() <= max_p)
    if max_bold_sd is not None:
        keep &= t.sd.to_numpy() <= max_bold_sd
    sel = ids[keep]
    n_pass = len(sel)
    if len(sel) > max_voxels:
        sel = t.loc[sel].r.sort_values(ascending=False).index[:max_voxels]
    return sel, len(ids), n_pass


def fit_noise(model, data, par_df, pars, resid_max_iter):
    from braincoder.optimize import ResidualFitter
    rf = ResidualFitter(model=model, data=data, paradigm=par_df, parameters=pars)
    return rf.fit(max_n_iterations=resid_max_iter, progressbar=False, method='gauss')


def decode(model, data, pars, omega, dof, mask, l2_norm, learning_rate, max_iter, min_iter, shape):
    from braincoder.optimize import StimulusFitter
    sf = StimulusFitter(model=model, data=data, omega=omega, parameters=pars, dof=dof)
    fit = sf.fit(l2_norm=l2_norm, learning_rate=learning_rate, max_n_iterations=max_iter, min_n_iterations=min_iter,
                 progressbar=False, mask=mask)
    return np.asarray(fit.values if hasattr(fit, 'values') else fit, dtype=np.float32).reshape(shape)


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


def check_prfs(subject, fold, source, bids, joint_tag='final2', rois=('V1', 'V2', 'V3'), resolution=50):
    """Recompute the fold fit's held-out R² with ``make_model`` (full paradigm, as in the fit) and compare it with
    the R² the fit wrote; for the ``prf_cv_roi*`` sources also compare the predictions with those of the fitting
    code's own model factory. Prints per ROI: n voxels, corr and max |ΔR²|, median prediction correlation."""
    from retsupp.modeling.cv.fit_prf_cv import make_fold_assignments
    sub = Subject(subject, bids_folder=bids)
    deriv = Path(bids) / 'derivatives'
    folder, tag = SOURCES[source][:2]
    stem = f'sub-{subject:02d}_fold-{fold}_{tag.format(joint_tag=joint_tag)}'
    cv = pd.read_csv(deriv / folder / f'sub-{subject:02d}' / f'{stem}_cvr2.tsv.gz', sep='\t').set_index('voxel_index')
    pars_all, roi_of = load_prfs(deriv, subject, fold, source, joint_tag)
    test_runs = sorted(make_fold_assignments(sub)[fold])
    with np.load(deriv / 'cleaned_bold_cache' / f'sub-{subject:02d}' / f'sub-{subject:02d}_kind-full_res-{resolution}.npz') as f:
        bold_all = f['bold']
    factory = None
    if folder.startswith('prf_cv_roi') and folder != 'prf_cv_roi_jointhrf':
        from retsupp.modeling.fit_prf import MODEL_CFG, build_model_factory
        parent = {'model-2': 2, 'model-5': 5, 'model-13': 3, 'model-14': 4, 'model-16': 6}[tag]
        factory = lambda grid: build_model_factory(MODEL_CFG[parent], grid, .2)
    for roi in rois:
        ids = roi_of.index[(roi_of.to_numpy() == roi)]
        ids = ids[pars_all.loc[ids].notna().all(axis=1).to_numpy()]
        pars = pars_all.loc[ids].reset_index(drop=True).astype(np.float32)
        obs, pred, pred_ref = [], [], []
        for ses, r in test_runs:
            par = sub.get_stimulus_with_distractors(session=ses, run=r, resolution=resolution, grid_radius=5.0,
                                                    distractor_shape='rectangle', distractor_long_side=1.5,
                                                    distractor_short_side=0.375).astype(np.float32).reshape(N_TR, -1)
            gx, gy = sub.get_extended_grid_coordinates(resolution=resolution, session=ses, run=r, grid_radius=5.0)
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
            y = slice_run(bold_all, sub, ses, r)[:, np.asarray(ids)].astype(np.float32)
            m = make_model(source, grid, par, None, pars)
            pars = pars[list(m.parameter_labels)]
            obs.append(y)
            pred.append(np.asarray(m.predict(paradigm=par, parameters=pars)))
            if factory is not None:
                pred_ref.append(np.asarray(factory(grid)(y, par).predict(paradigm=par, parameters=pars)))
        obs, pred = np.vstack(obs), np.vstack(pred)
        r2 = 1 - ((obs - pred) ** 2).sum(0) / ((obs - obs.mean(0)) ** 2).sum(0)
        stored = cv.loc[ids, 'r2_test'].to_numpy()
        msg = (f'{source} sub-{subject:02d} fold-{fold} {roi}: {len(ids)} voxels; R² recomputed vs stored: '
               f'corr {np.corrcoef(r2, stored)[0, 1]:.5f}, max |Δ| {np.nanmax(np.abs(r2 - stored)):.2e}, '
               f'median {np.median(r2):.4f} vs {np.median(stored):.4f}')
        if pred_ref:
            pr = np.vstack(pred_ref)
            c = [np.corrcoef(pred[:, v], pr[:, v])[0, 1] for v in range(pred.shape[1]) if pred[:, v].std() > 0]
            msg += f'; prediction corr with fitting factory: median {np.median(c):.6f}, min {np.min(c):.6f}'
        print(msg, flush=True)


def main(subject, session, run, rois, bids, source, decoders, joint_tag, max_voxels, max_bold_sd, l2_norm, learning_rate,
         max_iter, min_iter, resid_max_iter, resolution=50, grid_radius=5.0, out_root=None, selection='loo', min_r=None,
         max_p=.01, out_suffix=''):
    sub = Subject(subject, bids_folder=bids)
    if run not in sub.get_runs(session):
        print(f'sub-{subject:02d} has no ses-{session} run-{run}; nothing to do')
        return
    deriv = Path(bids) / 'derivatives'
    if selection == 'relnull' and min_r is None:
        raise ValueError('--selection relnull needs --min-r')
    out_root = Path(out_root or deriv / 'decoded_final_2026-11') / (
        (source if selection == 'loo' else f'{source}_relnull_r{min_r:g}') + out_suffix)
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
    if selection == 'loo':
        rel = pd.read_csv(deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}' / f'sub-{subject:02d}_fold-{fold}_reliability.tsv.gz',
                          sep='\t').set_index('voxel_index')
    else:
        from retsupp.modeling.cv.voxel_reliability_null import fold_path
        rel = pd.read_csv(fold_path(bids, subject, fold), sep='\t').set_index('voxel_index')
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
        if selection == 'loo':
            sel, n_roi = select_voxels(pars_all, roi_of, rel, roi, max_voxels, max_bold_sd)
            n_pass = len(sel)
        else:
            sel, n_roi, n_pass = select_voxels_relnull(pars_all, roi_of, rel, roi, max_voxels, max_bold_sd, min_r, max_p)
        if len(sel) < 20:
            print(f'{roi}: only {len(sel)} voxels, skipped', flush=True)
            continue
        print(f'{roi}: {len(sel)} voxels ({selection}: {n_pass} pass of {n_roi})', flush=True)
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
            noise[kind] = fit_noise(m, data, par_df, pars, resid_max_iter)
            print(f'  residual fit ({kind} paradigm) {time.time() - t1:.0f} s', flush=True)
        for dec in decoders:
            out = out_root / dec / f'sub-{subject:02d}'
            if (out / f'readout_ses-{session}_run-{run}_roi-{roi}.tsv').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            kind = 'items' if dec.endswith('_items') else 'bar'
            omega, dof = noise[kind]
            mask = (pars_of[kind] > 0) if dec.startswith('outline') else None
            decoded = decode(mk(kind)[0], data, pars, omega, dof, mask, l2_norm, learning_rate, max_iter, min_iter,
                             (T, resolution, resolution))
            ro = readout(decoded, paradigm, grid, aperture, hp).assign(subject=subject, session=session, run=run, roi=roi,
                                                                       decoder=dec, source=source, fold=fold, hp_location=hp,
                                                                       n_voxels=len(sel),
                                                                       **({} if selection == 'loo' else dict(
                                                                           n_voxels_pass=n_pass,
                                                                           selection=f'relnull_r{min_r:g}')))
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
    ap.add_argument('--selection', choices=['loo', 'relnull'], default='loo',
                    help='loo: training-run reliability > 0 (default); relnull: fold-wise r > --min-r, p <= --max-p')
    ap.add_argument('--min-r', type=float, default=None)
    ap.add_argument('--max-p', type=float, default=.01)
    ap.add_argument('--out-suffix', default='', help="appended to the output folder name, e.g. '_pooled'")
    ap.add_argument('--check-prfs', action='store_true',
                    help='only verify the model construction against the fold fit of --session/--run (see check_prfs)')
    a = ap.parse_args()
    if a.check_prfs:
        fold = fold_of(Subject(a.subject, bids_folder=a.bids_folder), a.session, a.run)
        check_prfs(a.subject, fold, a.prf_source, a.bids_folder, a.joint_tag, a.rois)
        raise SystemExit
    main(a.subject, a.session, a.run, a.rois, a.bids_folder, a.prf_source, a.decoders, a.joint_tag, a.max_voxels, a.max_bold_sd,
         a.l2_norm, a.learning_rate, a.max_n_iterations, a.min_n_iterations, a.resid_max_iter, out_root=a.out_root,
         selection=a.selection, min_r=a.min_r, max_p=a.max_p, out_suffix=a.out_suffix)
