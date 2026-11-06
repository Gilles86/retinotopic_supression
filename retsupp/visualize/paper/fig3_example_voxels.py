"""Example voxel time courses for Figure 3: BOLD, m0 and m7 predictions, events and bar position.

Per candidate (subject, ROI): among the fitted voxels with cross-run reliability > 0 and a pRF centre
within 2° of a ring location, the one with the highest m7 training R². The window is the stretch of
``--window`` s of one run with the most target and distractor events at that ring location. Runs on
the cluster; writes ``notes/data/fig3_example_voxel_sub-XX_ROI_2026-11.tsv``.

Usage::

    python -m retsupp.visualize.paper.fig3_example_voxels 23 V3
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = ['af_prf_cv_ladder_nested_full_bounded_maxsd20', 'af_prf_cv_ladder_nested_full_bounded']


def load_fit(deriv, model, subject, roi):
    for root in ROOTS:
        pkl = deriv / root / model / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                return pickle.load(f), root
    raise FileNotFoundError(f'{model} sub-{subject} {roi}')


def main(subject, roi, bids, window, out_dir):
    from braincoder.hrf import SPMHRFModel
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import bar_paradigm, sign_pattern_of
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
    sub = Subject(subject, bids)
    bold, par, ci, di, ti, grid, _, meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    par_bar = bar_paradigm(sub, meta, 50, 5.0)
    ring = get_ring_positions()
    hrf = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)
    preds, ids, pars = {}, None, None
    for name in ('m0_prf', 'm7_full'):
        fit, root = load_fit(deriv, name, subject, roi)
        p = fit['fit_pars_per_fold'][0]
        ids = np.asarray(fit['voxel_ids'])
        model = build_model(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                            ring_positions=ring, hrf_model=hrf, sign_pattern=sign_pattern_of(name), item_scale='free')
        preds[name] = np.asarray(model.predict(paradigm=par, parameters=p.reindex(columns=list(model.parameter_labels))))
        pars = p
        print(name, root, preds[name].shape, flush=True)
    y = bold[:, ids]
    r2 = 1 - ((y - preds['m7_full']) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    rel = pd.concat([pd.read_csv(f, sep='\t') for f in sorted((deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel = rel.groupby('voxel_index').reliability.mean().reindex(ids).to_numpy()
    dist = np.hypot(pars.x.to_numpy()[:, None] - ring[:, 0], pars.y.to_numpy()[:, None] - ring[:, 1])
    near = dist.min(1)
    ok = (rel > 0) & (near < 2)
    if not ok.any():
        raise ValueError('no voxel passes')
    v = np.where(ok)[0][np.argmax(r2[ok])]
    loc = dist[v].argmin()
    n_win = int(window / 1.6)
    best = (-1, None, None)
    for r in meta:
        ev = (ti[r.start:r.start + r.n_T, loc] > 0).astype(float) + (di[r.start:r.start + r.n_T, loc] > 0)
        count = np.convolve(ev, np.ones(n_win), 'valid')
        if count.max() > best[0]:
            best = (count.max(), r, int(count.argmax()))
    _, r, start = best
    s, e = r.start + start, r.start + start + n_win
    gx, gy = grid[:, 0], grid[:, 1]
    b = np.asarray(par_bar)[s:e].reshape(e - s, -1)
    w = b.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        bx, by = (b @ gx) / w, (b @ gy) / w
    out = pd.DataFrame(dict(t=np.arange(n_win) * 1.6, bold=y[s:e, v], m0=preds['m0_prf'][s:e, v], m7=preds['m7_full'][s:e, v],
                            target_here=ti[s:e, loc], singleton_here=di[s:e, loc], target_any=ti[s:e].sum(1), singleton_any=di[s:e].sum(1),
                            bar_x=bx, bar_y=by))
    for k, val in dict(subject=subject, roi=roi, voxel_id=ids[v], x=pars.x.iloc[v], y=pars.y.iloc[v], sd=pars.sd.iloc[v], r2_m7=r2[v],
                       reliability=rel[v], ring_x=ring[loc, 0], ring_y=ring[loc, 1], hp=r.hp, session=r.session, run=r.run).items():
        out[k] = val
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f'fig3_example_voxel_sub-{subject:02d}_{roi}_2026-11.tsv'
    out.to_csv(path, sep='\t', index=False)
    print('Saved', path, f'voxel {ids[v]}, R² {r2[v]:.3f}, run {r.session}-{r.run}, HP {r.hp}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subject', type=int)
    ap.add_argument('roi')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--window', type=float, default=75.)
    ap.add_argument('--out-dir', default='notes/data')
    a = ap.parse_args()
    main(a.subject, a.roi, a.bids_folder, a.window, a.out_dir)
