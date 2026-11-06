"""Event-locked average of what the gain fields add, for a Figure 3 panel that a single voxel cannot make.

Per subject and ROI, on the full-data nested fits (m0_prf, m7_full):

1. Voxels: cross-run reliability > 0 and a pRF centre within 2° of a ring location (the nearest one is
   the voxel's location). No R² criterion, so the selection does not favour either model.
2. Events at the voxel's location: target onsets, and distractor onsets split by whether that location
   is HP in the run. As a control, the same events at the opposite ring location.
3. Epochs from -3 to +15 TRs around each onset of the residual of m0 (data − m0) and of what m7 adds
   (m7 − m0), each divided by the voxel's BOLD SD over all runs; averaged over events per voxel, then
   over voxels.

If the gain fields capture event-locked structure that m0 misses, the two curves agree in shape.

Writes ``notes/data/fig3_event_average/sub-XX_2026-11.tsv`` (one row per ROI × event type × place × lag).

Usage::

    python -m retsupp.visualize.paper.fig3_event_average 23 --rois V1 V2 V3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig3_example_voxels import load_fit

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
PRE, POST = 3, 15
OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}


def onsets(x):
    on = x > 0
    return np.flatnonzero(on & ~np.r_[False, on[:-1]])


def main(subject, rois, bids, out_dir):
    from braincoder.hrf import SPMHRFModel
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import sign_pattern_of
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
    sub = Subject(subject, bids)
    bold, par, ci, di, ti, grid, _, meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    ring = get_ring_positions()
    assert np.allclose(ring[OPPOSITE[0]], -ring[0]), 'ring order changed'
    hrf = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)
    rel_all = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
        (deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel_all = rel_all.groupby('voxel_index').reliability.mean()
    # Onsets per run, per ring location; HP flag per run and location.
    events = []
    for r in meta:
        sl = slice(r.start, r.start + r.n_T)
        for loc in range(len(ring)):
            hp = bool(ci[r.start, loc] > 0)
            for kind, ind in (('target', ti), ('distractor', di)):
                for t in onsets(ind[sl, loc]):
                    if PRE <= t < r.n_T - POST:
                        events.append((kind, loc, hp, r.start + t))
    ev = pd.DataFrame(events, columns=['kind', 'loc', 'hp', 't'])
    lags = np.arange(-PRE, POST + 1)
    rows = []
    for roi in rois:
        try:
            fits = {name: load_fit(deriv, name, subject, roi)[0] for name in ('m0_prf', 'm7_full')}
        except FileNotFoundError as e:
            print('skip', roi, e, flush=True)
            continue
        ids = np.asarray(fits['m7_full']['voxel_ids'])
        assert np.array_equal(ids, np.asarray(fits['m0_prf']['voxel_ids'])), 'm0 and m7 voxel sets differ'
        preds = {}
        for name, fit in fits.items():
            p = fit['fit_pars_per_fold'][0]
            model = build_model(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                                ring_positions=ring, hrf_model=hrf, sign_pattern=sign_pattern_of(name), item_scale='free')
            preds[name] = np.asarray(model.predict(paradigm=par, parameters=p.reindex(columns=list(model.parameter_labels))))
        p7 = fits['m7_full']['fit_pars_per_fold'][0]
        y = bold[:, ids]
        sd = y.std(0)
        resid = (y - preds['m0_prf']) / sd
        added = (preds['m7_full'] - preds['m0_prf']) / sd
        rel = rel_all.reindex(ids).to_numpy()
        dist = np.hypot(p7.x.to_numpy()[:, None] - ring[:, 0], p7.y.to_numpy()[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        ok = (rel > 0) & (dist.min(1) < 2)
        print(f'{roi}: {ok.sum()} of {len(ids)} voxels', flush=True)
        for place in ('here', 'opposite'):
            for kind, hp_sel in (('target', None), ('distractor', True), ('distractor', False)):
                acc_r, acc_a, nv, ne = np.zeros(len(lags)), np.zeros(len(lags)), 0, 0
                for loc in range(len(ring)):
                    vox = np.flatnonzero(ok & (loc_v == loc))
                    if not len(vox):
                        continue
                    ev_loc = loc if place == 'here' else OPPOSITE[loc]
                    e = ev[(ev.kind == kind) & (ev['loc'] == ev_loc)]
                    if hp_sel is not None:
                        e = e[e.hp == hp_sel]
                    if e.empty:
                        continue
                    idx = e.t.to_numpy()[:, None] + lags[None, :]                     # (events, lags)
                    acc_r += resid[idx][:, :, vox].mean(0).sum(1)
                    acc_a += added[idx][:, :, vox].mean(0).sum(1)
                    nv += len(vox)
                    ne += len(e)
                if nv == 0:
                    continue
                label = kind if hp_sel is None else f'{kind}_{"hp" if hp_sel else "lp"}'
                for lag, a, b in zip(lags, acc_r / nv, acc_a / nv):
                    rows.append(dict(subject=subject, roi=roi, event=label, place=place, lag=int(lag), t=float(lag) * 1.6,
                                     data_minus_m0=a, m7_minus_m0=b, n_voxels=nv, n_events=ne))
    out = Path(out_dir) / 'fig3_event_average'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'sub-{subject:02d}_2026-11.tsv'
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)
    print('Saved', path, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='notes/data')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir)
