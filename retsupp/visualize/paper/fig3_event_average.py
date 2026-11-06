"""Event-related mean time series at the voxel's ring location, data against the m0 and m7 predictions.

For a Figure 3 panel that a single voxel cannot make. Per subject and ROI, on the full-data nested fits
(m0_prf, m7_full):

1. Voxels: cross-run reliability > 0 and a pRF centre within 2° of a ring location (the nearest one is
   the voxel's location). No R² criterion, so the selection favours neither model.
2. Trials: every search onset, classed 2 × 3: whether the voxel's location is the run's HP location or an
   LP location, and what lands there on that trial (the target, the distractor or a neutral item).
3. The bar response is removed from everything: the prediction of m0 with the bar alone (m0's pRFs,
   items left out) is subtracted from the data, from m0 and from m7. What remains is the response to
   the search items (and noise); m0 predicts the same response whatever lands at the location, m7
   lets it depend on the gain fields.
4. Epochs from -2 to +12 TRs around each onset, % signal; averaged over trials per voxel, then over
   voxels. Trials come every few seconds, so neighbouring trials overlap in every epoch; they are
   balanced across the classes, which differ only in what lands at the voxel's location.

Writes ``notes/data/fig3_event_average/sub-XX_2026-11.tsv`` (one row per ROI × class × lag).

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
PRE, POST = 2, 12


def onsets(x):
    on = x > 0
    return np.flatnonzero(on & ~np.r_[False, on[:-1]])


def main(subject, rois, bids, out_dir):
    from braincoder.hrf import SPMHRFModel
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import bar_paradigm, sign_pattern_of
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
    sub = Subject(subject, bids)
    bold, par, ci, di, ti, grid, _, meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    par_bar = bar_paradigm(sub, meta, 50, 5.0)
    assert np.shape(par_bar) == np.shape(par), (np.shape(par_bar), np.shape(par))
    ring = get_ring_positions()
    hrf = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)
    rel_all = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
        (deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel_all = rel_all.groupby('voxel_index').reliability.mean()
    # Trials (search onsets, from the target indicator) and what lands at each ring location.
    trials = []
    for r in meta:
        sl = slice(r.start, r.start + r.n_T)
        t_on = sorted(set(np.concatenate([onsets(ti[sl, k]) for k in range(len(ring))]).tolist()))
        for t in t_on:
            if not PRE <= t < r.n_T - POST:
                continue
            tt = r.start + t
            for loc in range(len(ring)):
                hp = ci[tt, loc] > 0
                what = 'target' if ti[tt, loc] > 0 else 'distractor' if di[tt, loc] > 0 else 'neutral'
                trials.append((loc, f'{what}_{"hp" if hp else "lp"}', tt))
    tr = pd.DataFrame(trials, columns=['loc', 'cls', 't'])
    print(tr.groupby('cls').size().to_string(), flush=True)
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
        pred = {}
        for name, fit in fits.items():
            p = fit['fit_pars_per_fold'][0]
            model = build_model(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                                ring_positions=ring, hrf_model=hrf, sign_pattern=sign_pattern_of(name), item_scale='free')
            pp = p.reindex(columns=list(model.parameter_labels))
            pred[name] = np.asarray(model.predict(paradigm=par, parameters=pp))
            if name == 'm0_prf':
                pred['bar'] = np.asarray(model.predict(paradigm=par_bar, parameters=pp))
        p7 = fits['m7_full']['fit_pars_per_fold'][0]
        rel = rel_all.reindex(ids).to_numpy()
        dist = np.hypot(p7.x.to_numpy()[:, None] - ring[:, 0], p7.y.to_numpy()[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        ok = (rel > 0) & (dist.min(1) < 2)
        print(f'{roi}: {ok.sum()} of {len(ids)} voxels', flush=True)
        series = {'data': bold[:, ids] - pred['bar'], 'm0': pred['m0_prf'] - pred['bar'], 'm7': pred['m7_full'] - pred['bar']}
        for cls in sorted(tr.cls.unique()):
            acc = {k: np.zeros(len(lags)) for k in series}
            nv = ne = 0
            for loc in range(len(ring)):
                vox = np.flatnonzero(ok & (loc_v == loc))
                e = tr[(tr['loc'] == loc) & (tr.cls == cls)]
                if not len(vox) or e.empty:
                    continue
                idx = e.t.to_numpy()[:, None] + lags[None, :]
                for k, s in series.items():
                    acc[k] += s[:, vox][idx].mean(0).sum(1)
                nv += len(vox)
                ne += len(e)
            if nv == 0:
                continue
            for i, lag in enumerate(lags):
                rows.append(dict(subject=subject, roi=roi, cls=cls, lag=int(lag), t=float(lag) * 1.6,
                                 **{k: acc[k][i] / nv for k in series}, n_voxels=nv, n_trials=ne))
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
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir)
