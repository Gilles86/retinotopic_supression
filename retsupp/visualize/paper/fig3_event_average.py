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
4. Epochs from -4 to +20 s around each exact onset, binned at 0.4 s: onsets are jittered against the
   TR grid (1.6 s), so each bin collects the samples whose time from onset falls in it. % signal; averaged
   over trials per voxel, then over voxels. Trials come every few seconds, so neighbouring trials overlap in every epoch; they are
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
T0, T1, BIN = -4.0, 20.0, 0.4          # epoch and bin width (s): onsets are jittered against the TR grid
N_BINS = int(round((T1 - T0) / BIN))
LOC = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}   # events' location codes -> ring channel (upper right, upper left, lower left, lower right)


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
    # Trials: exact search onsets (s) from the events, and what lands at each ring location.
    trials = []
    for r in meta:
        ev = sub.get_onsets(r.session, r.run)
        ev = ev[ev.event_type == 'target'].sort_values('onset')
        hp = int(np.argmax(ci[r.start]))
        tr_s = sub.get_tr(r.session, r.run)
        for _, e in ev.iterrows():
            t_ch = LOC.get(e.target_location, -1)
            d_ch = LOC.get(e.distractor_location, -1)
            for loc in range(len(ring)):
                what = 'target' if loc == t_ch else 'distractor' if loc == d_ch else 'neutral'
                trials.append((loc, f'{what}_{"hp" if loc == hp else "lp"}', r.start, r.n_T, float(e.onset), tr_s))
    tr = pd.DataFrame(trials, columns=['loc', 'cls', 'start', 'n_T', 'onset', 'tr'])
    print(tr.groupby('cls').size().to_string(), flush=True)
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
            num = {k: np.zeros(N_BINS) for k in series}
            den = np.zeros(N_BINS)
            nv = ne = 0
            for loc in range(len(ring)):
                vox = np.flatnonzero(ok & (loc_v == loc))
                e = tr[(tr['loc'] == loc) & (tr.cls == cls)]
                if not len(vox) or e.empty:
                    continue
                acc = {k: np.zeros(N_BINS) for k in series}
                cnt = np.zeros(N_BINS)
                for row in e.itertuples():
                    ks = np.arange(row.n_T)
                    tau = (ks + .5) * row.tr - row.onset
                    sel = (tau >= T0) & (tau < T1)
                    b = ((tau[sel] - T0) // BIN).astype(int)
                    for k, sr in series.items():
                        np.add.at(acc[k], b, sr[row.start + ks[sel]][:, vox].mean(1))
                    np.add.at(cnt, b, 1)
                good = cnt > 0
                for k in series:
                    num[k][good] += len(vox) * acc[k][good] / cnt[good]
                den[good] += len(vox)
                nv += len(vox)
                ne += len(e)
            if nv == 0:
                continue
            for b in range(N_BINS):
                if den[b] == 0:
                    continue
                rows.append(dict(subject=subject, roi=roi, cls=cls, lag=b, t=T0 + (b + .5) * BIN,
                                 **{k: num[k][b] / den[b] for k in series}, n_voxels=nv, n_trials=ne))
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
