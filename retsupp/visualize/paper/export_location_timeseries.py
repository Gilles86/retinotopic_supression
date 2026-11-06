"""Per subject: the mean time series of the voxels on each item location (bar response removed), plus the trials.

Input for the deconvolution (``deconvolve_location_responses.py``, nideconv), which runs locally on these small tables.

Per ROI and ring location: voxels with cross-run reliability > 0 and a pRF centre within 2° of that location
(full-data m7_full pRFs; no R² criterion). Their BOLD minus the prediction of m0 for the bar alone, averaged over voxels.

Writes to ``<out-dir>/location_timeseries/``:
``sub-XX_timeseries.tsv.gz`` (subject, roi, loc, n_voxels, session, run, tr_index, time, signal) and
``sub-XX_trials.tsv`` (session, run, onset, loc, cls: target/distractor/neutral × hp/lp at that location).

Usage::

    python -m retsupp.visualize.paper.export_location_timeseries 23
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig3_example_voxels import load_fit

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
LOC = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}   # events' location codes -> ring channel (upper right, upper left, lower left, lower right)


def main(subject, rois, bids, out_dir):
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
    rel_all = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
        (deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel_all = rel_all.groupby('voxel_index').reliability.mean()
    out = Path(out_dir) / 'location_timeseries'
    out.mkdir(parents=True, exist_ok=True)

    trials = []
    for r in meta:
        ev = sub.get_onsets(r.session, r.run)
        ev = ev[ev.event_type == 'target'].sort_values('onset')
        hp = int(np.argmax(ci[r.start]))
        for _, e in ev.iterrows():
            t_ch, d_ch = LOC.get(e.target_location, -1), LOC.get(e.distractor_location, -1)
            for loc in range(len(ring)):
                what = 'target' if loc == t_ch else 'distractor' if loc == d_ch else 'neutral'
                trials.append(dict(session=r.session, run=r.run, onset=float(e.onset), loc=loc,
                                   cls=f'{what}_{"hp" if loc == hp else "lp"}'))
    pd.DataFrame(trials).to_csv(out / f'sub-{subject:02d}_trials.tsv', sep='\t', index=False)

    rows = []
    for roi in rois:
        try:
            fit = load_fit(deriv, 'm0_prf', subject, roi)[0]
            p7 = load_fit(deriv, 'm7_full', subject, roi)[0]['fit_pars_per_fold'][0]
        except FileNotFoundError as e:
            print('skip', roi, e, flush=True)
            continue
        ids = np.asarray(fit['voxel_ids'])
        p = fit['fit_pars_per_fold'][0]
        model = build_model(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                            ring_positions=ring, hrf_model=hrf, sign_pattern=sign_pattern_of('m0_prf'), item_scale='free')
        bar = np.asarray(model.predict(paradigm=par_bar, parameters=p.reindex(columns=list(model.parameter_labels))))
        dist = np.hypot(p7.x.to_numpy()[:, None] - ring[:, 0], p7.y.to_numpy()[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        ok = (rel_all.reindex(ids).to_numpy() > 0) & (dist.min(1) < 2)
        sig = bold[:, ids] - bar
        for loc in range(len(ring)):
            vox = np.flatnonzero(ok & (loc_v == loc))
            if not len(vox):
                continue
            mean = sig[:, vox].mean(1)
            for r in meta:
                tr = sub.get_tr(r.session, r.run)
                k = np.arange(r.n_T)
                rows.append(pd.DataFrame(dict(subject=subject, roi=roi, loc=loc, n_voxels=len(vox), session=r.session, run=r.run,
                                              tr_index=k, time=(k + .5) * tr, signal=mean[r.start + k])))
        print(roi, 'done', flush=True)
    pd.concat(rows).to_csv(out / f'sub-{subject:02d}_timeseries.tsv.gz', sep='\t', index=False, float_format='%.5g')
    print('Saved', out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir)
