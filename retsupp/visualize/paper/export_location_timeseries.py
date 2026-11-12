"""Per subject: the mean time series of the voxels on each item location (bar response removed), plus the trials.

Input for the deconvolution (``deconvolve_location_responses.py``, nideconv), which runs locally on these small tables.

Per ROI and ring location: voxels with cross-run reliability > 0 and a pRF centre within 2° of that location
(full-data m7_full pRFs; no R² criterion). Their BOLD minus the prediction of m0 for the bar alone, averaged over voxels.

Writes to ``<out-dir>/location_timeseries/``:
``sub-XX_timeseries.tsv.gz`` (subject, roi, loc, n_voxels, session, run, hp, tr_index, time, signal = raw − bar, raw = the
voxels' mean BOLD, bar = m0's prediction for the bar alone, mean over the same voxels; hp: the location is the run's HP) and
``sub-XX_trials.tsv`` (session, run, onset, loc, cls: target/distractor/neutral × hp/lp at that location).

``--root <ladder root> [fallback roots]`` (another voxel selection, e.g. ``..._relnull_r0.2_roihrf_polished``, which has only m4_TDS
and m7_full): voxels, pRF centres and the bar prediction all come from that root's m7_full (all-data fit), the bar
prediction with every gain set to 0 (the m0_prf model with m7_full's pRFs and HRF table). ``--no-reliability-filter``
drops the reliability > 0 criterion, for roots whose selection already is a reliability criterion. Without these
options nothing changes.

Usage::

    python -m retsupp.visualize.paper.export_location_timeseries 23
    python -m retsupp.visualize.paper.export_location_timeseries 23 \
        --root af_prf_cv_ladder_nested_full_bounded_maxsd20_relnull_r0.2_roihrf_polished --no-reliability-filter \
        --hrf-table /shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv \
        --subdir location_timeseries_relnull_r0.2_2026-11-12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.fig3_example_voxels import load_fit

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
LOC = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}   # events' location codes -> ring channel (upper right, upper left, lower left, lower right)


def main(subject, rois, bids, out_dir, subdir='location_timeseries', distance_bins=None, nearest_item_bins=None,
         ladder_roots=None, hrf_table=None, root=None, reliability_filter=True):
    from braincoder.hrf import SPMHRFModel
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import GAINS, bar_paradigm, sign_pattern_of
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
    sub = Subject(subject, bids)
    bold, par, ci, di, ti, grid, _, meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    par_bar = bar_paradigm(sub, meta, 50, 5.0)
    ring = get_ring_positions()
    hrf = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)
    if ladder_roots:      # the final fits: their pRFs and their per-ROI HRF for the bar prediction
        import retsupp.visualize.paper.fig3_example_voxels as fev
        fev.ROOTS = list(ladder_roots)
    if hrf_table:
        from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table
        table = load_table(hrf_table)
    if reliability_filter:
        rel_all = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
            (deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
        rel_all = rel_all.groupby('voxel_index').reliability.mean()
    else:   # the root's own selection: every voxel it fitted
        rel_all = pd.Series(dtype=float)
    out = Path(out_dir) / subdir
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
    if nearest_item_bins:
        # Every search onset once, as one event type: the display drives all eight item positions at the same time.
        for r in meta:
            ev = sub.get_onsets(r.session, r.run)
            for on in ev[ev.event_type == 'target'].onset:
                trials.append(dict(session=r.session, run=r.run, onset=float(on), loc=-1, cls='search'))
    pd.DataFrame(trials).to_csv(out / f'sub-{subject:02d}_trials.tsv', sep='\t', index=False)

    rows = []
    for roi in rois:
        try:
            if root:
                fit = load_root_fit(deriv, root, 'm7_full', subject, roi)
                p7 = fit['fit_pars_per_fold'][0]
            else:
                fit = load_fit(deriv, 'm0_prf', subject, roi)[0]
                p7 = load_fit(deriv, 'm7_full', subject, roi)[0]['fit_pars_per_fold'][0]
        except FileNotFoundError as e:
            print('skip', roi, e, flush=True)
            continue
        ids = np.asarray(fit['voxel_ids'])
        p = fit['fit_pars_per_fold'][0]
        if root:   # bar alone: m7_full's pRFs, every gain at 0 (= the m0_prf model)
            p = p.copy()
            p[[g for g in GAINS if g in p.columns]] = 0.
        if not reliability_filter:
            rel_all = pd.Series(1., index=ids)
        if hrf_table:
            hrf = roi_hrf_model(sub, roi, table)
        model = build_model(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                            ring_positions=ring, hrf_model=hrf, sign_pattern=sign_pattern_of('m0_prf'), item_scale='free')
        bar = np.asarray(model.predict(paradigm=par_bar, parameters=p.reindex(columns=list(model.parameter_labels))))
        dist = np.hypot(p7.x.to_numpy()[:, None] - ring[:, 0], p7.y.to_numpy()[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        ok = (rel_all.reindex(ids).to_numpy() > 0) & (dist.min(1) < 2)
        if nearest_item_bins:
            # Every reliable voxel, binned by the distance of its pRF centre from the nearest of the eight items.
            rel_ok = rel_all.reindex(ids).to_numpy() > 0
            ang = np.deg2rad(np.arange(8) * 45.)
            items8 = 4. * np.stack([np.cos(ang), np.sin(ang)], 1)
            near = np.hypot(p7.x.to_numpy()[:, None] - items8[:, 0], p7.y.to_numpy()[:, None] - items8[:, 1]).min(1)
            for lo_, hi_ in zip(nearest_item_bins[:-1], nearest_item_bins[1:]):
                vox = np.flatnonzero(rel_ok & (near >= lo_) & (near < hi_))
                if len(vox) < 5:
                    continue
                sig = (bold[:, ids[vox]] - bar[:, vox]).mean(1)
                for r in meta:
                    tr = sub.get_tr(r.session, r.run)
                    k = np.arange(r.n_T)
                    rows.append(pd.DataFrame(dict(subject=subject, roi=roi, loc=-1, dbin=lo_, n_voxels=len(vox),
                                                  session=r.session, run=r.run, tr_index=k, time=(k + .5) * tr,
                                                  signal=sig[r.start + k])))
            print(roi, 'done', flush=True)
            continue
        if distance_bins:
            # Every reliable voxel, binned by the distance of its pRF centre from each item location in turn.
            rel_ok = rel_all.reindex(ids).to_numpy() > 0
            for loc in range(len(ring)):
                for lo_, hi_ in zip(distance_bins[:-1], distance_bins[1:]):
                    vox = np.flatnonzero(rel_ok & (dist[:, loc] >= lo_) & (dist[:, loc] < hi_))
                    if len(vox) < 5:
                        continue
                    raw = bold[:, ids[vox]].mean(1)
                    bar_m = bar[:, vox].mean(1)
                    for r in meta:
                        tr = sub.get_tr(r.session, r.run)
                        k = np.arange(r.n_T)
                        rows.append(pd.DataFrame(dict(subject=subject, roi=roi, loc=loc, dbin=lo_, n_voxels=len(vox),
                                                      session=r.session, run=r.run, hp=int(np.argmax(ci[r.start]) == loc),
                                                      tr_index=k, time=(k + .5) * tr, signal=raw[r.start + k] - bar_m[r.start + k],
                                                      raw=raw[r.start + k], bar=bar_m[r.start + k])))
            print(roi, 'done', flush=True)
            continue
        for loc in range(len(ring)):
            vox = np.flatnonzero(ok & (loc_v == loc))
            if not len(vox):
                continue
            raw = bold[:, ids[vox]].mean(1)
            bar_m = bar[:, vox].mean(1)
            for r in meta:
                tr = sub.get_tr(r.session, r.run)
                k = np.arange(r.n_T)
                rows.append(pd.DataFrame(dict(subject=subject, roi=roi, loc=loc, n_voxels=len(vox), session=r.session, run=r.run,
                                              hp=int(np.argmax(ci[r.start]) == loc), tr_index=k, time=(k + .5) * tr,
                                              signal=raw[r.start + k] - bar_m[r.start + k], raw=raw[r.start + k],
                                              bar=bar_m[r.start + k])))
        print(roi, 'done', flush=True)
    pd.concat(rows).to_csv(out / f'sub-{subject:02d}_timeseries.tsv.gz', sep='\t', index=False, float_format='%.5g')
    print('Saved', out, flush=True)


def load_root_fit(deriv, roots, model, subject, roi):
    """The fit from the first of ``roots`` that has it."""
    import pickle
    for root in ([roots] if isinstance(roots, str) else roots):
        pkl = Path(root if Path(root).is_absolute() else deriv / root) / model / f'sub-{subject:02d}' / \
            f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
        if pkl.exists():
            print(roi, 'from', root, flush=True)
            with open(pkl, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(f'{model} sub-{subject} {roi} in {roots}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    ap.add_argument('--subdir', default='location_timeseries')
    ap.add_argument('--nearest-item-bins', type=float, nargs='+', default=None,
                    help='Bin edges (deg) of pRF-centre distance from the nearest of the eight items; one search-onset event.')
    ap.add_argument('--distance-bins', type=float, nargs='+', default=None,
                    help='Bin edges (deg) of pRF-centre distance from each item location; writes one series per bin.')
    ap.add_argument('--ladder-root', nargs='+', default=None,
                    help='ladder roots for the m0/m7 fits (first hit wins); default the earlier canonical-HRF fits')
    ap.add_argument('--hrf-table', default=None, help='one HRF per subject x ROI for the bar prediction (with --ladder-root)')
    ap.add_argument('--root', nargs='+', default=None,
                    help='ladder root(s) (name under derivatives or path; first hit wins): voxels, pRFs and the bar '
                         'prediction (gains at 0) from m7_full; replaces --ladder-root')
    ap.add_argument('--no-reliability-filter', action='store_true',
                    help='keep every voxel of the fit (no cross-run reliability > 0 criterion)')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir, a.subdir, a.distance_bins, a.nearest_item_bins, a.ladder_root,
         a.hrf_table, a.root, not a.no_reliability_filter)
