"""Noise-free predictions of the refitted (polished) joint gain-field fit through the Fig. 3 pipeline, per HP term.

Voxel selection, bar removal and units exactly as the measured refit series of Fig. 3
(``export_location_timeseries.py --root <polished root>``): voxels reliable across runs (r > 0) whose m7_full pRF
centre lies within 2° of an item location, each assigned to its nearest location; signal = the model's prediction
minus the prediction for the bar alone (m7_full's pRFs with every gain at 0), averaged per location. The trial tables
are copied from the measured series, so the deconvolution (``deconvolve_location_responses.py``) is the same.

Variants (one output subdir each, ``<subdir>_<variant>``), all from the joint fit m7_full of ``--root`` unless noted:

* ``nohp``: no HP terms: sustained g_HP, g_LP := (g_HP + 3·g_LP)/4 and the distractor transient g_HP_dyn, g_LP_dyn :=
  their mean weighted by the distractor pulses at HP vs LP in that subject's runs (as ``predict_location_timeseries_terms``).
* ``S``: sustained HP − LP only (dS as fitted; the distractor transient HP − LP, dD, equalised as in nohp).
* ``SD``: the joint fit as fitted (dS and dD; m7_full has no target transient HP − LP term, dT).
* ``SDT`` / ``SDT0`` (only with ``--dt-root``): m8_full_dT (m7 plus dT) of that root, as fitted / with the target
  split equalised (weighted by the target pulses at HP vs LP). That root's pRFs are the ones held fixed (not polished).

Also ``<subdir>_SD/sub-XX_overlap.tsv``: per ROI and item location, voxel counts: the Fig. 3 selection (reliable, pRF
centre within 2°, nearest location) and pRF overlap (reliable, distance from the pRF centre to the item's edge
< FWHM/2 of the pRF centre, item radius 0.75° = size_stimuli/2; a voxel can overlap more than one item).

    python -m retsupp.visualize.paper.predict_location_timeseries_polished 3 --root <polished root> \\
        --hrf-table <tsv> --trials-from <location_timeseries_polished_2026-11-12> [--dt-root <reliable_roihrf root>]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.predict_location_timeseries_kappa import ROIS, load
from retsupp.visualize.paper.predict_location_timeseries_terms import equalise

ITEM_RADIUS = 0.75   # deg, size_stimuli / 2
FWHM = 2 * np.sqrt(2 * np.log(2))


def main(subject, rois, bids, out_dir, subdir, root, hrf_table, trials_from, dt_root=None):
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import GAINS, bar_paradigm, sign_pattern_of
    from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table
    from retsupp.modeling.target_split import TargetSplitDoG
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
    root = Path(root if Path(root).is_absolute() else deriv / root)
    sub = Subject(subject, bids)
    _bold, par, ci, di, ti, grid, _, meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0,
                                                                    item_shape='rectangle')
    del _bold
    par_bar = bar_paradigm(sub, meta, 50, 5.0)
    ring = get_ring_positions()
    table = load_table(hrf_table)
    rel_all = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
        (deriv / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel_all = rel_all.groupby('voxel_index').reliability.mean()
    w_d = float((np.asarray(di) * np.asarray(ci)).sum() / np.asarray(di).sum())
    w_t = float((np.asarray(ti) * np.asarray(ci)).sum() / np.asarray(ti).sum())
    variants = ('nohp', 'S', 'SD') + (('SDT', 'SDT0') if dt_root else ())
    outs = {v: Path(out_dir) / f'{subdir}_{v}' for v in variants}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(trials_from) / f'sub-{subject:02d}_trials.tsv', o / f'sub-{subject:02d}_trials.tsv')

    rows, info, overlap = {v: [] for v in variants}, [], []
    for roi in rois:
        try:
            m7 = load(root, 'm7_full', subject, roi)
        except FileNotFoundError as e:
            print('skip', roi, e, flush=True)
            continue
        m8 = None
        if dt_root:   # m8_full_dT is not complete: SDT / SDT0 only where it exists
            try:
                m8 = load(dt_root, 'm8_full_dT', subject, roi)
            except FileNotFoundError as e:
                print('no SDT for', roi, e, flush=True)
        ids = np.asarray(m7['voxel_ids'])
        hrf = roi_hrf_model(sub, roi, table)
        kw = dict(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                  ring_positions=ring, hrf_model=hrf)
        m7_model = build_model(**kw, sign_pattern=sign_pattern_of('m7_full'), item_scale='free')
        p7 = m7['fit_pars_per_fold'][0].reindex(columns=list(m7_model.parameter_labels))
        assert not p7.isna().any().any()
        # Bar alone: m7_full's pRFs with every gain at 0 (= the m0_prf model), as export_location_timeseries --root.
        bar_model = build_model(**kw, sign_pattern=sign_pattern_of('m0_prf'), item_scale='free')
        p0 = m7['fit_pars_per_fold'][0].copy()
        p0[[g for g in GAINS if g in p0.columns]] = 0.
        bar = np.asarray(bar_model.predict(paradigm=par_bar, parameters=p0.reindex(columns=list(bar_model.parameter_labels))))
        nodyn = equalise(p7, 'g_HP_dyn', 'g_LP_dyn', w_d)
        params = {'nohp': (m7_model, equalise(nodyn, 'g_HP', 'g_LP', .25)), 'S': (m7_model, nodyn), 'SD': (m7_model, p7)}
        g = lambda p, c: float(p[c].iloc[0])   # noqa: E731
        row = dict(subject=subject, roi=roi, w_d=w_d, w_t=w_t, dS=g(p7, 'g_HP') - g(p7, 'g_LP'),
                   dD=g(p7, 'g_HP_dyn') - g(p7, 'g_LP_dyn'), g_T=g(p7, 'g_T_dyn'), g_D_lp=g(p7, 'g_LP_dyn'),
                   item_scale=g(p7, 'item_scale'))
        if m8 is not None:
            if not np.array_equal(ids, np.asarray(m8['voxel_ids'])):
                raise ValueError(f'{roi}: m8_full_dT voxels differ from the polished m7_full')
            m8_model = TargetSplitDoG(grid_coordinates=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di,
                                      target_indicator=ti, ring_positions=ring, hrf_model=hrf,
                                      sign_pattern=sign_pattern_of('m8_full_dT'), item_scale_mode='free')
            p8 = m8['fit_pars_per_fold'][0].reindex(columns=list(m8_model.parameter_labels))
            assert not p8.isna().any().any()
            params['SDT'] = (m8_model, p8)
            params['SDT0'] = (m8_model, equalise(p8, 'g_T_HP_dyn', 'g_T_dyn', w_t))
            row.update(dT=g(p8, 'g_T_HP_dyn') - g(p8, 'g_T_dyn'), dS_m8=g(p8, 'g_HP') - g(p8, 'g_LP'),
                       dD_m8=g(p8, 'g_HP_dyn') - g(p8, 'g_LP_dyn'))
        info.append(row)
        pred = {v: np.asarray(m.predict(paradigm=par, parameters=p)) for v, (m, p) in params.items()}

        dist = np.hypot(p7.x.to_numpy()[:, None] - ring[:, 0], p7.y.to_numpy()[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        rel_ok = rel_all.reindex(ids).to_numpy() > 0
        ok = rel_ok & (dist.min(1) < 2)
        half = FWHM * p7.sd.to_numpy() / 2
        for loc in range(len(ring)):
            ov = rel_ok & (dist[:, loc] - ITEM_RADIUS < half)
            overlap.append(dict(subject=subject, roi=roi, loc=loc, n_fig3=int((ok & (loc_v == loc)).sum()),
                                n_overlap=int(ov.sum()), n_overlap_within2=int((ov & (dist[:, loc] < 2)).sum()),
                                n_reliable=int(rel_ok.sum()), n_fit=len(ids),
                                fwhm_median_fig3=float(np.median(2 * half[ok & (loc_v == loc)]))
                                if (ok & (loc_v == loc)).any() else np.nan))
            vox = np.flatnonzero(ok & (loc_v == loc))
            if not len(vox):
                continue
            bar_m = bar[:, vox].mean(1)
            for v in pred:
                raw = pred[v][:, vox].mean(1)
                for r in meta:
                    tr = sub.get_tr(r.session, r.run)
                    k = np.arange(r.n_T)
                    rows[v].append(pd.DataFrame(dict(
                        subject=subject, roi=roi, loc=loc, n_voxels=len(vox), session=r.session, run=r.run,
                        hp=int(np.argmax(ci[r.start]) == loc), tr_index=k, time=(k + .5) * tr,
                        signal=raw[r.start + k] - bar_m[r.start + k], raw=raw[r.start + k], bar=bar_m[r.start + k])))
        print(roi, 'done', flush=True)

    for v in variants:
        if not rows[v]:
            continue
        pd.concat(rows[v]).to_csv(outs[v] / f'sub-{subject:02d}_timeseries.tsv.gz', sep='\t', index=False,
                                  float_format='%.6g')
    pd.DataFrame(info).to_csv(outs['SD'] / f'sub-{subject:02d}_terms.tsv', sep='\t', index=False)
    pd.DataFrame(overlap).to_csv(outs['SD'] / f'sub-{subject:02d}_overlap.tsv', sep='\t', index=False)
    print('Saved', {v: str(o) for v, o in outs.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    ap.add_argument('--subdir', default='location_timeseries_polished_pred_2026-11-13')
    ap.add_argument('--root', required=True, help='refitted ladder root with m7_full (name under derivatives or path)')
    ap.add_argument('--dt-root', default=None, help='root with m8_full_dT (same voxels) for the SDT / SDT0 variants')
    ap.add_argument('--hrf-table', required=True)
    ap.add_argument('--trials-from', required=True, help='location time-series folder of the data (its trial tables)')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir, a.subdir, a.root, a.hrf_table, a.trials_from, a.dt_root)
