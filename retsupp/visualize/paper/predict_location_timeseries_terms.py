"""Which HP − LP term carries the model's predicted reduction at HP in Fig. 3? Noise-free predictions per term.

Same voxel selection and bar removal as ``predict_location_timeseries_kappa.py`` / ``export_location_timeseries.py``
(voxels reliable across runs with an m7_full pRF centre within 2° of an item location, minus m0's bar-only prediction,
averaged per location); the time series are then deconvolved like the data (``deconvolve_location_responses.py
--relative``). Variants (one output subdir each, ``<subdir>_<variant>``):

* ``nohp``: m7_full with every HP − LP difference removed: sustained g_HP, g_LP := (g_HP + 3·g_LP)/4 (the mean over the
  four locations) and distractor transient g_HP_dyn, g_LP_dyn := their mean weighted by the distractor pulses at HP vs
  LP in that subject's runs. The reference: it should predict no location effect.
* ``S``: m7_full with only the sustained HP − LP difference (the transient distractor gains equalised as in nohp).
* ``SD``: m7_full as fitted (sustained HP − LP and the distractor transient HP − LP, dD).
* ``SDT``: m8_full_dT as fitted (m7 plus a target transient gain at HP, g_T_HP_dyn: dT = g_T_HP_dyn − g_T_dyn).
* ``SDT0``: the same m8_full_dT parameters with the target split removed (g_T_HP_dyn, g_T_dyn := their mean weighted by
  the target pulses at HP vs LP), so ``SDT − SDT0`` is the part due to dT alone.

    python -m retsupp.visualize.paper.predict_location_timeseries_terms 3 \\
        --ladder-root <final DoG root> --hrf-table <tsv> --trials-from <location_timeseries_final2>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.predict_location_timeseries_kappa import ROIS, load

VARIANTS = ('nohp', 'S', 'SD', 'SDT', 'SDT0')


def equalise(p, a, b, w_a):
    """Set parameters a and b both to w_a·a + (1 − w_a)·b."""
    p = p.copy()
    m = w_a * p[a] + (1 - w_a) * p[b]
    p[a], p[b] = m, m
    return p


def main(subject, rois, bids, out_dir, subdir, ladder_root, hrf_table, trials_from):
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import bar_paradigm, sign_pattern_of
    from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table
    from retsupp.modeling.target_split import TargetSplitDoG
    from retsupp.utils.data import Subject
    deriv = Path(bids) / 'derivatives'
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
    # Share of the distractor (target) pulse mass at the HP location: the weights of the equalised gains.
    w_d = float((np.asarray(di) * np.asarray(ci)).sum() / np.asarray(di).sum())
    w_t = float((np.asarray(ti) * np.asarray(ci)).sum() / np.asarray(ti).sum())
    outs = {v: Path(out_dir) / f'{subdir}_{v}' for v in VARIANTS}
    for v, o in outs.items():
        o.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(trials_from) / f'sub-{subject:02d}_trials.tsv', o / f'sub-{subject:02d}_trials.tsv')

    rows, info = {v: [] for v in VARIANTS}, []
    for roi in rois:
        try:
            m0 = load(ladder_root, 'm0_prf', subject, roi)
            m7 = load(ladder_root, 'm7_full', subject, roi)
            m8 = load(ladder_root, 'm8_full_dT', subject, roi)
        except FileNotFoundError as e:
            print('skip', roi, e, flush=True)
            continue
        ids = np.asarray(m0['voxel_ids'])
        assert np.array_equal(ids, np.asarray(m7['voxel_ids'])) and np.array_equal(ids, np.asarray(m8['voxel_ids']))
        hrf = roi_hrf_model(sub, roi, table)
        kw = dict(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                  ring_positions=ring, hrf_model=hrf)
        bar_model = build_model(**kw, sign_pattern=sign_pattern_of('m0_prf'), item_scale='free')
        p0 = m0['fit_pars_per_fold'][0]
        bar = np.asarray(bar_model.predict(paradigm=par_bar, parameters=p0.reindex(columns=list(bar_model.parameter_labels))))
        m7_model = build_model(**kw, sign_pattern=sign_pattern_of('m7_full'), item_scale='free')
        p7 = m7['fit_pars_per_fold'][0].reindex(columns=list(m7_model.parameter_labels))
        m8_model = TargetSplitDoG(grid_coordinates=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di,
                                  target_indicator=ti, ring_positions=ring, hrf_model=hrf,
                                  sign_pattern=sign_pattern_of('m8_full_dT'), item_scale_mode='free')
        p8 = m8['fit_pars_per_fold'][0].reindex(columns=list(m8_model.parameter_labels))
        assert not (p7.isna().any().any() or p8.isna().any().any())
        nodyn = equalise(p7, 'g_HP_dyn', 'g_LP_dyn', w_d)
        params = {'nohp': (m7_model, equalise(nodyn, 'g_HP', 'g_LP', .25)), 'S': (m7_model, nodyn), 'SD': (m7_model, p7),
                  'SDT': (m8_model, p8), 'SDT0': (m8_model, equalise(p8, 'g_T_HP_dyn', 'g_T_dyn', w_t))}
        pred = {v: np.asarray(m.predict(paradigm=par, parameters=p)) for v, (m, p) in params.items()}
        g = lambda p, c: float(p[c].iloc[0])   # noqa: E731
        info.append(dict(subject=subject, roi=roi, w_d=w_d, w_t=w_t, dS=g(p7, 'g_HP') - g(p7, 'g_LP'),
                         dD=g(p7, 'g_HP_dyn') - g(p7, 'g_LP_dyn'), dT=g(p8, 'g_T_HP_dyn') - g(p8, 'g_T_dyn'),
                         dS_m8=g(p8, 'g_HP') - g(p8, 'g_LP'), dD_m8=g(p8, 'g_HP_dyn') - g(p8, 'g_LP_dyn')))

        dist = np.hypot(p7.x.to_numpy()[:, None] - ring[:, 0], p7.y.to_numpy()[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        ok = (rel_all.reindex(ids).to_numpy() > 0) & (dist.min(1) < 2)
        for loc in range(len(ring)):
            vox = np.flatnonzero(ok & (loc_v == loc))
            if not len(vox):
                continue
            bar_m = bar[:, vox].mean(1)
            for v in VARIANTS:
                raw = pred[v][:, vox].mean(1)
                for r in meta:
                    tr = sub.get_tr(r.session, r.run)
                    k = np.arange(r.n_T)
                    rows[v].append(pd.DataFrame(dict(
                        subject=subject, roi=roi, loc=loc, n_voxels=len(vox), session=r.session, run=r.run,
                        hp=int(np.argmax(ci[r.start]) == loc), tr_index=k, time=(k + .5) * tr,
                        signal=raw[r.start + k] - bar_m[r.start + k], raw=raw[r.start + k], bar=bar_m[r.start + k])))
        print(roi, 'done', flush=True)

    for v in VARIANTS:
        pd.concat(rows[v]).to_csv(outs[v] / f'sub-{subject:02d}_timeseries.tsv.gz', sep='\t', index=False,
                                  float_format='%.6g')
    pd.DataFrame(info).to_csv(outs['SD'] / f'sub-{subject:02d}_terms.tsv', sep='\t', index=False)
    print('Saved', {v: str(o) for v, o in outs.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    ap.add_argument('--subdir', default='location_timeseries_terms_pred_2026-11-11')
    ap.add_argument('--ladder-root', required=True, help='final DoG ladder root (m0_prf, m7_full, m8_full_dT)')
    ap.add_argument('--hrf-table', required=True)
    ap.add_argument('--trials-from', required=True, help='location time-series folder of the data (its trial tables)')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir, a.subdir, a.ladder_root, a.hrf_table, a.trials_from)
