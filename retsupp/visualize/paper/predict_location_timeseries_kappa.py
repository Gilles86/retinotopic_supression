"""Does the DoG centre/surround tilt κ predict the neighbour pattern of Fig. 3e and Fig. 6? Noise-free model predictions.

Per subject and ROI, the predicted BOLD of three full-data fits goes through exactly the voxel selection and bar
removal of ``export_location_timeseries.py`` (Fig. 3): voxels reliable across runs with an m7_full pRF centre within 2°
of an item location, minus m0's prediction for the bar alone, averaged per location. The time series are then
deconvolved like the data (``deconvolve_location_responses.py --relative``).

Variants (one output subdir each, ``<subdir>_<variant>``):

* ``m7``: the final m7_full fit (DoG gain-field model, sustained field on centre and surround alike).
* ``dogk``: the m8_full_dogk fit (one shared tilt κ: centre weight 1 + κ, surround 1 − κ).
* ``dogk0``: the same m8_full_dogk parameters with κ := 0, so ``dogk − dogk0`` is the part due to κ alone.

Also a readout for the decoded bar (Fig. 6), without decoding: per run and voxel the mean predicted bar drive
(prediction for the bar alone minus the voxel's baseline) of the V1–V3 voxels reliable across runs with a pRF centre
inside the bar aperture (eccentricity < 3.17°), by the quadrant of the pRF centre relative to the run's HP location.
Written to ``<subdir>_bar/sub-XX_barquadrant.tsv``.

    python -m retsupp.visualize.paper.predict_location_timeseries_kappa 3 \\
        --ladder-root <final DoG root> --dogk-root <dogcs root> --hrf-table <tsv>
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
EARLY = ('V1', 'V2', 'V3')
RELATIVE = {0: 'hp', 1: 'nextleft', 2: 'opposite', 3: 'nextright'}   # (quadrant - HP) mod 4, as deconvolve --relative


def load(root, model, subject, roi):
    pkl = Path(root) / model / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def main(subject, rois, bids, out_dir, subdir, ladder_root, dogk_root, hrf_table, trials_from):
    from retsupp.modeling.dog_centre_surround_field import DoGCentreSurroundField
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, build_model, get_ring_positions
    from retsupp.modeling.fit_cv_ladder_nested import bar_paradigm, sign_pattern_of
    from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table
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
    variants = ('m7', 'dogk', 'dogk0')
    outs = {v: Path(out_dir) / f'{subdir}_{v}' for v in variants + ('bar',)}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    for v in variants:   # same trials as the data
        shutil.copy(Path(trials_from) / f'sub-{subject:02d}_trials.tsv', outs[v] / f'sub-{subject:02d}_trials.tsv')
    hp_run = [int(np.argmax(ci[r.start])) for r in meta]

    rows, bars, info = {v: [] for v in variants}, [], []
    for roi in rois:
        try:
            m0 = load(ladder_root, 'm0_prf', subject, roi)
            m7 = load(ladder_root, 'm7_full', subject, roi)
            dk = load(dogk_root, 'm8_full_dogk', subject, roi)
        except FileNotFoundError as e:
            print('skip', roi, e, flush=True)
            continue
        ids = np.asarray(m0['voxel_ids'])
        assert np.array_equal(ids, np.asarray(m7['voxel_ids'])) and np.array_equal(ids, np.asarray(dk['voxel_ids']))
        hrf = roi_hrf_model(sub, roi, table)
        kw = dict(grid_coords=grid, paradigm=par, condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
                  ring_positions=ring, hrf_model=hrf)
        bar_model = build_model(**kw, sign_pattern=sign_pattern_of('m0_prf'), item_scale='free')
        p0 = m0['fit_pars_per_fold'][0]
        bar = np.asarray(bar_model.predict(paradigm=par_bar, parameters=p0.reindex(columns=list(bar_model.parameter_labels))))
        m7_model = build_model(**kw, sign_pattern=sign_pattern_of('m7_full'), item_scale='free')
        p7 = m7['fit_pars_per_fold'][0].reindex(columns=list(m7_model.parameter_labels))
        dk_model = DoGCentreSurroundField(grid_coordinates=grid, paradigm=par, hrf_model=hrf, condition_indicator=ci,
                                          dynamic_indicator=di, target_indicator=ti, ring_positions=ring,
                                          sign_pattern=sign_pattern_of('m8_full_dogk'), item_scale_mode='free')
        pk = dk['fit_pars_per_fold'][0].reindex(columns=list(dk_model.parameter_labels))
        pk0 = pk.copy()
        pk0['kappa'] = 0.0
        assert not (p7.isna().any().any() or pk.isna().any().any())
        models = {'m7': (m7_model, p7), 'dogk': (dk_model, pk), 'dogk0': (dk_model, pk0)}
        pred = {v: np.asarray(m.predict(paradigm=par, parameters=p)) for v, (m, p) in models.items()}
        kappa = float(pk['kappa'].iloc[0])
        info.append(dict(subject=subject, roi=roi, kappa=kappa, g_HP=float(pk['g_HP'].iloc[0]),
                         g_LP=float(pk['g_LP'].iloc[0]), sigma_AF=float(pk['sigma_AF'].iloc[0]),
                         sd_median=float(pk['sd'].median()),
                         srf_size_median=float(pk['srf_size'].median()) if 'srf_size' in pk else np.nan,
                         srf_amplitude_median=float(pk['srf_amplitude'].median()) if 'srf_amplitude' in pk else np.nan))

        # Fig. 3: item-location voxels (selection as export_location_timeseries).
        dist = np.hypot(p7.x.to_numpy()[:, None] - ring[:, 0], p7.y.to_numpy()[:, None] - ring[:, 1])
        loc_v = dist.argmin(1)
        rel_ok = rel_all.reindex(ids).to_numpy() > 0
        ok = rel_ok & (dist.min(1) < 2)
        for loc in range(len(ring)):
            vox = np.flatnonzero(ok & (loc_v == loc))
            if not len(vox):
                continue
            bar_m = bar[:, vox].mean(1)
            for v in variants:
                raw = pred[v][:, vox].mean(1)
                for r in meta:
                    tr = sub.get_tr(r.session, r.run)
                    k = np.arange(r.n_T)
                    rows[v].append(pd.DataFrame(dict(
                        subject=subject, roi=roi, loc=loc, n_voxels=len(vox), session=r.session, run=r.run,
                        hp=int(np.argmax(ci[r.start]) == loc), tr_index=k, time=(k + .5) * tr,
                        signal=raw[r.start + k] - bar_m[r.start + k], raw=raw[r.start + k], bar=bar_m[r.start + k])))

        # Fig. 6 readout: predicted bar drive per quadrant (no decoding).
        if roi in EARLY:
            ecc = np.hypot(p7.x.to_numpy(), p7.y.to_numpy())
            quad = np.argmin(np.abs(((np.arctan2(p7.y.to_numpy(), p7.x.to_numpy())[:, None]
                                      - np.arctan2(ring[:, 1], ring[:, 0])[None, :]) + np.pi) % (2 * np.pi) - np.pi), 1)
            sel = np.flatnonzero(rel_ok & (ecc < 3.17))
            for v in variants:
                m, p = models[v]
                drive = np.asarray(m.predict(paradigm=par_bar, parameters=p)) - p['baseline'].to_numpy()[None, :]
                for i, r in enumerate(meta):
                    d = drive[r.start:r.start + r.n_T, sel].mean(0)
                    for q in range(4):
                        vq = quad[sel] == q
                        if vq.any():
                            bars.append(dict(subject=subject, roi=roi, variant=v, session=r.session, run=r.run,
                                             quadrant=q, rel=RELATIVE[(q - hp_run[i]) % 4], n_voxels=int(vq.sum()),
                                             drive=float(d[vq].mean())))
        print(roi, f'κ = {kappa:+.3f}', 'done', flush=True)

    for v in variants:
        pd.concat(rows[v]).to_csv(outs[v] / f'sub-{subject:02d}_timeseries.tsv.gz', sep='\t', index=False,
                                  float_format='%.6g')
    pd.DataFrame(bars).to_csv(outs['bar'] / f'sub-{subject:02d}_barquadrant.tsv', sep='\t', index=False)
    pd.DataFrame(info).to_csv(outs['bar'] / f'sub-{subject:02d}_kappa_fit.tsv', sep='\t', index=False)
    print('Saved', {v: str(o) for v, o in outs.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11')
    ap.add_argument('--subdir', default='location_timeseries_kappa_pred')
    ap.add_argument('--ladder-root', required=True, help='final DoG ladder root (m0_prf, m7_full)')
    ap.add_argument('--dogk-root', required=True, help='root with the m8_full_dogk full-data fits')
    ap.add_argument('--hrf-table', required=True)
    ap.add_argument('--trials-from', required=True, help='location time-series folder of the data (its trial tables)')
    a = ap.parse_args()
    main(a.subject, a.rois, a.bids_folder, a.out_dir, a.subdir, a.ladder_root, a.dogk_root, a.hrf_table, a.trials_from)
