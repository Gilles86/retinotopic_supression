"""Recovery of the centre/surround tilt κ of the sustained HP field in the DoG gain-field model.

Can the data tell a sustained HP effect on the DoG pRF's centre (driving) pooling from one on its surround
(suppressive) pooling or on both (m7_full), at our effect size and noise? Per subject × ROI, from the final all-runs
DoG ladder fit (``--root``; frozen per-voxel pRFs and m7_full; HRF from ``--hrf-table``):

1. Truths (``dog_centre_surround_field``; centre weight 1 + κ, surround 1 − κ). ``none``: m7 with g_HP := g_LP.
   ``both`` (κ = 0, the m7 form): g_HP = g_LP + k·dS, dS the fitted m7 contrast (|dS| floored at ``--min-ds``).
   ``centre`` (κ = +1) and ``surround`` (κ = −1): g_LP as in m7, g_HP − g_LP set so that the RMS change of the
   predicted BOLD against the same κ without HP effect equals that of ``both``. Sizes k = ``--sizes``.
2. Noise: the real residuals under m7, phase-randomised per run with one set of phases for all voxels.
3. Refit as on the real data: m7 from the real m7 optimum (all shared terms), then the κ model from the m7 refit
   (κ = 0) and from κ = ±1 with half the gains; best kept. Per-voxel pRFs stay at the real fit.

Output (one row per truth × size × noise draw × fitted model, never overwritten):
``<out-dir>/sub-XX/sub-XX_roi-<ROI>_recovery.tsv`` with SSE, shared terms and the truth's values.

Usage::

    python -m retsupp.modeling.recover_dog_cs_field 3 --roi V3 --root <final DoG root> --hrf-table <tsv> \\
        --out-dir <bids>/derivatives/dog_cs_field_recovery_2026-11
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.decode.simulate_decode import phase_randomise
from retsupp.modeling.dog_centre_surround_field import DoGCentreSurroundField
from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, get_ring_positions, shared_pars_for
from retsupp.modeling.fit_cv_ladder_nested import KAPPA_STARTS, child_init, sign_pattern_of, shared_fit
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as DoGItems,
)
from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table as load_hrf_table
from retsupp.utils.data import Subject

REPORT = ['sigma_AF', 'g_HP', 'g_LP', 'kappa', 'item_scale']
TRUTH_KAPPA = {'both': 0.0, 'centre': 1.0, 'surround': -1.0}


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main(subject, roi, root, hrf_table, out_dir, sizes=(1.0, 4.0), n_noise=2, min_ds=0.02, seed=0,
         bids_folder='/shares/zne.uzh/gdehol/ds-retsupp'):
    out = Path(out_dir) / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_recovery.tsv'
    if out.exists():
        raise FileExistsError(out)
    with open(Path(root) / 'm7_full' / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl', 'rb') as f:
        d = pickle.load(f)
    m7 = d['fit_pars_per_fold'][0]
    voxel_ids = np.asarray(d['voxel_ids'])

    sub = Subject(subject, bids_folder)
    bold, par, ci, di, ti, grid, _masker, _run_meta = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    run_meta = _run_meta
    y = bold[:, voxel_ids]
    sp = sign_pattern_of('m7_full')
    kw = dict(grid_coordinates=grid, paradigm=par, hrf_model=roi_hrf_model(sub, roi, load_hrf_table(hrf_table)),
              condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
              ring_positions=get_ring_positions(), sign_pattern=sp, item_scale_mode='free')
    base_model = DoGItems(**kw)
    model = DoGCentreSurroundField(**kw)
    labels = list(model.parameter_labels)

    def predict(pars):
        return np.asarray(model.predict(paradigm=par, parameters=pars[labels]))

    p_real = m7.reindex(columns=labels)
    p_real['kappa'] = 0.0
    p7 = np.asarray(base_model.predict(paradigm=par, parameters=m7[list(base_model.parameter_labels)]))
    resid = y - predict(p_real)
    print(f'equality check: max |Δpred| vs m7 = {np.abs(predict(p_real) - p7).max():.3g}', flush=True)
    ds_fit = float(m7['g_HP'].iloc[0] - m7['g_LP'].iloc[0])
    ds_use = np.sign(ds_fit or -1.0) * max(abs(ds_fit), min_ds)

    truths = {('none', 0.0): p_real.assign(g_HP=p_real['g_LP'])}
    for k in sizes:
        p_both = p_real.assign(g_HP=p_real['g_LP'] + k * ds_use)
        target = rms(predict(p_both) - predict(truths[('none', 0.0)]))
        truths[('both', k)] = p_both
        for name in ('centre', 'surround'):
            p0 = p_real.assign(g_HP=p_real['g_LP'], kappa=TRUTH_KAPPA[name])
            ref = predict(p0)
            x = k * ds_use
            for _ in range(5):   # secant on the RMS change (close to linear in the HP contrast)
                r = rms(predict(p0.assign(g_HP=p0['g_LP'] + x)) - ref)
                x = x * target / max(r, 1e-12)
            truths[(name, k)] = p0.assign(g_HP=p0['g_LP'] + x)
        print(f'size {k}: dS both {k * ds_use:+.3f} (RMS {target:.4g}), centre '
              f'{float(truths[("centre", k)].g_HP.iloc[0] - truths[("centre", k)].g_LP.iloc[0]):+.3f}, surround '
              f'{float(truths[("surround", k)].g_HP.iloc[0] - truths[("surround", k)].g_LP.iloc[0]):+.3f}',
              flush=True)

    rng = np.random.default_rng(seed + 1000 * subject)
    noises = []
    for _ in range(n_noise):
        z = np.zeros_like(resid)
        for rm in run_meta:
            z[rm.start:rm.stop] = phase_randomise(resid[rm.start:rm.stop], rng)
        noises.append(z)

    free_m7 = [p for p in shared_pars_for(sp, 'free') if p != 'sigma_T_dyn']
    rows = []
    for (truth, k), p_true in truths.items():
        signal = predict(p_true)
        for r, z in enumerate(noises):
            y_sim = (signal + z).astype(np.float32)
            t0 = time.time()
            m7_hat, info = shared_fit(base_model, y_sim, par, m7.reindex(columns=list(base_model.parameter_labels)),
                                      free_m7, bounded=True, sigma_starts=(None,))
            fits = {'m7_full': (m7_hat, info['sse_final'])}
            init = child_init(m7_hat, model, 'm8_full_dogk')
            cands = [init]
            for kap in KAPPA_STARTS:
                cands.append(init.assign(kappa=kap, g_HP=init['g_HP'] / 2, g_LP=init['g_LP'] / 2))
            best = None
            for c in cands:
                pars, inf = shared_fit(model, y_sim, par, c, free_m7 + ['kappa'], bounded=True, sigma_starts=(None,))
                if best is None or inf['sse_final'] < best[1]:
                    best = (pars, inf['sse_final'])
            fits['m8_full_dogk'] = best
            for name, (pars, sse) in fits.items():
                row = dict(subject=subject, roi=roi, truth=truth, size=k, noise=r, model=name, sse=sse,
                           n_voxels=len(voxel_ids), ds_fit_real=ds_fit)
                row.update({c: float(pars[c].iloc[0]) if c in pars else 0.0 for c in REPORT})
                row.update({f'true_{c}': float(p_true[c].iloc[0]) for c in REPORT})
                rows.append(row)
            print(f'{truth} k={k} noise {r}: ' + ', '.join(f'{n} {s:.1f}' for n, (_, s) in fits.items())
                  + f" | κ̂ {float(best[0]['kappa'].iloc[0]):+.2f} | {time.time() - t0:.0f}s", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out.with_suffix('.partial.tsv'), sep='\t', index=False)
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    out.with_suffix('.partial.tsv').unlink(missing_ok=True)
    print(f'Written {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subject', type=int)
    ap.add_argument('--roi', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--hrf-table', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--sizes', type=float, nargs='+', default=[1.0, 4.0])
    ap.add_argument('--n-noise', type=int, default=2)
    ap.add_argument('--min-ds', type=float, default=0.02)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    a = ap.parse_args()
    main(a.subject, a.roi, a.root, a.hrf_table, a.out_dir, sizes=tuple(a.sizes), n_noise=a.n_noise,
         min_ds=a.min_ds, bids_folder=a.bids_folder)
