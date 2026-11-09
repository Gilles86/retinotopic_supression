"""Model and parameter recovery for the locus of the sustained HP effect in the DN gain-field model.

Can the data tell an HP effect on the stimulus drive (input gain, m7_full) from one on the neural baseline b
(m8_full_dnb) or on the surround baseline d (m8_full_dnd), at our effect size and noise? Per subject × ROI, from
the all-runs DN ladder fit (``--root``; the frozen per-voxel pRFs and m7_full):

1. Truths. ``none``: m7 with the HP input gain removed (g_HP := g_LP). ``input``: the same plus an input gain at HP,
   g_HP = g_LP + k·dS (dS the fitted m7 contrast). ``b``: no input HP effect, b scaled at HP only
   (w_b > 0, θ = 0). ``d``: same for d (w_d < 0, θ = 0: d lower at HP, i.e. higher opposite). The weights are set
   so the RMS change of the predicted BOLD equals that of the input truth, so all three effects are equally large.
   Sizes k = ``--sizes`` (1 = the fitted dS). If the fitted dS is tiny, |dS| is floored at ``--min-ds``.
2. Noise. The real residuals under m7, phase-randomised per run with one set of phases for all voxels
   (``simulate_decode.phase_randomise``: keeps each voxel's spectrum and the cross-voxel covariance).
3. Refit, as on the real data: m7 from the real m7 optimum (all shared terms), then b-only, d-only (θ starts 0,
   π/4, −π/4, π/2) and both from their fits. Per-voxel pRFs stay at the real fit.

Output (one row per truth × size × noise draw × fitted model, never overwritten):
``<out-dir>/sub-XX/sub-XX_roi-<ROI>_recovery.tsv`` with SSE, shared terms and the truth's values.

Usage::

    python -m retsupp.modeling.recover_dn_baseline_field 3 --roi V3AB --root <all-runs DN ladder root> \\
        --out-dir <bids>/derivatives/dn_baseline_field_recovery_2026-11
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.hrf import SPMHRFModel
from retsupp.decode.simulate_decode import phase_randomise
from retsupp.modeling.dn_attention_field import DNDynamicAttentionFieldPRF2D
from retsupp.modeling.dn_baseline_field import DNBaselineField
from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm, get_ring_positions, shared_pars_for
from retsupp.modeling.fit_cv_ladder_nested import (THETA_STARTS, child_init, extra_terms, shared_fit,
                                                   sign_pattern_of)
from retsupp.utils.data import Subject

VARIANTS = {'m8_full_dnb': ('b',), 'm8_full_dnd': ('d',), 'm8_full_dnbd': ('b', 'd')}
REPORT = ['sigma_AF', 'g_HP', 'g_LP', 'w_b', 'w_d', 'theta_bd', 'item_scale']


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main(subject, roi, root, out_dir, sizes=(1.0, 4.0), n_noise=2, min_ds=0.02, seed=0,
         bids_folder='/shares/zne.uzh/gdehol/ds-retsupp'):
    out = Path(out_dir) / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_recovery.tsv'
    if out.exists():
        raise FileExistsError(out)
    with open(Path(root) / 'm7_full' / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl', 'rb') as f:
        d = pickle.load(f)
    m7 = d['fit_pars_per_fold'][0]
    voxel_ids = np.asarray(d['voxel_ids'])

    sub = Subject(subject, bids_folder)
    bold, par, ci, di, ti, grid, _masker, run_meta = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    y = bold[:, voxel_ids]
    sp = sign_pattern_of('m7_full')
    kw = dict(grid_coordinates=grid, paradigm=par, hrf_model=SPMHRFModel(tr=1.6), condition_indicator=ci,
              dynamic_indicator=di, target_indicator=ti, ring_positions=get_ring_positions(), sign_pattern=sp,
              item_scale_mode='free')
    base_model = DNDynamicAttentionFieldPRF2D(**kw)
    models = {name: DNBaselineField(targets=t, **kw) for name, t in VARIANTS.items()}
    full_model = models['m8_full_dnbd']
    labels = list(full_model.parameter_labels)

    def predict(pars):
        return np.asarray(full_model.predict(paradigm=par, parameters=pars[labels]))

    p_real = m7.reindex(columns=labels)
    p_real[['w_b', 'w_d', 'theta_bd']] = 0.0
    resid = y - predict(p_real)
    p_none = p_real.copy()
    p_none['g_HP'] = p_none['g_LP']
    pred_none = predict(p_none)
    ds_fit = float(m7['g_HP'].iloc[0] - m7['g_LP'].iloc[0])
    ds_use = np.sign(ds_fit or -1.0) * max(abs(ds_fit), min_ds)

    truths = {}
    for k in sizes:
        p_in = p_none.copy()
        p_in['g_HP'] = p_none['g_LP'] + k * ds_use
        target = rms(predict(p_in) - pred_none)
        truths[('input', k)] = p_in
        for term, sign in (('w_b', 1.0), ('w_d', -1.0)):
            w = 0.1 * sign
            for _ in range(4):   # secant on the RMS change, which is close to linear in w
                p = p_none.copy()
                p[term] = w
                r = rms(predict(p) - pred_none)
                w = w * target / max(r, 1e-12)
            p = p_none.copy()
            p[term] = w
            truths[(term[-1], k)] = p
        print(f'size {k}: input dS {k * ds_use:+.3f}, RMS {target:.4g}; '
              f'w_b {truths[("b", k)]["w_b"].iloc[0]:+.3f}, w_d {truths[("d", k)]["w_d"].iloc[0]:+.3f}', flush=True)
    truths[('none', 0.0)] = p_none

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
            fits = {}
            m7_hat, info = shared_fit(base_model, y_sim, par, m7.reindex(columns=list(base_model.parameter_labels)),
                                      free_m7, bounded=True, sigma_starts=(None,))
            fits['m7_full'] = (m7_hat, info['sse_final'])
            for name in VARIANTS:
                model = models[name]
                parent = 'm8_full_dnb' if name == 'm8_full_dnbd' else 'm7_full'
                init = child_init(fits[parent][0], model, name)
                cands = [init]
                if name == 'm8_full_dnbd':
                    cands.append(child_init(fits['m8_full_dnd'][0], model, name))
                else:
                    for th in THETA_STARTS:
                        alt = init.copy()
                        alt['theta_bd'] = th
                        cands.append(alt)
                free = free_m7 + extra_terms(name)
                best = None
                for c in cands:
                    pars, inf = shared_fit(model, y_sim, par, c, free, bounded=True, sigma_starts=(None,))
                    if best is None or inf['sse_final'] < best[1]:
                        best = (pars, inf['sse_final'])
                fits[name] = best
            for name, (pars, sse) in fits.items():
                row = dict(subject=subject, roi=roi, truth=truth, size=k, noise=r, model=name, sse=sse,
                           n_voxels=len(voxel_ids), ds_fit_real=ds_fit)
                row.update({c: float(pars[c].iloc[0]) if c in pars else 0.0 for c in REPORT})
                row.update({f'true_{c}': float(p_true[c].iloc[0]) for c in REPORT})
                rows.append(row)
            print(f'{truth} k={k} noise {r}: ' + ', '.join(f'{n} {s:.1f}' for n, (_, s) in fits.items())
                  + f' | {time.time() - t0:.0f}s', flush=True)
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
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--sizes', type=float, nargs='+', default=[1.0, 4.0])
    ap.add_argument('--n-noise', type=int, default=2)
    ap.add_argument('--min-ds', type=float, default=0.02)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    a = ap.parse_args()
    main(a.subject, a.roi, a.root, a.out_dir, sizes=tuple(a.sizes), n_noise=a.n_noise, min_ds=a.min_ds,
         bids_folder=a.bids_folder)
