"""Parameter recovery for small pRFs: does the fitting procedure put a floor under V1 pRF size?

Simulates BOLD time courses of pRFs with known centre σ (0.1-1.4°) at V1-like eccentricities (0.75-3°), on the real
stimulus of one subject's training runs (fold 0, kind 'full' = bar + search items), with the canonical HRF (TR 1.6 s),
AR(1) noise scaled to a target R², and a fine "truth" pixel grid (``--truth-res``, default 200 px over ±5°, 0.05°/px)
so the fit never sees the grid that generated the data. Then fits them with the paper's chain
(``fit_prf_cv_roi`` + ``fit_prf_cv_roi_hrf_joint``):

    m1 Gaussian: grid (x, y in linspace(-3, 3, 12); σ in linspace(sd_grid_low, 4, 8)) + refine + GD (MODEL_CFG[1])
    m2 DoG from m1 (MODEL_CFG[2])
    joint HRF: stage 2 (one HRF shared by all voxels, plus baseline) and stage 3 (all free), 3000 iterations each

under one *condition* = (fitting resolution, lowest σ in the grid, ``sd_min`` of the model). The paper's settings are
resolution 50 (0.2°/px), grid σ from 1.0°, sd_min 0.2.

Truth families: 'gauss' (Gaussian pRF) and 'dog' (DoG, surround amplitude 0.07, surround 3 × centre: the median of
the real V1 fits).

Optional perturbations of the truth that the fit does not model: ``--eye-jitter`` (a random eye position per TR,
the whole stimulus shifted on the retina) and ``--truth-delay/--truth-dispersion`` (an HRF other than the canonical
one; the joint-HRF stage can absorb a change within the SPM family).

Output (never overwritten): ``<out_dir>/sim_<family>_r2-<r2>_<condition>.tsv`` with one row per simulated voxel:
true and recovered x, y, σ after m1, m2 and the joint-HRF stage, the fitted HRF, and R².

Usage::

    python -m retsupp.modeling.cv.sim_prf_size_floor --family dog --r2 .35 --condition paper50 \\
        --out-dir /shares/zne.uzh/gdehol/ds-retsupp/derivatives/sim_prf_size_floor_2026-11-12
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = {   # name: (fit resolution, lowest σ in the m1 grid, sd_min)
    'paper50': (50, 1.0, .2),
    'paper100': (100, 1.0, .2),
    'paper150': (150, 1.0, .2),
    'grid50': (50, .25, .2),
    'grid100': (100, .25, .2),
    'free50': (50, .15, .05),
    'free100': (100, .15, .05),
    'free150': (150, .15, .05),
}
TRUE_SD = [.1, .2, .3, .4, .5, .6, .8, 1.0, 1.4]
TRUE_ECC = [.75, 1.5, 2.25, 3.0]


def paradigm(sub, runs, res):
    pars, grid = [], None
    for ses, run in runs:
        p = sub.get_stimulus_with_distractors(session=ses, run=run, resolution=res, grid_radius=5.0,
                                              distractor_shape='rectangle', distractor_long_side=1.5,
                                              distractor_short_side=0.375).astype(np.float32)
        pars.append(p.reshape(p.shape[0], -1))
        if grid is None:
            gx, gy = sub.get_extended_grid_coordinates(resolution=res, session=ses, run=run, grid_radius=5.0)
            grid = np.stack((gx.ravel(), gy.ravel()), axis=1).astype(np.float32)
    return np.vstack(pars), grid


def true_parameters(family, n_rep, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for sd in TRUE_SD:
        for ecc in TRUE_ECC:
            for _ in range(n_rep):
                a = rng.uniform(0, 2 * np.pi)
                rows.append(dict(x=ecc * np.cos(a), y=ecc * np.sin(a), sd=sd, true_ecc=ecc))
    t = pd.DataFrame(rows)
    t['baseline'], t['amplitude'] = 0., 1.
    if family == 'dog':
        t['srf_amplitude'], t['srf_size'] = .07, 3.
    return t


def ar1_noise(rng, n_t, n_v, phi=.3):
    e = rng.standard_normal((n_t, n_v))
    for i in range(1, n_t):
        e[i] += phi * e[i - 1]
    return e / e.std(0, keepdims=True)


def jitter_paradigm(par, res, sd, rng):
    """Shift the stimulus of every TR by a random eye position (iid normal, SD ``sd`` degrees, whole pixels)."""
    if sd <= 0:
        return par
    px = 10. / (res - 1)
    out = par.reshape(-1, res, res).copy()
    shifts = np.round(rng.normal(0, sd, (len(out), 2)) / px).astype(int)
    for t, (dy, dx) in enumerate(shifts):
        out[t] = np.roll(out[t], (dy, dx), axis=(0, 1))
    return out.reshape(len(out), -1)


def main(family, r2_target, condition, out_dir, subject=23, truth_res=200, n_rep=6, seed=0, n2=3000, n3=3000,
         debug=False, eye_jitter=0., truth_delay=4.5, truth_dispersion=.75):
    from braincoder.hrf import SPMHRFModel
    from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF, GaussianPRF2DWithHRF
    from braincoder.optimize import ParameterFitter

    from retsupp.modeling.cv.fit_prf_cv import make_fold_assignments
    from retsupp.modeling.cv.fit_prf_cv_roi_hrf_joint import CANONICAL, fit_roi, flex_factory
    from retsupp.modeling.fit_prf import HRF, MODEL_CFG, _adapt_m2_from_m1, build_model_factory, gd_fit_scheduled
    from retsupp.utils.data import Subject

    res, sd_low, sd_min = CONDITIONS[condition]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = (f'_eye-{eye_jitter:g}' if eye_jitter > 0 else '') + (
        f'_hrf-{truth_delay:g}-{truth_dispersion:g}' if (truth_delay, truth_dispersion) != (4.5, .75) else '')
    out = out_dir / f'sim_{family}_r2-{r2_target:g}_{condition}{extra}.tsv'
    if out.exists():
        print(f'{out} exists, nothing to do')
        return
    t0 = time.time()
    sub = Subject(subject, Path('/shares/zne.uzh/gdehol/ds-retsupp') if Path('/shares').exists() else Path('/data/ds-retsupp'))
    runs = sorted(r for k, f in enumerate(make_fold_assignments(sub)) if k != 0 for r in f)
    if debug:
        runs = runs[:2]

    # Truth: fine grid, canonical HRF, AR(1) noise at the target R².
    truth = true_parameters(family, 1 if debug else n_rep, seed)
    par_t, grid_t = paradigm(sub, runs, truth_res)
    par_t = jitter_paradigm(par_t, truth_res, eye_jitter, np.random.default_rng(seed + 2))
    cls = DifferenceOfGaussiansPRF2DWithHRF if family == 'dog' else GaussianPRF2DWithHRF
    hrf = SPMHRFModel(tr=1.6, delay=truth_delay, dispersion=truth_dispersion)
    tm = cls(grid_coordinates=grid_t, paradigm=par_t, hrf_model=hrf)
    labels = list(tm.parameter_labels)
    signal = np.asarray(tm.predict(parameters=truth[labels].astype(np.float32)))
    del par_t, tm
    rng = np.random.default_rng(seed + 1)
    noise_sd = signal.std(0) * np.sqrt((1 - r2_target) / r2_target)
    data = (signal + ar1_noise(rng, *signal.shape) * noise_sd).astype(np.float32)
    print(f'{len(truth)} voxels, T = {data.shape[0]}; truth simulated ({time.time() - t0:.0f} s)', flush=True)

    par, grid = paradigm(sub, runs, res)
    n_iter = 50 if debug else None

    # m1: grid + refine + GD, as fit_prf.grid_fit but with the condition's σ floor in the grid.
    f1 = build_model_factory(MODEL_CFG[1], grid, sd_min)
    fitter = ParameterFitter(f1(data, par), data, par)
    g = fitter.fit_grid(np.linspace(-3, 3, 12), np.linspace(-3, 3, 12), np.linspace(max(sd_low, sd_min + 1e-3), 4., 8),
                        [0], [1], use_correlation_cost=True)
    g = fitter.refine_baseline_and_amplitude(g, l2_alpha=0.001)
    sched = lambda m: [(fx, lr, n_iter or it, atol) for fx, lr, it, atol in MODEL_CFG[m]['schedule']]
    m1 = gd_fit_scheduled(f1, data, par, g, 10000, sched(1)).drop(columns=['r2', 'theta', 'ecc'], errors='ignore')
    print(f'm1 done ({time.time() - t0:.0f} s)', flush=True)
    f2 = build_model_factory(MODEL_CFG[2], grid, sd_min)
    m2 = gd_fit_scheduled(f2, data, par, _adapt_m2_from_m1(m1), 10000, sched(2))
    m2 = m2.drop(columns=['r2', 'theta', 'ecc'], errors='ignore').reset_index(drop=True)
    print(f'm2 done ({time.time() - t0:.0f} s)', flush=True)

    # Joint HRF (fit_prf_cv_roi_hrf_joint, final2 recipe): stage 2 shared HRF + baseline, stage 3 all free.
    ff = flex_factory(MODEL_CFG[4], grid, sd_min, 2., .1)
    flab = list(ff(data, par).parameter_labels)
    s1 = m2.assign(**CANONICAL)[flab]
    s2 = fit_roi(ff, data, par, s1, [p for p in flab if p not in HRF and p != 'baseline'], .005, n_iter or n2, 50 if debug else 200, 1e-6)
    s3 = fit_roi(ff, data, par, s2, [], .001, n_iter or n3, 50 if debug else 200, 1e-6)
    print(f'joint HRF done ({time.time() - t0:.0f} s)', flush=True)

    def r2(pred):
        return 1 - ((data - pred) ** 2).sum(0) / ((data - data.mean(0)) ** 2).sum(0)

    df = truth.rename(columns={c: f'true_{c}' for c in ['x', 'y', 'sd', 'srf_amplitude', 'srf_size']})
    df = df.drop(columns=['baseline', 'amplitude'])
    for name, p in (('m1', m1), ('m2', m2), ('s3', s3)):
        for c in ['x', 'y', 'sd'] + (['srf_amplitude', 'srf_size'] if name != 'm1' else []):
            df[f'{name}_{c}'] = p[c].to_numpy()
    df['s3_r2'] = r2(np.asarray(ff(data, par).predict(parameters=s3)))
    df['s3_r2_truth_signal'] = 1 - ((signal - np.asarray(ff(data, par).predict(parameters=s3))) ** 2).sum(0) / (
        (signal - signal.mean(0)) ** 2).sum(0)
    df['hrf_delay'], df['hrf_dispersion'] = s3.hrf_delay.to_numpy(), s3.hrf_dispersion.to_numpy()
    df = df.assign(eye_jitter=eye_jitter, truth_delay=truth_delay, truth_dispersion=truth_dispersion, family=family, r2_target=r2_target, condition=condition, resolution=res, sd_grid_low=sd_low,
                   sd_min=sd_min, subject=subject, seconds=time.time() - t0)
    df.to_csv(out, sep='\t', index=False)
    print(df.groupby('true_sd')[['m1_sd', 'm2_sd', 's3_sd']].median().round(3).to_string())
    print(f'done ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', choices=['gauss', 'dog'], default='dog')
    ap.add_argument('--r2', type=float, default=.35)
    ap.add_argument('--condition', choices=list(CONDITIONS), default='paper50')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--subject', type=int, default=23)
    ap.add_argument('--truth-res', type=int, default=200)
    ap.add_argument('--n-rep', type=int, default=6)
    ap.add_argument('--eye-jitter', type=float, default=0., help='SD (deg) of a random eye position per TR in the truth')
    ap.add_argument('--truth-delay', type=float, default=4.5)
    ap.add_argument('--truth-dispersion', type=float, default=.75)
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    main(a.family, a.r2, a.condition, a.out_dir, a.subject, a.truth_res, a.n_rep, debug=a.debug, eye_jitter=a.eye_jitter,
         truth_delay=a.truth_delay, truth_dispersion=a.truth_dispersion)
