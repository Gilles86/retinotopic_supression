"""Is the sustained LP gain identified once the pRFs are free? Profile likelihoods and the profiled Hessian.

For one subject × area × gain-field model (m4_TDS, m7_full; DoG, bounded, all runs; the voxels and starts of the
reference polished root), with the joint Levenberg-Marquardt fit of ``joint_lm`` (pRFs + shared terms):

``dump``      loads the data once with the ladder's own loader (``build_data_and_paradigm``, rectangle items), keeps
              the voxels of the reference fit and stores BOLD, paradigm, indicators, the HRF and the starts
              (reference polished fit = warm; clamped fit with the m0 pRFs = cold) in one pickle per area.
``fit``       LM to convergence from the warm and the cold start; the Gauss-Newton Hessian over the shared terms
              with the pRFs profiled out (Schur complement) at the best point: eigenvalues, eigenvectors,
              correlations, Laplace SEs (σ² = SSE / (N − n_par), white noise: a lower bound on the true SE).
``profile``   fixes g_LP (or dS = g_HP − g_LP, m7_full) on a grid and refits everything else to convergence at each
              value, warm-started along the grid from the optimum outward: training SSE on all runs, and training /
              held-out SSE on the three run3 folds (one run per HP condition held out).

Outputs under ``derivatives/joint_optimisation_2026-11-13/`` (pickles) and small TSVs next to them; the plots and
the note are made locally from the TSVs (``visualize/plot_joint_identifiability.py``).

Usage::

    python -m retsupp.modeling.joint_identifiability dump 3 --roi V1 TO IPS
    python -m retsupp.modeling.joint_identifiability fit 3 --roi IPS --model m7_full
    python -m retsupp.modeling.joint_identifiability profile 3 --roi IPS --model m7_full --param g_LP
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.hrf import SPMHRFModel
from retsupp.modeling.fit_af_prf_cv_v2 import (build_data_and_paradigm, build_model, fixed_pars_for, get_ring_positions,
                                               make_folds, shared_pars_for, split_by_condition_with_target)
from retsupp.modeling.fit_cv_ladder_nested import raw_bounds, sign_pattern_of
from retsupp.modeling.joint_lm import JointProblem, lm_fit, normal_blocks, schur_hessian
from retsupp.modeling.roi_hrf import hrf_values, load_table

BIDS = Path('/shares/zne.uzh/gdehol/ds-retsupp')
REF = 'af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished'
CLAMPED = 'af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf'
HRF_TABLE = 'paper_2026-11/roi_hrf_dog_final2.tsv'
OUT = 'joint_optimisation_2026-11-13'
MODELS = ('m4_TDS', 'm7_full')
GRID = {'g_LP': np.array([-10, -5, -3, -2, -1.5, -1.25, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
                          0.1, 0.2]),
        'dS': np.array([-1, -0.6, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1, 2, 3])}
ZEROSUM = False  # --zerosum: sustained field without a condition-invariant part (g_LP = -g_HP / 3), m7_full only
GAIN_LO = None   # --gain-lo: lower bound of every sustained/transient gain instead of SHARED_BOUNDS' -10
SUMMARY = ['sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale']


def out_dir(bids):
    d = Path(bids) / 'derivatives' / OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(bids, subject, roi):
    return out_dir(bids) / 'cache' / f'sub-{subject:02d}_roi-{roi}.pkl'


def load_fit(bids, root, name, subject, roi):
    pkl = Path(bids) / 'derivatives' / root / name / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    return d['fit_pars_per_fold'][0], np.asarray(d['voxel_ids'])


def dump(subject, rois, bids):
    from retsupp.utils.data import Subject
    sub = Subject(subject, bids)
    bold, par, ci, di, ti, grid, _m, run_meta = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0,
                                                                         item_shape='rectangle')
    tab = load_table(Path(bids) / 'derivatives' / HRF_TABLE)
    for roi in rois:
        starts, ids = {}, None
        for name in MODELS:
            for kind, root in (('warm', REF), ('cold', CLAMPED)):
                p, v = load_fit(bids, root, name, subject, roi)
                if ids is None:
                    ids = v
                if not np.array_equal(ids, v):
                    raise ValueError(f'{name} {kind}: voxel set differs')
                starts[(name, kind)] = p
        d = dict(subject=subject, roi=roi, voxel_ids=ids, bold=bold[:, ids].astype(np.float32), paradigm=par,
                 ci=ci, di=di, ti=ti, grid=grid, run_meta=run_meta, tr=sub.get_tr(session=1, run=1),
                 hrf=hrf_values(subject, roi, tab), starts=starts)
        p = cache_path(bids, subject, roi)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            pickle.dump(d, f)
        print(f'{roi}: {ids.size} voxels -> {p}', flush=True)


def load_cache(bids, subject, roi):
    with open(cache_path(bids, subject, roi), 'rb') as f:
        return pickle.load(f)


def make_model(c, name, d):
    """The ladder's DoG gain-field model (``build_model``, free item scale) on the rows in ``d``."""
    hrf = SPMHRFModel(tr=c['tr'], delay=c['hrf']['hrf_delay'], dispersion=c['hrf']['hrf_dispersion'])
    return build_model(grid_coords=c['grid'], paradigm=d['paradigm'], condition_indicator=d['condition_indicator'],
                       dynamic_indicator=d['dynamic_indicator'], target_indicator=d['target_indicator'],
                       ring_positions=get_ring_positions(), hrf_model=hrf, sign_pattern=sign_pattern_of(name),
                       item_scale='free')


def splits(c):
    """[('all', train, None)] + the three run3 folds (train, held)."""
    whole = dict(bold=c['bold'], paradigm=c['paradigm'], condition_indicator=c['ci'], dynamic_indicator=c['di'],
                 target_indicator=c['ti'], run_meta=c['run_meta'])
    out = [('all', whole, None)]
    for label, held in make_folds(c['run_meta'], 'run3'):
        s = split_by_condition_with_target(bold=c['bold'], paradigm=c['paradigm'], condition_indicator=c['ci'],
                                           dynamic_indicator=c['di'], target_indicator=c['ti'],
                                           run_meta=c['run_meta'], held_out_runs=held)
        out.append((label, s['train'], s['held']))
    return out


def free_lists(model, name):
    sp = sign_pattern_of(name)
    shared = shared_pars_for(sp, 'free')
    fixed = fixed_pars_for(sp, 'free') or []
    labels = list(model.parameter_labels)
    free_sh = [p for p in shared if p != 'sigma_T_dyn']
    free_pv = [p for p in labels if p not in shared and p not in fixed]
    return free_sh, free_pv


def shared_map(free_sh, fix=None):
    """A, b, φ names for s = A φ + b. ``fix``: None (all free), ('g_LP', v) or ('dS', v): g_HP = g_LP + v."""
    n = len(free_sh)
    if fix is None:
        return None, None, list(free_sh)
    what, v = fix
    A, b = np.eye(n), np.zeros(n)
    if what == 'zerosum':
        # Sustained field with no condition-invariant part: g_HP + 3 g_LP = 0, so averaged over the four HP
        # conditions every ring position has gain 0 (g_LP = -g_HP / 3, dS = 4/3 g_HP).
        i_hp, i_lp = free_sh.index('g_HP'), free_sh.index('g_LP')
        A[i_lp, i_lp], A[i_lp, i_hp] = 0.0, -1.0 / 3.0
        drop = i_lp
    elif what == 'dS':
        i_hp, i_lp = free_sh.index('g_HP'), free_sh.index('g_LP')
        A[i_hp, i_hp], A[i_hp, i_lp], b[i_hp] = 0.0, 1.0, v
        drop = i_hp
    else:
        drop = free_sh.index(what)
        A[drop, drop], b[drop] = 0.0, v
    keep = [j for j in range(n) if j != drop]
    return A[:, keep], b, [free_sh[j] for j in keep]


def problem(model, name, y, paradigm, init, fix=None):
    free_sh, free_pv = free_lists(model, name)
    if ZEROSUM and fix is None:
        fix = ('zerosum', None)
    A, b, phi_names = shared_map(free_sh, fix)
    prob = JointProblem(model, y, paradigm, init.reindex(columns=list(model.parameter_labels)), free_pv, free_sh,
                        A=A, b=b, phi_names=phi_names)
    rb = dict(zip(free_sh, raw_bounds(model, init.reindex(columns=list(model.parameter_labels)), free_sh)))
    if GAIN_LO is not None:   # free gains: raw = value
        rb = {n: ((max(GAIN_LO, b[0]), b[1]) if n.startswith('g_') else b) for n, b in rb.items()}
    return prob, [rb[n] for n in phi_names]


def tag():
    return ('' if GAIN_LO is None else f'_glo{GAIN_LO:g}') + ('_zerosum' if ZEROSUM else '')


def summary(pars, prefix=''):
    row = {f'{prefix}{k}': float(pars[k].iloc[0]) for k in SUMMARY if k in pars}
    row[f'{prefix}dS'] = float(pars['g_HP'].iloc[0] - pars['g_LP'].iloc[0])
    row[f'{prefix}amp_median'] = float(pars['amplitude'].median())
    row[f'{prefix}baseline_mean'] = float(pars['baseline'].mean())
    row[f'{prefix}srf_amp_median'] = float(pars['srf_amplitude'].median())
    return row


def hessian_report(prob, x, phi, sse, n_obs):
    """Profiled GN Hessian (JᵀJ, so ΔSSE ≈ δᵀHδ) in φ (raw) and in natural units (gains: same; σ, item scale:
    through the softplus slope)."""
    pred, j_pv, j_phi = prob.jacobians(x, phi)
    U, W, Vs, g_pv, g_s = normal_blocks(prob.y, pred, j_pv, j_phi)
    H = schur_hessian(U, W, Vs)
    # d natural / d raw for each φ, by central differences of the model transform
    eps = 1e-2
    slopes = []
    for i in range(len(phi)):
        up, dn = phi.copy(), phi.copy()
        up[i] += eps
        dn[i] -= eps
        pu, pd_ = prob.pars(x, up), prob.pars(x, dn)
        nm = prob.phi_names[i]
        slopes.append(float(pu[nm].iloc[0] - pd_[nm].iloc[0]) / (2 * eps))
    Dinv = np.diag(1.0 / np.asarray(slopes))
    Hn = Dinv @ H @ Dinv
    n_par = x.size + phi.size
    s2 = sse / (n_obs - n_par)
    cov = np.linalg.pinv(Hn) * s2
    se = np.sqrt(np.abs(np.diag(cov)))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.outer(se, se)
    # eigen-analysis on the correlation-scaled Hessian (unit-free): small eigenvalues = flat directions
    dsc = np.sqrt(np.diag(Hn))
    Hs = Hn / np.outer(dsc, dsc)
    ev, evec = np.linalg.eigh(Hs)
    evn, evecn = np.linalg.eigh(Hn)
    return dict(names=list(prob.phi_names), H_raw=H, H_nat=Hn, slopes=slopes, sigma2=s2, n_obs=n_obs, n_par=n_par,
                se=se, corr=corr, eig_scaled=ev, evec_scaled=evec, cond_scaled=float(ev.max() / max(ev.min(), 1e-300)),
                eig_nat=evn, evec_nat=evecn, cond_nat=float(evn.max() / max(evn.min(), 1e-300)),
                rel_sse_per_se=s2 / sse)


def fit(subject, roi, name, bids, max_iter, ftol, kinds=('warm', 'cold')):
    c = load_cache(bids, subject, roi)
    whole = splits(c)[0][1]
    m = make_model(c, name, whole)
    y = whole['bold']
    res = {}
    for kind in kinds:
        init = c['starts'][(name, kind)]
        prob, bnds = problem(m, name, y, whole['paradigm'], init)
        print(f'== {roi} {name} {kind}: start SSE {prob.sse(*prob.start()):.4f}', flush=True)
        x, phi, info = lm_fit(prob, bounds=bnds, max_iter=max_iter, ftol=ftol)
        pars = prob.pars(x, phi)
        res[kind] = dict(pars=pars, info=info, start=summary(init.reindex(columns=list(m.parameter_labels))),
                         end=summary(pars), x=x, phi=phi)
        print(f'   {kind}: ' + ', '.join(f'{k}={v:.4f}' for k, v in res[kind]['end'].items()), flush=True)
    best = min(res, key=lambda k: res[k]['info']['sse_final'])
    prob, _ = problem(m, name, y, whole['paradigm'], res[best]['pars'])
    h = hessian_report(prob, *prob.start(), res[best]['info']['sse_final'], y.size)
    res['hessian'] = h
    res['best'] = best
    print('   SE (white noise): ' + ', '.join(f'{n}={s:.4f}' for n, s in zip(h['names'], h['se'])))
    print(f'   cond (scaled) {h["cond_scaled"]:.3g}; smallest eigvec: ' + ', '.join(
        f'{n} {v:+.2f}' for n, v in zip(h['names'], h['evec_scaled'][:, 0])))
    p = out_dir(bids) / 'fits' / f'sub-{subject:02d}_roi-{roi}_{name}_lm{tag()}.pkl'
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'wb') as f:
        pickle.dump(res, f)
    print(f'-> {p}')


def profile(subject, roi, name, param, bids, max_iter, ftol, split_labels=None, grid=None, from_start=False):
    """``from_start``: begin at the reference polished fit (warm start) instead of the LM end point of ``fit``; the
    'optimum' row is then the free LM fit from there after ``max_iter`` iterations."""
    c = load_cache(bids, subject, roi)
    if from_start:
        best_pars = c['starts'][(name, 'warm')]
    else:
        with open(out_dir(bids) / 'fits' / f'sub-{subject:02d}_roi-{roi}_{name}_lm{tag()}.pkl', 'rb') as f:
            fitres = pickle.load(f)
        best_pars = fitres[fitres['best']]['pars']
    grid = np.asarray(GRID[param] if grid is None else grid, dtype=float)
    if param == 'g_LP' and GAIN_LO is not None:
        grid = grid[grid >= GAIN_LO]
    rows = []
    t0 = time.time()
    for label, train, held in splits(c):
        if split_labels and label not in split_labels:
            continue
        m_tr = make_model(c, name, train)
        m_he = None if held is None else make_model(c, name, held)
        # unconstrained optimum on this split (from the all-runs optimum), then out along the grid both ways
        prob0, b0 = problem(m_tr, name, train['bold'], train['paradigm'], best_pars)
        x, phi, info = lm_fit(prob0, bounds=b0, max_iter=max_iter, ftol=ftol, verbose=False)
        opt = prob0.pars(x, phi)
        v_opt = float(opt['g_LP'].iloc[0]) if param == 'g_LP' else float(opt['g_HP'].iloc[0] - opt['g_LP'].iloc[0])

        def record(pars, value, it, kind, inf=None):
            last = (inf or {}).get('log', [{}])[-1]
            row = dict(subject=subject, roi=roi, model=name, param=param, value=value, split=label, kind=kind,
                       gain_lo=GAIN_LO, n_vox=len(pars), n_obs=int(train['bold'].size), nit=it,
                       reason=(inf or {}).get('reason'), last_rel=last.get('rel', np.nan),
                       sse_train=float(((train['bold'] - np.asarray(m_tr.predict(paradigm=train['paradigm'],
                                                                                parameters=pars))) ** 2).sum()))
            if held is not None:
                row['sse_held'] = float(((held['bold'] - np.asarray(m_he.predict(paradigm=held['paradigm'],
                                                                                parameters=pars))) ** 2).sum())
                row['n_obs_held'] = int(held['bold'].size)
            row.update(summary(pars))
            rows.append(row)
        record(opt, v_opt, info['nit'], 'optimum', info)
        print(f'  [{label}] optimum {param}={v_opt:.4f} SSE {info["sse_final"]:.4f} ({info["nit"]} it) '
              f'| {time.time() - t0:.0f}s', flush=True)
        for side in (grid[grid >= v_opt], grid[grid < v_opt][::-1]):
            start = opt
            for v in side:
                prob, bnds = problem(m_tr, name, train['bold'], train['paradigm'], start, fix=(param, float(v)))
                x, phi, info = lm_fit(prob, bounds=bnds, max_iter=max_iter, ftol=ftol, verbose=False)
                pars = prob.pars(x, phi)
                record(pars, float(v), info['nit'], 'grid', info)
                print(f'  [{label}] {param}={v:+.2f}: SSE {info["sse_final"]:.4f} ({info["nit"]} it, '
                      f'{info["reason"]}) | {time.time() - t0:.0f}s', flush=True)
                start = pars
    df = pd.DataFrame(rows).sort_values(['split', 'value'])
    stag = ('' if not split_labels else '_' + '-'.join(split_labels)) + ('_fromstart' if from_start else '')
    p = out_dir(bids) / 'profiles' / f'sub-{subject:02d}_roi-{roi}_{name}_profile-{param}{tag()}{stag}.tsv'
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep='\t', index=False)
    print(f'-> {p}')


def hessians(subject, roi, name, bids):
    """Profiled GN Hessian over the shared terms at the reference starts (warm = reference polished fit, cold =
    clamped fit) and at the LM end points of ``fit`` (all in the reported parametrisation)."""
    c = load_cache(bids, subject, roi)
    whole = splits(c)[0][1]
    m = make_model(c, name, whole)
    y = whole['bold']
    pts = {f'start_{k}': c['starts'][(name, k)] for k in ('warm', 'cold')}
    fp = out_dir(bids) / 'fits' / f'sub-{subject:02d}_roi-{roi}_{name}_lm{tag()}.pkl'
    if fp.exists():
        with open(fp, 'rb') as f:
            fr = pickle.load(f)
        pts.update({f'lm_{k}': fr[k]['pars'] for k in ('warm', 'cold') if k in fr})
    out, rows = {}, []
    for label, pars in pts.items():
        prob, _ = problem(m, name, y, whole['paradigm'], pars)
        x, phi = prob.start()
        sse = prob.sse(x, phi)
        h = hessian_report(prob, x, phi, sse, y.size)
        out[label] = dict(h, summary=summary(pars.reindex(columns=list(m.parameter_labels))), sse=sse)
        print(f'== {roi} {name} {label}: SSE {sse:.2f}; cond (correlation-scaled) {h["cond_scaled"]:.3g}')
        print('   SE: ' + ', '.join(f'{n}={v:.4f}' for n, v in zip(h['names'], h['se'])))
        for j in range(2):
            print(f'   eig {h["eig_scaled"][j]:.3g}: ' + ', '.join(
                f'{n} {v:+.2f}' for n, v in zip(h['names'], h['evec_scaled'][:, j]) if abs(v) > 0.1))
        if 'g_LP' in h['names']:
            i = h['names'].index('g_LP')
            print('   corr(g_LP, ·): ' + ', '.join(f'{n} {h["corr"][i, j]:+.2f}' for j, n in enumerate(h['names'])
                                                  if j != i))
        for j, n in enumerate(h['names']):
            rows.append(dict(subject=subject, roi=roi, model=name, point=label, gain_lo=GAIN_LO, sse=sse, term=n,
                             value=out[label]['summary'].get(n, np.nan), se=h['se'][j],
                             cond_scaled=h['cond_scaled'], eig_min_scaled=h['eig_scaled'][0],
                             evec_min=h['evec_scaled'][j, 0], sigma2=h['sigma2'],
                             **{f'corr_{k}': h['corr'][j, kk] for kk, k in enumerate(h['names'])}))
    p = out_dir(bids) / 'hessians' / f'sub-{subject:02d}_roi-{roi}_{name}_hessian{tag()}'
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(out, f)
    pd.DataFrame(rows).to_csv(p.with_suffix('.tsv'), sep='\t', index=False)
    print(f'-> {p}.tsv')


def check(subject, roi, name, bids):
    """Diagnostics: Jacobian vs central differences per column, and predicted vs actual SSE change of damped steps."""
    from retsupp.modeling.joint_lm import schur_step
    c = load_cache(bids, subject, roi)
    whole = splits(c)[0][1]
    m = make_model(c, name, whole)
    prob, bnds = problem(m, name, whole['bold'], whole['paradigm'], c['starts'][(name, 'warm')])
    x, phi = prob.start()
    pred, j_pv, j_phi = prob.jacobians(x, phi)
    for i, nm in enumerate(prob.per_voxel):
        for eps in (1e-2, 1e-3):
            e = np.zeros_like(x)
            e[:, i] = eps
            fd = (prob.predict(x + e, phi) - prob.predict(x - e, phi)) / (2 * eps)
            print(f'pv {nm} eps {eps}: max|fd-J|/max|fd| = {np.abs(fd - j_pv[:, :, i]).max() / np.abs(fd).max():.2e}')
    for i, nm in enumerate(prob.phi_names):
        for eps in (1e-2, 1e-3):
            e = np.zeros_like(phi)
            e[i] = eps
            fd = (prob.predict(x, phi + e) - prob.predict(x, phi - e)) / (2 * eps)
            den = max(np.abs(fd).max(), 1e-30)
            print(f'sh {nm} eps {eps}: max|fd-J|/max|fd| = {np.abs(fd - j_phi[:, :, i]).max() / den:.2e} '
                  f'(max|fd| {den:.3g})')
    print('bounds', dict(zip(prob.phi_names, bnds)))
    x3, p3, _ = lm_fit(prob, bounds=bnds, max_iter=12, debug=True)
    pr3, jp3, js3 = prob.jacobians(x3, p3)
    pars3 = prob.pars(x3, p3)
    print('non-finite pred:', int((~np.isfinite(pr3)).sum()))
    for i, nm in enumerate(prob.per_voxel):
        bad = np.where(~np.isfinite(jp3[:, :, i]).all(0) | ~np.isfinite(jp3[:, :, i]).any(0) * 0 |
                       (~np.isfinite(jp3[:, :, i])).any(0))[0]
        if bad.size:
            print(f'NaN in d/d{nm} for voxels {bad[:10]}: raw {np.round(x3[bad[:5]], 3).tolist()}')
            print(pars3.iloc[bad[:5]][prob.per_voxel])
    for i, nm in enumerate(prob.phi_names):
        bad = np.where((~np.isfinite(js3[:, :, i])).any(0))[0]
        if bad.size:
            print(f'NaN in d/d{nm} for {bad.size} voxels, e.g. {bad[:10]}')
    U, W, Vs, g_pv, g_s = normal_blocks(prob.y, pred, j_pv, j_phi)
    f = float(((prob.y - pred) ** 2).sum())
    print('phi', dict(zip(prob.phi_names, np.round(phi, 4))), 'g_s', np.round(g_s, 3))
    for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5):
        for hold_sh in (False, True):
            hold = np.ones(len(phi), bool) if hold_sh else None
            d_pv, d_s = schur_step(U, W, Vs, g_pv, g_s, lam, hold)
            jd = np.einsum('tvi,vi->tv', j_pv, d_pv) + np.einsum('tvi,i->tv', j_phi, d_s)
            pred_dec = 2 * (jd * (prob.y - pred)).sum() - (jd ** 2).sum()
            f_new = prob.sse(x + d_pv, phi + d_s)
            print(f'λ {lam:g} hold_shared {hold_sh}: predicted ΔSSE {-pred_dec:.4f}, actual {f_new - f:.4f}, '
                  f'|d_s| {np.abs(d_s).max():.3g}, max|d_pv| per col {np.round(np.abs(d_pv).max(0), 4)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['dump', 'fit', 'profile', 'check', 'hessian'])
    ap.add_argument('--zerosum', action='store_true',
                    help='fit: sustained field with zero condition-average (g_LP = -g_HP/3); m7_full only.')
    ap.add_argument('--gain-lo', type=float, default=None,
                    help='Lower bound of the gains (e.g. -1: the field cannot zero the drive) instead of -10.')
    ap.add_argument('subject', type=int)
    ap.add_argument('--roi', nargs='+', required=True)
    ap.add_argument('--model', default='m7_full', choices=list(MODELS))
    ap.add_argument('--param', default='g_LP', choices=['g_LP', 'dS'])
    ap.add_argument('--splits', nargs='*', default=None, help='profile: only these splits (all, run3-0, ...)')
    ap.add_argument('--grid', nargs='*', type=float, default=None)
    ap.add_argument('--from-start', action='store_true', help='profile: start at the reference fit, not at fit\'s end')
    ap.add_argument('--bids-folder', default=str(BIDS))
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--ftol', type=float, default=1e-10)
    a = ap.parse_args()
    GAIN_LO = a.gain_lo
    ZEROSUM = a.zerosum
    if a.stage == 'dump':
        dump(a.subject, a.roi, a.bids_folder)
    else:
        for roi in a.roi:
            if a.stage == 'hessian':
                hessians(a.subject, roi, a.model, a.bids_folder)
            elif a.stage == 'check':
                check(a.subject, roi, a.model, a.bids_folder)
            elif a.stage == 'fit':
                fit(a.subject, roi, a.model, a.bids_folder, a.max_iter, a.ftol)
            else:
                profile(a.subject, roi, a.model, a.param, a.bids_folder, a.max_iter, a.ftol, a.splits, a.grid,
                        a.from_start)
