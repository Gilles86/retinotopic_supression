"""Nested CV ladder: every gain-field model is fit from its parent's optimum.

The first ladder (``fit_af_prf_cv_v2``, one independent fit per model) refit
all ~3.5k per-voxel pRF parameters for every model, and richer models often
ended with a *lower* training R² than the model nested inside them. The
ΔCV-R² between attention models (~1e-3) was at the level of that optimiser
noise. Here, per (subject, ROI) and CV fold:

1. ``m00_bar`` — vanilla DoG pRF on the bar-only stimulus: full gradient
   descent from the m4 mean-fit template, run to a plateau.
2. ``m0_prf`` — + rectangle search items with a free drive scale: full
   gradient descent warm-started from the m00 fold fit.
3. Every gain-field model **freezes the per-voxel pRFs at the m0 fold fit**
   and fits only its shared parameters (gains, σ's, item scale) with
   L-BFGS-B, starting from its parent's fitted shared parameters with the
   new terms at their null value (gain 0; an untied HP gain starts at its
   LP partner). A start at the parent optimum reproduces the parent's
   prediction exactly, L-BFGS only accepts descent steps, and the best of
   several σ starts is kept, so training SSE is never above the parent's.

Models run in parent order and each writes its outputs (the canonical
``cv_helpers.write_cv_tsvs`` trio + a pickle with the fold fits) as soon
as all folds are done; a rerun skips models whose pickle exists.

Polish (``--polish``): after the clamped chain, every model is refit once
more with **all** parameters free (per-voxel pRFs + shared terms), by
L-BFGS from its clamped optimum, so training SSE can only go down. Written
to a sibling ``..._polished`` root; the clamped fits are never touched. The
experiment: do the gains drift once the pRFs are released, and does held-out
prediction improve or degrade?

``--cv-scheme full`` fits one "fold" on all runs (no held-out data, CV-R² is
NaN): the main parameter estimates under the same nested procedure.

``--bounded`` boxes the shared terms (``SHARED_BOUNDS``) in both fits.

Output: ``derivatives/af_prf_cv_ladder_nested[_<scheme>][_bounded][_polished]/<model>/sub-XX/``.
Existing outputs are loaded, never overwritten.

Usage::

    python -m retsupp.modeling.fit_cv_ladder_nested 3 --roi V1 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp [--polish]
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize

from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter
from retsupp.modeling.cv_helpers import per_voxel_r2, write_cv_tsvs
from retsupp.modeling.fit_af_prf_cv_v2 import (
    ALL_SHARED_PARS,
    CV_SCHEMES,
    build_data_and_paradigm,
    build_model,
    choose_voxel_mask,
    fixed_pars_for,
    get_ring_positions,
    make_folds,
    make_init_pars,
    shared_pars_for,
    split_by_condition_with_target,
)
from retsupp.utils.data import Subject

# name: (sus_hp, sus_lp, dyn_hp, dyn_lp, tgt, item_scale, parent)
LADDER = {
    'm00_bar':   ('zero', 'zero', 'zero', 'zero', 'zero', 'none', None),
    'm0_prf':    ('zero', 'zero', 'zero', 'zero', 'zero', 'free', 'm00_bar'),
    'm1_T':      ('zero', 'zero', 'zero', 'zero', 'free', 'free', 'm0_prf'),
    'm2_D':      ('zero', 'zero', 'tied', 'free', 'zero', 'free', 'm0_prf'),
    'm3_S':      ('tied', 'free', 'zero', 'zero', 'zero', 'free', 'm0_prf'),
    'm3b_TD':    ('zero', 'zero', 'tied', 'free', 'free', 'free', 'm1_T'),
    'm4_TDS':    ('tied', 'free', 'tied', 'free', 'free', 'free', 'm3b_TD'),
    'm5_TDS_dS': ('free', 'free', 'tied', 'free', 'free', 'free', 'm4_TDS'),
    'm6_TDS_dD': ('tied', 'free', 'free', 'free', 'free', 'free', 'm4_TDS'),
    'm7_full':   ('free', 'free', 'free', 'free', 'free', 'free', 'm5_TDS_dS'),
}
GAINS = ['g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn']
# An untied HP gain starts at its LP partner (that is what 'tied' meant).
TIE_PARTNER = {'g_HP': 'g_LP', 'g_HP_dyn': 'g_LP_dyn'}
PER_VOXEL_COLS = ['x', 'y', 'sd', 'baseline', 'amplitude',
                  'srf_amplitude', 'srf_size']
SIGMA_STARTS = (None, 0.5, 1.0, 2.0, 4.0)   # None = parent's σ's
LBFGS_ROUNDS = 5   # restarts from the previous end point (fresh Hessian)
POLISH_ROUNDS = 3
# --bounded: box on the shared terms. Unbounded, a few % of fold fits ran to
# σ_dyn ~0.03° or ~1e4° with |gain| up to 5e5 -- directions the data do not
# constrain (a pinprick far below the 1.5° item, or a uniform field that
# trades off against pRF amplitude).
SHARED_BOUNDS = {'sigma_AF': (0.5, 8.0), 'sigma_dyn': (0.5, 8.0),
                 'item_scale': (1e-3, 50.0),
                 **{g: (-10.0, 10.0) for g in GAINS}}
POLISH_MAX_ITER = 3000
SCHEMES = tuple(CV_SCHEMES) + ('full',)


def sign_pattern_of(name):
    s = LADDER[name]
    return dict(zip(GAINS, s[:5]))


def gd_fit(model, bold_train, paradigm, init_pars, sign_pattern, item_scale,
           max_n_iterations, learning_rate):
    """Full per-voxel gradient descent (m00 / m0), run to a plateau."""
    fitter = ParameterFitter(model, pd.DataFrame(bold_train), paradigm)
    t0 = time.time()
    fit_pars = fitter.fit(
        init_pars=init_pars,
        max_n_iterations=max_n_iterations,
        min_n_iterations=1000,
        lag=200,
        r2_atol=1e-7,
        shared_pars=shared_pars_for(sign_pattern, item_scale),
        fixed_pars=fixed_pars_for(sign_pattern, item_scale),
        learning_rate=learning_rate,
        progressbar=False,
    )
    info = dict(optimizer='adam_gd', seconds=time.time() - t0)
    return fit_pars, info


def make_sse_fn(model, bold_train, paradigm, raw_pars, free_ix):
    """SSE and gradient as a function of the free shared raw parameters."""
    n_pars = raw_pars.shape[1]
    y = tf.constant(bold_train, dtype=tf.float64)
    par_ = model.stimulus._clean_paradigm(paradigm)
    raw = tf.constant(raw_pars, dtype=tf.float32)
    mask = np.zeros(n_pars, dtype=np.float32)
    mask[free_ix] = 1.0
    mask = tf.constant(mask)
    ix = tf.constant(np.asarray(free_ix, dtype=np.int32)[:, None])

    @tf.function
    def sse_and_grad(v):
        with tf.GradientTape() as tape:
            tape.watch(v)
            full = tf.scatter_nd(ix, v, [n_pars])
            params = raw * (1.0 - mask) + full[tf.newaxis, :] * mask
            pred = model._predict(
                par_[tf.newaxis, ...],
                model._transform_parameters_forward(params)[tf.newaxis, ...],
                None)[0]
            # Residuals and sum in float64: in float32 the summation noise
            # (~1e-7 relative) stops L-BFGS before the optimum.
            sse = tf.reduce_sum((y - tf.cast(pred, tf.float64)) ** 2)
        return sse, tape.gradient(sse, v)

    def fun(v64):
        sse, g = sse_and_grad(tf.constant(v64, dtype=tf.float32))
        return float(sse.numpy()), g.numpy().astype(np.float64)

    return fun


def raw_bounds(model, init_pars, cols):
    """SHARED_BOUNDS of ``cols`` mapped to raw (optimiser) space."""
    labels = list(model.parameter_labels)
    lo = init_pars[labels].iloc[:1].copy()
    hi = lo.copy()
    for c in cols:
        lo[c], hi[c] = SHARED_BOUNDS[c]
    r = np.asarray(model._transform_parameters_backward(
        np.vstack([lo.values, hi.values]).astype(np.float32)), dtype=np.float64)
    out = []
    for c in cols:
        a, b = r[0, labels.index(c)], r[1, labels.index(c)]
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f'non-finite raw bound for {c}: {a}, {b}')
        out.append((min(a, b), max(a, b)))
    return out


def shared_fit(model, bold_train, paradigm, init_pars, free_shared,
               max_iter=2000, bounded=False):
    """L-BFGS-B over the shared parameters, per-voxel pRFs frozen.

    Starts from ``init_pars`` (the parent optimum, new terms at null) and
    from alternative σ starts; keeps the lowest training SSE. The parent
    start is always among the candidates and L-BFGS never ends above its
    start, so the result is never worse than the parent on training data.
    """
    labels = list(model.parameter_labels)
    free_ix = [labels.index(p) for p in free_shared]
    sigma_ix = [labels.index(p) for p in free_shared if p.startswith('sigma')]
    raw0 = model._transform_parameters_backward(
        init_pars[labels].values.astype(np.float32))
    raw0 = np.asarray(raw0)
    fun = make_sse_fn(model, bold_train, paradigm, raw0, free_ix)
    bnds = raw_bounds(model, init_pars, free_shared) if bounded else None

    def clip(x):
        if bnds is None:
            return x
        return np.clip(x, [b[0] for b in bnds], [b[1] for b in bnds])

    t0 = time.time()
    f_parent, _ = fun(clip(raw0[0, free_ix].astype(np.float64)))
    best = None
    starts = []
    for s in SIGMA_STARTS:
        x0 = raw0[0, free_ix].astype(np.float64).copy()
        if s is not None:
            if not sigma_ix:
                continue
            tmp = init_pars[labels].iloc[:1].copy()
            for k in sigma_ix:
                tmp.iloc[0, k] = s
            raw_s = np.asarray(model._transform_parameters_backward(
                tmp.values.astype(np.float32)))
            for j, k in enumerate(free_ix):
                if k in sigma_ix:
                    x0[j] = raw_s[0, k]
        x0 = clip(x0)
        f_start, _ = fun(x0)
        f_end, x_end, nit = f_start, x0, 0
        for _round in range(LBFGS_ROUNDS):
            res = minimize(fun, x_end, jac=True, method='L-BFGS-B',
                           bounds=bnds,
                           options=dict(maxiter=max_iter, maxfun=4 * max_iter,
                                        ftol=1e-14, gtol=1e-9))
            nit += int(res.nit)
            if not res.fun < f_end - 1e-9 * abs(f_end):
                break
            f_end, x_end = res.fun, res.x
        starts.append(dict(sigma_start=s, f_start=f_start, f_end=f_end,
                           nit=nit, rounds=_round + 1,
                           status=int(res.status),
                           message=str(res.message)))
        if best is None or f_end < best[0]:
            best = (f_end, x_end)

    raw_best = raw0.copy()
    raw_best[:, free_ix] = best[1].astype(np.float32)
    fit_pars = pd.DataFrame(
        np.asarray(model._transform_parameters_forward(raw_best)),
        columns=labels, index=init_pars.index)
    _, g = fun(best[1])
    info = dict(optimizer='lbfgs_shared', seconds=time.time() - t0,
                sse_parent_start=f_parent, sse_final=best[0],
                grad_norm=float(np.linalg.norm(g)), starts=starts)
    return fit_pars, info


def child_init(parent_pars, model, name):
    """Parent optimum with the child's new terms at their null value."""
    labels = list(model.parameter_labels)
    init = parent_pars.reindex(columns=labels).copy()
    sp = sign_pattern_of(name)
    parent_sp = sign_pattern_of(LADDER[name][6])
    for g in GAINS:
        if parent_sp[g] == 'zero' and sp[g] != 'zero':
            init[g] = 0.0
        if parent_sp[g] == 'tied' and sp[g] == 'free':
            init[g] = init[TIE_PARTNER[g]]
    if init[['sigma_AF', 'sigma_dyn']].isna().any().any():
        raise ValueError('parent pars lack σ columns')
    init['sigma_T_dyn'] = init['sigma_dyn']
    return init


def polish_fit(model, bold_train, paradigm, init_pars, free_shared,
               free_per_voxel, bounded=False):
    """L-BFGS-B over all free parameters, from the clamped optimum.

    Per-voxel columns ``free_per_voxel`` get one value per voxel, the
    ``free_shared`` columns one value broadcast to every voxel. Never ends
    above its start.
    """
    labels = list(model.parameter_labels)
    n_vox, n_pars = len(init_pars), len(labels)
    pv_ix = [labels.index(p) for p in free_per_voxel]
    sh_ix = [labels.index(p) for p in free_shared]
    raw0 = np.asarray(model._transform_parameters_backward(
        init_pars[labels].values.astype(np.float32)))

    def selector(ix):
        e = np.zeros((len(ix), n_pars), dtype=np.float32)
        e[np.arange(len(ix)), ix] = 1.0
        return tf.constant(e)
    e_pv, e_sh = selector(pv_ix), selector(sh_ix)
    keep = np.ones(n_pars, dtype=np.float32)
    keep[pv_ix + sh_ix] = 0.0
    keep = tf.constant(keep)
    raw = tf.constant(raw0, dtype=tf.float32)
    y = tf.constant(bold_train, dtype=tf.float64)
    par_ = model.stimulus._clean_paradigm(paradigm)
    n_pv = n_vox * len(pv_ix)

    @tf.function
    def sse_and_grad(v):
        with tf.GradientTape() as tape:
            tape.watch(v)
            pv = tf.reshape(v[:n_pv], [n_vox, len(pv_ix)])
            sh = v[n_pv:][tf.newaxis, :]
            params = (raw * keep[tf.newaxis, :] + tf.matmul(pv, e_pv)
                      + tf.matmul(sh, e_sh))
            pred = model._predict(
                par_[tf.newaxis, ...],
                model._transform_parameters_forward(params)[tf.newaxis, ...],
                None)[0]
            sse = tf.reduce_sum((y - tf.cast(pred, tf.float64)) ** 2)
        return sse, tape.gradient(sse, v)

    def fun(v64):
        sse, g = sse_and_grad(tf.constant(v64, dtype=tf.float32))
        return float(sse.numpy()), g.numpy().astype(np.float64)

    x0 = np.concatenate([raw0[:, pv_ix].ravel(),
                         raw0[0, sh_ix]]).astype(np.float64)
    bnds = None
    if bounded:
        bnds = [(None, None)] * n_pv + raw_bounds(model, init_pars,
                                                  free_shared)
        lo = np.array([-np.inf if b[0] is None else b[0] for b in bnds])
        hi = np.array([np.inf if b[1] is None else b[1] for b in bnds])
        x0 = np.clip(x0, lo, hi)
    t0 = time.time()
    f_start, _ = fun(x0)
    f_end, x_end, nit = f_start, x0, 0
    for _round in range(POLISH_ROUNDS):
        res = minimize(fun, x_end, jac=True, method='L-BFGS-B', bounds=bnds,
                       options=dict(maxiter=POLISH_MAX_ITER,
                                    maxfun=2 * POLISH_MAX_ITER,
                                    ftol=1e-14, gtol=1e-9))
        nit += int(res.nit)
        if not res.fun < f_end - 1e-9 * abs(f_end):
            break
        f_end, x_end = res.fun, res.x

    raw_best = raw0.copy()
    raw_best[:, pv_ix] = x_end[:n_pv].reshape(n_vox, len(pv_ix))
    raw_best[:, sh_ix] = x_end[n_pv:][np.newaxis, :]
    fit_pars = pd.DataFrame(
        np.asarray(model._transform_parameters_forward(
            raw_best.astype(np.float32))),
        columns=labels, index=init_pars.index)
    _, g = fun(x_end)
    info = dict(optimizer='lbfgs_all', seconds=time.time() - t0,
                sse_clamped=f_start, sse_final=f_end, nit=nit,
                rounds=_round + 1, status=int(res.status),
                message=str(res.message),
                grad_norm=float(np.linalg.norm(g)),
                grad_norm_shared=float(np.linalg.norm(g[n_pv:])))
    return fit_pars, info


def bar_paradigm(sub, run_meta, resolution, grid_radius):
    return np.vstack([
        sub.get_bar_stimulus(session=rm.session, run=rm.run,
                             resolution=resolution,
                             grid_radius=grid_radius
                             ).astype(np.float32).reshape(rm.n_T, -1)
        for rm in run_meta])


def main(subject, bids_folder, roi, cv_scheme='loco', resolution=50,
         grid_radius=5.0, max_voxels=500, p_signal_thr=0.5,
         gd_iterations=8000, learning_rate=0.01, models=None,
         output_root=None, polish=False, bounded=False):
    if cv_scheme not in SCHEMES:
        raise ValueError(f'cv_scheme must be in {SCHEMES}')
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    root = Path(output_root) if output_root else (
        bids_folder / 'derivatives' / (
            'af_prf_cv_ladder_nested'
            + ('' if cv_scheme == 'loco' else f'_{cv_scheme}')
            + ('_bounded' if bounded else '')))
    polish_root = root.parent / f'{root.name}_polished'
    models = models or list(LADDER)

    (bold, par_items, ci, di, ti, grid_coords, _masker,
     run_meta) = build_data_and_paradigm(
        sub, resolution=resolution, grid_radius=grid_radius,
        item_shape='rectangle')
    par_bar = bar_paradigm(sub, run_meta, resolution, grid_radius)

    prf_pars = pd.DataFrame(
        sub.get_prf_parameters_volume(model=4, return_images=False))
    voxel_mask, descr = choose_voxel_mask(
        sub, roi, prf_pars, use_psignal=True, p_signal_thr=p_signal_thr,
        r2_thr=0.05, model_label=4, max_voxels=max_voxels)
    print(f'{descr}: {voxel_mask.sum()} voxels')
    voxel_ids = np.where(voxel_mask)[0]
    ring_positions = get_ring_positions()
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)
    if cv_scheme == 'full':
        def whole(par):
            return dict(train=dict(
                bold=bold, paradigm=par, condition_indicator=ci,
                dynamic_indicator=di, target_indicator=ti,
                run_meta=run_meta), held=None)
        folds = []
        splits = [(whole(par_items), whole(par_bar))]
    else:
        folds = make_folds(run_meta, cv_scheme)
        splits = []
    for _label, held_runs in folds:
        kw = dict(bold=bold, condition_indicator=ci, dynamic_indicator=di,
                  target_indicator=ti, run_meta=run_meta,
                  held_out_runs=held_runs)
        try:
            s_items = split_by_condition_with_target(paradigm=par_items, **kw)
            s_bar = split_by_condition_with_target(paradigm=par_bar, **kw)
        except RuntimeError as e:
            print(f'skipping fold {_label}: {e}')
            splits.append(None)
            continue
        splits.append((s_items, s_bar))

    fits: dict[str, list] = {}

    def load(name, root):
        pkl = root / name / f'sub-{subject:02d}' / \
            f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                return pickle.load(f)['fit_pars_per_fold']
        return None

    # Resolve the ancestors every requested model needs.
    needed = []
    for name in models:
        chain, n = [], name
        while n is not None:
            chain.append(n)
            n = LADDER[n][6]
        for n in reversed(chain):
            if n not in needed:
                needed.append(n)

    stages = [('clamped', n) for n in needed]
    if polish:
        stages += [('polish', n) for n in needed]
    polished: dict[str, list] = {}
    for stage, name in stages:
        out_root = root if stage == 'clamped' else polish_root
        store = fits if stage == 'clamped' else polished
        cached = load(name, out_root)
        if cached is not None:
            print(f'== {name} [{stage}]: loaded existing fit')
            store[name] = cached
            continue
        sus_hp, sus_lp, dyn_hp, dyn_lp, tgt, item_scale, parent = LADDER[name]
        sp = sign_pattern_of(name)
        print(f'\n===== {name} [{stage}] (parent {parent}) =====')
        fold_pars, cv_r2s, train_r2s, shared_dicts, infos = [], [], [], [], []
        tr_meta, he_meta = [], []
        for k, split in enumerate(splits):
            if split is None:
                nan = np.full(voxel_ids.size, np.nan, dtype=np.float32)
                for lst in (cv_r2s, train_r2s):
                    lst.append(nan.copy())
                for lst in (fold_pars, shared_dicts, infos):
                    lst.append(None)
                tr_meta.append([])
                he_meta.append([])
                continue
            s = split[1] if item_scale == 'none' else split[0]
            train, held = s['train'], s['held']
            y_tr = train['bold'][:, voxel_mask]
            y_he = None if held is None else held['bold'][:, voxel_mask]

            def mk(d):
                return build_model(
                    grid_coords=grid_coords, paradigm=d['paradigm'],
                    condition_indicator=d['condition_indicator'],
                    dynamic_indicator=d['dynamic_indicator'],
                    target_indicator=d['target_indicator'],
                    ring_positions=ring_positions, hrf_model=hrf_model,
                    sign_pattern=sp, item_scale=item_scale)
            m_tr = mk(train)
            labels = list(m_tr.parameter_labels)

            if stage == 'polish':
                fixed = fixed_pars_for(sp, item_scale) or []
                shared = shared_pars_for(sp, item_scale)
                free_sh = [p for p in shared if p != 'sigma_T_dyn']
                free_pv = [p for p in labels
                           if p not in shared and p not in fixed]
                init = fits[name][k].reindex(columns=labels)
                pars, info = polish_fit(m_tr, y_tr, train['paradigm'], init,
                                        free_sh, free_pv, bounded=bounded)
                print(f'  fold {k}: SSE clamped {info["sse_clamped"]:.2f}'
                      f' -> {info["sse_final"]:.2f} | nit {info["nit"]} | '
                      f'|grad| {info["grad_norm"]:.3g} (shared '
                      f'{info["grad_norm_shared"]:.3g}) | '
                      f'{info["seconds"]:.0f}s')
            elif name == 'm00_bar':
                init = make_init_pars(prf_pars, voxel_mask, sp,
                                      sigma_af_init=2.0, sigma_dyn_init=2.0,
                                      item_scale=item_scale)
                init = init.reindex(columns=labels)
                init = ParameterFitter(m_tr, pd.DataFrame(y_tr),
                                       train['paradigm']
                                       ).refine_baseline_and_amplitude(
                    init, l2_alpha=1e-3)
                pars, info = gd_fit(m_tr, y_tr, train['paradigm'], init, sp,
                                    item_scale, gd_iterations, learning_rate)
            elif name == 'm0_prf':
                init = fits['m00_bar'][k].copy()
                init['item_scale'] = 1.0
                init = init.reindex(columns=labels)
                pars, info = gd_fit(m_tr, y_tr, train['paradigm'], init, sp,
                                    item_scale, gd_iterations, learning_rate)
            else:
                init = child_init(fits[parent][k], m_tr, name)
                free = [p for p in shared_pars_for(sp, item_scale)
                        if p != 'sigma_T_dyn']
                pars, info = shared_fit(m_tr, y_tr, train['paradigm'],
                                        init, free, bounded=bounded)
                print(f'  fold {k}: SSE parent-start {info["sse_parent_start"]:.2f}'
                      f' -> {info["sse_final"]:.2f} | |grad| '
                      f'{info["grad_norm"]:.3g} | {info["seconds"]:.0f}s')

            tr_r2 = per_voxel_r2(y_tr, np.asarray(
                m_tr.predict(paradigm=train['paradigm'], parameters=pars)))
            if held is None:
                cv_r2 = np.full(voxel_ids.size, np.nan, dtype=np.float32)
            else:
                m_he = mk(held)
                cv_r2 = per_voxel_r2(y_he, np.asarray(
                    m_he.predict(paradigm=held['paradigm'],
                                 parameters=pars)))
            print(f'  fold {k}: train R² mean {np.nanmean(tr_r2):.5f} | '
                  f'CV-R² median {np.nanmedian(cv_r2):.5f}')
            info['train_sse_total'] = float(
                ((y_tr - np.asarray(m_tr.predict(
                    paradigm=train['paradigm'], parameters=pars))) ** 2).sum())
            fold_pars.append(pars)
            cv_r2s.append(cv_r2)
            train_r2s.append(tr_r2)
            shared_dicts.append(pars.reindex(
                columns=ALL_SHARED_PARS + ['item_scale']).iloc[0].to_dict())
            infos.append(info)
            tr_meta.append([(r.session, r.run, r.hp) for r in train['run_meta']])
            he_meta.append([] if held is None else
                           [(r.session, r.run, r.hp) for r in held['run_meta']])

        store[name] = fold_pars
        out_dir = out_root / name / f'sub-{subject:02d}'
        pkl = out_dir / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
        if pkl.exists():   # never overwrite an existing fit
            raise FileExistsError(pkl)
        out_dir.mkdir(parents=True, exist_ok=True)
        shared_labels = ALL_SHARED_PARS + ['item_scale']
        write_cv_tsvs(
            out_dir, subject=subject, roi=roi, model=name,
            voxel_ids=voxel_ids, cv_r2_per_fold=cv_r2s,
            train_r2_per_fold=train_r2s,
            fit_pars_per_fold=[None if p is None else
                               p.reindex(columns=PER_VOXEL_COLS + shared_labels)
                               for p in fold_pars],
            shared_pars_per_fold=shared_dicts,
            shared_par_labels=shared_labels,
            per_voxel_par_cols=PER_VOXEL_COLS,
            train_run_meta_per_fold=tr_meta, held_run_meta_per_fold=he_meta,
            selector='psignal', p_signal_thr=p_signal_thr,
            extra_meta=dict(parent=parent, sign_pattern=sp,
                            item_scale=item_scale, cv_scheme=cv_scheme,
                            max_voxels=max_voxels,
                            gd_iterations=gd_iterations, stage=stage,
                            bounded=bounded,
                            optimizer_info=infos))
        with open(pkl, 'wb') as f:
            pickle.dump(dict(fit_pars_per_fold=fold_pars, parent=parent,
                             optimizer_info=infos, voxel_ids=voxel_ids,
                             cv_scheme=cv_scheme, stage=stage), f)
        print(f'Saved {name} [{stage}] -> {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subject', type=int)
    ap.add_argument('--roi', required=True)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--cv-scheme', choices=list(SCHEMES), default='loco')
    ap.add_argument('--gd-iterations', type=int, default=8000)
    ap.add_argument('--learning-rate', type=float, default=0.01)
    ap.add_argument('--max-voxels', type=int, default=500)
    ap.add_argument('--output-root', default=None,
                    help='Override the output root (e.g. for smoke tests).')
    ap.add_argument('--models', nargs='*', default=None,
                    help='Subset (ancestors are fit or loaded as needed).')
    ap.add_argument('--polish', action='store_true',
                    help='Then refit every model with all parameters free '
                         '(written to <root>_polished).')
    ap.add_argument('--bounded', action='store_true',
                    help='Box the shared terms (SHARED_BOUNDS); root gets '
                         'a _bounded suffix.')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.roi, cv_scheme=a.cv_scheme,
         gd_iterations=a.gd_iterations, learning_rate=a.learning_rate,
         models=a.models, max_voxels=a.max_voxels,
         output_root=a.output_root, polish=a.polish, bounded=a.bounded)
