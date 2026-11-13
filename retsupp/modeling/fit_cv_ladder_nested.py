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

``--loss nll`` fits the shared terms to the Gaussian negative log-likelihood
with each voxel's noise variance profiled out, sum_v (T/2) log SSE_v, instead
of the summed SSE, so a voxel counts by its fit relative to its own noise and
not by its scale. ``--base-root`` reuses that root's m00 and m0 fits instead
of refitting them (their per-voxel pRFs and the voxel set must match).

``--family dn`` uses the divisive-normalization pRF (``dn_attention_field``)
instead of the DoG: m00 starts from the whole-brain m5 fit, the gain fields are
the same. Root gets a _dn suffix.

``--family gauss`` uses a Gaussian pRF: the DoG classes with the surround amplitude fixed at 0
(``gauss_attention_field``); m00 starts from x, y, sd of the Gaussian fold fits with the joint per-ROI HRF
(``cv/fit_prf_cv_roi_hrf_joint.py --family gauss``, stage 3, fold mean; m4 where missing). Root gets a _gauss suffix.

``--hp-rotation K`` (1-3) relabels every run's HP location as the ring position K
steps away, for a null distribution of the HP-specific terms. Only the labels move:
the data, stimulus and distractor and target events stay as they were. Models with
tied gains do not depend on the labels, so with ``--base-root`` they are loaded from
there. Root gets a _hprot<K> suffix.

``--roi`` takes several ROIs; the data are then loaded once. ``--warm-root`` starts m00 and m0 from
earlier fits of the same voxels (with ``--gd-iterations`` and ``--gd-min-iterations`` set low, a refit
after a small change such as the HRF is cheap). ``--sigma-starts`` and ``--lbfgs-rounds`` set the
search of the shared fits.

``--models m8_full_dT`` fits the target gain split by target at HP vs elsewhere (``target_split``) from
m7_full; all its ancestors are loaded from ``--base-root``. ``--models m8_full_adjopp`` likewise splits the
sustained LP gain into the two locations next to HP and the one opposite it (``lp_split``; one new term,
g_adj_minus_opp = g_adj − g_opp, with g_LP kept as the mean of the three LP gains). ``--models m7_full_sigtied``
ties the transient width to the sustained one (σ_dyn := σ_AF, ``sigma_tied``; one term fewer than m7_full),
refitting σ_AF and the gains from m7_full.

``--models m8_full_dnb m8_full_dnd m8_full_dnbd`` (``--family dn`` only) add a second sustained field that scales
the DN neural baseline (w_b), the surround baseline (w_d) or both, with one shape (σ_AF, HP/LP pattern θ) shared
(``dn_baseline_field``). Started from m7_full with the weights at 0 (checked to reproduce m7's prediction) and
from three other θ; the joint model also starts from both one-weight fits. A Gauss-Newton Hessian (JᵀJ) of the
SSE over the shared terms is stored (``hessian_corr``) to show whether w_b and w_d trade off.

``--models m8_full_dogk`` (DoG only) splits the sustained field between the centre (driving) and surround
(suppressive) pooling with one shared tilt κ: centre weight 1 + κ, surround 1 − κ (``dog_centre_surround_field``;
κ = 0 is m7_full, checked at the start, κ = ±1 field on one pooling only). Started from m7_full and from κ = ±1;
``m8_full_dogcen`` / ``m8_full_dogsur`` fit the κ = +1 / −1 special cases (also used as dogk starts when listed
before it). A Gauss-Newton Hessian is stored for dogk (κ against g_HP, g_LP).

``--frozen-prf`` (with a CV scheme and an all-runs ``--base-root``) is the cheap held-out test for models that
differ only in shared terms: per-voxel pRFs and m00-m5 are taken from the all-runs fit (same in every fold, so
their leak into the held-out runs is the same for all models), m7_full and the requested models are refitted per
fold. Root gets a _frozenprf suffix.

``--reliable-only`` keeps the selected voxels with model-free cross-run reliability > 0 (mean over
the ``prf_cv_reliability`` folds). The m00, m0, m1 and m3b fits come from ``--base-root``, restricted
to those voxels, so m4 onward refit the shared terms on them. Root gets a _reliable suffix.

``--selection-table <dir> --min-r R`` selects the voxels without p_signal and without the 500 cap: ROI voxels with
model-free cross-run reliability r > R and p <= ``--max-p`` (0.01) against a circular-shift null
(``cv/voxel_reliability_null.py``), plus ``--max-bold-sd`` as usual. Root gets a _relnull_r<R> suffix.

``--polish-method lm`` replaces the alternating polish by a joint Levenberg-Marquardt fit of the per-voxel pRFs and
the shared terms (``joint_lm``: damped Gauss-Newton with the voxel blocks eliminated by the Schur complement, as in
bundle adjustment; float64 solves on the float32 model), run to convergence (``--polish-lm-iter``,
``--polish-lm-ftol``). See ``notes/joint_optimisation_2026-11-13.md``.

``--warm-polish-root <polished root>`` (with ``--polish --polish-method alt|lm --polish-models ...``) fits only the
all-free models, skipping the clamped chain: each starts from the shared terms of that root's fit of the same
model, subject and area, with per-voxel pRFs from it where the voxel was fitted there and from the whole-brain m4
pRF (baseline/amplitude by least squares, then Adam on those pRFs with the shared terms fixed) elsewhere. For a
new voxel selection this replaces m00 ... m7 + polish by one warm polish. The alternating polish saves a
checkpoint after every round (``*_polish-ckpt.pkl`` next to the output) and resumes from it.

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
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize

from braincoder.hrf import SPMHRFModel  # noqa: F401
from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, hrf_values, load_table as load_hrf_table
from braincoder.optimize import ParameterFitter
from retsupp.modeling.cv_helpers import per_voxel_r2, write_cv_tsvs
from retsupp.modeling.dn_attention_field import DNDynamicAttentionFieldPRF2D, init_from_m5
from retsupp.modeling.target_split import NEW as T_HP, TargetSplitDN, TargetSplitDoG
from retsupp.modeling.lp_split import NEW as LP_D, LPSplitDN, LPSplitDoG
from retsupp.modeling.sigma_tied import SigmaTiedDN, SigmaTiedDoG
from retsupp.modeling.dn_baseline_field import NEW as BF_TERMS, DNBaselineField
from retsupp.modeling.dog_centre_surround_field import NEW as KAPPA, DoGCentreSurroundField
from retsupp.modeling.gauss_attention_field import gauss_init, gaussian
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial as DoGFactorial,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as DoGFactorialItems,
)
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
    'm8_full_dT': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
    'm8_full_adjopp': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
    'm7_full_sigtied': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
    'm8_full_dnb': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
    'm8_full_dnd': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
    'm8_full_dnbd': ('free', 'free', 'free', 'free', 'free', 'free', 'm8_full_dnb'),
    'm8_full_dogk': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
    'm8_full_dogcen': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
    'm8_full_dogsur': ('free', 'free', 'free', 'free', 'free', 'free', 'm7_full'),
}
# Models fitted on top of finished ladders, run only when asked: the target gain split by target at HP vs
# elsewhere (``target_split``) and the sustained LP gain split into next to HP vs opposite (``lp_split``).
# Each adds one shared term to its parent, started at the value that reproduces the parent.
SPLIT_TERMS = {'m8_full_dT': T_HP, 'm8_full_adjopp': LP_D}
# m7_full with one width (σ_dyn := σ_AF): one term fewer, so it is fitted from m7 but is not above it.
SIGMA_TIED = ('m7_full_sigtied',)
# DN only: a second sustained field (σ_AF, HP/LP pattern θ) scales the neural baseline b (w_b), the surround
# baseline d (w_d), or both (``dn_baseline_field``). Fitted from m7_full with the weights at 0; the joint model
# starts from the b-only fit and, as a second candidate, from the d-only fit.
BASELINE_FIELD = {'m8_full_dnb': ('b',), 'm8_full_dnd': ('d',), 'm8_full_dnbd': ('b', 'd')}
EXTRA_PARENTS = {'m8_full_dnbd': ('m8_full_dnd',)}
THETA_STARTS = (0.785398, -0.785398, 1.570796)   # extra θ starts (parent σ's): uniform, HP vs LP, LP only
# DoG only: the sustained field split between the centre (driving) and surround (suppressive) pooling by a shared
# tilt κ (``dog_centre_surround_field``): centre weight 1 + κ, surround 1 − κ, κ = 0 is m7_full exactly. dogk fits κ
# (from κ = 0, checked to reproduce m7, and from κ = ±1); dogcen / dogsur are the special cases κ = +1 / −1 (field on
# the centre / surround pooling only), fitted only when asked.
CS_FIELD = {'m8_full_dogk': None, 'm8_full_dogcen': 1.0, 'm8_full_dogsur': -1.0}
KAPPA_STARTS = (1.0, -1.0)   # extra κ starts of dogk (parent σ's and gains)
TARGET_SPLIT = tuple(SPLIT_TERMS) + SIGMA_TIED + tuple(BASELINE_FIELD) + tuple(CS_FIELD)
GAINS = ['g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn']
# Models whose fit cannot depend on which ring position is labelled HP.
LABEL_BLIND = ('m1_T', 'm2_D', 'm3_S', 'm3b_TD', 'm4_TDS')
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
                 **{g: (-10.0, 10.0) for g in GAINS + [T_HP, LP_D]},
                 'w_b': (-5.0, 5.0), 'w_d': (-5.0, 5.0), 'theta_bd': (-6.3, 6.3),
                 KAPPA: (-2.0, 2.0)}
POLISH_MAX_ITER = 3000
SCHEMES = tuple(CV_SCHEMES) + ('full',)


def extra_terms(name):
    """Shared terms a model adds to the m7 set."""
    if name in SPLIT_TERMS:
        return [SPLIT_TERMS[name]]
    if name in BASELINE_FIELD:
        return [t for t, k in zip(BF_TERMS, ('b', 'd', None))
                if k is None or k in BASELINE_FIELD[name]]
    if name in CS_FIELD:
        return [KAPPA] if CS_FIELD[name] is None else []
    return []


def sign_pattern_of(name):
    s = LADDER[name]
    return dict(zip(GAINS, s[:5]))


def gd_fit(model, bold_train, paradigm, init_pars, sign_pattern, item_scale,
           max_n_iterations, learning_rate, min_n_iterations=1000):
    """Full per-voxel gradient descent (m00 / m0), run to a plateau."""
    fitter = ParameterFitter(model, pd.DataFrame(bold_train), paradigm)
    t0 = time.time()
    fit_pars = fitter.fit(
        init_pars=init_pars,
        max_n_iterations=max_n_iterations,
        min_n_iterations=min_n_iterations,
        lag=200,
        r2_atol=1e-7,
        shared_pars=shared_pars_for(sign_pattern, item_scale),
        fixed_pars=fixed_pars_for(sign_pattern, item_scale),
        learning_rate=learning_rate,
        progressbar=False,
    )
    info = dict(optimizer='adam_gd', seconds=time.time() - t0)
    return fit_pars, info


def make_sse_fn(model, bold_train, paradigm, raw_pars, free_ix, loss='sse'):
    """Objective and gradient as a function of the free shared raw parameters.

    ``loss='sse'``: summed SSE. ``loss='nll'``: sum_v (T/2) log SSE_v.
    """
    n_timepoints = float(bold_train.shape[0])
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
            sse_v = tf.reduce_sum((y - tf.cast(pred, tf.float64)) ** 2, axis=0)
            if loss == 'nll':
                sse = 0.5 * n_timepoints * tf.reduce_sum(tf.math.log(sse_v))
            else:
                sse = tf.reduce_sum(sse_v)
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
               max_iter=2000, bounded=False, loss='sse', sigma_starts=SIGMA_STARTS,
               lbfgs_rounds=LBFGS_ROUNDS, hessian=False):
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
    fun = make_sse_fn(model, bold_train, paradigm, raw0, free_ix, loss=loss)
    bnds = raw_bounds(model, init_pars, free_shared) if bounded else None

    def clip(x):
        if bnds is None:
            return x
        return np.clip(x, [b[0] for b in bnds], [b[1] for b in bnds])

    t0 = time.time()
    f_parent, _ = fun(clip(raw0[0, free_ix].astype(np.float64)))
    best = None
    starts = []
    for s in sigma_starts:
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
        for _round in range(lbfgs_rounds):
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
                loss=loss, sse_parent_start=f_parent, sse_final=best[0],
                grad_norm=float(np.linalg.norm(g)), starts=starts)
    if hessian:
        info.update(gn_hessian(model, paradigm, raw_best, free_ix, free_shared))
    return fit_pars, info


def gn_hessian(model, paradigm, raw_pars, free_ix, labels):
    """Gauss-Newton Hessian JᵀJ of the SSE over the free shared terms (raw space), at ``raw_pars``.

    J is the Jacobian of all predictions (T × V) with respect to each shared term, by forward-mode
    differentiation (one pass per term; the shared value sits in every voxel's row). JᵀJ is positive
    semi-definite, and its inverse is proportional to the Laplace covariance of the shared terms, so the
    correlations show which terms trade off (e.g. w_b against w_d). Not the full Hessian: residual
    curvature is ignored, which is standard for least squares near the optimum.
    """
    par_ = model.stimulus._clean_paradigm(paradigm)
    raw = tf.constant(np.asarray(raw_pars, dtype=np.float32))
    cols = []
    for k in free_ix:
        tangent = np.zeros(raw.shape, dtype=np.float32)
        tangent[:, k] = 1.0
        with tf.autodiff.ForwardAccumulator(raw, tf.constant(tangent)) as acc:
            pred = model._predict(par_[tf.newaxis, ...],
                                  model._transform_parameters_forward(raw)[tf.newaxis, ...], None)
        cols.append(np.asarray(acc.jvp(pred), dtype=np.float64).ravel())
    jac = np.stack(cols, axis=1)
    hess = jac.T @ jac
    cov = np.linalg.pinv(hess)   # θ has no curvature where both weights are 0
    sd = np.sqrt(np.abs(np.diag(cov)))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.outer(sd, sd)
    return dict(hessian_labels=list(labels), hessian=hess, hessian_corr=corr,
                hessian_eig=np.linalg.eigvalsh(hess))


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
    if T_HP in labels and init[T_HP].isna().all():
        init[T_HP] = init['g_T_dyn']   # split starts unsplit
    if LP_D in labels and init[LP_D].isna().all():
        init[LP_D] = 0.0               # g_adj = g_opp = g_LP
    for t in BF_TERMS:
        if t in labels and init[t].isna().all():
            init[t] = 0.0              # weights 0: the parent's prediction (θ = 0: field at HP only)
    if KAPPA in labels and init[KAPPA].isna().all():
        init[KAPPA] = 0.0              # κ = 0: the parent's prediction (fixed-κ classes override it)
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


def polish_alternating(model, bold_train, paradigm, init_pars, free_shared, fixed, shared_all,
                       bounded=False, rounds=2, learning_rate=1e-3, max_iter=3000, checkpoint=None, tol=0.0):
    """Block-coordinate polish, robust to float32 gradient noise (where one L-BFGS over all parameters can stall
    with an ABNORMAL line search): per-voxel pRFs by Adam gradient descent with the shared terms fixed, then the
    shared terms by bounded L-BFGS with the pRFs fixed, ``rounds`` times. Each step is kept only if it lowers the
    training SSE, so the result is never above the clamped start.

    ``checkpoint``: a pickle path written after every round (best parameters so far); a rerun resumes from the
    round after the last one saved, so a task that hits its time limit loses at most one round.

    ``tol`` > 0: stop once a round lowers the training SSE by less than ``tol`` (relative); ``converged`` in the
    returned info says whether that happened before ``rounds``."""
    y = np.asarray(bold_train, dtype=np.float64)

    def sse(p):
        return float(((y - np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)) ** 2).sum())
    t0 = time.time()
    labels = list(model.parameter_labels)
    best = init_pars[labels].copy()
    f0 = fbest = sse(best)
    log = []
    start_round = 0
    if checkpoint is not None and Path(checkpoint).exists():
        with open(checkpoint, 'rb') as f:
            ck = pickle.load(f)
        best, fbest, f0, log = ck['best'].reindex(columns=labels), ck['fbest'], ck['f0'], ck['log']
        start_round = ck['round'] + 1
        print(f'    resuming after alt round {ck["round"]} from {checkpoint} (SSE {fbest:.2f})')
    freeze = list(dict.fromkeys([p for p in (fixed or []) + list(shared_all) if p in labels]))
    converged = bool(log) and tol > 0 and log[-1].get('rel_change', np.inf) < tol
    for r in range(start_round, rounds):
        if converged:
            break
        f_round_start = fbest
        fitter = ParameterFitter(model, pd.DataFrame(bold_train), paradigm)
        pv = fitter.fit(init_pars=best, max_n_iterations=max_iter, min_n_iterations=200, lag=100, r2_atol=1e-8,
                        fixed_pars=freeze, learning_rate=learning_rate, progressbar=False)
        pv = pv.reindex(columns=labels)
        f_pv = sse(pv)
        if f_pv < fbest:
            best, fbest = pv, f_pv
        sh, info = shared_fit(model, bold_train, paradigm, best, free_shared, bounded=bounded, sigma_starts=(None,))
        sh = sh.reindex(columns=labels)
        f_sh = sse(sh)
        if f_sh < fbest:
            best, fbest = sh, f_sh
        rel = (f_round_start - fbest) / f_round_start
        converged = tol > 0 and rel < tol
        log.append(dict(round=r, sse_after_prf=f_pv, sse_after_shared=f_sh, sse_best=fbest, rel_change=rel,
                        seconds=time.time() - t0))
        print(f'    alt round {r}: SSE pRF step {f_pv:.2f}, shared step {f_sh:.2f} (start {f0:.2f}) '
              f'| rel change {rel:.2e} | {time.time() - t0:.0f}s', flush=True)
        if checkpoint is not None:
            _atomic_pickle(checkpoint, dict(round=r, best=best, fbest=fbest, f0=f0, log=log))
    return best, dict(optimizer='alt_gd_lbfgs', seconds=time.time() - t0, sse_clamped=f0, sse_final=fbest,
                      nit=0, rounds=rounds, status=0, message='alternating', grad_norm=float('nan'),
                      grad_norm_shared=float('nan'), log=log, converged=converged, tol=tol)


def _atomic_pickle(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    tmp.replace(path)


ROI_ALIASES = {'V3AB': ['V3A', 'V3B'], 'LO': ['LO1', 'LO2'], 'TO': ['TO1', 'TO2'], 'VO': ['VO1', 'VO2']}


def roi_mask(sub, roi, n_vox):
    """BOLD-mask voxels of ``roi``, built as in ``select_roi_voxels_psignal`` (same aliases, same masker)."""
    masker = sub.get_bold_mask(return_masker=True)
    masker.fit()
    out = np.zeros(n_vox, dtype=bool)
    for r in ROI_ALIASES.get(roi, [roi]):
        out |= masker.transform(sub.get_retinotopic_roi(roi=r, bold_space=True)).astype(bool).ravel()
    return out


def select_from_table(sub, roi, prf_pars, table_dir, min_r, max_p):
    """Voxels of ``roi`` with cross-run reliability r > ``min_r`` and circular-shift p <= ``max_p``
    (``cv/voxel_reliability_null.py``), and a finite whole-brain DoG pRF (the start of the fits). No p_signal,
    no cap on the number of voxels."""
    s = sub.subject_id
    tab = pd.read_csv(Path(table_dir) / f'sub-{s:02d}' / f'sub-{s:02d}_reliability_null.tsv.gz', sep='\t')
    n_vox = len(prf_pars)
    in_roi = roi_mask(sub, roi, n_vox)
    ok = np.zeros(n_vox, dtype=bool)
    sel = tab[(tab.r > min_r) & (tab.p <= max_p)].voxel_index.to_numpy()
    ok[sel] = True
    covered = np.zeros(n_vox, dtype=bool)
    covered[tab.voxel_index.to_numpy()] = True
    finite = np.isfinite(prf_pars[['x', 'y', 'sd', 'amplitude']].to_numpy()).all(1)
    mask = in_roi & ok & finite
    descr = (f'ROI {roi} | reliability r > {min_r:g}, null p <= {max_p:g}: {(in_roi & ok).sum()} of {in_roi.sum()} '
             f'ROI voxels ({(in_roi & ~covered).sum()} not in the table), {(in_roi & ok & ~finite).sum()} without '
             f'a finite m4 pRF dropped')
    if mask.sum() == 0:
        raise RuntimeError(f'No voxels survive: {descr}.')
    return mask, descr


def bar_paradigm(sub, run_meta, resolution, grid_radius):
    return np.vstack([
        sub.get_bar_stimulus(session=rm.session, run=rm.run,
                             resolution=resolution,
                             grid_radius=grid_radius
                             ).astype(np.float32).reshape(rm.n_T, -1)
        for rm in run_meta])


def main(subject, bids_folder, rois, cv_scheme='loco', resolution=50,
         grid_radius=5.0, max_voxels=500, p_signal_thr=0.5,
         gd_iterations=8000, learning_rate=0.01, models=None,
         output_root=None, polish=False, bounded=False, max_bold_sd=None,
         loss='sse', base_roots=(), family='dog', hp_rotation=0, reliable_only=False,
         warm_roots=(), gd_min_iterations=1000, sigma_starts=SIGMA_STARTS, lbfgs_rounds=LBFGS_ROUNDS,
         hrf_table=None, frozen_prf=False, polish_models=None, polish_method='lbfgs',
         polish_suffix='_polished', polish_rounds=2, polish_max_iter=3000, polish_tol=0.0,
         polish_lm_iter=300, polish_lm_ftol=1e-10,
         selection_table=None, min_r=None, max_p=0.01, warm_polish_root=None):
    if cv_scheme not in SCHEMES:
        raise ValueError(f'cv_scheme must be in {SCHEMES}')
    if frozen_prf and (cv_scheme == 'full' or not base_roots):
        raise ValueError('--frozen-prf needs a CV scheme and --base-root (a full-scheme root)')
    if hp_rotation and not base_roots:
        raise ValueError('--hp-rotation needs --base-root for the tied models')
    if polish and loss != 'sse':
        raise ValueError('--polish only implements the SSE loss')
    if polish and family != 'dog':
        raise ValueError('--polish only implements the DoG family')
    if selection_table and (reliable_only or min_r is None):
        raise ValueError('--selection-table needs --min-r and replaces --reliable-only')
    if warm_polish_root and not (polish and polish_models and polish_method in ('alt', 'lm')):
        raise ValueError('--warm-polish-root needs --polish --polish-method alt|lm --polish-models ...')
    per_voxel_cols = PER_VOXEL_COLS + (
        ['neural_baseline', 'surround_baseline'] if family == 'dn' else [])
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    root = Path(output_root) if output_root else (
        bids_folder / 'derivatives' / (
            'af_prf_cv_ladder_nested'
            + ('' if cv_scheme == 'loco' else f'_{cv_scheme}')
            + ('_bounded' if bounded else '')
            + (f'_maxsd{max_bold_sd:g}' if max_bold_sd else '')
            + ('' if loss == 'sse' else f'_{loss}')
            + ('' if family == 'dog' else f'_{family}')
            + (f'_hprot{hp_rotation}' if hp_rotation else '')
            + ('_reliable' if reliable_only else '')
            + (f'_relnull_r{min_r:g}' if selection_table else '')
            + ('_roihrf' if hrf_table else '')
            + ('_frozenprf' if frozen_prf else '')))
    polish_root = root.parent / f'{root.name}{polish_suffix}'
    hrf_tab = load_hrf_table(hrf_table)
    models = models or [m for m in LADDER if m not in TARGET_SPLIT]

    (bold, par_items, ci, di, ti, grid_coords, _masker,
     run_meta) = build_data_and_paradigm(
        sub, resolution=resolution, grid_radius=grid_radius,
        item_shape='rectangle')
    ci = np.roll(ci, hp_rotation, axis=1)   # columns follow the ring order
    par_bar = bar_paradigm(sub, run_meta, resolution, grid_radius)

    prf_pars = pd.DataFrame(
        sub.get_prf_parameters_volume(model=4, return_images=False))
    def fit_one(roi):
        if selection_table:
            voxel_mask, descr = select_from_table(sub, roi, prf_pars, selection_table, min_r, max_p)
        else:
            voxel_mask, descr = choose_voxel_mask(
                sub, roi, prf_pars, use_psignal=True, p_signal_thr=p_signal_thr,
                r2_thr=0.05, model_label=4, max_voxels=max_voxels)
        print(f'{descr}: {voxel_mask.sum()} voxels')
        if max_bold_sd:
            # Applied after the top-N ranking, so the result is a strict subset of
            # the voxels of the fit without this limit.
            sd = bold.std(0)
            drop = voxel_mask & (sd > max_bold_sd)
            voxel_mask = voxel_mask & ~drop
            print(f'BOLD SD > {max_bold_sd:g}: dropped {drop.sum()} voxels '
                  f'{np.where(drop)[0].tolist()}, {voxel_mask.sum()} left')
        if reliable_only:
            rel = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
                (bids_folder / 'derivatives' / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob(
                    f'sub-{subject:02d}_fold-*_reliability.tsv.gz'))])
            rel = rel.groupby('voxel_index').reliability.mean()
            reliable = np.zeros(voxel_mask.size, dtype=bool)
            reliable[rel.index[rel > 0]] = True
            print(f'Reliability > 0: {(voxel_mask & reliable).sum()} of {voxel_mask.sum()} voxels')
            voxel_mask = voxel_mask & reliable
        voxel_ids = np.where(voxel_mask)[0]
        ring_positions = get_ring_positions()
        hrf_model = roi_hrf_model(sub, roi, hrf_tab)   # canonical without --hrf-table
        print(f'HRF: {hrf_values(subject, roi, hrf_tab)}')
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

        def ckpt_path(name, k, what):
            return (polish_root / name / f'sub-{subject:02d}'
                    / f'sub-{subject:02d}_roi-{roi}_fold-{k}_{what}.pkl')

        def warm_polish_init(name, k, m_tr, y_tr, paradigm, sp, item_scale):
            """Start of the all-free fit of ``name`` on the current voxels without the clamped chain.

            Shared terms (gains, σ's, item scale): the ``--warm-polish-root`` fit of the same model, subject and
            area (fitted on another voxel set). Per-voxel pRFs: that fit where the voxel was in it; elsewhere the
            whole-brain DoG pRF (model 4) with baseline and amplitude refitted by least squares, then Adam on those
            voxels' pRFs with the shared terms fixed at the warm values (exact: with the shared terms fixed the
            voxels are independent), as m00/m0 do (``--gd-iterations``, ``--learning-rate``). The alternating
            polish then refits everything. Saved as a checkpoint, so a rerun does not redo it."""
            ck = ckpt_path(name, k, 'warm-init')
            if ck.exists():
                with open(ck, 'rb') as f:
                    d = pickle.load(f)
                print(f'  warm init {name} fold {k}: loaded {ck}')
                return d['init'], d['info']
            labels = list(m_tr.parameter_labels)
            pkl = Path(warm_polish_root) / name / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
            with open(pkl, 'rb') as f:
                old = pickle.load(f)
            folds_old = old['fit_pars_per_fold']
            p_old = folds_old[min(k, len(folds_old) - 1)]
            p_old = p_old.set_index(pd.Index(np.asarray(old['voxel_ids'])))
            shared_all = shared_pars_for(sp, item_scale) + (fixed_pars_for(sp, item_scale) or [])
            per_voxel = [c for c in labels if c not in shared_all]
            init = make_init_pars(prf_pars, voxel_mask, sp, sigma_af_init=2.0, sigma_dyn_init=2.0,
                                  item_scale=item_scale).reindex(columns=labels)
            for c in labels:
                if c not in per_voxel:
                    if c not in p_old.columns:
                        raise ValueError(f'{pkl} lacks shared term {c}')
                    init[c] = float(p_old[c].iloc[0])
            hit = np.isin(voxel_ids, p_old.index)
            pv_ix = [labels.index(c) for c in per_voxel]
            init.iloc[np.where(hit)[0], pv_ix] = p_old.loc[voxel_ids[hit], per_voxel].to_numpy()
            new = np.where(~hit)[0]
            t0 = time.time()
            info = dict(warm_root=str(warm_polish_root), n_warm=int(hit.sum()), n_new=int(new.size))
            if new.size:
                y_new = pd.DataFrame(np.asarray(y_tr)[:, new])
                sub_init = init.iloc[new].reset_index(drop=True)
                sub_init = ParameterFitter(m_tr, y_new, paradigm).refine_baseline_and_amplitude(
                    sub_init, l2_alpha=1e-3)
                fitter = ParameterFitter(m_tr, y_new, paradigm)
                pv = fitter.fit(init_pars=sub_init.reindex(columns=labels), max_n_iterations=gd_iterations,
                                min_n_iterations=gd_min_iterations, lag=200, r2_atol=1e-7, fixed_pars=shared_all,
                                learning_rate=learning_rate, progressbar=False)
                init.iloc[new, pv_ix] = pv.reindex(columns=labels)[per_voxel].to_numpy()
            info['seconds_new_voxels'] = time.time() - t0
            if init[labels].isna().any().any():
                raise ValueError(f'{name} fold {k}: NaN in the warm start')
            print(f'  warm init {name} fold {k}: {hit.sum()} voxels from {warm_polish_root}, {new.size} new '
                  f'(m4 + Adam on their pRFs, {info["seconds_new_voxels"]:.0f}s)', flush=True)
            _atomic_pickle(ck, dict(init=init, info=info))
            return init, info

        def warm(name, k, init):
            """Start from an earlier fit of the same model and voxels where one exists."""
            for wr in warm_roots:
                pkl = Path(wr) / name / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
                if not pkl.exists():
                    continue
                with open(pkl, 'rb') as f:
                    old = pickle.load(f)
                folds_old = old['fit_pars_per_fold']
                p_old = folds_old[min(k, len(folds_old) - 1)]
                if p_old is None:
                    continue
                p_old = p_old.set_index(pd.Index(np.asarray(old['voxel_ids'])))
                hit = np.isin(voxel_ids, p_old.index)
                cols = [c for c in init.columns if c in p_old.columns]
                out = init.copy()
                out.loc[hit, cols] = p_old.loc[voxel_ids[hit], cols].to_numpy()
                print(f'  warm start {name} fold {k}: {hit.sum()} of {len(hit)} voxels from {wr}')
                return out
            return init

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
                chain.extend(EXTRA_PARENTS.get(n, ()))
                n = LADDER[n][6]
            for n in reversed(chain):
                if n not in needed:
                    needed.append(n)

        stages = [('clamped', n) for n in needed]
        if warm_polish_root:
            # Only the all-free fits of the named models, each started from the earlier polished fit (no clamped
            # chain): see warm_polish_init. Where that root has no fit of this subject and area (no voxels in its
            # selection), the full chain is run instead, from the whole-brain pRFs as without the option.
            have = [(Path(warm_polish_root) / n / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
                     ).exists() for n in polish_models]
            if all(have):
                stages = []
            else:
                print(f'No warm-start fit of {roi} in {warm_polish_root}: running the full chain')
        use_warm = warm_polish_root and not stages
        if polish:
            # --polish-models restricts the all-free pass to these models (each from its own clamped optimum)
            stages += [('polish', n) for n in needed if not polish_models or n in polish_models]
        polished: dict[str, list] = {}
        for stage, name in stages:
            out_root = root if stage == 'clamped' else polish_root
            store = fits if stage == 'clamped' else polished
            cached = load(name, out_root)
            from_base = (('m00_bar', 'm0_prf') + (LABEL_BLIND if hp_rotation else ())
                         + (('m1_T', 'm3b_TD') if reliable_only else ()))
            if any(m in TARGET_SPLIT for m in models):
                # The split is fitted on top of finished ladders: every ancestor comes from --base-root.
                from_base = from_base + tuple(n for n in needed if n not in TARGET_SPLIT)
            if frozen_prf:
                # Per-voxel pRFs and the ladder below m7 come from the all-runs fit; m7 and the variants are
                # refitted per fold (their shared terms only), so held-out prediction compares the shared
                # gain-field terms with the pRFs identical in every model.
                from_base = tuple(n for n in needed if n not in TARGET_SPLIT and n != 'm7_full')
            if cached is None and base_roots and name in from_base:
                # First root holding this fit wins, so an overlay root goes first.
                for base in base_roots:
                    cached = load(name, Path(base))
                    if cached is not None:
                        break
                if cached is None:
                    raise FileNotFoundError(f'{name} not in {base_roots}')
                if reliable_only:
                    pkl = Path(base) / name / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
                    with open(pkl, 'rb') as f:
                        base_ids = np.asarray(pickle.load(f)['voxel_ids'])
                    keep = np.isin(base_ids, voxel_ids)
                    if keep.sum() != voxel_ids.size:
                        raise ValueError(f'{name} in {base}: lacks {voxel_ids.size - keep.sum()} voxels')
                    cached = [None if p is None else p.loc[keep].reset_index(drop=True) for p in cached]
                if any(p is not None and len(p) != voxel_ids.size for p in cached):
                    raise ValueError(f'{name} in {base}: voxel set differs')
                if frozen_prf and len(cached) == 1:
                    cached = cached * len(splits)
            if cached is not None:
                print(f'== {name} [{stage}]: loaded existing fit')
                store[name] = cached
                continue
            sus_hp, sus_lp, dyn_hp, dyn_lp, tgt, item_scale, parent = LADDER[name]
            sp = sign_pattern_of(name)
            if family == 'dn' and item_scale == 'none':
                item_scale = 'fixed'   # one class; the bar paradigm has no item pixels
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
                s = split[1] if LADDER[name][5] == 'none' else split[0]
                train, held = s['train'], s['held']
                y_tr = train['bold'][:, voxel_mask]
                y_he = None if held is None else held['bold'][:, voxel_mask]

                extra = extra_terms(name)

                def mk(d, name=name):
                    split_cls = {T_HP: (TargetSplitDoG, TargetSplitDN),
                                 LP_D: (LPSplitDoG, LPSplitDN)}.get(SPLIT_TERMS.get(name))
                    if name in SIGMA_TIED:
                        split_cls = (SigmaTiedDoG, SigmaTiedDN)
                    if name in CS_FIELD:
                        if family != 'dog':
                            raise ValueError(f'{name} needs --family dog')
                        return DoGCentreSurroundField(
                            kappa_fixed=CS_FIELD[name], item_scale_mode=item_scale,
                            grid_coordinates=grid_coords, paradigm=d['paradigm'], hrf_model=hrf_model,
                            condition_indicator=d['condition_indicator'],
                            dynamic_indicator=d['dynamic_indicator'],
                            target_indicator=d['target_indicator'],
                            ring_positions=ring_positions, sign_pattern=sp)
                    if name in BASELINE_FIELD:
                        if family != 'dn':
                            raise ValueError(f'{name} needs --family dn')
                        return DNBaselineField(
                            targets=BASELINE_FIELD[name],
                            grid_coordinates=grid_coords, paradigm=d['paradigm'],
                            hrf_model=hrf_model,
                            condition_indicator=d['condition_indicator'],
                            dynamic_indicator=d['dynamic_indicator'],
                            target_indicator=d['target_indicator'],
                            ring_positions=ring_positions, sign_pattern=sp,
                            item_scale_mode=item_scale)
                    if family == 'dn':
                        return (split_cls[1] if split_cls else DNDynamicAttentionFieldPRF2D)(
                            grid_coordinates=grid_coords, paradigm=d['paradigm'],
                            hrf_model=hrf_model,
                            condition_indicator=d['condition_indicator'],
                            dynamic_indicator=d['dynamic_indicator'],
                            target_indicator=d['target_indicator'],
                            ring_positions=ring_positions, sign_pattern=sp,
                            item_scale_mode=item_scale)
                    kw = dict(grid_coordinates=grid_coords, paradigm=d['paradigm'],
                              condition_indicator=d['condition_indicator'],
                              dynamic_indicator=d['dynamic_indicator'],
                              target_indicator=d['target_indicator'],
                              ring_positions=ring_positions, hrf_model=hrf_model, sign_pattern=sp)
                    if family == 'gauss':
                        # The DoG classes with the surround amplitude fixed at 0 (gauss_attention_field).
                        if split_cls:
                            return gaussian(split_cls[0])(item_scale_mode=item_scale, **kw)
                        if item_scale == 'none':
                            return gaussian(DoGFactorial)(**kw)
                        return gaussian(DoGFactorialItems)(item_scale_mode=item_scale, **kw)
                    if split_cls:
                        return split_cls[0](item_scale_mode=item_scale, **kw)
                    return build_model(grid_coords=kw.pop('grid_coordinates'), item_scale=item_scale, **kw)
                m_tr = mk(train)
                labels = list(m_tr.parameter_labels)

                if stage == 'polish':
                    fixed = fixed_pars_for(sp, item_scale) or []
                    shared = shared_pars_for(sp, item_scale)
                    free_sh = [p for p in shared if p != 'sigma_T_dyn']
                    free_pv = [p for p in labels
                               if p not in shared and p not in fixed]
                    warm_info = None
                    if use_warm:
                        init, warm_info = warm_polish_init(name, k, m_tr, y_tr, train['paradigm'], sp, item_scale)
                    else:
                        init = fits[name][k].reindex(columns=labels)
                    if polish_method == 'lm':
                        from retsupp.modeling.joint_lm import polish_lm
                        pars, info = polish_lm(m_tr, y_tr, train['paradigm'], init, free_sh, free_pv,
                                               bounds=raw_bounds(m_tr, init, free_sh) if bounded else None,
                                               max_iter=polish_lm_iter, ftol=polish_lm_ftol)
                        if warm_info is not None:
                            info['warm_start'] = warm_info
                    elif polish_method == 'alt':
                        pars, info = polish_alternating(m_tr, y_tr, train['paradigm'], init, free_sh, fixed,
                                                        shared, bounded=bounded, rounds=polish_rounds,
                                                        max_iter=polish_max_iter,
                                                        checkpoint=ckpt_path(name, k, 'polish-ckpt'),
                                                        tol=polish_tol)
                        if warm_info is not None:
                            info['warm_start'] = warm_info
                    else:
                        pars, info = polish_fit(m_tr, y_tr, train['paradigm'], init,
                                                free_sh, free_pv, bounded=bounded)
                    print(f'  fold {k}: SSE clamped {info["sse_clamped"]:.2f}'
                          f' -> {info["sse_final"]:.2f} | nit {info["nit"]} | '
                          f'|grad| {info["grad_norm"]:.3g} (shared '
                          f'{info["grad_norm_shared"]:.3g}) | '
                          f'{info["seconds"]:.0f}s')
                elif name == 'm00_bar' and family == 'dn':
                    m5 = init_from_m5(sub, voxel_mask)
                    init = make_init_pars(m5, np.ones(len(m5), dtype=bool), sp,
                                          sigma_af_init=2.0, sigma_dyn_init=2.0,
                                          item_scale=item_scale)
                    init['neural_baseline'] = m5['neural_baseline'].values
                    init['surround_baseline'] = m5['surround_baseline'].values
                    init = init.reindex(columns=labels)
                    init = warm(name, k, init)
                    pars, info = gd_fit(m_tr, y_tr, train['paradigm'], init, sp,
                                        item_scale, gd_iterations, learning_rate, gd_min_iterations)
                elif name == 'm00_bar':
                    template = prf_pars
                    if family == 'gauss':
                        template, n_folds, have = gauss_init(bids_folder, subject, prf_pars)
                        print(f'  Gaussian template: {np.isin(voxel_ids, have).sum()} of {voxel_ids.size} voxels '
                              f'from the joint-HRF Gaussian fold fits ({n_folds} folds), the rest from m4')
                    init = make_init_pars(template, voxel_mask, sp,
                                          sigma_af_init=2.0, sigma_dyn_init=2.0,
                                          item_scale=item_scale)
                    init = init.reindex(columns=labels)
                    init = ParameterFitter(m_tr, pd.DataFrame(y_tr),
                                           train['paradigm']
                                           ).refine_baseline_and_amplitude(
                        init, l2_alpha=1e-3)
                    init = warm(name, k, init)
                    pars, info = gd_fit(m_tr, y_tr, train['paradigm'], init, sp,
                                        item_scale, gd_iterations, learning_rate, gd_min_iterations)
                elif name == 'm0_prf':
                    init = fits['m00_bar'][k].copy()
                    init['item_scale'] = 1.0
                    init = init.reindex(columns=labels)
                    init = warm(name, k, init)
                    pars, info = gd_fit(m_tr, y_tr, train['paradigm'], init, sp,
                                        item_scale, gd_iterations, learning_rate, gd_min_iterations)
                else:
                    init = child_init(fits[parent][k], m_tr, name)
                    free = [p for p in shared_pars_for(sp, item_scale)
                            if p != 'sigma_T_dyn'] + extra
                    starts = sigma_starts
                    if name in SIGMA_TIED:
                        # One width: σ_AF carries it ('parent' start = the parent's σ_AF); the parent's
                        # σ_dyn is an extra start.
                        free = [p for p in free if p != 'sigma_dyn']
                        starts = tuple(sigma_starts) + (float(fits[parent][k]['sigma_dyn'].iloc[0]),)
                    if name in BASELINE_FIELD or CS_FIELD.get(name, 1.0) is None:
                        # Equality check: the start (weights 0) must reproduce the parent's prediction.
                        p_par = np.asarray(mk(train, parent).predict(
                            paradigm=train['paradigm'], parameters=fits[parent][k]))
                        p_new = np.asarray(m_tr.predict(paradigm=train['paradigm'], parameters=init))
                        dev = float(np.abs(p_new - p_par).max())
                        print(f'  fold {k}: equality check vs {parent}: max |Δpred| = {dev:.3g}')
                        if not dev <= 1e-4 * max(1.0, float(np.abs(p_par).max())):
                            raise AssertionError(f'{name} start differs from {parent}: {dev}')
                    cands = [(init, starts)]
                    if frozen_prf and name == 'm7_full':
                        # Also start from the all-runs m7 optimum.
                        full = load('m7_full', Path(base_roots[0]))
                        if full is not None:
                            cands.append((full[0].reindex(columns=labels), (None,)))
                    if name in BASELINE_FIELD and parent == 'm7_full':
                        # Other HP/LP patterns of the new field, from the parent's σ's.
                        for th in THETA_STARTS:
                            alt = init.copy()
                            alt['theta_bd'] = th
                            cands.append((alt, (None,)))
                    if name in CS_FIELD:
                        sus = ['g_HP', 'g_LP']
                        if CS_FIELD[name] is None:
                            # The κ = ±1 ends, from the parent's gains and σ's, and the fixed-κ fits if done.
                            for kap in KAPPA_STARTS:
                                alt = init.copy()
                                alt[KAPPA] = kap
                                alt[sus] = alt[sus] / 2.0
                                cands.append((alt, (None,)))
                            for ep in ('m8_full_dogcen', 'm8_full_dogsur'):
                                if ep in fits and fits[ep][k] is not None:
                                    cands.append((fits[ep][k].reindex(columns=labels), (None,)))
                        else:
                            # Field on one pooling only (weight 2g): also start from half the parent's gains.
                            alt = init.copy()
                            alt[sus] = alt[sus] / 2.0
                            cands.append((alt, (None,)))
                    for ep in EXTRA_PARENTS.get(name, ()):
                        # e.g. the joint model also starts from the d-only optimum (w_b = 0).
                        cands.append((child_init(fits[ep][k], m_tr, name), (None,)))
                    best_fit, cand_log = None, []
                    for c_init, c_starts in cands:
                        c_pars, c_info = shared_fit(
                            m_tr, y_tr, train['paradigm'], c_init, free, bounded=bounded, loss=loss,
                            sigma_starts=c_starts, lbfgs_rounds=lbfgs_rounds,
                            hessian=name in BASELINE_FIELD or name == 'm8_full_dogk')
                        cand_log.append(dict(
                            theta_start=float(c_init['theta_bd'].iloc[0]) if 'theta_bd' in c_init else None,
                            kappa_start=float(c_init[KAPPA].iloc[0]) if KAPPA in c_init else None,
                            sse_start=c_info['sse_parent_start'], sse_final=c_info['sse_final'],
                            seconds=c_info['seconds']))
                        if best_fit is None or c_info['sse_final'] < best_fit[1]['sse_final']:
                            best_fit = (c_pars, c_info)
                    pars, info = best_fit
                    info['sse_parent_start'] = cand_log[0]['sse_start']   # the parent's own optimum
                    info['seconds'] = sum(c['seconds'] for c in cand_log)
                    if len(cand_log) > 1:
                        info['candidates'] = cand_log
                        print('  candidates: ' + ', '.join(
                            (f"θ0={c['theta_start']:.2f}" if c['theta_start'] is not None else
                             f"κ0={c['kappa_start']:.2f}" if c['kappa_start'] is not None else f'start {i}')
                            + f": {c['sse_final']:.2f}" for i, c in enumerate(cand_log)))
                    if 'hessian_corr' in info:
                        lab = info['hessian_labels']
                        corr = info['hessian_corr']
                        new_ix = [lab.index(t) for t in extra if t in lab]
                        print('  shared fit: ' + ', '.join(
                            f'{t}={pars[t].iloc[0]:.4g}' for t in ['g_HP', 'g_LP'] + extra))
                        for i in new_ix:
                            print(f'  corr({lab[i]}, ·): ' + ', '.join(
                                f'{lab[j]} {corr[i, j]:+.2f}' for j in range(len(lab)) if j != i))
                    print(f'  fold {k}: SSE parent-start {info["sse_parent_start"]:.2f}'
                          f' -> {info["sse_final"]:.2f} | |grad| '
                          f'{info["grad_norm"]:.3g} | {info["seconds"]:.0f}s')

                if pars[labels].isna().any().any():
                    raise ValueError(f'{name} fold {k}: NaN in the fitted parameters')
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
                    columns=ALL_SHARED_PARS + ['item_scale'] + extra).iloc[0].to_dict())
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
            shared_labels = ALL_SHARED_PARS + ['item_scale'] + extra_terms(name)
            write_cv_tsvs(
                out_dir, subject=subject, roi=roi, model=name,
                voxel_ids=voxel_ids, cv_r2_per_fold=cv_r2s,
                train_r2_per_fold=train_r2s,
                fit_pars_per_fold=[None if p is None else
                                   p.reindex(columns=per_voxel_cols + shared_labels)
                                   for p in fold_pars],
                shared_pars_per_fold=shared_dicts,
                shared_par_labels=shared_labels,
                per_voxel_par_cols=per_voxel_cols,
                train_run_meta_per_fold=tr_meta, held_run_meta_per_fold=he_meta,
                selector=('relnull' if selection_table else 'psignal'), p_signal_thr=p_signal_thr,
                extra_meta=dict(parent=parent, sign_pattern=sp,
                                item_scale=item_scale, cv_scheme=cv_scheme,
                                max_voxels=max_voxels,
                                gd_iterations=gd_iterations, stage=stage,
                                bounded=bounded, loss=loss, hp_rotation=hp_rotation,
                                selection_table=selection_table, min_r=min_r, max_p=max_p,
                                warm_polish_root=warm_polish_root if use_warm else None,
                                optimizer_info=infos))
            with open(pkl, 'wb') as f:
                pickle.dump(dict(fit_pars_per_fold=fold_pars, parent=parent,
                                 optimizer_info=infos, voxel_ids=voxel_ids,
                                 cv_scheme=cv_scheme, stage=stage, loss=loss), f)
            print(f'Saved {name} [{stage}] -> {out_dir}')

    failed = []
    for roi in ([rois] if isinstance(rois, str) else list(rois)):
        print(f'\n########## {roi} ##########', flush=True)
        try:
            fit_one(roi)
        except Exception:
            traceback.print_exc()
            failed.append(roi)
    if failed:
        raise RuntimeError(f'failed ROIs: {failed}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subject', type=int)
    ap.add_argument('--roi', nargs='+', required=True, help='One or more ROIs; the data are loaded once.')
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
    ap.add_argument('--polish-models', nargs='+', default=None,
                    help='With --polish: refit only these models with all parameters free (default: every model '
                         'fitted or loaded).')
    ap.add_argument('--polish-method', choices=['lbfgs', 'alt', 'lm'], default='lbfgs',
                    help='lbfgs: one L-BFGS over all parameters; alt: alternate Adam on the pRFs and L-BFGS on the '
                         'shared terms (robust to float32 gradient noise); lm: joint Levenberg-Marquardt over pRFs '
                         'and shared terms with the per-voxel blocks eliminated by the Schur complement (joint_lm).')
    ap.add_argument('--polish-lm-iter', type=int, default=300, help='LM polish: maximum iterations.')
    ap.add_argument('--polish-lm-ftol', type=float, default=1e-10,
                    help='LM polish: stop after 3 accepted steps that each lower training SSE by less than this '
                         '(relative).')
    ap.add_argument('--polish-suffix', default='_polished', help='Suffix of the polish root (tests).')
    ap.add_argument('--polish-rounds', type=int, default=2, help='Alternating polish: rounds (maximum with --polish-tol).')
    ap.add_argument('--polish-tol', type=float, default=0.0,
                    help='Alternating polish: stop when a round lowers training SSE by less than this (relative).')
    ap.add_argument('--polish-max-iter', type=int, default=3000, help='Alternating polish: Adam iterations per round.')
    ap.add_argument('--selection-table', default=None,
                    help='Directory of cv/voxel_reliability_null.py tables (derivatives/prf_cv_roi_reliability_null): '
                         'select ROI voxels with r > --min-r and p <= --max-p instead of p_signal / top-N; root gets '
                         'a _relnull_r<min-r> suffix.')
    ap.add_argument('--min-r', type=float, default=None)
    ap.add_argument('--max-p', type=float, default=0.01)
    ap.add_argument('--warm-polish-root', default=None,
                    help='With --polish --polish-method alt --polish-models: skip the clamped chain and start each '
                         'polished model from this polished root (shared terms; pRFs where the voxel was fitted, '
                         'whole-brain m4 + Adam elsewhere).')
    ap.add_argument('--bounded', action='store_true',
                    help='Box the shared terms (SHARED_BOUNDS); root gets '
                         'a _bounded suffix.')
    ap.add_argument('--max-bold-sd', type=float, default=None,
                    help='Drop voxels whose BOLD SD (percent signal) exceeds '
                         'this; root gets a _maxsd<value> suffix.')
    ap.add_argument('--loss', choices=['sse', 'nll'], default='sse',
                    help='Objective of the shared-term fits; nll adds a _nll '
                         'suffix to the root.')
    ap.add_argument('--family', choices=['dog', 'dn', 'gauss'], default='dog',
                    help='pRF model under the gain fields; dn / gauss add a _dn / _gauss suffix.')
    ap.add_argument('--base-root', nargs='+', default=[],
                    help='Reuse m00 and m0 from these roots, first one holding the fit wins.')
    ap.add_argument('--hp-rotation', type=int, choices=[0, 1, 2, 3], default=0,
                    help='Relabel HP as the ring position this many steps away.')
    ap.add_argument('--reliable-only', action='store_true',
                    help='Keep selected voxels with cross-run reliability > 0, frozen pRFs from --base-root.')
    ap.add_argument('--warm-root', nargs='+', default=[],
                    help='Start m00 and m0 from earlier fits in these roots (same voxels where present).')
    ap.add_argument('--gd-min-iterations', type=int, default=1000)
    ap.add_argument('--sigma-starts', nargs='+', default=None,
                    help="σ starts of the shared fits; 'parent' is the parent's σ. Default: parent .5 1 2 4.")
    ap.add_argument('--lbfgs-rounds', type=int, default=LBFGS_ROUNDS)
    ap.add_argument('--hrf-table', default=None,
                    help='TSV of one HRF per subject x ROI (build_roi_hrf_table.py); root gets a _roihrf suffix. '
                         'Do not combine with --base-root fits made with another HRF.')
    ap.add_argument('--frozen-prf', action='store_true',
                    help='CV with the per-voxel pRFs (and m00-m5) of the all-runs --base-root; m7_full and the '
                         'requested models are refitted per fold. Root gets a _frozenprf suffix.')
    a = ap.parse_args()
    starts = SIGMA_STARTS if a.sigma_starts is None else tuple(None if v == 'parent' else float(v) for v in a.sigma_starts)
    main(a.subject, a.bids_folder, a.roi, cv_scheme=a.cv_scheme,
         gd_iterations=a.gd_iterations, learning_rate=a.learning_rate,
         models=a.models, max_voxels=a.max_voxels,
         output_root=a.output_root, polish=a.polish, bounded=a.bounded,
         max_bold_sd=a.max_bold_sd, loss=a.loss, base_roots=a.base_root, family=a.family,
         hp_rotation=a.hp_rotation, reliable_only=a.reliable_only, warm_roots=a.warm_root,
         gd_min_iterations=a.gd_min_iterations, sigma_starts=starts, lbfgs_rounds=a.lbfgs_rounds,
         hrf_table=a.hrf_table, frozen_prf=a.frozen_prf, polish_models=a.polish_models,
         polish_method=a.polish_method, polish_suffix=a.polish_suffix, polish_tol=a.polish_tol,
         polish_rounds=a.polish_rounds, polish_max_iter=a.polish_max_iter,
         polish_lm_iter=a.polish_lm_iter, polish_lm_ftol=a.polish_lm_ftol,
         selection_table=a.selection_table, min_r=a.min_r, max_p=a.max_p, warm_polish_root=a.warm_polish_root)
