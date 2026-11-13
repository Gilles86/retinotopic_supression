"""Joint Levenberg-Marquardt fit of per-voxel pRFs and shared gain-field terms (block Schur complement).

The gain-field models have ~7 free parameters per voxel (DoG pRF: x, y, sd, baseline, amplitude, srf_amplitude,
srf_size) and ~10 shared terms (gains, σ's, item scale) that enter every voxel. The Gauss-Newton matrix JᵀJ is then
block-arrowhead: one small dense block per voxel (U_v), the voxel-shared couplings (W_v) and one dense shared block
(V). As in bundle adjustment, each damped GN step eliminates the voxel blocks with the Schur complement

    S = V + λD_s − Σ_v W_vᵀ (U_v + λD_v)⁻¹ W_v,        S δ_s = g_s − Σ_v W_vᵀ (U_v + λD_v)⁻¹ g_v,
    δ_v = (U_v + λD_v)⁻¹ (g_v − W_v δ_s),

so one step costs one Jacobian (one forward-mode pass per per-voxel column, all voxels at once since voxels are
independent, plus one per shared term) and V small solves. The model runs in float32 (as everywhere in the ladder);
residuals, JᵀJ, the Schur complement and the solves are float64. Steps are accepted only if the training SSE drops
(Marquardt damping D = diag JᵀJ, λ adapted), so the result never ends above its start.

The shared terms can be reparametrised linearly, s = A φ + b (raw space): a profile likelihood fixes one term (drop
its column, put its value in b) and a contrast such as dS = g_HP − g_LP is fixed by writing g_HP = g_LP + dS. Shared
terms are boxed by projection (bounds on φ in raw space; a term at a bound whose step points outward is held for
that step).

``schur_hessian`` returns the Gauss-Newton Hessian of the SSE over φ with the pRFs profiled out (λ = 0) at a point,
the matrix whose inverse × σ² is the Laplace covariance of the shared terms.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import tensorflow as tf


class JointProblem:
    """Prediction and Jacobians of one gain-field model on one voxel set.

    ``per_voxel``: free per-voxel columns. ``shared``: shared columns that may move (raw space; tied or fixed shared
    columns stay at their ``init_pars`` value). ``A``, ``b``: s = A φ + b maps the optimised shared coordinates φ to
    the raw values of ``shared`` (default: identity, φ = s)."""

    def __init__(self, model, y, paradigm, init_pars, per_voxel, shared, A=None, b=None, phi_names=None):
        self.model = model
        self.labels = list(model.parameter_labels)
        self.y = np.asarray(y, dtype=np.float64)
        self.par_ = model.stimulus._clean_paradigm(paradigm)
        self.pv_ix = [self.labels.index(c) for c in per_voxel]
        self.sh_ix = [self.labels.index(c) for c in shared]
        self.per_voxel, self.shared = list(per_voxel), list(shared)
        self.raw0 = np.asarray(model._transform_parameters_backward(
            init_pars[self.labels].values.astype(np.float32)), dtype=np.float32)
        self.index = init_pars.index
        n_sh = len(shared)
        self.A = np.eye(n_sh) if A is None else np.asarray(A, dtype=np.float64)
        s0 = self.raw0[0, self.sh_ix].astype(np.float64)
        self.b = np.zeros(n_sh) if b is None else np.asarray(b, dtype=np.float64)
        self.phi_names = list(phi_names) if phi_names is not None else list(shared)
        self.phi0 = np.linalg.lstsq(self.A, s0 - self.b, rcond=None)[0]
        self.n_vox = self.raw0.shape[0]
        self._pred = tf.function(self._pred_impl, reduce_retracing=True)
        self._jvp = tf.function(self._jvp_impl, reduce_retracing=True)

    # ---- parameters ---------------------------------------------------------------------------------------------
    def raw(self, x_pv, phi):
        r = self.raw0.copy()
        r[:, self.pv_ix] = x_pv
        r[:, self.sh_ix] = (self.A @ phi + self.b)[np.newaxis, :]
        return r

    def start(self):
        return self.raw0[:, self.pv_ix].astype(np.float64), self.phi0.copy()

    def pars(self, x_pv, phi):
        r = self.raw(x_pv, phi).astype(np.float32)
        return pd.DataFrame(np.asarray(self.model._transform_parameters_forward(r)), columns=self.labels,
                            index=self.index)

    # ---- model --------------------------------------------------------------------------------------------------
    def _pred_impl(self, raw):
        m = self.model
        return m._predict(self.par_[tf.newaxis, ...], m._transform_parameters_forward(raw)[tf.newaxis, ...], None)[0]

    def _jvp_impl(self, raw, tangent):
        with tf.autodiff.ForwardAccumulator(raw, tangent) as acc:
            pred = self._pred_impl(raw)
        return acc.jvp(pred)

    def predict(self, x_pv, phi):
        return np.asarray(self._pred(tf.constant(self.raw(x_pv, phi).astype(np.float32))), dtype=np.float64)

    def sse(self, x_pv, phi):
        return float(((self.y - self.predict(x_pv, phi)) ** 2).sum())

    def jacobians(self, x_pv, phi):
        """Prediction (T, V), per-voxel Jacobian (T, V, p) and Jacobian over φ (T, V, q), float64."""
        raw = tf.constant(self.raw(x_pv, phi).astype(np.float32))
        pred = np.asarray(self._pred(raw), dtype=np.float64)
        T, V = pred.shape
        j_pv = np.empty((T, V, len(self.pv_ix)))
        for i, k in enumerate(self.pv_ix):
            t = np.zeros(raw.shape, dtype=np.float32)
            t[:, k] = 1.0          # voxel v's prediction depends only on row v: column v is d pred_v / d raw[v, k]
            j_pv[:, :, i] = np.asarray(self._jvp(raw, tf.constant(t)))
        j_s = np.empty((T, V, len(self.sh_ix)))
        for i, k in enumerate(self.sh_ix):
            t = np.zeros(raw.shape, dtype=np.float32)
            t[:, k] = 1.0          # a shared term sits in every row (the model reads it from row 0 or from each row)
            j_s[:, :, i] = np.asarray(self._jvp(raw, tf.constant(t)))
        bad = ~np.isfinite(j_pv)
        self.n_nonfinite = int(bad.sum())
        if self.n_nonfinite:   # e.g. d/dx of a voxel whose surround width underflowed: hold that entry this step
            j_pv[bad] = 0.0
        j_phi = j_s @ self.A
        j_phi[~np.isfinite(j_phi)] = 0.0
        return pred, j_pv, j_phi


def normal_blocks(y, pred, j_pv, j_phi):
    r = y - pred
    U = np.einsum('tvi,tvj->vij', j_pv, j_pv)
    W = np.einsum('tvi,tvj->vij', j_pv, j_phi)
    Vs = np.einsum('tvi,tvj->ij', j_phi, j_phi)
    g_pv = np.einsum('tvi,tv->vi', j_pv, r)
    g_s = np.einsum('tvi,tv->i', j_phi, r)
    return U, W, Vs, g_pv, g_s


def _damped_inv(U, lam):
    p = U.shape[-1]
    d = np.diagonal(U, axis1=1, axis2=2)
    floor = 1e-9 * max(float(d.max()), 1e-30)
    d = np.maximum(d, 1e-6 * np.median(d, axis=0)[np.newaxis, :])   # column floor: flat directions still damped
    Ud = U + lam * np.einsum('vi,ij->vij', np.maximum(d, floor), np.eye(p)) + floor * np.eye(p)
    return np.linalg.inv(Ud)


def schur_step(U, W, Vs, g_pv, g_s, lam, hold=None):
    """Damped GN step for the block-arrowhead system; ``hold`` (bool, len q) keeps those φ at their value."""
    Ui = _damped_inv(U, lam)
    UiW = np.einsum('vij,vjk->vik', Ui, W)
    Uig = np.einsum('vij,vj->vi', Ui, g_pv)
    q = Vs.shape[0]
    ds_ = np.diag(Vs)
    S = Vs + lam * np.diag(np.maximum(ds_, 1e-9 * max(ds_.max(), 1e-30))) - np.einsum('vji,vjk->ik', W, UiW)
    rhs = g_s - np.einsum('vji,vj->i', W, Uig)
    d_s = np.zeros(q)
    free = np.ones(q, bool) if hold is None else ~np.asarray(hold)
    if free.any():
        Sf = S[np.ix_(free, free)]
        d_s[free] = np.linalg.solve(Sf + 1e-12 * np.trace(Sf) / max(free.sum(), 1) * np.eye(free.sum()), rhs[free])
    d_pv = Uig - np.einsum('vij,j->vi', UiW, d_s)
    return d_pv, d_s


def schur_hessian(U, W, Vs):
    """JᵀJ over φ with the per-voxel blocks profiled out (undamped Schur complement; pinv on singular U_v)."""
    Ui = np.linalg.pinv(U, hermitian=True)
    return Vs - np.einsum('vji,vjk,vkl->il', W, Ui, W)


PV_BOUNDS = {'sd': (-15.0, None), 'srf_amplitude': (-15.0, None), 'srf_size': (-15.0, None)}


def lm_fit(prob, x_pv=None, phi=None, bounds=None, max_iter=300, ftol=1e-10, n_small=3, lam0=1e-3,
           max_seconds=None, verbose=True, log_every=10, debug=False, pv_bounds=PV_BOUNDS, max_step=0.25):
    """Levenberg-Marquardt over all free parameters of ``prob``.

    ``bounds``: list of (lo, hi) per φ (raw space) or None. Stops after ``n_small`` consecutive accepted steps that
    lower the SSE by less than ``ftol`` (relative), when the damping exceeds 1e10 (no descent step left at float32
    precision), at ``max_iter`` or ``max_seconds``. Returns (x_pv, phi, info)."""
    if x_pv is None or phi is None:
        x0, p0 = prob.start()
        x_pv = x0 if x_pv is None else x_pv
        phi = p0 if phi is None else phi
    # Per-voxel raw floors (softplus(-15) = 3e-7: a surround amplitude or width that small is zero for the prediction,
    # but lower raw values make its derivatives underflow to NaN). Steps larger than max(1, max_step·|value|) per raw
    # parameter (max(0.5, max_step·|φ|) for shared terms) are treated as rejected (a trust region through λ).
    pv_lo = np.array([(pv_bounds or {}).get(c, (None, None))[0] for c in prob.per_voxel], dtype=float)
    pv_lo = np.where(np.isnan(pv_lo), -np.inf, pv_lo)
    x_pv = np.maximum(x_pv, pv_lo[np.newaxis, :])
    lo = hi = None
    if bounds is not None:
        lo = np.array([-np.inf if b is None or b[0] is None else b[0] for b in bounds])
        hi = np.array([np.inf if b is None or b[1] is None else b[1] for b in bounds])
        phi = np.clip(phi, lo, hi)
    t0 = time.time()
    lam = lam0
    pred, j_pv, j_phi = prob.jacobians(x_pv, phi)
    f = float(((prob.y - pred) ** 2).sum())
    f0 = f
    log = [dict(it=0, sse=f, lam=lam, seconds=0.0)]
    small, it, reason = 0, 0, 'max_iter'
    n_evals = 1
    while it < max_iter:
        it += 1
        U, W, Vs, g_pv, g_s = normal_blocks(prob.y, pred, j_pv, j_phi)
        hold = None
        if lo is not None:
            # g_s is the descent direction (+Jᵀr): at a bound and pointing outward → hold.
            hold = ((phi <= lo + 1e-12) & (g_s < 0)) | ((phi >= hi - 1e-12) & (g_s > 0))
        accepted = False
        while lam < 1e10:
            d_pv, d_s = schur_step(U, W, Vs, g_pv, g_s, lam, hold)
            if (np.abs(d_s) > np.maximum(0.5, max_step * np.abs(phi))).any() or not np.isfinite(d_s).all():
                if debug:
                    print(f'      trial λ {lam:.2e}: shared step too large ({np.abs(d_s).max():.3g})', flush=True)
                lam *= 4.0
                continue
            cap = np.maximum(1.0, max_step * np.abs(x_pv))
            n_capped = int((np.abs(d_pv) > cap).sum())
            d_pv = np.clip(d_pv, -cap, cap)   # nearly flat per-voxel directions (e.g. surround of a voxel without one)
            phi_new = phi + d_s
            if lo is not None:
                phi_new = np.clip(phi_new, lo, hi)
            x_new = np.maximum(x_pv + d_pv, pv_lo[np.newaxis, :])
            f_new = prob.sse(x_new, phi_new)
            n_evals += 1
            if debug:
                print(f'      trial λ {lam:.2e}: ΔSSE {f_new - f:.4f} | capped {n_capped} | hold {None if hold is None else np.where(hold)[0]}'
                      f' | clipped {np.where(phi_new != phi + d_s)[0]} | max|d_s| {np.abs(d_s).max():.3g}', flush=True)
            if np.isfinite(f_new) and f_new < f:
                accepted = True
                break
            lam *= 4.0
        if not accepted:
            reason = 'no_descent'
            break
        rel = (f - f_new) / f
        x_pv, phi, f = x_new, phi_new, f_new
        lam = max(lam / 3.0, 1e-12)
        small = small + 1 if rel < ftol else 0
        log.append(dict(it=it, sse=f, rel=rel, lam=lam, seconds=time.time() - t0))
        if verbose and (it % log_every == 0 or it <= 3):
            print(f'    lm it {it}: SSE {f:.4f} (rel {rel:.2e}) λ {lam:.1e} | {time.time() - t0:.0f}s', flush=True)
        if small >= n_small:
            reason = 'ftol'
            break
        if max_seconds is not None and time.time() - t0 > max_seconds:
            reason = 'time'
            break
        pred, j_pv, j_phi = prob.jacobians(x_pv, phi)
    # gradient at the end point (for the record)
    pred, j_pv, j_phi = prob.jacobians(x_pv, phi)
    U, W, Vs, g_pv, g_s = normal_blocks(prob.y, pred, j_pv, j_phi)
    info = dict(optimizer='lm_schur', n_nonfinite_jac=getattr(prob, 'n_nonfinite', 0), sse_start=f0, sse_final=f, nit=it, n_evals=n_evals, reason=reason,
                seconds=time.time() - t0, lam_final=lam, grad_norm=float(2 * np.sqrt((g_pv ** 2).sum() + (g_s ** 2).sum())),
                grad_norm_shared=float(2 * np.linalg.norm(g_s)), log=log)
    if verbose:
        print(f'    lm done ({reason}): SSE {f0:.4f} -> {f:.4f} in {it} it, {info["seconds"]:.0f}s', flush=True)
    return x_pv, phi, info


def polish_lm(model, bold_train, paradigm, init_pars, free_shared, free_per_voxel, bounds=None, max_iter=300,
              ftol=1e-10, max_seconds=None):
    """Ladder entry point (``--polish-method lm``): joint LM over per-voxel pRFs and shared terms from ``init_pars``.

    ``bounds``: raw-space (lo, hi) per ``free_shared`` term (``raw_bounds``) or None. Same info keys as the other
    polish methods."""
    labels = list(model.parameter_labels)
    prob = JointProblem(model, bold_train, paradigm, init_pars.reindex(columns=labels), free_per_voxel, free_shared)
    x, phi, info = lm_fit(prob, bounds=bounds, max_iter=max_iter, ftol=ftol, max_seconds=max_seconds)
    pars = prob.pars(x, phi)
    info.update(sse_clamped=info['sse_start'], status=0, message=info['reason'])
    return pars, info
