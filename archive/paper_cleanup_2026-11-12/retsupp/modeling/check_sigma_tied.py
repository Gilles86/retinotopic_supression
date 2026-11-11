"""Check ``sigma_tied`` on simulated inputs: the tied model equals its parent with σ_dyn = σ_T_dyn = σ_AF,
ignores the σ_dyn it is given, and differs from the parent when the parent's widths differ.

    python -m retsupp.modeling.check_sigma_tied
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from braincoder.hrf import SPMHRFModel
from retsupp.modeling.dn_attention_field import DNDynamicAttentionFieldPRF2D
from retsupp.modeling.fit_af_prf_cv_v2 import get_ring_positions
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial_items as Items,
)
from retsupp.modeling.sigma_tied import SigmaTiedDN, SigmaTiedDoG

rng = np.random.default_rng(0)
T, V, R = 120, 7, 30
lin = np.linspace(-5, 5, R)
gx, gy = np.meshgrid(lin, lin)
grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
paradigm = (rng.random((T, grid.shape[0])) < 0.15).astype(np.float32)
ci = np.zeros((T, 4), np.float32)
ci[np.arange(T), np.repeat(np.arange(4), T // 4)] = 1
di = (rng.random((T, 4)) < 0.1).astype(np.float32) * rng.random((T, 4)).astype(np.float32)
ti = (rng.random((T, 4)) < 0.1).astype(np.float32) * rng.random((T, 4)).astype(np.float32)
sp = dict(g_HP='free', g_LP='free', g_HP_dyn='free', g_LP_dyn='free', g_T_dyn='free')
kw = dict(grid_coordinates=grid, paradigm=paradigm, hrf_model=SPMHRFModel(tr=1.6),
          condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
          ring_positions=get_ring_positions(), sign_pattern=sp, item_scale_mode='free')

pars = pd.DataFrame(dict(
    x=rng.uniform(-3, 3, V), y=rng.uniform(-3, 3, V), sd=rng.uniform(.5, 2, V),
    baseline=rng.normal(0, 1, V), amplitude=rng.uniform(.5, 2, V),
    srf_amplitude=rng.uniform(.1, .5, V), srf_size=rng.uniform(2, 4, V),
    sigma_AF=2.0, g_HP=-0.3, g_LP=0.15, sigma_dyn=1.0, g_HP_dyn=-0.2, g_LP_dyn=0.1,
    g_T_dyn=0.4, sigma_T_dyn=1.0, item_scale=0.7)).astype(np.float32)


def predict(model, p):
    # Through the raw space, as the fitter does: backward, then forward inside predict.
    raw = np.asarray(model._transform_parameters_backward(p[model.parameter_labels].to_numpy()))
    usable = pd.DataFrame(np.asarray(model._transform_parameters_forward(raw)), columns=model.parameter_labels)
    return np.asarray(model.predict(paradigm=paradigm, parameters=usable)), usable


def compare(parent, child, p):
    tied = p.copy()
    tied['sigma_dyn'] = tied['sigma_T_dyn'] = tied['sigma_AF']
    base, _ = predict(parent, tied)
    new, usable = predict(child, p)           # p has σ_dyn = 1 ≠ σ_AF: the tied model must ignore it
    err = np.abs(new - base).max()
    print(f'{type(child).__name__}: max |pred - parent with σ_dyn = σ_AF| {err:.3g} (prediction SD {base.std():.3g}); '
          f'fitted σ_dyn, σ_T_dyn = {usable.sigma_dyn.iloc[0]:.3f}, {usable.sigma_T_dyn.iloc[0]:.3f}')
    assert err < 1e-5 * max(1.0, np.abs(base).max())
    assert np.allclose(usable[['sigma_dyn', 'sigma_T_dyn']], 2.0, atol=1e-5)
    untied, _ = predict(parent, p)
    print(f'  parent with σ_dyn = 1 vs tied: max |diff| {np.abs(untied - new).max():.3g} (must be > 0)')
    assert np.abs(untied - new).max() > 1e-3


compare(Items(**kw), SigmaTiedDoG(**kw), pars)
dn = pars.copy()
dn['srf_size'] = 2.0
dn['neural_baseline'] = 1.0
dn['surround_baseline'] = 2.0
compare(DNDynamicAttentionFieldPRF2D(**kw), SigmaTiedDN(**kw), dn)
print('OK')
