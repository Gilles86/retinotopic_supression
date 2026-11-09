"""Check ``lp_split`` on simulated inputs: at g_adj_minus_opp = 0 the prediction equals the parent's,
and at a nonzero value the sustained field matches a direct numpy computation.

    python -m retsupp.modeling.check_lp_split
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
from retsupp.modeling.lp_split import NEW, LPSplitDN, LPSplitDoG, _sustained_field

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
ring = get_ring_positions()
sp = dict(g_HP='free', g_LP='free', g_HP_dyn='free', g_LP_dyn='free', g_T_dyn='free')
kw = dict(grid_coordinates=grid, paradigm=paradigm, hrf_model=SPMHRFModel(tr=1.6),
          condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
          ring_positions=ring, sign_pattern=sp, item_scale_mode='free')

pars = pd.DataFrame(dict(
    x=rng.uniform(-3, 3, V), y=rng.uniform(-3, 3, V), sd=rng.uniform(.5, 2, V),
    baseline=rng.normal(0, 1, V), amplitude=rng.uniform(.5, 2, V),
    srf_amplitude=rng.uniform(.1, .5, V), srf_size=rng.uniform(2, 4, V),
    sigma_AF=2.0, g_HP=-0.3, g_LP=0.15, sigma_dyn=1.5, g_HP_dyn=-0.2, g_LP_dyn=0.1,
    g_T_dyn=0.4, sigma_T_dyn=1.5, item_scale=0.7)).astype(np.float32)


def compare(parent, child, p):
    base = np.asarray(parent.predict(paradigm=paradigm, parameters=p))
    for d in (0.0,):
        q = p.copy()
        q[NEW] = d
        new = np.asarray(child.predict(paradigm=paradigm, parameters=q))
        print(f'{type(child).__name__}: max |pred - parent| at {NEW}=0: {np.abs(new - base).max():.3g} '
              f'(prediction SD {base.std():.3g})')
        assert np.abs(new - base).max() < 1e-5 * max(1.0, np.abs(base).max())
    q = p.copy()
    q[NEW] = 0.9
    new = np.asarray(child.predict(paradigm=paradigm, parameters=q))
    print(f'  at {NEW}=0.9: max |pred - parent| {np.abs(new - base).max():.3g} (must be > 0)')
    assert np.abs(new - base).max() > 1e-3


compare(Items(**kw), LPSplitDoG(**kw), pars)
dn = pars.copy()
dn['srf_size'] = 2.0
dn['neural_baseline'] = 1.0
dn['surround_baseline'] = 2.0
compare(DNDynamicAttentionFieldPRF2D(**kw), LPSplitDN(**kw), dn)

# Sustained field against numpy: run with HP at ring 0 (UR) -> UL, LR next, LL opposite.
m = LPSplitDoG(**kw)
q = pars.copy()
q[NEW] = 0.9
f = np.asarray(_sustained_field(m, q.to_numpy()[None], np.float32(0.9)))
g_adj, g_opp = 0.15 + 0.3, 0.15 - 0.6
w = np.array([[-0.3, g_adj, g_opp, g_adj],     # c=0: UR HP; UL, LR next; LL opposite
              [g_adj, -0.3, g_adj, g_opp],
              [g_opp, g_adj, -0.3, g_adj],
              [g_adj, g_opp, g_adj, -0.3]])
a = np.exp(-((grid[None, :, 0] - ring[:, None, 0]) ** 2 + (grid[None, :, 1] - ring[:, None, 1]) ** 2) / (2 * 2.0 ** 2))
ref = np.maximum(1 + w @ a, 0)
print(f'sustained field vs numpy: max |diff| {np.abs(f - ref).max():.3g}; '
      f'mean over three LP gains {(2 * g_adj + g_opp) / 3:.3f} = g_LP 0.150')
assert np.abs(f - ref).max() < 1e-5
print('OK')
