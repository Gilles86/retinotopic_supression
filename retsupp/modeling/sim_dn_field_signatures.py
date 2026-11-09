"""Simulated signatures of the sustained HP field in the DN gain-field models (no fitting, no data).

Reference model (c, m7, ``dn_attention_field``): the field multiplies the stimulus, in numerator and
denominator. Variants (``dn_baseline_field``): the field scales the neural baseline b (a) or the surround
baseline d (b). Synthetic voxels sit on the four diagonals (the ring directions) at 1-3° eccentricity, with
DN parameters in the range of the m7 DN fits (sub-03, V1-hV4); bars sweep the 3.17° aperture in four
directions while the eight search items (4° ring) stay on; a few TRs show the items only.

Readout, per voxel and bar position class, neural response N (before the HRF, which is linear):
``centre`` (bar over the pRF centre, distance < sd), ``surround`` (bar outside the centre but within the
surround, 1.5 sd < distance < 2 sd·srf_size), ``items_only`` (no bar). Each is compared with the same voxel
without the HP field: change = (N_field − N_nofield) / N_nofield(centre), i.e. in units of the voxel's
own bar-at-centre response, so that classes with small or negative responses stay comparable. Voxels are grouped by quadrant relative to HP: at HP, next to HP (the
two neighbours), opposite. HP is the upper-right ring position (condition 0) throughout.

Output: ``notes/data/dn_field_signatures_2026-11.tsv`` (one row per variant × parameter set × voxel × class).

Usage::

    KERAS_BACKEND=tensorflow python -m retsupp.modeling.sim_dn_field_signatures
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.hrf import SPMHRFModel
from retsupp.modeling.dn_baseline_field import DNBaselineField
from retsupp.modeling.fit_af_prf_cv_v2 import get_ring_positions

APERTURE, BAR_WIDTH, RING_ECC, ITEM_RADIUS = 3.17, 0.8, 4.0, 0.6
QUADRANT = {0: 'HP', 1: 'next', 2: 'opposite', 3: 'next'}   # ring steps from HP (UR, UL, LL, LR)
GAINS = ['g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn']


def paradigm(resolution=50, grid_radius=5.0, step=0.2, n_blank=20):
    x = np.linspace(-grid_radius, grid_radius, resolution)
    gx, gy = np.meshgrid(x, -x)
    grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    r = np.hypot(grid[:, 0], grid[:, 1])
    items = np.zeros(len(grid), np.float32)
    for k in range(8):
        ang = np.pi / 4 + k * np.pi / 4
        items[np.hypot(grid[:, 0] - RING_ECC * np.cos(ang), grid[:, 1] - RING_ECC * np.sin(ang))
              <= ITEM_RADIUS] = 1.0
    frames, bars = [], []
    for ori in np.deg2rad([0, 45, 90, 135]):
        n = np.array([np.cos(ori), np.sin(ori)])
        for pos in np.arange(-APERTURE, APERTURE + 1e-6, step):
            bar = ((np.abs(grid @ n - pos) <= BAR_WIDTH / 2) & (r <= APERTURE)).astype(np.float32)
            frames.append(np.maximum(bar, items))
            bars.append((ori, pos))
    for _ in range(n_blank):
        frames.append(items.copy())
        bars.append((np.nan, np.nan))
    return grid, np.array(frames, np.float32), np.array(bars)


def voxels():
    rows = []
    for k, ecc, sd, srf, b in itertools.product(range(4), [1.0, 2.0, 3.0], [0.6, 1.2],
                                                [1.5, 3.0, 6.0, 12.0], [0.5, 3.0]):
        ang = np.pi / 4 + k * np.pi / 2
        rows.append(dict(ring=k, quadrant=QUADRANT[k], ecc=ecc, x=ecc * np.cos(ang), y=ecc * np.sin(ang),
                         sd=sd, srf_size=srf, neural_baseline=b))
    v = pd.DataFrame(rows)
    v['baseline'], v['amplitude'], v['srf_amplitude'], v['surround_baseline'] = 0.0, 80.0, 0.01, 0.02
    return v


VARIANTS = {   # name: shared terms that differ from the no-field model
    'none': {},
    'c_input_gain': {'g_HP': -0.3},
    'c_input_gain_pos': {'g_HP': +0.3},
    'a_neural_baseline': {'w_b': 1.0},
    'b_surround_baseline': {'w_d': 1.0},
}


def main(out, sigma_af=1.5):
    grid, par, bars = paradigm()
    T = len(par)
    ci = np.zeros((T, 4), np.float32)
    ci[:, 0] = 1.0                                  # HP = upper right
    zeros = np.zeros((T, 4), np.float32)
    sp = {g: 'free' for g in GAINS}
    model = DNBaselineField(targets=('b', 'd'), grid_coordinates=grid, paradigm=par,
                            hrf_model=SPMHRFModel(tr=1.6), condition_indicator=ci, dynamic_indicator=zeros,
                            target_indicator=zeros, ring_positions=get_ring_positions(), sign_pattern=sp,
                            item_scale_mode='free')
    vox = voxels()
    base = vox.assign(sigma_AF=sigma_af, sigma_dyn=1.0, sigma_T_dyn=1.0, item_scale=0.75,
                      w_b=0.0, w_d=0.0, theta_bd=0.0, **{g: 0.0 for g in GAINS})
    # Bar-to-pRF-centre distance per TR and voxel.
    n = np.c_[np.cos(bars[:, 0]), np.sin(bars[:, 0])]
    dist = np.abs(n @ vox[['x', 'y']].to_numpy().T - bars[:, 1:2])      # (T, V), NaN for items-only
    centre = dist < vox.sd.to_numpy()
    surround = (dist > 1.5 * vox.sd.to_numpy()) & (dist < 2 * vox.sd.to_numpy() * vox.srf_size.to_numpy())
    blank = np.isnan(bars[:, 0])[:, None] & np.ones(len(vox), bool)

    resp = {}
    for name, terms in VARIANTS.items():
        p = base.copy()
        for k_, v_ in terms.items():
            p[k_] = v_
        resp[name] = np.asarray(model._basis_predictions(
            par[np.newaxis], p[model.parameter_labels].to_numpy(np.float32)[np.newaxis]))[0]
    with np.errstate(invalid='ignore'):
        ref = np.nanmean(np.where(centre, resp['none'], np.nan), 0)
    rows = []
    for name in VARIANTS:
        if name == 'none':
            continue
        for cls, mask in [('centre', centre), ('surround', surround), ('items_only', blank)]:
            m0 = np.where(mask, resp['none'], np.nan)
            m1 = np.where(mask, resp[name], np.nan)
            with np.errstate(invalid='ignore', divide='ignore'):
                r0, r1 = np.nanmean(m0, 0), np.nanmean(m1, 0)
            out_v = vox.assign(variant=name, bar_class=cls, resp_nofield=r0, resp_field=r1,
                               rel_change=(r1 - r0) / ref, n_trs=mask.sum(0))
            rows.append(out_v)
    df = pd.concat(rows, ignore_index=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    summ = (df.groupby(['variant', 'bar_class', 'quadrant']).rel_change.median().unstack('quadrant')
            [['HP', 'next', 'opposite']])
    summ['next_minus_opp'] = summ['next'] - summ['opposite']
    print((100 * summ).round(2).to_string())
    big = df[(df.variant == 'c_input_gain') & (df.bar_class == 'centre')]
    print('\nc_input_gain, centre, median % change by surround size (srf_size) and quadrant:')
    print((100 * big.groupby(['srf_size', 'quadrant']).rel_change.median().unstack()).round(2).to_string())
    print('\nc_input_gain, centre, median % change by eccentricity and quadrant:')
    print((100 * big.groupby(['ecc', 'quadrant']).rel_change.median().unstack()).round(2).to_string())
    print(f'\nWritten {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='notes/data/dn_field_signatures_2026-11.tsv')
    ap.add_argument('--sigma-af', type=float, default=1.5)
    a = ap.parse_args()
    main(a.out, a.sigma_af)
