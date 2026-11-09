"""Simulated signatures of a sustained HP field on the DoG pRF's centre vs surround pooling (no fitting, no data).

Model: ``dog_centre_surround_field`` (κ = 0 is m7_full: the field on both poolings; κ = +1 centre (driving)
pooling only; κ = −1 surround (suppressive) pooling only). Field at HP only: g_HP = −0.3 (suppression; also +0.3),
g_LP = 0, σ_AF = 1.5°. Synthetic voxels on the four diagonals (the ring directions) at 1-3° eccentricity with DoG
parameters in the range of the final DoG fits (m7_full, reliable voxels, per-ROI HRF; median sd 0.8°, srf_size
1.3-3.2 with a tail of very wide surrounds, surround volume srf_amplitude·srf_size² 0.65-1.15 of the centre's).
Bars sweep the 3.17° aperture in four directions with the eight search items (4° ring) on; a few TRs show the
items only. Same paradigm and readout as ``sim_dn_field_signatures``: response before the HRF per voxel and bar
class (``centre``: bar over the pRF centre; ``surround``: bar outside the centre but within 2 surround sd;
``items_only``), change against the same voxel without field, in units of its own bar-at-centre response; grouped
by quadrant relative to HP (HP, next to HP, opposite).

Output: ``notes/data/dog_cs_field_signatures_2026-11.tsv``.

Usage::

    KERAS_BACKEND=tensorflow python -m retsupp.modeling.sim_dog_cs_field_signatures
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.hrf import SPMHRFModel
from retsupp.modeling.dog_centre_surround_field import DoGCentreSurroundField
from retsupp.modeling.fit_af_prf_cv_v2 import get_ring_positions
from retsupp.modeling.sim_dn_field_signatures import QUADRANT, paradigm

GAINS = ['g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn']
VARIANTS = {   # name: (g_HP, κ)
    'm7_both': (-0.3, 0.0),
    'centre_only': (-0.3, 1.0),
    'surround_only': (-0.3, -1.0),
    'tilt_centre_0.5': (-0.3, 0.5),
    'tilt_surround_0.5': (-0.3, -0.5),
    'm7_both_pos': (0.3, 0.0),
    'centre_only_pos': (0.3, 1.0),
    'surround_only_pos': (0.3, -1.0),
}


def voxels():
    rows = []
    for k, ecc, sd, srf, vol in itertools.product(range(4), [1.0, 2.0, 3.0], [0.6, 1.2],
                                                  [1.3, 1.8, 3.2, 20.0], [0.6, 0.9]):
        ang = np.pi / 4 + k * np.pi / 2
        rows.append(dict(ring=k, quadrant=QUADRANT[k], ecc=ecc, x=ecc * np.cos(ang), y=ecc * np.sin(ang),
                         sd=sd, srf_size=srf, surround_volume=vol, srf_amplitude=vol / srf ** 2))
    v = pd.DataFrame(rows)
    v['baseline'], v['amplitude'] = 0.0, 1.0
    return v


def main(out, sigma_af=1.5, item_scale=2.0):
    grid, par, bars = paradigm()
    T = len(par)
    ci = np.zeros((T, 4), np.float32)
    ci[:, 0] = 1.0                                  # HP = upper right
    zeros = np.zeros((T, 4), np.float32)
    model = DoGCentreSurroundField(
        grid_coordinates=grid, paradigm=par, hrf_model=SPMHRFModel(tr=1.6), condition_indicator=ci,
        dynamic_indicator=zeros, target_indicator=zeros, ring_positions=get_ring_positions(),
        sign_pattern={g: 'free' for g in GAINS}, item_scale_mode='free')
    vox = voxels()
    base = vox.assign(sigma_AF=sigma_af, sigma_dyn=1.0, sigma_T_dyn=1.0, item_scale=item_scale, kappa=0.0,
                      **{g: 0.0 for g in GAINS})
    n = np.c_[np.cos(bars[:, 0]), np.sin(bars[:, 0])]
    dist = np.abs(n @ vox[['x', 'y']].to_numpy().T - bars[:, 1:2])
    sd, srf = vox.sd.to_numpy(), vox.srf_size.to_numpy()
    centre = dist < sd
    surround = (dist > 1.5 * sd) & (dist < 2 * sd * srf)
    items_only = np.isnan(bars[:, 0])[:, None] & np.ones(len(vox), bool)

    def resp(p):
        return np.asarray(model._basis_predictions(
            par[np.newaxis], p[model.parameter_labels].to_numpy(np.float32)[np.newaxis]))[0]
    r0 = resp(base)
    with np.errstate(invalid='ignore'):
        ref = np.nanmean(np.where(centre, r0, np.nan), 0)
    rows = []
    for name, (g_hp, kappa) in VARIANTS.items():
        r1 = resp(base.assign(g_HP=g_hp, kappa=kappa))
        for cls, mask in [('centre', centre), ('surround', surround), ('items_only', items_only)]:
            with np.errstate(invalid='ignore'):
                a0 = np.nanmean(np.where(mask, r0, np.nan), 0)
                a1 = np.nanmean(np.where(mask, r1, np.nan), 0)
            rows.append(vox.assign(variant=name, g_HP=g_hp, kappa=kappa, bar_class=cls, resp_nofield=a0,
                                   resp_field=a1, rel_change=(a1 - a0) / ref, n_trs=mask.sum(0)))
    df = pd.concat(rows, ignore_index=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    summ = (df.groupby(['variant', 'bar_class'], sort=False).apply(
        lambda d: d.groupby('quadrant').rel_change.median()).reindex(columns=['HP', 'next', 'opposite']))
    summ['next_minus_opp'] = summ['next'] - summ['opposite']
    print('Median % change of the response (units: own bar-at-centre response), field at HP:')
    print((100 * summ).round(2).to_string())
    for by in ('srf_size', 'surround_volume', 'ecc'):
        sub = df[df.bar_class.eq('centre') & df.variant.isin(['m7_both', 'centre_only', 'surround_only'])]
        print(f'\nBar on centre, median % change by {by}:')
        print((100 * sub.groupby(['variant', by, 'quadrant']).rel_change.median().unstack()).round(2).to_string())
    print(f'\nWritten {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='notes/data/dog_cs_field_signatures_2026-11.tsv')
    ap.add_argument('--sigma-af', type=float, default=1.5)
    ap.add_argument('--item-scale', type=float, default=2.0)
    a = ap.parse_args()
    main(a.out, a.sigma_af, a.item_scale)
