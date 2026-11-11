"""Does a population-level HP gain predict a new subject's BOLD?

Zero-free-parameter model comparison for the HP-specific gains. Base model
is the bounded, nested, full-data ``m4_TDS`` fit of the subject (location
blind: ``g_HP := g_LP``, ``g_HP_dyn := g_LP_dyn``). With everything else
fixed the prediction is linear in the HP gains, so adding an HP-specific
offset δ changes the SSE by

    ΔSSE(δ) = δ² a − 2 δ b,   a = Σ X²,   b = Σ X (Y − P0)

with ``X = ∂P/∂g_HP`` (or ``∂P/∂g_HP_dyn``). δ is chosen offline from the
*other* subjects (leave-one-subject-out), so nothing is fitted to the
subject being evaluated. This script only writes the per-voxel,
per-HP-condition sufficient statistics.

X is a finite difference with step ``H``; ``lin_err`` compares the forward
and backward differences.

Outputs (never overwritten) under
``derivatives/af_fixed_group_gain/sub-XX/sub-XX_roi-R_stats.tsv.gz``.

Usage::

    python -m retsupp.modeling.fit_fixed_group_gain 3 --rois V1 V3AB \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.hrf import SPMHRFModel
from retsupp.modeling.fit_af_prf_cv_v2 import (
    build_data_and_paradigm,
    build_model,
    get_ring_positions,
)
from retsupp.modeling.fit_cv_ladder_nested import TIE_PARTNER, sign_pattern_of
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
BASE_ROOT = 'af_prf_cv_ladder_nested_full_bounded'
BASE_MODEL = 'm4_TDS'
H = 0.05
TERMS = {'S': 'g_HP', 'D': 'g_HP_dyn'}


def fit_roi(sub, roi, data, out_path):
    bold, paradigm, ci, di, ti, grid_coords, run_meta = data
    stem = f'sub-{sub.subject_id:02d}_roi-{roi}'
    pkl = (sub.bids_folder / 'derivatives' / BASE_ROOT / BASE_MODEL
           / f'sub-{sub.subject_id:02d}' / f'{stem}_cv-fits.pkl')
    with open(pkl, 'rb') as f:
        fit = pickle.load(f)
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)
    # All gains free, so g_HP can move away from g_LP.
    model = build_model(
        grid_coords=grid_coords, paradigm=paradigm,
        condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
        ring_positions=get_ring_positions(), hrf_model=hrf_model,
        sign_pattern=sign_pattern_of('m7_full'), item_scale='free')
    pars = fit['fit_pars_per_fold'][0].reindex(
        columns=list(model.parameter_labels))
    for g, partner in TIE_PARTNER.items():
        pars[g] = pars[partner]
    assert not pars.isna().any().any(), pars.columns[pars.isna().any()]
    pars = pars.astype(np.float32)
    Y = bold[:, fit['voxel_ids']].astype(np.float64)

    def pred(p):
        return np.asarray(model.predict(paradigm=paradigm, parameters=p),
                          dtype=np.float64)

    P0 = pred(pars)
    R = Y - P0
    X, lin_err = {}, {}
    for k, g in TERMS.items():
        up, dn = pars.copy(), pars.copy()
        up[g] += H
        dn[g] -= H
        fwd = (pred(up) - P0) / H
        bwd = (P0 - pred(dn)) / H
        X[k] = .5 * (fwd + bwd)
        lin_err[k] = np.abs(fwd - bwd).max() / max(np.abs(X[k]).max(), 1e-12)

    rows = []
    for hp in sorted({rm.hp for rm in run_meta}):
        t = np.concatenate([np.arange(rm.start, rm.stop)
                            for rm in run_meta if rm.hp == hp])
        y, r = Y[t], R[t]
        rows.append(pd.DataFrame(dict(
            voxel_id=fit['voxel_ids'], hp=hp, n_T=len(t),
            sst=((y - y.mean(0)) ** 2).sum(0), sse0=(r ** 2).sum(0),
            a_S=(X['S'][t] ** 2).sum(0), b_S=(X['S'][t] * r).sum(0),
            a_D=(X['D'][t] ** 2).sum(0), b_D=(X['D'][t] * r).sum(0),
            c_SD=(X['S'][t] * X['D'][t]).sum(0))))
    df = pd.concat(rows)
    vox = pars[['x', 'y', 'sd']].reset_index(drop=True)
    vox['voxel_id'] = fit['voxel_ids']
    df = df.merge(vox, on='voxel_id')
    df.insert(0, 'roi', roi)
    df.insert(0, 'subject', sub.subject_id)
    for g in ('g_LP', 'g_LP_dyn', 'sigma_AF', 'sigma_dyn'):
        df[g] = float(pars[g].iloc[0])
    df['lin_err_S'], df['lin_err_D'] = lin_err['S'], lin_err['D']
    print(f'{stem}: {len(fit["voxel_ids"])} voxels, lin_err '
          f'S={lin_err["S"]:.2e} D={lin_err["D"]:.2e}, '
          f'own δ̂_S={df.b_S.sum() / df.a_S.sum():+.4f} '
          f'δ̂_D={df.b_D.sum() / df.a_D.sum():+.4f}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)


def main(subject, bids_folder, rois, output_subdir='af_fixed_group_gain'):
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    todo = {roi: out_dir / f'sub-{subject:02d}_roi-{roi}_stats.tsv.gz' for roi in rois}
    todo = {roi: p for roi, p in todo.items() if not p.exists()}
    if not todo:
        print('All outputs exist, nothing to do')
        return
    (bold, paradigm, ci, di, ti, grid_coords, _masker,
     run_meta) = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0,
                                         item_shape='rectangle')
    data = (bold, paradigm, ci, di, ti, grid_coords, run_meta)
    for roi, path in todo.items():
        fit_roi(sub, roi, data, path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--output-subdir', default='af_fixed_group_gain')
    args = ap.parse_args()
    main(args.subject, args.bids_folder, args.rois, args.output_subdir)
