"""Leave-one-subject-out test of every ladder mechanism, with nothing fitted
to the subject being evaluated.

Each mechanism is added to its parent's bounded full-data fit of the
subject (the nested ladder base). The mechanism's strength enters
(near-)linearly, so its effect on the SSE for an offset δ is

    ΔSSE(δ) = δ² a − 2 δ b,   a = Σ X²,   b = Σ X (Y − P0),

with ``X = ∂P/∂θ`` at the base. δ is chosen offline from the *other*
subjects. A mechanism's spatial width (σ_AF / σ_dyn) is not linear; it is
set to the median of the other subjects' fitted widths in the child model.

=========  ===========  ======================================  ==========
mechanism  base         perturbed parameter(s)                  width from
=========  ===========  ======================================  ==========
items      m00_bar      item_scale (base 1 = bar strength)      --
T          m0_prf       g_T_dyn                                 m1_T
D          m0_prf       g_HP_dyn = g_LP_dyn (location blind)    m2_D
S          m0_prf       g_HP = g_LP (location blind)            m3_S
dS         m4_TDS       g_HP                                    (base)
dD         m4_TDS       g_HP_dyn                                (base)
=========  ===========  ======================================  ==========

X is a central difference with step ``H``; ``lin_err`` compares the
forward and backward differences.

Outputs (never overwritten) under
``derivatives/af_fixed_group_mechanisms/sub-XX/sub-XX_roi-R_stats.tsv.gz``.

Usage::

    python -m retsupp.modeling.fit_fixed_group_mechanisms 3 --rois V3AB \\
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
SUBJECTS = range(1, 31)
BASE_ROOT = 'af_prf_cv_ladder_nested_full_bounded'
H = 0.05
# name: (base model, perturbed parameters, (width parameter, child model))
MECHANISMS = {
    'items': ('m00_bar', ['item_scale'], None),
    'T': ('m0_prf', ['g_T_dyn'], ('sigma_dyn', 'm1_T')),
    'D': ('m0_prf', ['g_HP_dyn', 'g_LP_dyn'], ('sigma_dyn', 'm2_D')),
    'S': ('m0_prf', ['g_HP', 'g_LP'], ('sigma_AF', 'm3_S')),
    'dS': ('m4_TDS', ['g_HP'], None),
    'dD': ('m4_TDS', ['g_HP_dyn'], None),
}


def fit_path(bids_folder, model, subject, roi):
    return (Path(bids_folder) / 'derivatives' / BASE_ROOT / model
            / f'sub-{subject:02d}' / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl')


def load_fit(bids_folder, model, subject, roi):
    with open(fit_path(bids_folder, model, subject, roi), 'rb') as f:
        return pickle.load(f)


def loso_width(bids_folder, par, model, subject, roi):
    """Median of the other subjects' fitted width in the child model."""
    vals = []
    for s in SUBJECTS:
        if s == subject or not fit_path(bids_folder, model, s, roi).exists():
            continue
        vals.append(float(load_fit(bids_folder, model, s, roi)
                          ['fit_pars_per_fold'][0][par].iloc[0]))
    return float(np.median(vals)), len(vals)


def fit_roi(sub, roi, data, out_path):
    bold, paradigm, ci, di, ti, grid_coords, run_meta = data
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)
    # Everything free, so each mechanism can move away from its base value.
    model = build_model(
        grid_coords=grid_coords, paradigm=paradigm,
        condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
        ring_positions=get_ring_positions(), hrf_model=hrf_model,
        sign_pattern=sign_pattern_of('m7_full'), item_scale='free')

    def pred(p):
        return np.asarray(model.predict(paradigm=paradigm, parameters=p),
                          dtype=np.float64)

    conds = {hp: np.concatenate([np.arange(rm.start, rm.stop)
                                 for rm in run_meta if rm.hp == hp])
             for hp in sorted({rm.hp for rm in run_meta})}
    out = []
    for mech, (base, perturb, width) in MECHANISMS.items():
        fit = load_fit(sub.bids_folder, base, sub.subject_id, roi)
        pars = fit['fit_pars_per_fold'][0].reindex(
            columns=list(model.parameter_labels))
        if pars['item_scale'].isna().all():
            pars['item_scale'] = 1.0           # m00: items at bar strength
        for g, partner in TIE_PARTNER.items():
            pars[g] = pars[partner]
        width_val, n_width = np.nan, 0
        if width is not None:
            width_val, n_width = loso_width(sub.bids_folder, *width,
                                            sub.subject_id, roi)
            pars[width[0]] = width_val
        pars['sigma_T_dyn'] = pars['sigma_dyn']
        assert not pars.isna().any().any(), pars.columns[pars.isna().any()]
        pars = pars.astype(np.float32)
        Y = bold[:, fit['voxel_ids']].astype(np.float64)

        P0 = pred(pars)
        up, dn = pars.copy(), pars.copy()
        for p in perturb:
            up[p] += H
            dn[p] -= H
        fwd = (pred(up) - P0) / H
        bwd = (P0 - pred(dn)) / H
        X = .5 * (fwd + bwd)
        lin_err = np.abs(fwd - bwd).max() / max(np.abs(X).max(), 1e-12)
        R = Y - P0
        for hp, t in conds.items():
            y, r, x = Y[t], R[t], X[t]
            out.append(pd.DataFrame(dict(
                mech=mech, voxel_id=fit['voxel_ids'], hp=hp,
                x=pars['x'].values, y=pars['y'].values, sd=pars['sd'].values,
                sst=((y - y.mean(0)) ** 2).sum(0), sse0=(r ** 2).sum(0),
                a=(x ** 2).sum(0), b=(x * r).sum(0),
                lin_err=lin_err, width=width_val, n_width=n_width)))
        print(f'sub-{sub.subject_id:02d} {roi} {mech}: '
              f'{len(fit["voxel_ids"])} voxels, width={width_val:.3f}, '
              f'lin_err={lin_err:.2e}, own δ̂={(X * R).sum() / (X ** 2).sum():+.4f}')
    df = pd.concat(out)
    df.insert(0, 'roi', roi)
    df.insert(0, 'subject', sub.subject_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)


def main(subject, bids_folder, rois, output_subdir='af_fixed_group_mechanisms'):
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
    ap.add_argument('--output-subdir', default='af_fixed_group_mechanisms')
    args = ap.parse_args()
    main(args.subject, args.bids_folder, args.rois, args.output_subdir)
