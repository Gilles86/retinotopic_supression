"""Learning curves of the gain fields on top of the nested full-data fit.

Takes the bounded, nested, full-data ladder fit (``m7_full`` in
``af_prf_cv_ladder_nested_full_bounded``) as fixed — pRFs, HRF, σ_AF,
σ_dyn and item_scale — and re-estimates the five gains
(``g_HP``, ``g_LP``, ``g_HP_dyn``, ``g_LP_dyn``, ``g_T_dyn``) with
time structure.

With everything but the gains fixed, the prediction is linear in the
gains and in the paradigm, so every regressor is a difference of two
model predictions:

- ``X[k, r] = pred(gain k = 1, paradigm of run r only)
  − pred(all gains 0, paradigm of run r only)``
- ``B[r] = pred(all gains 0, paradigm of run r only) − baseline``: the
  gain-free stimulus drive of run r; its weight is a per-run amplitude
  (habituation / arousal control).
- Ramp regressors use the paradigm multiplied by a per-TR modulator
  instead of a run mask (exact, because the model is linear in the
  paradigm).

The decomposition is checked against the model's own prediction with
the fitted gains (``recon_rel_err``).

Three designs, each weighted OLS over all voxels of the ROI (weights
1 / residual variance of the full fit), after removing a per-voxel,
per-run intercept:

``runwise``
    Per-run amplitude + all five gains per run (72 regressors).
``position``
    Per-run amplitude + five gains per position of the run within its
    HP block (0/1/2).
``parametric``
    Per-run amplitude + within-run amplitude ramp + for each gain a
    mean, a slope over continuous time within the HP block (in runs,
    TR-resolved) and a slope over block number across the experiment.

Outputs (never overwritten) under
``derivatives/af_learning_gains_nested/sub-XX/``:
``sub-XX_roi-R_{runwise,position,parametric,check}.tsv``.

Usage::

    python -m retsupp.modeling.fit_learning_gains_nested 3 --rois V1 V3AB \\
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
from retsupp.modeling.fit_cv_ladder_nested import GAINS, sign_pattern_of
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
BASE_ROOT = 'af_prf_cv_ladder_nested_full_bounded'
BASE_MODEL = 'm7_full'
KINDS = dict(zip(GAINS, ['S_HP', 'S_LP', 'D_HP', 'D_LP', 'T']))


def block_structure(run_meta):
    """Position of each run within its block of consecutive same-HP runs,
    and the block number across the experiment (both 0-based)."""
    pos, blk = [], []
    for i, rm in enumerate(run_meta):
        new = i == 0 or rm.hp != run_meta[i - 1].hp
        blk.append(0 if i == 0 else blk[-1] + new)
        pos.append(0 if new else pos[-1] + 1)
    return np.array(pos), np.array(blk)


def wls(regs, y, w):
    """Weighted OLS of y (T, V) on a list of (T, V) regressors."""
    X = np.stack([r.ravel() for r in regs]).astype(np.float64)
    wt = np.tile(w, y.shape[0])
    Xw = X * wt
    XtX = Xw @ X.T
    beta = np.linalg.solve(XtX, Xw @ y.ravel())
    resid = y.ravel() - beta @ X
    dof = y.size - len(regs)
    se = np.sqrt(np.diag(np.linalg.inv(XtX)) * (resid ** 2 * wt).sum() / dof)
    return beta, se


def fit_roi(sub, roi, data, out_dir):
    bold, paradigm, ci, di, ti, grid_coords, run_meta = data
    stem = f'sub-{sub.subject_id:02d}_roi-{roi}'
    outs = {n: out_dir / f'{stem}_{n}.tsv'
            for n in ('runwise', 'position', 'parametric', 'check')}
    if any(p.exists() for p in outs.values()):
        print(f'{stem}: outputs exist, skipping')
        return
    pkl = (sub.bids_folder / 'derivatives' / BASE_ROOT / BASE_MODEL
           / f'sub-{sub.subject_id:02d}' / f'{stem}_cv-fits.pkl')
    with open(pkl, 'rb') as f:
        fit = pickle.load(f)
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)
    model = build_model(
        grid_coords=grid_coords, paradigm=paradigm,
        condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
        ring_positions=get_ring_positions(), hrf_model=hrf_model,
        sign_pattern=sign_pattern_of(BASE_MODEL), item_scale='free')
    pars = fit['fit_pars_per_fold'][0].reindex(
        columns=list(model.parameter_labels)).astype(np.float32)
    Y = bold[:, fit['voxel_ids']].astype(np.float64)

    def pred(p, par=paradigm):
        return np.asarray(model.predict(paradigm=par, parameters=p),
                          dtype=np.float64)

    p0 = pars.copy()
    p0[GAINS] = 0.0
    units = {}
    for g in GAINS:
        units[g] = p0.copy()
        units[g][g] = 1.0
    baseline = pred(p0, np.zeros_like(paradigm))

    # --- linearity check on the full time series ---
    full = pred(pars)
    base_full = pred(p0)
    recon = base_full + sum(float(pars[g].iloc[0]) * (pred(units[g]) - base_full)
                            for g in GAINS)
    recon_err = np.abs(recon - full).max() / max(np.abs(full - baseline).max(), 1e-12)
    w = 1.0 / np.var(Y - full, axis=0)

    # --- per-run regressors ---
    pos, blk = block_structure(run_meta)
    B, X = [], {g: [] for g in GAINS}
    for rm in run_meta:
        par_r = np.zeros_like(paradigm)
        par_r[rm.start:rm.stop] = paradigm[rm.start:rm.stop]
        b_r = pred(p0, par_r)
        B.append(b_r - baseline)
        for g in GAINS:
            X[g].append(pred(units[g], par_r) - b_r)

    # --- TR-resolved modulators (time within HP block in runs; block number) ---
    m_wb = np.zeros(paradigm.shape[0])
    m_blk = np.zeros(paradigm.shape[0])
    for i, rm in enumerate(run_meta):
        m_wb[rm.start:rm.stop] = pos[i] + (np.arange(rm.n_T) + .5) / rm.n_T
        m_blk[rm.start:rm.stop] = blk[i]
    m_wb -= m_wb.mean()
    m_blk -= m_blk.mean()
    ramp = {}
    for name, m in (('wb', m_wb), ('blk', m_blk)):
        par_m = paradigm * m[:, np.newaxis].astype(paradigm.dtype)
        b_m = pred(p0, par_m)
        ramp[name] = {'B': b_m - baseline}
        for g in GAINS:
            ramp[name][g] = pred(units[g], par_m) - b_m

    # --- remove a per-voxel intercept within each run (Frisch–Waugh) ---
    def demean(a):
        a = a.copy()
        for rm in run_meta:
            a[rm.start:rm.stop] -= a[rm.start:rm.stop].mean(0)
        return a

    y = demean(Y)
    B = [demean(b) for b in B]
    X = {g: [demean(x) for x in X[g]] for g in GAINS}
    for name in ramp:
        ramp[name] = {k: demean(v) for k, v in ramp[name].items()}

    meta = dict(subject=sub.subject_id, roi=roi)
    runs = [dict(session=rm.session, run=rm.run, hp=rm.hp,
                 position=int(pos[i]), block=int(blk[i]))
            for i, rm in enumerate(run_meta)]

    # runwise
    regs = B + [X[g][i] for g in GAINS for i in range(len(run_meta))]
    names = ([('amplitude', i) for i in range(len(run_meta))]
             + [(KINDS[g], i) for g in GAINS for i in range(len(run_meta))])
    beta, se = wls(regs, y, w)
    runwise = pd.DataFrame([dict(**meta, **runs[i], kind=k, g=b, se=s)
                            for (k, i), b, s in zip(names, beta, se)])

    # position within HP block
    regs, names = list(B), [('amplitude', f'run{i}') for i in range(len(run_meta))]
    for g in GAINS:
        for p in sorted(set(pos)):
            regs.append(sum(X[g][i] for i in range(len(run_meta)) if pos[i] == p))
            names.append((KINDS[g], f'pos{p}'))
    beta, se = wls(regs, y, w)
    position = pd.DataFrame([dict(**meta, kind=k, level=l, g=b, se=s)
                             for (k, l), b, s in zip(names, beta, se)])

    # parametric
    regs = list(B) + [ramp['wb']['B']]
    names = [('amplitude', f'run{i}') for i in range(len(run_meta))] + [('amplitude', 'within_block')]
    for g in GAINS:
        regs += [sum(X[g]), ramp['wb'][g], ramp['blk'][g]]
        names += [(KINDS[g], 'mean'), (KINDS[g], 'within_block'), (KINDS[g], 'across_blocks')]
    beta, se = wls(regs, y, w)
    parametric = pd.DataFrame([dict(**meta, kind=k, term=t, g=b, se=s)
                               for (k, t), b, s in zip(names, beta, se)])

    check = pd.DataFrame([dict(**meta, n_voxels=Y.shape[1],
                               recon_rel_err=float(recon_err),
                               **{f'fit_{g}': float(pars[g].iloc[0]) for g in GAINS})])
    print(check.T.to_string())
    print(parametric.to_string())
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (('runwise', runwise), ('position', position),
                     ('parametric', parametric), ('check', check)):
        df.to_csv(outs[name], sep='\t', index=False)
    print(f'Saved {out_dir}/{stem}_*.tsv')


def main(subject, bids_folder, rois, output_subdir='af_learning_gains_nested'):
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    (bold, paradigm, ci, di, ti, grid_coords, _masker,
     run_meta) = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0,
                                         item_shape='rectangle')
    data = (bold, paradigm, ci, di, ti, grid_coords, run_meta)
    for roi in rois:
        print(f'\n===== sub-{subject:02d} {roi} =====')
        fit_roi(sub, roi, data, out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--output-subdir', default='af_learning_gains_nested')
    args = ap.parse_args()
    main(args.subject, args.bids_folder, args.rois, args.output_subdir)
