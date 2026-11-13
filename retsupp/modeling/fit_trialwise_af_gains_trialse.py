"""Per-trial gain-field gains with their sampling covariance (addendum 6 of notes/rt_power_plan_2026-11-13.md).

Same BOLD-level method as ``fit_trialwise_af_gains`` (the root's m7_full fits held fixed, each area's signal voxels,
one regressor per trial and kind: target ``T``, distractor ``D_HP`` / ``D_LP``, global ``G``, previous-distractor
location ``P``), but the per-trial gains are estimated by weighted OLS (no ridge, so no shrinkage toward the canonical
gains) and come with an uncertainty for the measurement model.

Uncertainty: circular-shift placebo. The residual of the full canonical model (Y − prediction, all voxels of the area)
is circularly shifted in time within each run, by a random offset of at least ``MIN_SHIFT`` volumes, and projected
through the same estimator, ``(X'WX)⁻¹ X'W r_shift``. The spread of these placebo estimates over ``N_SHIFTS`` shifts is
the sampling (co)variance of each trial's gains under the data's own spatial and temporal noise structure (noise shared
across voxels included). The residual still holds the trial-to-trial gain variation of other trials, so the SEs lean
conservative. The offsets are the same for every area of a subject, so area estimates can be pooled afterwards with
their cross-area error correlation intact (``collect_trialse.py``).

Outputs per subject × area under ``derivatives/<output-subdir>/sub-XX/``:
``sub-XX_roi-R_trialse.tsv`` (trial_idx, kind, g, g0, se_shift, se_iid) and
``sub-XX_roi-R_trialse_shifts.npz`` (placebo estimates (N_SHIFTS, K) float32, trial_idx, kind, g, g0).

Usage (cluster)::

    python -m retsupp.modeling.fit_trialwise_af_gains_trialse 5 --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
        --ladder-root <root> --hrf-table <tsv>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.modeling import fit_trialwise_af_gains as fa
from retsupp.utils.data import Subject

AREAS = [a for g in fa.GROUPS.values() for a in g]   # the nine areas of Fig. 7
N_SHIFTS = 200
MIN_SHIFT = 20      # volumes (32 s): far beyond the HRF and the trial's own window
BATCH = 20


def shift_offsets(subject, n_runs, n_vol, n_shifts=N_SHIFTS):
    rng = np.random.default_rng(1000 + subject)
    return rng.integers(MIN_SHIFT, n_vol - MIN_SHIFT + 1, size=(n_shifts, n_runs))


def area_estimates(sub, roi, data, trials, u, offsets, n_vol):
    bold, paradigm, cond_ind, dyn_ind, grid, tgt_ind = data
    af, model, pars = fa.load_ladder_fit(sub, roi, data)
    vox = af['voxel_mask_indices']
    beta = float(pars['item_scale'].iloc[0])
    paradigm = paradigm * (1.0 + (beta - 1.0) * model._tf_item_mask.numpy())[np.newaxis, :]
    Y = bold[:, vox]
    g_HP, g_LP, g_T = (float(pars[c].iloc[0]) for c in ('g_HP_dyn', 'g_LP_dyn', 'g_T_dyn'))
    sigma_dyn = float(pars['sigma_dyn'].iloc[0])
    pred_full = np.asarray(model.predict(parameters=pars))
    pars0 = pars.copy()
    pars0[['g_HP_dyn', 'g_LP_dyn', 'g_T_dyn']] = 0.0
    pred0 = np.asarray(model.predict(parameters=pars0))

    amp = pars['amplitude'].values.astype(np.float32)
    kern = fa.voxel_kernels(model, pars, Y.shape[0], length=24)
    D, D_glob = fa.ring_drives(model, pars, paradigm, grid, sigma_dyn)
    field = g_HP * dyn_ind * cond_ind + g_LP * dyn_ind * (1 - cond_ind) + g_T * tgt_ind
    phasic = fa.convolve_cols(np.einsum('tl,ltv->tv', field, D), kern, amp)
    target = pred_full - pred0
    recon_err = np.abs(phasic - target).max() / max(np.abs(target).max(), 1e-12)
    print(f'{roi}: {len(vox)} voxels, phasic reconstruction rel. error {recon_err:.2e}', flush=True)
    if recon_err > 1e-2:
        raise RuntimeError(f'{roi}: linear reconstruction does not match the model ({recon_err:.2e})')

    resid0 = (Y - pred0).astype(np.float64)
    resid_full = (Y - pred_full).astype(np.float32)
    w = 1.0 / np.var(resid_full, axis=0)
    regs, meta = fa.trial_regressors(trials, u, D, D_glob, kern, amp)
    g0 = meta['kind'].map({'T': g_T, 'D_HP': g_HP, 'D_LP': g_LP, 'G': 0.0, 'P': 0.0}).values.astype(np.float64)
    (M, b, _c), = fa.normal_equations(regs, resid0, w, [np.ones(len(vox), dtype=bool)])
    del resid0
    g = fa.ridge(M, b, g0, 0.0)
    Minv = np.linalg.inv(M)
    cond = np.linalg.cond(M)
    print(f'{roi}: K = {len(g)} per-trial regressors, cond(X\'WX) = {cond:.2e}', flush=True)

    # Placebo estimates from the circularly shifted residual.
    n_runs = offsets.shape[1]
    assert resid_full.shape[0] == n_runs * n_vol, (resid_full.shape, n_runs, n_vol)
    rw = resid_full * w.astype(np.float32)
    t0 = time.time()
    draws = np.zeros((offsets.shape[0], len(g)), dtype=np.float64)
    for s0 in range(0, offsets.shape[0], BATCH):
        offs = offsets[s0:s0 + BATCH]
        rs = np.empty((len(offs),) + rw.shape, dtype=np.float32)
        for r in range(n_runs):
            sl = slice(r * n_vol, (r + 1) * n_vol)
            for i, o in enumerate(offs):
                rs[i, sl] = np.roll(rw[sl], int(o), axis=0)
        Xtr = np.zeros((len(offs), len(g)))
        for k, (si, win) in enumerate(regs):
            Xtr[:, k] = np.einsum('wv,bwv->b', win, rs[:, si:si + win.shape[0]])
        draws[s0:s0 + BATCH] = Xtr @ Minv   # Minv symmetric
        del rs
    print(f'{roi}: {offsets.shape[0]} placebo shifts in {time.time() - t0:.0f} s', flush=True)
    se_shift = draws.std(0, ddof=1)
    se_iid = np.sqrt(np.clip(np.diag(Minv), 0, None))
    for kind in sorted(meta['kind'].unique()):
        sel = (meta['kind'] == kind).values
        print(f'{roi} {kind}: n {sel.sum()}, median SE shift {np.median(se_shift[sel]):.3g}, iid {np.median(se_iid[sel]):.3g}, '
              f'SD of estimates {g[sel].std():.3g}, mean placebo {draws[:, sel].mean():.2g}', flush=True)
    est = meta.assign(g=g, g0=g0, se_shift=se_shift, se_iid=se_iid, n_voxels=len(vox), cond=cond)
    return est, draws.astype(np.float32)


def main(subject, bids_folder, rois, output_subdir, ladder_roots, hrf_table, n_shifts):
    from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm as build_items_data
    from retsupp.modeling.roi_hrf import load_table
    fa.VOXELS = 'signal'
    fa.LADDER_ROOTS = tuple(Path(bids_folder) / 'derivatives' / r if not str(r).startswith('/') else Path(r)
                            for r in ladder_roots)
    fa.HRF_TABLE = load_table(hrf_table)
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    bold, paradigm, cond_ind, dyn_ind, tgt_ind, grid, _masker, _meta = build_items_data(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    paradigm, grid = np.asarray(paradigm), np.asarray(grid)
    trials, u, n_t = fa.build_trials(sub)
    assert n_t == paradigm.shape[0], (n_t, paradigm.shape)
    trials = fa.add_rt_bins(trials)
    runs = trials[['session', 'run']].drop_duplicates()
    n_vol = sub.get_n_volumes(1, 1)
    assert n_t == len(runs) * n_vol, (n_t, len(runs), n_vol)
    offsets = shift_offsets(subject, len(runs), n_vol, n_shifts)
    trials.to_csv(out_dir / f'sub-{subject:02d}_trials.tsv', sep='\t', index=False)
    data = (bold, paradigm, cond_ind, dyn_ind, grid, tgt_ind)
    for roi in rois:
        print(f'\n===== sub-{subject:02d} {roi} =====', flush=True)
        try:
            est, draws = area_estimates(sub, roi, data, trials, u, offsets, n_vol)
        except fa.NoFit as e:
            print(f'skipped {roi}: {e}')
            continue
        stem = out_dir / f'sub-{subject:02d}_roi-{roi}_trialse'
        est.insert(0, 'roi', roi)
        est.insert(0, 'subject', subject)
        est.to_csv(f'{stem}.tsv', sep='\t', index=False)
        np.savez_compressed(f'{stem}_shifts.npz', draws=draws, trial_idx=est['trial_idx'].to_numpy(),
                            kind=est['kind'].to_numpy().astype(str), g=est['g'].to_numpy(), g0=est['g0'].to_numpy(),
                            offsets=offsets)
        print(f'Saved {stem}.tsv / _shifts.npz', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject', type=int)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--rois', nargs='+', default=AREAS)
    p.add_argument('--output-subdir', default='af_trialwise_trialse')
    p.add_argument('--ladder-root', nargs='+', required=True)
    p.add_argument('--hrf-table', required=True)
    p.add_argument('--n-shifts', type=int, default=N_SHIFTS)
    a = p.parse_args()
    main(a.subject, a.bids_folder, a.rois, a.output_subdir, a.ladder_root, a.hrf_table, a.n_shifts)
