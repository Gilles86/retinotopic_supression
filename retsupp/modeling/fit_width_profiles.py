"""Field widths conditional on a reliable gain (2026-11-12): per-run sufficient statistics on the final joint fit.

Per subject x ROI, on the subject's final joint gain-field fit (m7_full; DoG pRFs, per-ROI HRF), every parameter is
held at the joint fit except the gains of one field, which enter the prediction linearly (the gain map is
1 + Σ g·Gaussian, added to the drive, with no clamp active at the values used), so they are least-squares coefficients:

    P(g) = P_0 + Σ_k g_k X_k,   X_k = [P(g_k = H) − P_0] / H,   P_0 = the prediction with those gains at 0.

Stored per run r (rows of that run, summed over the ROI's voxels): A_r = Σ X_k X_l, b_r = Σ X_k (Y − P_0) and
c_r = Σ (Y − P_0)². Any subset of runs then gives the gains (Σ A)⁻¹ Σ b and the error c − b'A⁻¹b, so the
leave-one-run-out jackknife and the profile over the width are done offline (``field_widths_reliable``).

* ``gains``: the five gains at the fitted widths (sustained at HP g_HP and at the LP locations g_LP; target g_T_dyn,
  with g_T_HP_dyn tied to it as in the joint fit; distractor at HP g_HP_dyn and at LP g_LP_dyn), all five set to 0 in
  P_0. Linearity check: ``lin_err`` = max |forward − backward difference| / max |X| with step ±H.
* ``sus_profile``: σ_AF on a log grid (0.5–8°); at each σ the sustained gains (g_HP, g_LP) are re-solved, everything
  else (pRFs, HRF, item scale, transient gains and width) fixed at the joint fit.
* ``dyn_profile``: σ_dyn (= σ_T_dyn, tied as in the fit) on the same grid; the target and the two distractor gains are
  re-solved, the sustained field and everything else fixed.

Output ``derivatives/<output-subdir>/sub-XX/sub-XX_roi-R_widthprof.npz``.

    python -m retsupp.modeling.fit_width_profiles 3 --rois V3   # canary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import retsupp.modeling.fit_fixed_group_mechanisms as fgm
from retsupp.modeling.fit_af_prf_cv_v2 import build_data_and_paradigm
from retsupp.modeling.roi_hrf import load_table as load_hrf_table
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']
GRID = np.geomspace(0.5, 8.0, 17)
H = 0.5
GAINS = {'g_HP': ['g_HP'], 'g_LP': ['g_LP'], 'g_T': ['g_T_dyn', 'g_T_HP_dyn'],
         'g_D_HP': ['g_HP_dyn'], 'g_D_LP': ['g_LP_dyn']}
SUS, DYN = ['g_HP', 'g_LP'], ['g_T', 'g_D_HP', 'g_D_LP']


def set_gain(p, name, value):
    p = p.copy()
    for col in GAINS[name]:
        if col in p:
            p[col] = value
    return p


def run_stats(X, R, run_meta):
    """Per run: A (k, k), b (k,), c — sums over that run's rows and all voxels."""
    A = np.stack([np.einsum('ktv,ltv->kl', X[:, rm.start:rm.stop], X[:, rm.start:rm.stop]) for rm in run_meta])
    b = np.stack([np.einsum('ktv,tv->k', X[:, rm.start:rm.stop], R[rm.start:rm.stop]) for rm in run_meta])
    c = np.array([(R[rm.start:rm.stop] ** 2).sum() for rm in run_meta])
    return A, b, c


def design(pred, pars, names, Y):
    """P_0 with the named gains at 0, the regressors X (forward difference with step H) and R = Y − P_0."""
    p0 = pars.copy()
    for n in names:
        p0 = set_gain(p0, n, 0.)
    P0 = pred(p0)
    X = np.stack([(pred(set_gain(p0, n, H)) - P0) / H for n in names])
    return P0, X, Y - P0


def fit_roi(sub, roi, data, out_dir, overlay_root):
    bold, paradigm, _ci, _di, _ti, _grid, run_meta = data
    model = fgm.build_full_model(sub, data, roi)
    fit, pars, _w, _n = fgm.base_parameters(sub, model, roi, 'm7_full', None, overlay_root)
    Y = bold[:, fit['voxel_ids']].astype(np.float64)

    def pred(p):
        return np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)

    names = list(GAINS)
    P0, X, R = design(pred, pars, names, Y)
    # linearity: backward difference from the same base
    p0 = pars.copy()
    for n in names:
        p0 = set_gain(p0, n, 0.)
    lin = []
    for k, n in enumerate(names):
        bwd = (P0 - pred(set_gain(p0, n, -H))) / H
        lin.append(np.abs(X[k] - bwd).max() / max(np.abs(X[k]).max(), 1e-12))
    A, b, c = run_stats(X, R, run_meta)
    fitted = {n: float(pars[GAINS[n][0]].iloc[0]) for n in names}
    out = dict(subject=sub.subject_id, roi=roi, n_voxels=Y.shape[1], gain_names=np.array(names),
               A=A, b=b, c=c, lin_err=np.array(lin), fitted_gains=np.array([fitted[n] for n in names]),
               sigma_AF=float(pars['sigma_AF'].iloc[0]), sigma_dyn=float(pars['sigma_dyn'].iloc[0]),
               runs=np.array([[rm.session, rm.run] for rm in run_meta]), grid=GRID)
    for field, gnames, wcols in (('sus', SUS, ['sigma_AF']), ('dyn', DYN, ['sigma_dyn', 'sigma_T_dyn'])):
        As, bs, cs = [], [], []
        for s in GRID:
            ps = pars.copy()
            for col in wcols:
                if col in ps:
                    ps[col] = s
            _P0, Xs, Rs = design(pred, ps, gnames, Y)
            a_, b_, c_ = run_stats(Xs, Rs, run_meta)
            As.append(a_)
            bs.append(b_)
            cs.append(c_)
        out[f'{field}_A'], out[f'{field}_b'], out[f'{field}_c'] = np.stack(As), np.stack(bs), np.stack(cs)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f'sub-{sub.subject_id:02d}_roi-{roi}_widthprof.npz', **out)
    g_all = np.linalg.solve(A.sum(0), b.sum(0))
    print(f'sub-{sub.subject_id:02d} {roi}: {Y.shape[1]} voxels; lin_err max {max(lin):.1e}; gains (re-solved) '
          + ', '.join(f'{n} {g:+.4f} (fit {fitted[n]:+.4f})' for n, g in zip(names, g_all))
          + f'; σ_AF {out["sigma_AF"]:.2f}, σ_dyn {out["sigma_dyn"]:.2f}', flush=True)


def main(subject, bids_folder, rois, output_subdir, base_root, hrf_table, overlay_root):
    fgm.BASE_ROOT = base_root
    fgm.HRF_TABLE = load_hrf_table(hrf_table)
    fgm.FAMILY = 'dog'
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    todo = [r for r in rois if not (out_dir / f'sub-{subject:02d}_roi-{r}_widthprof.npz').exists()]
    todo = [r for r in todo if fgm.fit_path(bids_folder, 'm7_full', subject, r, overlay_root).exists()]
    if not todo:
        print('Nothing to do')
        return
    bold, paradigm, ci, di, ti, grid, _masker, run_meta = build_data_and_paradigm(
        sub, resolution=50, grid_radius=5.0, item_shape='rectangle')
    data = (bold, paradigm, ci, di, ti, grid, run_meta)
    for roi in todo:
        fit_roi(sub, roi, data, out_dir, overlay_root)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--output-subdir', default='field_widths_reliable_2026-11-12')
    ap.add_argument('--base-root', default='af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf')
    ap.add_argument('--hrf-table', default='/shares/zne.uzh/gdehol/ds-retsupp/derivatives/paper_2026-11/roi_hrf_dog_final2.tsv')
    ap.add_argument('--overlay-root', default=None)
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.rois, a.output_subdir, a.base_root, a.hrf_table, a.overlay_root)
