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
dT         m4_TDS       g_T_HP_dyn (target at HP; starts at g_T_dyn)  (base)
kappa      m7_full      κ, DoG centre/surround tilt (dogk)      (base)
dnw        m7_full      w_b = w_d = δ, DN baseline field at HP  (base)
dnratio    m7_full      w_b = +δ, w_d = −δ (b/d ratio) at HP    (base)
=========  ===========  ======================================  ==========

kappa (``--family dog``) uses ``DoGCentreSurroundField``, dnw / dnratio (``--family dn``) ``DNBaselineField`` with
θ = 0 (field only at HP, as dS perturbs g_HP alone) and σ_AF the subject's own; the new terms start at 0, where
both reproduce m7_full (checked on every call, max |Δpred| printed and asserted).

X is a central difference with step ``H``; ``lin_err`` compares the
forward and backward differences.

DN responses curve with the gains, so this approximation fails there. ``--exact``
evaluates the actual predictions on a grid of δ instead and saves each voxel's
ΔSSE(δ); ``loso_fixed_group_exact`` does the test from those curves.

``--hp-rotation K`` relabels HP K ring steps (null for the conditional dS and dD); ``--sessions S`` uses only
session S's runs for a and b (split-half reliability of per-subject estimates).

Outputs (never overwritten) under
``derivatives/af_fixed_group_mechanisms/sub-XX/sub-XX_roi-R_stats.tsv.gz``, or
``sub-XX_roi-R_curves.npz`` with ``--exact``.

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

from braincoder.hrf import SPMHRFModel  # noqa: F401
from retsupp.modeling.roi_hrf import hrf_model as roi_hrf_model, load_table as load_hrf_table
from retsupp.modeling.fit_af_prf_cv_v2 import (
    build_data_and_paradigm,
    get_ring_positions,
)
from retsupp.modeling.fit_cv_ladder_nested import GAINS, LADDER, TIE_PARTNER, sign_pattern_of
from retsupp.modeling.dn_baseline_field import DNBaselineField
from retsupp.modeling.dog_centre_surround_field import DoGCentreSurroundField
from retsupp.modeling.lp_split import LPSplitDoG
from retsupp.modeling.target_split import TargetSplitDN, TargetSplitDoG
from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
SUBJECTS = range(1, 31)
BASE_ROOT = 'af_prf_cv_ladder_nested_full_bounded'
HRF_TABLE = None   # one HRF per subject x ROI (roi_hrf.load_table); canonical when None
H = 0.05
FAMILY = 'dog'
# name: (base model, perturbed parameters, (width parameter, child model))
MECHANISMS = {
    'items': ('m00_bar', ['item_scale'], None),
    'T': ('m0_prf', ['g_T_dyn'], ('sigma_dyn', 'm1_T')),
    'D': ('m0_prf', ['g_HP_dyn', 'g_LP_dyn'], ('sigma_dyn', 'm2_D')),
    'S': ('m0_prf', ['g_HP', 'g_LP'], ('sigma_AF', 'm3_S')),
    'dS': ('m4_TDS', ['g_HP'], None),
    'dD': ('m4_TDS', ['g_HP_dyn'], None),
    'dT': ('m4_TDS', ['g_T_HP_dyn'], None),
    # Distractor gain on top of the target term (m3b_TD vs m1_T). σ_dyn stays the subject's own from m1_T, since the
    # distractor shares it with the target and changing it would change the target term of the base.
    'DonT': ('m1_T', ['g_HP_dyn', 'g_LP_dyn'], None),
    # Terms of the m8 models on top of the final m7_full; their own model class (MECH_MODEL), new terms at 0.
    'kappa': ('m7_full', ['kappa'], None),
    'dnw': ('m7_full', ['w_b', 'w_d'], None),
    'dnratio': ('m7_full', ['w_b', 'w_d'], None),
    # Sustained LP gain split, next to HP minus opposite HP (lp_split), on top of m7_full; e.g. per session (--sessions).
    'adjopp': ('m7_full', ['g_adj_minus_opp'], None),
}
# Mechanisms with their own model class (the others use build_full_model), and the family it needs.
MECH_MODEL = {'kappa': 'dog', 'dnw': 'dn', 'dnratio': 'dn', 'adjopp': 'dog'}
# Direction of the perturbation per parameter (default +1 each).
PERTURB_SIGN = {'dnratio': [1.0, -1.0]}
# Lower and upper δ of the --exact grid. items stops at item_scale = 0.
EXACT_RANGE = {'items': (-1., .5), 'T': (-1.5, 1.5), 'D': (-1.5, 1.5),
               'S': (-.4, .4), 'dS': (-.8, .8), 'dD': (-1.5, 1.5),
               'dT': (-1.5, 1.5), 'DonT': (-1.5, 1.5),
               'kappa': (-1., 1.), 'dnw': (-1.5, 1.5), 'dnratio': (-.5, .5), 'adjopp': (-.8, .8)}   # sub-03 canary: minima within ±0.15
N_GRID = 25


def exact_grid(lo, hi):
    """N_GRID values of δ from lo to hi, including 0 and densest around it."""
    u = np.sinh(np.linspace(-2.5, 2.5, N_GRID)) / np.sinh(2.5)
    return np.where(u < 0, -u * lo, u * hi)


def fit_path(bids_folder, model, subject, roi, overlay_root=None):
    """The fit in ``overlay_root`` if that subject x ROI finished there, else in BASE_ROOT.

    m7_full is the last model of the ladder, so its presence means the whole
    chain exists and base and child models never mix roots.
    """
    tail = (f'sub-{subject:02d}', f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl')
    deriv = Path(bids_folder) / 'derivatives'
    if overlay_root and (deriv / overlay_root / 'm7_full').joinpath(*tail).exists():
        return (deriv / overlay_root / model).joinpath(*tail)
    return (deriv / BASE_ROOT / model).joinpath(*tail)


def load_fit(bids_folder, model, subject, roi, overlay_root=None):
    with open(fit_path(bids_folder, model, subject, roi, overlay_root), 'rb') as f:
        return pickle.load(f)


def loso_width(bids_folder, par, model, subject, roi, overlay_root=None):
    """Median of the other subjects' fitted width in the child model."""
    vals = []
    for s in SUBJECTS:
        if s == subject or not fit_path(bids_folder, model, s, roi, overlay_root).exists():
            continue
        vals.append(float(load_fit(bids_folder, model, s, roi, overlay_root)
                          ['fit_pars_per_fold'][0][par].iloc[0]))
    return float(np.median(vals)), len(vals)


def build_full_model(sub, data, roi):
    _bold, paradigm, ci, di, ti, grid_coords, _run_meta = data
    hrf_model = roi_hrf_model(sub, roi, HRF_TABLE)   # canonical without --hrf-table
    # Everything free, so each mechanism can move away from its base value.
    # The target-split class: at g_T_HP_dyn = g_T_dyn it predicts exactly what the parent does.
    if FAMILY == 'dn':
        return TargetSplitDN(
            grid_coordinates=grid_coords, paradigm=paradigm,
            condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
            ring_positions=get_ring_positions(), hrf_model=hrf_model,
            sign_pattern=sign_pattern_of('m7_full'), item_scale_mode='free')
    cls = TargetSplitDoG
    if FAMILY == 'gauss':   # the DoG class with the surround amplitude fixed at 0 (as the Gaussian ladder)
        from retsupp.modeling.gauss_attention_field import gaussian
        cls = gaussian(TargetSplitDoG)
    return cls(
        grid_coordinates=grid_coords, paradigm=paradigm,
        condition_indicator=ci, dynamic_indicator=di, target_indicator=ti,
        ring_positions=get_ring_positions(), hrf_model=hrf_model,
        sign_pattern=sign_pattern_of('m7_full'), item_scale_mode='free')


def mech_model(sub, data, roi, mech):
    """The model a mechanism is evaluated with: its own m8 class, else the target-split class."""
    if mech not in MECH_MODEL:
        return build_full_model(sub, data, roi)
    if MECH_MODEL[mech] != FAMILY:
        raise ValueError(f'{mech} needs --family {MECH_MODEL[mech]}')
    _bold, paradigm, ci, di, ti, grid_coords, _run_meta = data
    kw = dict(grid_coordinates=grid_coords, paradigm=paradigm, condition_indicator=ci, dynamic_indicator=di,
              target_indicator=ti, ring_positions=get_ring_positions(),
              hrf_model=roi_hrf_model(sub, roi, HRF_TABLE), sign_pattern=sign_pattern_of('m7_full'),
              item_scale_mode='free')
    if mech == 'adjopp':
        return LPSplitDoG(**kw)
    if FAMILY == 'dog':
        return DoGCentreSurroundField(**kw)
    return DNBaselineField(targets=('b', 'd'), **kw)


def perturbed(pars, mech, delta):
    p = pars.copy()
    perturb = MECHANISMS[mech][1]
    for name, sgn in zip(perturb, PERTURB_SIGN.get(mech, [1.0] * len(perturb))):
        p[name] += sgn * delta
    return p


def base_parameters(sub, model, roi, base, width, overlay_root):
    """A mechanism's base fit, with its width taken from the other subjects."""
    fit = load_fit(sub.bids_folder, base, sub.subject_id, roi, overlay_root)
    labels = list(model.parameter_labels)
    pars = fit['fit_pars_per_fold'][0].reindex(columns=labels)
    if pars['item_scale'].isna().all():
        pars['item_scale'] = 1.0           # m00: items at bar strength
    # A gain the base ties (or zeroes) starts at its partner; a free one (m7_full) keeps its fitted value.
    pattern = dict(zip(GAINS, LADDER[base][:5]))
    for g, partner in TIE_PARTNER.items():
        if pattern[g] != 'free':
            pars[g] = pars[partner]
    width_val, n_width = np.nan, 0
    if width is not None:
        width_val, n_width = loso_width(sub.bids_folder, *width,
                                        sub.subject_id, roi, overlay_root)
        pars[width[0]] = width_val
    if 'sigma_T_dyn' in labels:
        pars['sigma_T_dyn'] = pars['sigma_dyn']
    if 'g_T_HP_dyn' in labels:
        pars['g_T_HP_dyn'] = pars['g_T_dyn']
    for new in ('kappa', 'w_b', 'w_d', 'theta_bd', 'g_adj_minus_opp'):   # m8 terms at the value that reproduces m7 (θ = 0: HP only)
        if new in labels:
            pars[new] = 0.0
    assert not pars.isna().any().any(), pars.columns[pars.isna().any()]
    return fit, pars.astype(np.float32), width_val, n_width


def fit_roi(sub, roi, data, out_path, overlay_root=None, sessions=None, run_parity=None):
    bold, paradigm, _ci, _di, _ti, _grid, run_meta = data

    # Sufficient statistics only over the runs of the requested sessions (the fits stay as they are).
    used = [rm for rm in run_meta if (sessions is None or rm.session in sessions)
            and (run_parity is None or rm.run % 2 == (1 if run_parity == 'odd' else 0))]
    conds = {hp: np.concatenate([np.arange(rm.start, rm.stop) for rm in used if rm.hp == hp])
             for hp in sorted({rm.hp for rm in used})}
    out = []
    for mech, (base, perturb, width) in MECHANISMS.items():
        model = mech_model(sub, data, roi, mech)

        def pred(p):
            return np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)

        fit, pars, width_val, n_width = base_parameters(sub, model, roi, base, width, overlay_root)
        Y = bold[:, fit['voxel_ids']].astype(np.float64)

        P0 = pred(pars)
        up, dn = perturbed(pars, mech, H), perturbed(pars, mech, -H)
        fwd = (pred(up) - P0) / H
        bwd = (P0 - pred(dn)) / H
        X = .5 * (fwd + bwd)
        lin_err = np.abs(fwd - bwd).max() / max(np.abs(X).max(), 1e-12)
        R = Y - P0
        t_all = np.concatenate(list(conds.values()))
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
              f'lin_err={lin_err:.2e}, own δ̂={(X[t_all] * R[t_all]).sum() / (X[t_all] ** 2).sum():+.4f}')
    df = pd.concat(out)
    df.insert(0, 'roi', roi)
    df.insert(0, 'subject', sub.subject_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)


def fit_roi_exact(sub, roi, data, out_path, overlay_root=None):
    """Each voxel's ΔSSE(δ) from the actual predictions, per mechanism."""
    bold, paradigm, _ci, _di, _ti, _grid, run_meta = data
    ref_model = build_full_model(sub, data, roi)
    conds = [np.concatenate([np.arange(rm.start, rm.stop) for rm in run_meta if rm.hp == hp])
             for hp in sorted({rm.hp for rm in run_meta})]
    saved = {}
    for mech, (base, perturb, width) in MECHANISMS.items():
        model = mech_model(sub, data, roi, mech)
        fit, pars, width_val, n_width = base_parameters(sub, model, roi, base, width, overlay_root)
        Y = bold[:, fit['voxel_ids']].astype(np.float64)
        if mech in MECH_MODEL:
            # At δ = 0 the m8 model must predict exactly what the parent does.
            _f, ref_pars, _w, _n = base_parameters(sub, ref_model, roi, base, width, overlay_root)
            P_ref = np.asarray(ref_model.predict(paradigm=paradigm, parameters=ref_pars), dtype=np.float64)
            P_0 = np.asarray(model.predict(paradigm=paradigm, parameters=pars), dtype=np.float64)
            err = np.abs(P_0 - P_ref).max()
            print(f'sub-{sub.subject_id:02d} {roi} {mech}: max |pred(δ=0) - pred({base})| = {err:.2e} '
                  f'(max |pred| {np.abs(P_ref).max():.2e})', flush=True)
            assert err <= 1e-4 + 1e-5 * np.abs(P_ref).max(), f'{mech}: δ=0 does not reproduce {base}'   # float32: ~2e-5
        grid = exact_grid(*EXACT_RANGE[mech])
        sse = []
        for delta in grid:
            p = perturbed(pars, mech, delta)
            P = np.asarray(model.predict(paradigm=paradigm, parameters=p), dtype=np.float64)
            assert np.isfinite(P).all(), f'{mech} δ={delta}: non-finite prediction'
            sse.append(((Y - P) ** 2).sum(0))
        sse = np.stack(sse, 1)                                 # (V, K)
        dsse = sse - sse[:, [N_GRID // 2]]
        saved.update({f'{mech}_grid': grid, f'{mech}_dsse': dsse.astype(np.float32),
                      f'{mech}_voxel_ids': np.asarray(fit['voxel_ids']),
                      f'{mech}_sst': sum(((Y[t] - Y[t].mean(0)) ** 2).sum(0) for t in conds),
                      f'{mech}_width': width_val, f'{mech}_n_width': n_width})
        total = dsse.sum(0)
        print(f'sub-{sub.subject_id:02d} {roi} {mech}: {Y.shape[1]} voxels, '
              f'width={width_val:.3f}, best δ on grid={grid[total.argmin()]:+.3f}, '
              f'ΔSSE={total.min():.2f}', flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, subject=sub.subject_id, roi=roi, **saved)


def main(subject, bids_folder, rois, output_subdir='af_fixed_group_mechanisms',
         overlay_root=None, base_root=None, family='dog', step=None, exact=False, hp_rotation=0,
         mechanisms=None, sessions=None, hrf_table=None, run_parity=None):
    global BASE_ROOT, FAMILY, H, MECHANISMS, HRF_TABLE
    HRF_TABLE = load_hrf_table(hrf_table)
    # The m8 terms (kappa, dnw, dnratio) are family-specific and run only when listed.
    mechanisms = mechanisms or [m for m in MECHANISMS if m not in MECH_MODEL]
    MECHANISMS = {m: MECHANISMS[m] for m in mechanisms}
    BASE_ROOT, FAMILY, H = base_root or BASE_ROOT, family, step or H
    sub = Subject(subject, bids_folder=Path(bids_folder))
    out_dir = Path(bids_folder) / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    tail = 'curves.npz' if exact else 'stats.tsv.gz'
    todo = {roi: out_dir / f'sub-{subject:02d}_roi-{roi}_{tail}' for roi in rois}
    todo = {roi: p for roi, p in todo.items() if not p.exists()}
    bases = {m[0] for m in MECHANISMS.values()}
    for roi in [r for r in todo if not all(fit_path(bids_folder, b, subject, r, overlay_root).exists()
                                           for b in bases)]:
        print(f'sub-{subject:02d} {roi}: base fit missing (no reliable voxels?), skipped')
        del todo[roi]
    if not todo:
        print('All outputs exist (or have no base fit), nothing to do')
        return
    (bold, paradigm, ci, di, ti, grid_coords, _masker,
     run_meta) = build_data_and_paradigm(sub, resolution=50, grid_radius=5.0,
                                         item_shape='rectangle')
    ci = np.roll(ci, hp_rotation, axis=1)   # relabel HP, as in fit_cv_ladder_nested --hp-rotation
    data = (bold, paradigm, ci, di, ti, grid_coords, run_meta)
    for roi, path in todo.items():
        if exact:
            assert sessions is None, '--sessions is not implemented with --exact'
            fit_roi_exact(sub, roi, data, path, overlay_root)
        else:
            fit_roi(sub, roi, data, path, overlay_root, sessions, run_parity)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--rois', nargs='+', default=ROIS)
    ap.add_argument('--output-subdir', default='af_fixed_group_mechanisms')
    ap.add_argument('--overlay-root', default=None,
                    help='Derivatives folder whose finished subject x ROI fits replace BASE_ROOT')
    ap.add_argument('--base-root', default=None, help='Fits to use instead of BASE_ROOT')
    ap.add_argument('--family', choices=['dog', 'dn', 'gauss'], default='dog',
                    help='gauss: the Gaussian ladder fits (DoG classes with the surround amplitude fixed at 0)')
    ap.add_argument('--step', type=float, default=None,
                    help='Central-difference step; DN is less linear in the gains')
    ap.add_argument('--hp-rotation', type=int, choices=[0, 1, 2, 3], default=0,
                    help='Relabel HP K ring steps away, for a null of the conditional dS and dD. Their base '
                         '(m4_TDS) ties HP and LP gains, so its fits do not depend on the labels and need no '
                         'relabelled base root; mechanisms on label-dependent bases would.')
    ap.add_argument('--sessions', type=int, nargs='+', choices=[1, 2], default=None,
                    help='Sufficient statistics from these sessions only')
    ap.add_argument('--run-parity', choices=['odd', 'even'], default=None,
                    help='Sufficient statistics from odd or even runs only: each half holds every HP location '
                         '(split-half reliability; a session split confounds session-wide gain with dS)')
    ap.add_argument('--exact', action='store_true',
                    help='Save ΔSSE on a grid of δ from the actual predictions')
    ap.add_argument('--mechanisms', nargs='+', default=None, choices=list(MECHANISMS),
                    help='Only these mechanisms (default all but the m8 terms kappa, dnw, dnratio)')
    ap.add_argument('--hrf-table', default=None,
                    help='TSV of one HRF per subject x ROI (build_roi_hrf_table.py); must match the fits used')
    args = ap.parse_args()
    main(args.subject, args.bids_folder, args.rois, args.output_subdir, args.overlay_root,
         args.base_root, args.family, args.step, args.exact, args.hp_rotation,
         args.mechanisms, args.sessions, hrf_table=args.hrf_table, run_parity=args.run_parity)
