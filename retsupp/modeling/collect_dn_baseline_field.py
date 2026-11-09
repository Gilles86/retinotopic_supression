"""Collect the DN baseline-field fits (m8_full_dnb / dnd / dnbd) into one table.

One row per subject × ROI × model: training SSE of the parent (m7_full, or the b-only fit for the joint model)
and of the fit, the shared terms, and derived quantities:

- ``w_amp`` = (w_b + w_d)/2: scaling b and d together is the same as dividing the amplitude a, so this is an
  amplitude field (close to an input gain constant over the pRF);
- ``w_ratio`` = w_b − w_d: the change in log(b/d), the DN-specific direction;
- ``b_HP``, ``b_LP`` = w_b·cos θ, w_b·sin θ (and ``d_HP``, ``d_LP``): field weight at HP and at the LP
  locations; ``db`` = b_HP − b_LP and ``dd`` = d_HP − d_LP are the HP-specific effects;
- Gauss-Newton correlations (inverse of JᵀJ over the shared terms) for w_b vs w_d, w_amp vs w_ratio, and of
  each weight with g_HP and g_LP, where stored (fits from commit 26a9135 on).

With ``--cv-root`` (a ``--frozen-prf`` run3 root) it writes instead one row per subject × ROI × fold × model with
the held-out R² per voxel averaged over voxels (``cv_r2``) and its paired difference from m7_full (``dcv_r2``).

Usage::

    python -m retsupp.modeling.collect_dn_baseline_field --root <ladder root> --out notes/data/<name>.tsv
    python -m retsupp.modeling.collect_dn_baseline_field --cv-root <frozenprf root> --out notes/data/<name>.tsv
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ('m8_full_dnb', 'm8_full_dnd', 'm8_full_dnbd')
SHARED = ['sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale',
          'w_b', 'w_d', 'theta_bd']


def gn_corr(info):
    """Correlations from the stored Gauss-Newton Hessian, incl. the rotated (w_amp, w_ratio) basis."""
    if 'hessian' not in info:
        return {}
    lab = list(info['hessian_labels'])
    cov = np.linalg.pinv(np.asarray(info['hessian']))
    out = {}

    def corr(u, v):
        a, b = u @ cov @ v, np.sqrt((u @ cov @ u) * (v @ cov @ v))
        return float(a / b) if b > 0 else np.nan

    def e(name, scale=1.0):
        x = np.zeros(len(lab))
        if name in lab:
            x[lab.index(name)] = scale
        return x
    for w in ('w_b', 'w_d'):
        if w in lab:
            for g in ('g_HP', 'g_LP', 'sigma_AF', 'theta_bd'):
                out[f'corr_{w}_{g}'] = corr(e(w), e(g))
    if 'w_b' in lab and 'w_d' in lab:
        out['corr_w_b_w_d'] = corr(e('w_b'), e('w_d'))
        amp, ratio = 0.5 * (e('w_b') + e('w_d')), e('w_b') - e('w_d')
        out['corr_amp_ratio'] = corr(amp, ratio)
        for g in ('g_HP', 'g_LP'):
            out[f'corr_amp_{g}'] = corr(amp, e(g))
            out[f'corr_ratio_{g}'] = corr(ratio, e(g))
    out['hessian_cond'] = float(np.abs(info['hessian_eig']).max()
                                / max(np.abs(info['hessian_eig']).min(), 1e-300))
    return out


def main(root, out, base_root=None):
    root = Path(root)
    rows = []
    sources = [(root, m) for m in MODELS] + ([(Path(base_root), 'm7_full')] if base_root else [])
    for src, model in sources:
        subs = {p.parent.name for p in (root / MODELS[0]).glob('sub-*/*.pkl')}
        for pkl in sorted((src / model).glob('sub-*/sub-*_roi-*_cv-fits.pkl')):
            if model == 'm7_full' and pkl.parent.name not in subs:
                continue
            sub, roi = re.match(r'sub-(\d+)_roi-(.+)_cv-fits.pkl', pkl.name).groups()
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            for k, (pars, info) in enumerate(zip(d['fit_pars_per_fold'], d['optimizer_info'])):
                if pars is None:
                    continue
                row = dict(subject=int(sub), roi=roi, model=model, fold=k, n_voxels=len(pars),
                           sse_parent=info.get('sse_parent_start', np.nan),
                           sse_fit=info.get('sse_final', np.nan),
                           seconds=info.get('seconds', np.nan))
                row.update({c: float(pars[c].iloc[0]) for c in SHARED if c in pars})
                row.update(gn_corr(info))
                rows.append(row)
    df = pd.DataFrame(rows)
    for c in ('w_b', 'w_d', 'theta_bd'):
        if c not in df:
            df[c] = np.nan
    df['dsse_rel'] = (df.sse_fit - df.sse_parent) / df.sse_parent
    th = df.theta_bd
    df['w_amp'] = 0.5 * (df.w_b + df.w_d)
    df['w_ratio'] = df.w_b - df.w_d
    for w in ('b', 'd'):
        df[f'{w}_HP'] = df[f'w_{w}'] * np.cos(th)
        df[f'{w}_LP'] = df[f'w_{w}'] * np.sin(th)
        df[f'd{w}'] = df[f'{w}_HP'] - df[f'{w}_LP']
    df['dS'] = df.g_HP - df.g_LP
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    print(df.groupby('model')[['dsse_rel', 'w_b', 'w_d', 'db', 'dd', 'dS']].median().to_string())
    print(f'{len(df)} rows -> {out}')


def collect_cv(cv_root, out):
    cv_root = Path(cv_root)
    rows = []
    for f in sorted((cv_root / 'm7_full').glob('sub-*/sub-*_roi-*_cv-r2.tsv')):
        ref = pd.read_csv(f, sep='\t')
        for model in ('m7_full',) + MODELS:
            g = cv_root / model / f.parent.name / f.name
            if not g.exists():
                continue
            d = pd.read_csv(g, sep='\t').merge(ref[['fold', 'voxel_id', 'cv_r2', 'train_r2']],
                                                on=['fold', 'voxel_id'], suffixes=('', '_m7'))
            d['dcv'] = d.cv_r2 - d.cv_r2_m7
            d['dtrain'] = d.train_r2 - d.train_r2_m7
            agg = d.groupby('fold').agg(cv_r2=('cv_r2', 'mean'), dcv_r2=('dcv', 'mean'),
                                        dtrain_r2=('dtrain', 'mean'), n_voxels=('voxel_id', 'size'))
            for fold, r in agg.iterrows():
                rows.append(dict(subject=int(d.subject.iloc[0]), roi=d.roi.iloc[0], model=model, fold=fold, **r))
    df = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    print(df.groupby('model')[['dcv_r2', 'dtrain_r2']].mean().to_string())
    print(f'{len(df)} rows -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=None)
    ap.add_argument('--cv-root', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-root', default=None, help='Ladder root holding m7_full, added for comparison.')
    a = ap.parse_args()
    if a.cv_root:
        collect_cv(a.cv_root, a.out)
    else:
        main(a.root, a.out, a.base_root)
