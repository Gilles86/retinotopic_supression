"""Collect the DoG centre/surround tilt fits (m8_full_dogk, and dogcen / dogsur where fitted) and test them.

All-runs fits (``--root``, with ``--base-root`` holding m7_full): one row per subject × ROI × model with training
SSE of m7 and of the fit, the shared terms, dS = g_HP − g_LP (common gain), the effective HP contrasts on the
centre and surround pooling, dS_cen = (1 + κ)·dS and dS_sur = (1 − κ)·dS, and Gauss-Newton correlations of κ with
g_HP, g_LP, dS and σ_AF (inverse of JᵀJ over the shared terms). Then, per ROI: Wilcoxon of κ against 0 and of
dS(dogk) − dS(m7), BH across ROIs; pooled per subject (mean over ROIs) the same tests.

Held-out (``--cv-root``, a ``--frozen-prf`` run3 root): one row per subject × ROI × fold × model with the mean
held-out R² over voxels and its paired difference from m7_full refitted in the same fold; group test of the subject
mean (over ROIs and folds) of ΔCV-R² against 0, and per ROI with BH.

Usage::

    python -m retsupp.modeling.collect_dog_cs_field --root <dogcs root> --base-root <final DoG root> \\
        --out notes/data/dog_cs_field_2026-11.tsv
    python -m retsupp.modeling.collect_dog_cs_field --cv-root <frozenprf root> --out notes/data/dog_cs_field_cv_2026-11.tsv
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

MODELS = ('m8_full_dogk', 'm8_full_dogcen', 'm8_full_dogsur')
SHARED = ['sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale', 'kappa']


def gn_corr(info):
    if 'hessian' not in info:
        return {}
    lab = list(info['hessian_labels'])
    cov = np.linalg.pinv(np.asarray(info['hessian']))

    def e(*terms):
        x = np.zeros(len(lab))
        for t, s in terms:
            x[lab.index(t)] = s
        return x

    def corr(u, v):
        b = np.sqrt((u @ cov @ u) * (v @ cov @ v))
        return float(u @ cov @ v / b) if b > 0 else np.nan
    k = e(('kappa', 1.0))
    return dict(corr_kappa_g_HP=corr(k, e(('g_HP', 1.0))), corr_kappa_g_LP=corr(k, e(('g_LP', 1.0))),
                corr_kappa_dS=corr(k, e(('g_HP', 1.0), ('g_LP', -1.0))),
                corr_kappa_gmean=corr(k, e(('g_HP', 0.25), ('g_LP', 0.75))),
                corr_kappa_sigma_AF=corr(k, e(('sigma_AF', 1.0))))


def test(df, col, by='roi'):
    """Wilcoxon of ``col`` against 0 per ``by`` level (BH across levels) and pooled per subject."""
    rows = []
    for lev, d in df.groupby(by):
        x = d[col].dropna()
        p = wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
        rows.append(dict(level=lev, n=len(x), median=x.median(), p=p))
    r = pd.DataFrame(rows)
    ok = r.p.notna()
    r['q_bh'] = np.nan
    r.loc[ok, 'q_bh'] = multipletests(r.p[ok], method='fdr_bh')[1]
    pooled = df.groupby('subject')[col].mean().dropna()
    r = pd.concat([r, pd.DataFrame([dict(level='pooled (subject mean)', n=len(pooled), median=pooled.median(),
                                         p=wilcoxon(pooled).pvalue, q_bh=np.nan)])], ignore_index=True)
    return r.assign(variable=col)


def main(root, base_root, out):
    root = Path(root)
    rows = []
    for src, model in [(root, m) for m in MODELS] + [(Path(base_root), 'm7_full')]:
        for pkl in sorted((src / model).glob('sub-*/sub-*_roi-*_cv-fits.pkl')):
            sub, roi = re.match(r'sub-(\d+)_roi-(.+)_cv-fits.pkl', pkl.name).groups()
            if model == 'm7_full' and not (root / MODELS[0] / pkl.parent.name / pkl.name).exists():
                continue
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            pars, info = d['fit_pars_per_fold'][0], d['optimizer_info'][0]
            row = dict(subject=int(sub), roi=roi, model=model, n_voxels=len(pars),
                       sse_m7=info.get('sse_parent_start', np.nan), sse_fit=info.get('train_sse_total', np.nan),
                       sse_shared=info.get('sse_final', np.nan))
            row.update({c: float(pars[c].iloc[0]) for c in SHARED if c in pars})
            row.update(gn_corr(info))
            rows.append(row)
    df = pd.DataFrame(rows)
    df['kappa'] = df['kappa'].fillna(0.0)
    df['dS'] = df.g_HP - df.g_LP
    df['dS_cen'] = (1 + df.kappa) * df.dS
    df['dS_sur'] = (1 - df.kappa) * df.dS
    m7 = df[df.model.eq('m7_full')].set_index(['subject', 'roi'])
    df['dsse_rel'] = df.sse_shared / df.set_index(['subject', 'roi']).index.map(m7.sse_shared) - 1
    df['dS_m7'] = df.set_index(['subject', 'roi']).index.map(m7.dS)
    df['ddS'] = df.dS - df.dS_m7
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    k = df[df.model.eq('m8_full_dogk')]
    tests = pd.concat([test(k, 'kappa'), test(k, 'ddS'), test(k, 'dS'), test(df[df.model.eq('m7_full')], 'dS')
                       .assign(variable='dS_m7')], ignore_index=True)
    tests.to_csv(out.with_name(out.stem + '_tests.tsv'), sep='\t', index=False)
    pd.set_option('display.width', 200)
    print(k.groupby('roi')[['kappa', 'dS_m7', 'dS', 'dS_cen', 'dS_sur', 'dsse_rel', 'corr_kappa_g_HP',
                            'corr_kappa_g_LP', 'corr_kappa_dS']].median().round(3).to_string())
    print(tests.round(4).to_string())
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
    tests = []
    for model in MODELS:
        d = df[df.model.eq(model)].groupby(['subject', 'roi'], as_index=False).dcv_r2.mean()
        if len(d):
            tests.append(test(d, 'dcv_r2').assign(model=model))
    if tests:
        tests = pd.concat(tests, ignore_index=True)
        tests.to_csv(out.with_name(out.stem + '_tests.tsv'), sep='\t', index=False)
        print(tests.to_string())
    print(df.groupby('model')[['dcv_r2', 'dtrain_r2']].mean().to_string())
    print(f'{len(df)} rows -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=None)
    ap.add_argument('--base-root', default=None, help='Ladder root holding m7_full.')
    ap.add_argument('--cv-root', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cv_root:
        collect_cv(a.cv_root, a.out)
    else:
        main(a.root, a.base_root, a.out)
