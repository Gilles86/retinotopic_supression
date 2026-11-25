"""Group statistics of the joint zero-sum gain-field terms (update-3 checks, 2026-11-25).

Estimates: the joint zero-sum fit itself (main model m5_TDS_dS_zs: zero-sum sustained field, tied distractor gain).
Per participant x area, dS = g_HP - g_LP (= 4/3 g_HP), the target gain and the distractor gain, in %, taken from the
fit's own gains (``fitted_gains`` of ``fit_width_profiles``). Their uncertainty: the per-run sufficient statistics
(A_r, b_r) of the gains with everything else fixed at the joint fit, reparametrised to the model's three free gains
(g_HP = 3/4 d, g_LP = -1/4 d, g_D_HP = g_D_LP = D), and a leave-one-run-out jackknife of that least-squares solution.
The full-data solution reproduces the fitted gains (median |difference| ~1e-5) except where the fit sits on a bound;
those cells are flagged (``at_bound``) and keep the fitted value with the jackknife SE of the unbounded solution.
SEs are conditional on the pRFs, HRF, widths and item weight of the joint fit.

Analyses (``--analyses``):
  wilcoxon   median over participants, bootstrap 95% CI, Wilcoxon p, BH q (nine areas; three pathways)
  bayes      hierarchical model of trialwise_meta_bayes (Student-t; pathway = mean of its areas)
  calib      odd-even calibration per area and pathway, median SE, voxels
  inflate    the model with each pathway's SEs scaled so that its calibration is 0.67
  nose       the model without the SEs (each estimate weighted equally)
  hierarchy  area-level slope on log2 mean pRF size inside the hierarchical model; split-half within-participant
             correlation of target gain and dS

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.gain_meta_zs \\
        --root /data/ds-retsupp/derivatives/field_widths_zs_lm_2026-11-14 --family dog
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import fdr, median_ci
from retsupp.visualize.paper.paper_style import ROI_GROUPS, ROI_ORDER

PW = {'early': ['V1', 'V2', 'V3'], 'dorsal': ['V3AB', 'IPS'], 'occipitotemporal': ['hV4', 'VO', 'LO', 'TO']}
PW_OF = {a: g for g, aa in PW.items() for a in aa}
NAMES = ['g_HP', 'g_LP', 'g_T', 'g_D_HP', 'g_D_LP']
M = np.array([[.75, 0, 0], [-.25, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])   # gains = M @ (dS, T, D)
TERMS = ['dS', 'target', 'distractor']
SIZE_TABLE = 'notes/data/hierarchy_gain_terms_zs_lm_2026-11-13_n28_prf_size_axis.tsv'


def jk(A, b):
    R = len(A)
    full = np.linalg.solve(A.sum(0), b.sum(0))
    loo = np.stack([np.linalg.solve(A.sum(0) - A[r], b.sum(0) - b[r]) for r in range(R)])
    se = np.sqrt((R - 1) / R * ((loo - loo.mean(0)) ** 2).sum(0))
    return full, loo, se


def collect(root, exclude=(1, 2)):
    rows = []
    for f in sorted(glob.glob(f'{root}/sub-*/*_widthprof.npz')):
        z = np.load(f, allow_pickle=True)
        s, roi = int(z['subject']), str(z['roi'])
        if s in exclude or roi not in ROI_ORDER:
            continue
        assert list(z['gain_names']) == NAMES, z['gain_names']
        A = np.einsum('ij,rjk,kl->ril', M.T, z['A'], M)
        b = np.einsum('ij,rj->ri', M.T, z['b'])
        full, loo, se = jk(A, b)
        odd = z['runs'][:, 1] % 2 == 1
        go = np.linalg.solve(A[odd].sum(0), b[odd].sum(0))
        ge = np.linalg.solve(A[~odd].sum(0), b[~odd].sum(0))
        fg = np.asarray(z['fitted_gains'], float)
        fit = np.array([fg[0] - fg[1], fg[2], fg[3]])
        for k, term in enumerate(TERMS):
            rows.append(dict(subject=s, roi=roi, term=term, y=100 * fit[k], y_lin=100 * full[k], se=100 * se[k],
                             halfdiff=100 * abs(go[k] - ge[k]) / 2, y_odd=100 * go[k], y_even=100 * ge[k],
                             n_voxels=int(z['n_voxels']), n_runs=len(A),
                             at_bound=bool(abs(full[k] - fit[k]) > 1e-3)))
    d = pd.DataFrame(rows)
    d['pathway'] = d.roi.map(PW_OF)
    return d


# ---------------------------------------------------------------- (a) median / Wilcoxon / BH

def wilcoxon_table(t):
    rows = []
    def add(level, unit, v):
        m, lo, hi = median_ci(v.to_numpy())
        rows.append(dict(level=level, unit=unit, n=len(v), median=m, ci_lo=lo, ci_hi=hi, k_neg=int((v < 0).sum()),
                         p=stats.wilcoxon(v).pvalue))
    add('overall', 'all', t.groupby('subject').y.median())
    for g, aa in PW.items():
        add('group', g, t[t.roi.isin(aa)].groupby('subject').y.median())
    for a in ROI_ORDER:
        add('area', a, t[t.roi == a].y)
    r = pd.DataFrame(rows)
    for lev in ('group', 'area'):
        k = r.level == lev
        r.loc[k, 'q'] = fdr(r.loc[k, 'p'])
    return r


# ---------------------------------------------------------------- (b) hierarchical model

def bayes(t, likelihood='t', use_se=True, se_col='se', seed=0):
    from retsupp.visualize.paper import trialwise_meta_bayes as tmb
    tmb.GROUPS, tmb.AREAS, tmb.GROUP_OF = PW, [a for aa in PW.values() for a in aa], PW_OF
    tt = t[['subject', 'roi', 'y']].assign(se=t[se_col]).reset_index(drop=True)
    idata, areas, groups = tmb.fit(tt, likelihood=likelihood, use_se=use_se, seed=seed)
    s, diag = tmb.summarize(idata, tt, areas, groups, t.term.iloc[0], 'x')
    return s.drop(columns='variant'), diag


def slope_model(t, x_of, areas, likelihood='t', seed=0, draws=2000, tune=2000):
    """θ[s, a] = α + β·x[a] + v[a] + w[s], x = log2 mean pRF size (centred over the areas used); β in units of y per
    doubling. Likelihood as in the main model (Student-t, scale sqrt(SE² + τ²))."""
    import pymc as pm
    t = t[t.roi.isin(areas)].reset_index(drop=True)
    subs = sorted(t.subject.unique())
    x = np.array([x_of[a] for a in areas])
    xc = x - x.mean()
    ai = t.roi.map({a: i for i, a in enumerate(areas)}).values
    si = t.subject.map({s: i for i, s in enumerate(subs)}).values
    y, se = t.y.values, t.se.values
    sy = float(np.std(y))
    with pm.Model(coords=dict(area=areas, subject=subs)):
        al = pm.Normal('alpha', 0, 2 * sy)
        be = pm.Normal('beta', 0, 2 * sy / max(xc.std(), 1e-6))
        sd_a = pm.HalfNormal('sd_area', sy)
        sd_s = pm.HalfNormal('sd_subject', sy)
        tau = pm.HalfNormal('tau', sy)
        v = sd_a * pm.Normal('v_z', 0, 1, dims='area')
        w = sd_s * pm.Normal('w_z', 0, 1, dims='subject')
        theta = al + be * xc[ai] + v[ai] + w[si]
        sig = pm.math.sqrt(se ** 2 + tau ** 2)
        if likelihood == 't':
            pm.StudentT('y', nu=pm.Gamma('nu', 2, .1), mu=theta, sigma=sig, observed=y)
        else:
            pm.Normal('y', mu=theta, sigma=sig, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=4, target_accept=.99, random_seed=seed, progressbar=False,
                          nuts_sampler='nutpie')
    post = idata.posterior
    return post['beta'].values.ravel(), post['alpha'].values.ravel(), int(idata.sample_stats['diverging'].sum())


def hdi(x):
    import arviz as az
    lo, hi = az.hdi(np.asarray(x), hdi_prob=.95)
    return dict(mean=float(np.mean(x)), hdi_lo=float(lo), hdi_hi=float(hi), P=float(max((x > 0).mean(), (x < 0).mean())))


def hierarchy(d, size_table=SIZE_TABLE):
    size = pd.read_csv(size_table, sep='\t').set_index('roi').prf_size_deg
    x_of = np.log2(size).to_dict()
    rows, draws = [], {}
    sets = {'all nine': ROI_ORDER, 'early + dorsal': PW['early'] + PW['dorsal'],
            'early + occipitotemporal': PW['early'] + PW['occipitotemporal']}
    for term in TERMS:
        for name, areas in sets.items():
            if term != 'target' and name != 'all nine':
                continue
            b, a, div = slope_model(d[d.term == term], x_of, areas)
            draws[(term, name)] = (b, a)
            rows.append(dict(term=term, areas=name, quantity='slope per doubling', divergences=div, **hdi(b)))
            rows.append(dict(term=term, areas=name, quantity='slope / |mean| (relative)', divergences=div,
                             **hdi(b / np.abs(a))))
    bt, at = draws[('target', 'all nine')]
    bd, ad = draws[('dS', 'all nine')]
    rows.append(dict(term='target - dS', areas='all nine', quantity='difference of slopes (independent fits)',
                     **hdi(bt - bd)))
    rows.append(dict(term='target - |dS|', areas='all nine', quantity='difference of relative slopes, dS sign-flipped',
                     **hdi(bt / np.abs(at) - (-bd) / np.abs(ad))))
    return pd.DataFrame(rows)


def within_corr(d):
    """Per participant, Spearman over areas of target gain vs dS; same estimates and split halves (target from odd
    runs, dS from even runs and vice versa, so that their estimation errors are independent)."""
    out = []
    w = d.pivot_table(index=['subject', 'roi'], columns='term', values=['y', 'y_odd', 'y_even']).dropna()
    for s, g in w.groupby(level=0):
        if len(g) < 5:
            continue
        r_same = stats.spearmanr(g[('y', 'target')], g[('y', 'dS')]).statistic
        r_x1 = stats.spearmanr(g[('y_odd', 'target')], g[('y_even', 'dS')]).statistic
        r_x2 = stats.spearmanr(g[('y_even', 'target')], g[('y_odd', 'dS')]).statistic
        out.append(dict(subject=s, rho_same=r_same, rho_split=(r_x1 + r_x2) / 2))
    o = pd.DataFrame(out)
    res = []
    for c in ('rho_same', 'rho_split'):
        m, lo, hi = median_ci(o[c].dropna().to_numpy())
        res.append(dict(estimate=c, median=m, ci_lo=lo, ci_hi=hi, k_neg=int((o[c] < 0).sum()), n=int(o[c].notna().sum()),
                        p=stats.wilcoxon(o[c].dropna()).pvalue))
    return pd.DataFrame(res)


# ---------------------------------------------------------------- calibration

def calibration(d):
    rows = []
    for term in TERMS:
        t = d[d.term == term]
        units = [('all', ROI_ORDER)] + list(PW.items()) + [(a, [a]) for a in ROI_ORDER]
        for unit, aa in units:
            x = t[t.roi.isin(aa)]
            rows.append(dict(term=term, unit=unit, calibration=np.median(x.halfdiff / x.se), median_se=x.se.median(),
                             median_voxels=x.n_voxels.median(), n=len(x)))
    return pd.DataFrame(rows)


def main(a):
    out = Path(a.out or f'notes/data/gain_meta_zs_{a.family}_2026-11-25')
    out.mkdir(parents=True, exist_ok=True)
    d = collect(a.root)
    d.to_csv(out / 'estimates.tsv', sep='\t', index=False)
    print(f'{d.subject.nunique()} participants, {len(d) // 3} cells; at a bound:',
          d.groupby('term').at_bound.sum().to_dict(), flush=True)
    cal = calibration(d)
    cal.to_csv(out / 'calibration.tsv', sep='\t', index=False)
    if 'wilcoxon' in a.analyses:
        pd.concat([wilcoxon_table(d[d.term == t]).assign(term=t) for t in TERMS]).to_csv(
            out / 'wilcoxon.tsv', sep='\t', index=False)
    rows, diags = [], []
    for term in TERMS:
        t = d[d.term == term].copy()
        runs = []
        if 'bayes' in a.analyses:
            runs.append(('t', dict()))
            runs.append(('normal', dict(likelihood='normal')))
        if 'inflate' in a.analyses:
            c = cal[(cal.term == term)].set_index('unit').calibration
            t['se_infl'] = t.se * t.pathway.map(lambda g: c[g] / .67)
            runs.append(('t_inflated', dict(se_col='se_infl')))
        if 'nose' in a.analyses:
            runs.append(('t_nose', dict(use_se=False)))
        for name, kw in runs:
            s, dg = bayes(t, **kw)
            rows.append(s.assign(variant=name))
            diags.append({**dg, 'term': term, 'variant': name})
            print(term, name, dg, flush=True)
    if rows:
        pd.concat(rows).to_csv(out / 'bayes.tsv', sep='\t', index=False)
        pd.DataFrame(diags).to_csv(out / 'bayes_diagnostics.tsv', sep='\t', index=False)
    if 'hierarchy' in a.analyses:
        hierarchy(d).to_csv(out / 'hierarchy.tsv', sep='\t', index=False)
        within_corr(d).to_csv(out / 'within_corr.tsv', sep='\t', index=False)
    print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/data/ds-retsupp/derivatives/field_widths_zs_lm_2026-11-14')
    ap.add_argument('--family', default='dog')
    ap.add_argument('--out', default=None)
    ap.add_argument('--analyses', nargs='+', default=['wilcoxon', 'bayes', 'inflate', 'nose', 'hierarchy'])
    main(ap.parse_args())
