"""Gain–RT slopes of the trial-wise gains with more power, as fixed in ``notes/rt_power_plan_2026-11-13.md``.

Input: collected parametric tables of ``fit_trialwise_af_gains`` (per subject × area × gain: the slope on the RT
modulator, design with the global gain, and its leave-one-run-out jackknife SE ``se_jk``), one with z(log RT) and one
with the residualised RT (``--rt-modulator residual``). Distractor slope: mean of the HP- and LP-trial slopes, SE of
that mean (as ``trialwise_meta_bayes``).

Hierarchical models over subjects and areas (θ[s, a, k] = group mean[group(a), k] + area[a, k] + subject[s, k]; overall
effect = mean of the nine area means), for the terms k ∈ (target, distractor, global):

* ``separate`` — one model per term, independent HalfNormal area and subject SDs (the current Fig. 7 model);
* ``joint`` — the three terms in one model, area and subject effects multivariate normal with LKJ(2) correlations.

Likelihood:

* ``weighted`` — y ~ Normal(θ, √(SE² + τ²)) (the current model);
* ``me`` — measurement error: latent slope λ ~ Normal(θ, τ), y ~ Normal(λ, SE), λ sampled explicitly.

Priors as in ``trialwise_meta_bayes`` (per term, sy = SD of y: group means Normal(0, 2·sy), SDs and τ HalfNormal(sy)).
nutpie, 4 × 2000 draws after 2000 tuning, target_accept .99.

Robustness 4 (``--bins``): gains per residual-RT quintile (scheme rq5 of the bins table), centred within subject × area,
hierarchical monotonic-trend model per term: y[s, a, q] ~ Normal(β[s, a]·(m_q − mean m), σ_area), m_q = Σ_{j<q} ζ_j
(ζ ~ Dirichlet(1, 1, 1, 1), shape shared), β[s, a] = group mean + area + subject; β = change from the fastest to the
slowest quintile.

    python -m retsupp.visualize.paper.rt_power_meta --zlogrt <parametric.tsv> --resid <parametric.tsv> \
        --resid-bins <bins.tsv> --out notes/data/rt_power_2026-11-13
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as ptt

GROUPS = {'early': ['V1', 'V2', 'V3'], 'dorsal': ['V3AB', 'IPS'], 'occipitotemporal': ['hV4', 'LO', 'TO', 'VO']}
AREAS = [a for g in GROUPS.values() for a in g]
GROUP_OF = {a: g for g, aa in GROUPS.items() for a in aa}
TERMS = {'target': ['T'], 'distractor': ['D_HP', 'D_LP'], 'global': ['G']}
SEED = 0
DRAWS = dict(draws=2000, tune=2000, chains=4, target_accept=0.99, nuts_sampler='nutpie', progressbar=False)


def slope_table(path, min_subject=3):
    """Wide table: one row per subject × area with y_<term> and se_<term> for the three RT slopes."""
    p = pd.read_csv(path, sep='\t')
    p = p[(p.with_global.astype(str).str.lower() == 'true') & (p.modulator == 'rt') & p.roi.isin(AREAS)
          & (p.subject >= min_subject)]
    out = []
    for term, kinds in TERMS.items():
        d = p[p.kind.isin(kinds)]
        t = d.groupby(['subject', 'roi']).agg(y=('g', 'mean'), se=('se_jk', lambda s: np.sqrt((s ** 2).sum()) / len(s)),
                                              n=('g', 'size'))
        t = t[(t.n == len(kinds)) & np.isfinite(t.se) & (t.se > 0)].drop(columns='n')
        out.append(t.rename(columns={'y': f'y_{term}', 'se': f'se_{term}'}))
    return pd.concat(out, axis=1)


def fit(t, terms, joint, me, seed=None):
    seed = SEED if seed is None else seed
    """t: wide table (index subject, roi). Returns idata, areas, groups."""
    t = t.dropna(subset=[f'y_{k}' for k in terms]).reset_index()
    areas = [a for a in AREAS if a in set(t.roi)]
    groups = [g for g in GROUPS if any(GROUP_OF[a] == g for a in areas)]
    subs = sorted(t.subject.unique())
    ai = t.roi.map({a: i for i, a in enumerate(areas)}).values
    si = t.subject.map({s: i for i, s in enumerate(subs)}).values
    gi_area = np.array([groups.index(GROUP_OF[a]) for a in areas])
    Y = t[[f'y_{k}' for k in terms]].to_numpy()
    SE = t[[f'se_{k}' for k in terms]].to_numpy()
    sy = Y.std(0)
    K = len(terms)
    coords = dict(area=areas, group=groups, subject=subs, obs=np.arange(len(t)), term=terms)
    with pm.Model(coords=coords):
        gm = pm.Normal('group_mean', 0, 2 * sy, dims=('group', 'term'))
        if joint:
            ca, _, _ = pm.LKJCholeskyCov('chol_area', n=K, eta=2.0, sd_dist=pm.HalfNormal.dist(sigma=sy, shape=K),
                                         compute_corr=True, store_in_trace=True)
            cs, _, _ = pm.LKJCholeskyCov('chol_subject', n=K, eta=2.0, sd_dist=pm.HalfNormal.dist(sigma=sy, shape=K),
                                         compute_corr=True, store_in_trace=True)
            v = pm.Deterministic('v', ptt.dot(pm.Normal('v_z', 0, 1, dims=('area', 'term')), ca.T), dims=('area', 'term'))
            w = pm.Deterministic('w', ptt.dot(pm.Normal('w_z', 0, 1, dims=('subject', 'term')), cs.T),
                                 dims=('subject', 'term'))
        else:
            sd_a = pm.HalfNormal('sd_area', sy, dims='term')
            sd_s = pm.HalfNormal('sd_subject', sy, dims='term')
            v = pm.Deterministic('v', sd_a * pm.Normal('v_z', 0, 1, dims=('area', 'term')), dims=('area', 'term'))
            w = pm.Deterministic('w', sd_s * pm.Normal('w_z', 0, 1, dims=('subject', 'term')), dims=('subject', 'term'))
        tau = pm.HalfNormal('tau', sy, dims='term')
        am = pm.Deterministic('area_mean', gm[gi_area] + v, dims=('area', 'term'))
        pm.Deterministic('mu', am.mean(0), dims='term')
        theta = am[ai] + w[si]
        if me:
            lam = pm.Deterministic('latent', theta + tau * pm.Normal('lat_z', 0, 1, dims=('obs', 'term')),
                                   dims=('obs', 'term'))
            pm.Normal('y', mu=lam, sigma=SE, observed=Y, dims=('obs', 'term'))
        else:
            pm.Normal('y', mu=theta, sigma=ptt.sqrt(SE ** 2 + tau ** 2), observed=Y, dims=('obs', 'term'))
        idata = pm.sample(random_seed=seed, **DRAWS)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=seed, progressbar=False)
    return idata, t, areas, groups


def summarize(idata, t, areas, groups, terms, label):
    post = idata.posterior
    rows, diags = [], []
    for k in terms:
        def add(level, unit, x):
            x = np.asarray(x).ravel()
            lo, hi = az.hdi(x, hdi_prob=.95)
            rows.append(dict(analysis=label, term=k, level=level, unit=unit, mean=x.mean(), hdi_lo=lo, hdi_hi=hi,
                             p_gt0=(x > 0).mean(), p_lt0=(x < 0).mean(), credible=bool(lo > 0 or hi < 0)))
        add('overall', 'all', post['mu'].sel(term=k))
        for g in groups:
            add('group', g, post['group_mean'].sel(group=g, term=k))
        for a in areas:
            add('area', a, post['area_mean'].sel(area=a, term=k))
        yrep = idata.posterior_predictive['y'].sel(term=k).stack(s=('chain', 'draw')).values
        y = t[f'y_{k}'].to_numpy()
        lo, hi = np.percentile(yrep, [2.5, 97.5], axis=1)
        inside = []
        for a in areas:
            sel = (t.roi == a).values
            l, h = np.percentile(yrep[sel].mean(0), [2.5, 97.5])
            inside.append(l <= y[sel].mean() <= h)
        diags.append(dict(analysis=label, term=k, ppc_obs_coverage95=float(((y >= lo) & (y <= hi)).mean()),
                          ppc_area_means_inside=int(np.sum(inside)), n_areas=len(areas)))
    names = [n for n in ['mu', 'group_mean', 'area_mean', 'tau', 'sd_area', 'sd_subject', 'chol_area_stds',
                         'chol_subject_stds', 'chol_area_corr', 'chol_subject_corr'] if n in post]
    s = az.summary(idata, var_names=names)
    s = s[s.index.map(lambda i: not (('corr' in i) and (i.split('[')[1].split(',')[0].strip() ==
                                                         i.split(',')[1].rstrip(']').strip())))]
    conv = dict(analysis=label, n_obs=len(t), n_subjects=t.subject.nunique(),
                divergences=int(idata.sample_stats['diverging'].sum()), rhat_max=float(s.r_hat.max()),
                ess_bulk_min=float(s.ess_bulk.min()), ess_tail_min=float(s.ess_tail.min()))
    corr = []
    for nm in ('chol_area_corr', 'chol_subject_corr'):
        if nm in post:
            c = post[nm].values.reshape(-1, len(terms), len(terms))
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    x = c[:, i, j]
                    lo, hi = az.hdi(x, hdi_prob=.95)
                    corr.append(dict(analysis=label, level=nm.split('_')[1], pair=f'{terms[i]}–{terms[j]}',
                                     mean=x.mean(), hdi_lo=lo, hdi_hi=hi))
    return rows, [{**d, **conv} for d in diags], corr


def fit_bins(path, term, seed=0, min_subject=3):
    b = pd.read_csv(path, sep='\t')
    b = b[(b.scheme == 'rq5') & (b.with_global.astype(str).str.lower() == 'true') & b.kind.isin(TERMS[term])
          & b.roi.isin(AREAS) & (b.subject >= min_subject) & b['bin'].str.startswith('q')]
    d = b.groupby(['subject', 'roi', 'bin']).g.mean().unstack('bin').dropna()
    d = d.sub(d.mean(1), axis=0)
    t = d.stack().rename('y').reset_index()
    t['q'] = t['bin'].str[1].astype(int) - 1
    areas = [a for a in AREAS if a in set(t.roi)]
    groups = [g for g in GROUPS if any(GROUP_OF[a] == g for a in areas)]
    subs = sorted(t.subject.unique())
    ai = t.roi.map({a: i for i, a in enumerate(areas)}).values
    si = t.subject.map({s: i for i, s in enumerate(subs)}).values
    gi_area = np.array([groups.index(GROUP_OF[a]) for a in areas])
    sy = float(t.y.std())
    with pm.Model(coords=dict(area=areas, group=groups, subject=subs, step=np.arange(4))):
        zeta = pm.Dirichlet('zeta', np.ones(4), dims='step')
        m = ptt.concatenate([ptt.zeros(1), ptt.cumsum(zeta)])
        mc = m - m.mean()
        gm = pm.Normal('group_mean', 0, 2 * sy, dims='group')
        v = pm.Deterministic('v', pm.HalfNormal('sd_area', sy) * pm.Normal('v_z', 0, 1, dims='area'), dims='area')
        w = pm.Deterministic('w', pm.HalfNormal('sd_subject', sy) * pm.Normal('w_z', 0, 1, dims='subject'), dims='subject')
        am = pm.Deterministic('area_mean', gm[gi_area] + v, dims='area')
        pm.Deterministic('mu', am.mean())
        sig = pm.HalfNormal('sigma', sy, dims='area')
        pm.Normal('y', mu=(am[ai] + w[si]) * mc[t.q.values], sigma=sig[ai], observed=t.y.values)
        idata = pm.sample(random_seed=seed, **DRAWS)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=seed, progressbar=False)
    x = idata.posterior['mu'].values.ravel()
    lo, hi = az.hdi(x, hdi_prob=.95)
    s = az.summary(idata, var_names=['mu', 'group_mean', 'area_mean', 'sd_area', 'sd_subject', 'sigma', 'zeta'])
    yrep = idata.posterior_predictive['y'].stack(s=('chain', 'draw')).values
    plo, phi = np.percentile(yrep, [2.5, 97.5], axis=1)
    zeta = idata.posterior['zeta'].mean(('chain', 'draw')).values
    return dict(analysis='4: residual-RT quintiles, monotonic trend', term=term, level='overall', unit='all',
                mean=x.mean(), hdi_lo=lo, hdi_hi=hi, p_gt0=(x > 0).mean(), p_lt0=(x < 0).mean(),
                credible=bool(lo > 0 or hi < 0), n_obs=len(t), n_subjects=len(subs),
                divergences=int(idata.sample_stats['diverging'].sum()), rhat_max=float(s.r_hat.max()),
                ess_bulk_min=float(s.ess_bulk.min()),
                ppc_obs_coverage95=float(((t.y.values >= plo) & (t.y.values <= phi)).mean()),
                zeta=' '.join(f'{z:.2f}' for z in zeta))


ANALYSES = {   # label: (data, joint, me)
    'baseline (refit): z(log RT), separate, SE-weighted': ('zlogrt', False, False),
    'step 1 only: residual RT': ('resid', False, False),
    'step 2 only: joint model (LKJ)': ('zlogrt', True, False),
    'step 3 only: measurement error': ('zlogrt', False, True),
    'main (1 + 2 + 3)': ('resid', True, True),
    '5: main on r > 0.2 (own voxels)': ('resid_r02', True, True),
}


def main(sources, out, bins=None, only=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tabs = {k: slope_table(v) for k, v in sources.items() if v}
    terms = list(TERMS)
    rows, diags, corrs = [], [], []
    for label, (src, joint, me) in ANALYSES.items():
        if src not in tabs or (only and label not in only):
            continue
        if joint:
            fits = [(terms, fit(tabs[src], terms, joint, me))]
        else:
            fits = [([k], fit(tabs[src][[f'y_{k}', f'se_{k}']], [k], joint, me)) for k in terms]
        for ks, (idata, t, areas, groups) in fits:
            r, d, c = summarize(idata, t, areas, groups, ks, label)
            rows += r
            diags += d
            corrs += c
            print(label, ks, d, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'summary.tsv', sep='\t', index=False)
    pd.DataFrame(diags).to_csv(out / 'diagnostics.tsv', sep='\t', index=False)
    pd.DataFrame(corrs).to_csv(out / 'correlations.tsv', sep='\t', index=False)
    if bins:
        b = pd.DataFrame([fit_bins(bins, k) for k in terms])
        b.to_csv(out / 'quintile_trend.tsv', sep='\t', index=False)
        print(b.round(4).to_string(index=False))
    print(res[res.level == 'overall'].round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zlogrt', required=True)
    ap.add_argument('--resid', default=None)
    ap.add_argument('--resid-r02', default=None)
    ap.add_argument('--resid-bins', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', nargs='+', default=None, help='analysis labels to run (default: all with data)')
    ap.add_argument('--draws', type=int, default=2000, help='draws and tuning steps per chain')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    DRAWS.update(draws=a.draws, tune=a.draws)
    SEED = a.seed
    main(dict(zlogrt=a.zlogrt, resid=a.resid, resid_r02=a.resid_r02), a.out, a.resid_bins, a.only)
