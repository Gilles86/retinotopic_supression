"""Hierarchical Bayesian meta-analysis of the BOLD-level gain modulators (parametric trial-wise model).

Input: the collected parametric table of ``fit_trialwise_af_gains`` (``*_parametric.tsv``): per subject × area × kind ×
modulator the estimate ``g`` and its leave-one-run-out jackknife SE ``se_jk``.

Per question (a modulator of one gain), over the nine areas::

    y[s, a] ~ Normal(θ[s, a], sqrt(se[s, a]² + τ²))
    θ[s, a] = m[group(a)] + v[a] + w[s]

m: group means (fixed effects, Normal(0, 2·sd(y)); early V1–V3, dorsal V3AB IPS, occipitotemporal hV4 LO TO VO);
v: area within group; w: subject; τ: residual subject × area heterogeneity beyond the sampling error. The overall
effect μ is the mean of the nine area means. Non-centred, HalfNormal SDs scaled to the spread of y. Variants for sensitivity: Student-t likelihood (``--likelihood t``)
and no SE weighting (``--no-se``).

Questions: target gain × target repeat; distractor gain (mean of HP and LP trials) × distractor repeat;
target / distractor / global gain × z(log RT) (design with the global gain).

Relative variants (``rel*``) divide each area's estimates and SEs by the area's typical base gain: the median over
subjects of |g| of that subject × area's m7 fit (target: g_T_dyn; distractor: mean of g_HP_dyn and g_LP_dyn). The
median of the absolute value is used because the signed median is close to 0 in some areas (target V1; distractor V1,
V3, LO, IPS), where a ratio to it is undefined. Relative effects are comparable across fits whose absolute gains differ
in scale. The global gain has no base gain (prior 0), so it has no relative variant.

    python -m retsupp.visualize.paper.trialwise_meta_bayes --param notes/data/<...>_parametric.tsv \\
        --check notes/data/<...>_check.tsv --out notes/data/trialwise_meta_bayes_2026-11
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

GROUPS = {'early': ['V1', 'V2', 'V3'], 'dorsal': ['V3AB', 'IPS'], 'occipitotemporal': ['hV4', 'LO', 'TO', 'VO']}
AREAS = [a for g in GROUPS.values() for a in g]
GROUP_OF = {a: g for g, aa in GROUPS.items() for a in aa}
QUESTIONS = {   # name: (with_global, kinds averaged, modulator)
    'T_target_repeat': (False, ['T'], 'target_repeat'),
    'D_dist_repeat': (False, ['D_HP', 'D_LP'], 'dist_repeat'),
    'T_rt': (True, ['T'], 'rt'),
    'D_rt': (True, ['D_HP', 'D_LP'], 'rt'),
    'G_rt': (True, ['G'], 'rt'),
}


def question_table(p, with_global, kinds, modulator, units):
    d = p[(p.with_global.astype(str).str.lower() == str(with_global).lower()) & p.kind.isin(kinds)
          & (p.modulator == modulator) & p.roi.isin(units)]
    # Mean over the kinds (HP and LP distractor trials); SE of a mean of two estimates from different trials.
    t = d.groupby(['subject', 'roi']).agg(y=('g', 'mean'), se=('se_jk', lambda s: np.sqrt((s ** 2).sum()) / len(s)),
                                          n=('g', 'size')).reset_index()
    t = t[t.n == len(kinds)].dropna()
    return t[np.isfinite(t.se) & (t.se > 0)].reset_index(drop=True)


def fit(t, likelihood='normal', use_se=True, seed=0, draws=2000, tune=2000):
    areas = [a for a in AREAS if a in set(t.roi)]
    groups = [g for g in GROUPS if any(GROUP_OF[a] == g for a in areas)]
    subs = sorted(t.subject.unique())
    ai = t.roi.map({a: i for i, a in enumerate(areas)}).values
    gi_area = np.array([groups.index(GROUP_OF[a]) for a in areas])
    si = t.subject.map({s: i for i, s in enumerate(subs)}).values
    y, se = t.y.values, (t.se.values if use_se else np.zeros(len(t)))
    sy = float(np.std(y))
    coords = dict(area=areas, group=groups, subject=subs, obs=np.arange(len(t)))
    with pm.Model(coords=coords) as m:
        # Group means as fixed effects (three groups cannot inform a between-group SD); areas vary around their
        # group's mean, subjects around 0. The overall effect is the mean of the area means (each area equally).
        gm = pm.Normal('group_mean', 0, 2 * sy, dims='group')
        sd_a = pm.HalfNormal('sd_area', sy)
        sd_s = pm.HalfNormal('sd_subject', sy)
        tau = pm.HalfNormal('tau', sy)
        v = pm.Deterministic('v', sd_a * pm.Normal('v_z', 0, 1, dims='area'), dims='area')
        w = pm.Deterministic('w', sd_s * pm.Normal('w_z', 0, 1, dims='subject'), dims='subject')
        am = pm.Deterministic('area_mean', gm[gi_area] + v, dims='area')
        pm.Deterministic('mu', am.mean())
        theta = am[ai] + w[si]
        sigma = pm.math.sqrt(se ** 2 + tau ** 2)
        if likelihood == 't':
            nu = pm.Gamma('nu', 2, 0.1)
            pm.StudentT('y', nu=nu, mu=theta, sigma=sigma, observed=y, dims='obs')
        else:
            pm.Normal('y', mu=theta, sigma=sigma, observed=y, dims='obs')
        idata = pm.sample(draws=draws, tune=tune, chains=4, target_accept=0.99, random_seed=seed, progressbar=False,
                          nuts_sampler='nutpie')
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=seed, progressbar=False)
    return idata, areas, groups


def summarize(idata, t, areas, groups, question, variant):
    post = idata.posterior
    rows = []

    def add(level, unit, x):
        x = np.asarray(x).ravel()
        lo, hi = az.hdi(x, hdi_prob=.95)
        rows.append(dict(question=question, variant=variant, level=level, unit=unit, mean=x.mean(), hdi_lo=lo,
                         hdi_hi=hi, p_gt0=(x > 0).mean(), p_lt0=(x < 0).mean()))
    add('overall', 'all', post['mu'])
    for g in groups:
        add('group', g, post['group_mean'].sel(group=g))
    for a in areas:
        add('area', a, post['area_mean'].sel(area=a))
    s = az.summary(idata, var_names=['mu', 'sd_area', 'sd_subject', 'tau', 'group_mean', 'area_mean'])
    div = int(idata.sample_stats['diverging'].sum())
    # Posterior predictive check: per-observation 95% coverage, and the per-area mean of y against its predictive band.
    yrep = idata.posterior_predictive['y'].stack(s=('chain', 'draw')).values      # (obs, S)
    lo, hi = np.percentile(yrep, [2.5, 97.5], axis=1)
    cover = float(((t.y.values >= lo) & (t.y.values <= hi)).mean())
    area_cover = []
    for a in areas:
        sel = (t.roi == a).values
        rm = yrep[sel].mean(0)
        l, h = np.percentile(rm, [2.5, 97.5])
        area_cover.append(l <= t.y.values[sel].mean() <= h)
    diag = dict(question=question, variant=variant, n_obs=len(t), n_subjects=t.subject.nunique(), divergences=div,
                rhat_max=float(s.r_hat.max()), ess_bulk_min=float(s.ess_bulk.min()), ess_tail_min=float(s.ess_tail.min()),
                ppc_obs_coverage95=cover, ppc_area_means_inside=int(np.sum(area_cover)), n_areas=len(areas))
    return pd.DataFrame(rows), diag


def base_scale(check):
    """Per area and gain: median over subjects of |base gain| of the m7 fit."""
    c = pd.read_csv(check, sep='\t')
    c = c[c.roi.isin(AREAS)].assign(D=lambda d: (d.g_HP_dyn + d.g_LP_dyn) / 2)
    return {'T': c.groupby('roi').g_T_dyn.apply(lambda x: x.abs().median()),
            'D': c.groupby('roi').D.apply(lambda x: x.abs().median())}


def main(param, check, out, variants):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(param, sep='\t')
    scale = base_scale(check)
    pd.DataFrame(scale).rename_axis('roi').to_csv(out / 'base_scale.tsv', sep='\t')
    summ, diags, data = [], [], []
    for q, (wg, kinds, mod) in QUESTIONS.items():
        t0 = question_table(p, wg, kinds, mod, AREAS)
        gain = kinds[0][0]
        for variant in variants:
            if variant.startswith('rel') and gain not in scale:
                continue
            t = t0.copy()
            if variant.startswith('rel'):
                t['y'] /= t.roi.map(scale[gain])
                t['se'] /= t.roi.map(scale[gain])
            lik = 't' if variant.endswith('_t') else 'normal'
            idata, areas, groups = fit(t, likelihood=lik, use_se=not variant.endswith('_nose'))
            s, d = summarize(idata, t, areas, groups, q, variant)
            print(q, variant, d, flush=True)
            print(s.round(4).to_string(index=False), flush=True)
            summ.append(s)
            diags.append(d)
            if variant in ('abs', 'rel'):
                idata.to_netcdf(out / f'{q}_{variant}_idata.nc')
                data.append(t.assign(question=q, variant=variant))
    pd.concat(summ).to_csv(out / 'summary.tsv', sep='\t', index=False)
    pd.DataFrame(diags).to_csv(out / 'diagnostics.tsv', sep='\t', index=False)
    pd.concat(data).to_csv(out / 'data.tsv', sep='\t', index=False)
    print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--param', required=True)
    ap.add_argument('--check', required=True, help='Collected check table (base gains per subject × area)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--variants', nargs='+', default=['abs', 'abs_t', 'abs_nose', 'rel', 'rel_t', 'rel_nose'])
    a = ap.parse_args()
    main(a.param, a.check, a.out, a.variants)
