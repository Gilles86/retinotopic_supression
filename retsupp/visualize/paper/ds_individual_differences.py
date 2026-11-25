"""dS (joint zero-sum fit) vs the behavioural HP benefit between participants, with the reliability of both.

dS per participant x area: ``gain_meta_zs`` estimates (joint zero-sum fit, jackknife SEs). Per participant summaries:
(i) median over the nine areas (as the earlier analysis); (ii) the participant's posterior deviation in the Student-t
hierarchical model (trialwise_meta_bayes.fit: area means + participant deviation), which weights each area by its
precision and shrinks noisy participants; (iii) the same model with the participant deviation regressed on z(benefit)
(slope in % of drive per SD of benefit). Benefit: LP − HP distractor RT (correct trials; rt_links_between_subjects).
Reliability: split halves over runs (odd vs even) for dS (median over areas of the half estimates) and for the benefit
(odd vs even runs), Spearman-Brown corrected; the ceiling of an observable correlation is sqrt(rel_dS * rel_benefit).

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.ds_individual_differences
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.richter_followup import N_BOOT, RNG_SEED, load_trials


def spearman(a, b):
    j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    rho, p = stats.spearmanr(j.a, j.b)
    rng = np.random.default_rng(RNG_SEED)
    bs = [stats.spearmanr(*j.iloc[rng.integers(0, len(j), len(j))].T.values)[0] for _ in range(N_BOOT)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=len(j), rho=rho, ci_lo=lo, ci_hi=hi, p=p)


def sb(r):
    return 2 * r / (1 + r)


def main(a):
    from retsupp.visualize.paper import trialwise_meta_bayes as tmb
    from retsupp.visualize.paper.gain_meta_zs import PW, PW_OF
    import pymc as pm
    import arviz as az
    tmb.GROUPS, tmb.AREAS, tmb.GROUP_OF = PW, [x for aa in PW.values() for x in aa], PW_OF
    e = pd.read_csv(a.estimates, sep='\t')
    e = e[e.term == 'dS']
    t = load_trials(a.trials)
    ok = t[t.ok]
    d = ok.groupby(['subject', 'dist_cond']).rt_ms.mean().unstack()
    benefit = (d.LP - d.HP)
    benefit = benefit[benefit.index >= 3]
    odd = ok.run % 2 == 1
    bo = ok[odd].groupby(['subject', 'dist_cond']).rt_ms.mean().unstack()
    be = ok[~odd].groupby(['subject', 'dist_cond']).rt_ms.mean().unstack()
    b_odd, b_even = (bo.LP - bo.HP).reindex(benefit.index), (be.LP - be.HP).reindex(benefit.index)

    med = e.groupby('subject').y.median()
    idata, areas, groups = tmb.fit(e[['subject', 'roi', 'y', 'se']].reset_index(drop=True), likelihood='t')
    w = idata.posterior['w'].mean(('chain', 'draw')).to_series()
    w.index = w.index.astype(int)

    rows = [dict(summary='median over areas', **spearman(med, benefit)),
            dict(summary='hierarchical participant deviation', **spearman(w, benefit))]

    # in-model regression of the participant deviation on z(benefit)
    subs = sorted(e.subject.unique())
    z = ((benefit - benefit.mean()) / benefit.std()).reindex(subs).values
    ai = e.roi.map({x: i for i, x in enumerate(tmb.AREAS)}).values
    si = e.subject.map({s: i for i, s in enumerate(subs)}).values
    gi = np.array([list(PW).index(PW_OF[x]) for x in tmb.AREAS])
    y, se = e.y.values, e.se.values
    sy = float(np.std(y))
    with pm.Model():
        gm = pm.Normal('gm', 0, 2 * sy, shape=3)
        v = pm.HalfNormal('sd_a', sy) * pm.Normal('vz', 0, 1, shape=len(tmb.AREAS))
        beta = pm.Normal('beta', 0, sy)
        w2 = beta * z + pm.HalfNormal('sd_s', sy) * pm.Normal('wz', 0, 1, shape=len(subs))
        tau = pm.HalfNormal('tau', sy)
        pm.StudentT('y', nu=pm.Gamma('nu', 2, .1), mu=gm[gi][ai] + v[ai] + w2[si], sigma=pm.math.sqrt(se ** 2 + tau ** 2),
                    observed=y)
        idb = pm.sample(2000, tune=2000, chains=4, target_accept=.99, random_seed=0, progressbar=False,
                        nuts_sampler='nutpie')
    bx = idb.posterior['beta'].values.ravel()
    lo, hi = az.hdi(bx, hdi_prob=.95)
    reg = dict(beta_per_sd_benefit=bx.mean(), hdi_lo=lo, hdi_hi=hi, P=max((bx > 0).mean(), (bx < 0).mean()),
               divergences=int(idb.sample_stats['diverging'].sum()))

    # reliability
    r_ds = stats.spearmanr(e.groupby('subject').y_odd.median(), e.groupby('subject').y_even.median()).statistic
    r_b = stats.spearmanr(b_odd, b_even, nan_policy='omit').statistic
    rel = dict(ds_split_half=r_ds, ds_spearman_brown=sb(r_ds), benefit_split_half=r_b, benefit_spearman_brown=sb(r_b),
               ceiling=np.sqrt(max(sb(r_ds), 0) * max(sb(r_b), 0)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'spearman.tsv', sep='\t', index=False)
    pd.DataFrame([reg]).to_csv(out / 'in_model_regression.tsv', sep='\t', index=False)
    pd.DataFrame([rel]).to_csv(out / 'reliability.tsv', sep='\t', index=False)
    print(pd.DataFrame(rows).round(3).to_string(), reg, rel, f'benefit median {benefit.median():.1f} ms', sep='\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--estimates', default='notes/data/gain_meta_zs_dog_2026-11-25/estimates.tsv')
    ap.add_argument('--trials', default='notes/data/behaviour_trials_2026-11-11.tsv.gz')
    ap.add_argument('--out', default='notes/data/ds_individual_differences_zs_2026-11-25')
    main(ap.parse_args())
