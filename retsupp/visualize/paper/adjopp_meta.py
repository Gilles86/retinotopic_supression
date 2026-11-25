"""The LP-split zero-sum model (m6_TDS_dS_adjopp_zs) through the hierarchical model (update 4, item 7; 2026-11-25).

Input: ``fit_gain_stats_adjopp`` per-run sufficient statistics (A_r, b_r) of the gains g_HP, g_LP, δ, g_T, g_D_HP,
g_D_LP on each participant x area's joint zero-sum LP-split fit. The model's free gains are h = (dS, δ, T, D), with
g_HP = ¾ dS, g_LP = −¼ dS (zero-sum) and one distractor gain; δ = g_next − g_opp. Per cell: the fit's own values and a
leave-one-run-out jackknife SE of each contrast (re-solving h by least squares under these constraints, everything else
held at the fit), with odd-even calibration. Contrasts (%): dS (HP − mean LP), δ (next − opposite), HP − next
(dS − δ/3), HP − opposite (dS + 2δ/3).

Statistics as the gain-field terms: Student-t hierarchical model (``gain_meta_zs.bayes``; pathway = mean of its areas),
each area on its own data (no pooling), and the median / Wilcoxon / BH table.

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.adjopp_meta \\
        --root /data/ds-retsupp/derivatives/adjopp_gain_stats_zs_2026-11-25
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.visualize.paper.paper_style import ROI_ORDER

NAMES = ['g_HP', 'g_LP', 'delta', 'g_T', 'g_D_HP', 'g_D_LP']
M = np.array([[.75, 0, 0, 0], [-.25, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
# contrasts of h = (dS, δ, T, D)
CONTRASTS = {'dS': [1, 0, 0, 0], 'next_minus_opp': [0, 1, 0, 0], 'hp_minus_next': [1, -1 / 3, 0, 0],
             'hp_minus_opp': [1, 2 / 3, 0, 0], 'target': [0, 0, 1, 0], 'distractor': [0, 0, 0, 1]}


def collect(root, exclude=(1, 2)):
    rows = []
    for f in sorted(glob.glob(f'{root}/sub-*/*_gainstats.npz')):
        z = np.load(f, allow_pickle=True)
        s, roi = int(z['subject']), str(z['roi'])
        if s in exclude or roi not in ROI_ORDER:
            continue
        assert list(z['gain_names']) == NAMES, z['gain_names']
        A = np.einsum('ij,rjk,kl->ril', M.T, z['A'], M)
        b = np.einsum('ij,rj->ri', M.T, z['b'])
        R = len(A)
        full = np.linalg.solve(A.sum(0), b.sum(0))
        loo = np.stack([np.linalg.solve(A.sum(0) - A[r], b.sum(0) - b[r]) for r in range(R)])
        odd = z['runs'][:, 1] % 2 == 1
        go = np.linalg.solve(A[odd].sum(0), b[odd].sum(0))
        ge = np.linalg.solve(A[~odd].sum(0), b[~odd].sum(0))
        fg = np.asarray(z['fitted_gains'], float)
        h_fit = np.array([fg[0] - fg[1], fg[2], fg[3], fg[4]])
        for name, c in CONTRASTS.items():
            c = np.asarray(c)
            v = loo @ c
            rows.append(dict(subject=s, roi=roi, term=name, y=100 * h_fit @ c, y_lin=100 * full @ c,
                             se=100 * np.sqrt((R - 1) / R * ((v - v.mean()) ** 2).sum()),
                             halfdiff=100 * abs((go - ge) @ c) / 2, n_voxels=int(z['n_voxels']),
                             at_bound=bool(abs((full - h_fit) @ c) > 1e-3)))
    return pd.DataFrame(rows)


def own_data(t, seed=0):
    """Each area on its own data: y ~ Student-t(ν, μ_a, √(SE² + τ²)), no pooling over areas or participants."""
    import arviz as az
    import pymc as pm
    rows = []
    for roi in ROI_ORDER:
        x = t[t.roi == roi]
        if len(x) < 5:
            continue
        y, se, sy = x.y.values, x.se.values, float(np.std(x.y.values))
        with pm.Model():
            mu = pm.Normal('mu', 0, 2 * sy)
            tau = pm.HalfNormal('tau', sy)
            pm.StudentT('y', nu=pm.Gamma('nu', 2, .1), mu=mu, sigma=pm.math.sqrt(se ** 2 + tau ** 2), observed=y)
            idata = pm.sample(2000, tune=2000, chains=4, target_accept=.99, random_seed=seed, progressbar=False,
                              nuts_sampler='nutpie')
        m = idata.posterior['mu'].values.ravel()
        lo, hi = az.hdi(m, hdi_prob=.95)
        rows.append(dict(unit=roi, mean=m.mean(), hdi_lo=lo, hdi_hi=hi, p_gt0=(m > 0).mean(), p_lt0=(m < 0).mean(),
                         n=len(x), divergences=int(idata.sample_stats['diverging'].sum())))
    return pd.DataFrame(rows)


def main(a):
    from retsupp.visualize.paper.gain_meta_zs import bayes, wilcoxon_table
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = collect(a.root)
    d.to_csv(out / 'estimates.tsv', sep='\t', index=False)
    cal = d.groupby('term').apply(lambda x: np.median(x.halfdiff / x.se), include_groups=False)
    print(f'{d.subject.nunique()} participants, {len(d) // len(CONTRASTS)} cells; calibration (0.67 expected):',
          cal.round(2).to_dict(), '; at a bound:', d.groupby('term').at_bound.sum().to_dict(), flush=True)
    wil, bay, own, diags = [], [], [], []
    for term in a.terms:
        t = d[d.term == term].copy()
        wil.append(wilcoxon_table(t).assign(term=term))
        s, dg = bayes(t)
        bay.append(s.assign(term=term))
        diags.append({**dg, 'term': term})
        own.append(own_data(t).assign(term=term))
        print(term, dg, flush=True)
    pd.concat(wil).to_csv(out / 'wilcoxon.tsv', sep='\t', index=False)
    pd.concat(bay).to_csv(out / 'bayes.tsv', sep='\t', index=False)
    pd.concat(own).to_csv(out / 'own_data.tsv', sep='\t', index=False)
    pd.DataFrame(diags).to_csv(out / 'bayes_diagnostics.tsv', sep='\t', index=False)
    print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/data/ds-retsupp/derivatives/adjopp_gain_stats_zs_2026-11-25')
    ap.add_argument('--out', default='notes/data/adjopp_meta_zs_2026-11-25')
    ap.add_argument('--terms', nargs='+', default=['next_minus_opp', 'hp_minus_next', 'hp_minus_opp', 'dS'])
    main(ap.parse_args())
