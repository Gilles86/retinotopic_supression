"""Fig. 3 (measured responses at the item locations) through the hierarchical model of the gain-field terms.

Inputs: the jackknife deconvolutions of ``deconvolve_location_responses.py --jackknife`` (plain: item type x HP/LP;
``--relative``: item type x HP / next left / next right / opposite), column ``left_out`` = 'none' (all runs), one
'<session>_<run>' per run left out, 'odd' and 'even'.

Per subject x unit (an area; for the Wilcoxon reproduction also a pathway and all areas, ROIs pooled weighted by voxel
count as in ``fig45_common.item_timecourses``): each class's time course minus its pre-onset mean (-4 to 0 s), the mean
over the window of Fig. 3 (3.2 to 6.4 s), then the contrasts
  item:     target − neutral ((T_HP + T_LP)/2 − (N_HP + N_LP)/2), distractor − neutral (LP), target − distractor (LP),
            in % signal;
  location: HP − opposite, next − opposite, HP − next (items pooled, next = mean of the two neighbours), in % signal
            ('abs') and in % of the subject's mean response over the twelve item x location cells ('rel', as Fig. 3e;
            cells whose mean is not positive are left out).
Each with its leave-one-run-out jackknife SE and half its odd − even difference (calibration, 0.67 expected).

Statistics: Wilcoxon on the all-run values (per area BH over nine; pathways pooled by voxels, BH over three, as the
manuscript), the Student-t hierarchical model of ``gain_meta_zs.bayes`` (pathway = mean of its areas), and each area on
its own data (no pooling).

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.fig3_meta \\
        --plain <folder> --relative <folder> --out notes/data/fig3_meta_2026-11-25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import fdr, median_ci
from retsupp.visualize.paper.gain_meta_zs import PW, PW_OF, bayes
from retsupp.visualize.paper.paper_style import ROI_ORDER

PEAK = (3.2, 6.4)
ITEMS = ('target', 'distractor', 'neutral')
PATHWAY_NAMES = {'early': 'Early', 'dorsal': 'Dorsal', 'occipitotemporal': 'Occipitotemporal'}


def load(folder, exclude=(1, 2)):
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*.tsv'))])
    d = d[d.roi.isin(ROI_ORDER) & ~d.subject.isin(exclude)].copy()
    d['t'] = d.t.round(2)
    if 'left_out' not in d:
        d['left_out'] = 'none'
    return d


def windows(d, groups):
    """Per left_out x subject x unit x class: baseline-corrected window mean (ROIs and locations pooled by voxels)."""
    d = d.assign(unit=d.roi.map(groups)).dropna(subset=['unit'])
    d['w'] = d.response * d.n_voxels
    g = d.groupby(['left_out', 'subject', 'unit', 'cls', 't'])
    p = (g.w.sum() / g.n_voxels.sum()).rename('r').reset_index()
    base = p[p.t < 0].groupby(['left_out', 'subject', 'unit', 'cls']).r.mean()
    win = p[p.t.between(*PEAK)].groupby(['left_out', 'subject', 'unit', 'cls']).r.mean()
    return (win - base).unstack('cls')


def item_contrasts(pk):
    need = [f'{i}_{h}' for i in ITEMS for h in ('hp', 'lp')]
    pk = pk.dropna(subset=need)
    return pd.DataFrame({'target − neutral': (pk.target_hp + pk.target_lp) / 2 - (pk.neutral_hp + pk.neutral_lp) / 2,
                         'distractor − neutral': pk.distractor_lp - pk.neutral_lp,
                         'target − distractor': pk.target_lp - pk.distractor_lp})


def loc_contrasts(pk):
    locs = ('hp', 'nextleft', 'nextright', 'opposite')
    pk = pk.dropna(subset=[f'{i}_{l}' for i in ITEMS for l in locs])
    m = {l: pk[[f'{i}_{l}' for i in ITEMS]].mean(1) for l in locs}
    nxt = (m['nextleft'] + m['nextright']) / 2
    mean12 = sum(m.values()) / 4
    out = {}
    for k, v in (('HP − opposite', m['hp'] - m['opposite']), ('next − opposite', nxt - m['opposite']),
                 ('HP − next', m['hp'] - nxt)):
        out[f'{k} (abs)'] = v
        out[f'{k} (rel)'] = (100 * v / mean12).where(mean12 > 0)
    return pd.DataFrame(out)


def per_cell(c):
    """c: contrasts indexed by (left_out, subject, unit). Returns per subject x unit x term: y (all runs), SE, halfdiff."""
    rows = []
    for (s, u), g in c.groupby(level=['subject', 'unit']):
        g = g.droplevel(['subject', 'unit'])
        if 'none' not in g.index:
            continue
        loo = g.loc[[i for i in g.index if i not in ('none', 'odd', 'even')]]
        R = len(loo)
        for term in c.columns:
            full = g.loc['none', term]
            v = loo[term].dropna()
            if not np.isfinite(full) or len(v) < R or R < 4:
                continue
            se = np.sqrt((R - 1) / R * ((v - v.mean()) ** 2).sum())
            hd = abs(g.loc['odd', term] - g.loc['even', term]) / 2 if {'odd', 'even'} <= set(g.index) else np.nan
            rows.append(dict(subject=s, roi=u, term=term, y=full, se=se, halfdiff=hd, n_runs=R))
    return pd.DataFrame(rows)


def wilcoxon_rows(full, level):
    rows = []
    for u, g in full.groupby('unit'):
        for term in full.columns.drop('unit'):
            v = g[term].dropna()
            if len(v) < 6:
                continue
            m, lo, hi = median_ci(v.to_numpy())
            rows.append(dict(level=level, unit=u, term=term, n=len(v), mean=v.mean(), sem=v.sem(), median=m,
                             ci_lo=lo, ci_hi=hi, k_neg=int((v < 0).sum()), p=stats.wilcoxon(v).pvalue))
    r = pd.DataFrame(rows)
    for term in r.term.unique():
        k = r.term == term
        r.loc[k, 'q'] = fdr(r.loc[k, 'p'])
    return r


def own_data(t, seed=0):
    """Each area alone: y ~ Student-t(ν, μ, sqrt(SE² + τ²)), no pooling over areas or participants."""
    import arviz as az
    import pymc as pm
    rows = []
    for roi, g in t.groupby('roi'):
        y, se = g.y.to_numpy(), g.se.to_numpy()
        sy = float(np.std(y)) or 1.
        with pm.Model():
            mu = pm.Normal('mu', 0, 2 * sy)
            tau = pm.HalfNormal('tau', sy)
            pm.StudentT('y', nu=pm.Gamma('nu', 2, .1), mu=mu, sigma=pm.math.sqrt(se ** 2 + tau ** 2), observed=y)
            idata = pm.sample(2000, tune=2000, chains=4, target_accept=.99, random_seed=seed, progressbar=False,
                              nuts_sampler='nutpie')
        x = idata.posterior['mu'].values.ravel()
        lo, hi = az.hdi(x, hdi_prob=.95)
        rows.append(dict(unit=roi, mean=x.mean(), hdi_lo=lo, hdi_hi=hi, p_gt0=(x > 0).mean(), p_lt0=(x < 0).mean(),
                         n=len(y), divergences=int(idata.sample_stats['diverging'].sum())))
    return pd.DataFrame(rows)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by_area = {r: r for r in ROI_ORDER}
    by_path = {r: PATHWAY_NAMES[g] for r, g in PW_OF.items()}
    all_ = {r: 'All areas' for r in ROI_ORDER}
    cells, wil = [], []
    for kind, folder, fun in (('item', a.plain, item_contrasts), ('location', a.relative, loc_contrasts)):
        d = load(folder)
        for groups, level in ((by_area, 'area'), (by_path, 'pathway'), (all_, 'all')):
            c = fun(windows(d, groups))
            full = c.xs('none', level='left_out').reset_index(level='unit')
            wil.append(wilcoxon_rows(full, level).assign(kind=kind))
            if level == 'area':
                cells.append(per_cell(c).assign(kind=kind))
        print(kind, 'done', flush=True)
    cells = pd.concat(cells)
    cells['pathway'] = cells.roi.map(PW_OF)
    cells.to_csv(out / 'estimates.tsv', sep='\t', index=False)
    wil = pd.concat(wil)
    wil.to_csv(out / 'wilcoxon.tsv', sep='\t', index=False)
    cal = cells.groupby(['term', 'pathway']).apply(lambda x: np.median(x.halfdiff / x.se), include_groups=False)
    cal = pd.concat([cal.unstack(), cells.groupby('term').apply(lambda x: np.median(x.halfdiff / x.se),
                                                                include_groups=False).rename('all')], axis=1)
    cal.to_csv(out / 'calibration.tsv', sep='\t')
    print(cal.round(2).to_string(), flush=True)
    if a.no_bayes:
        return
    post, own, diags = [], [], []
    for term in a.terms or cells.term.unique():
        t = cells[cells.term == term].copy()
        s, dg = bayes(t)
        post.append(s.assign(term=term))
        diags.append({**dg, 'term': term})
        own.append(own_data(t).assign(term=term))
        print(term, dg, flush=True)
    pd.concat(post).to_csv(out / 'bayes.tsv', sep='\t', index=False)
    pd.concat(own).to_csv(out / 'own_data.tsv', sep='\t', index=False)
    pd.DataFrame(diags).to_csv(out / 'bayes_diagnostics.tsv', sep='\t', index=False)
    print('Wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plain', required=True, help='jackknife deconvolution, item type x HP/LP')
    ap.add_argument('--relative', required=True, help='jackknife deconvolution, --relative')
    ap.add_argument('--out', default='notes/data/fig3_meta_2026-11-25')
    ap.add_argument('--terms', nargs='+', default=None)
    ap.add_argument('--no-bayes', action='store_true')
    main(ap.parse_args())
