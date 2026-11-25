"""dS with its measurement error: precision-weighted and hierarchical tests on the zero-sum main model.

Input: the per-run sufficient statistics of ``fit_width_profiles`` on the zero-sum main model (gains of the sustained
and transient fields re-solved by least squares, pRFs, HRF and widths fixed at the joint fit). Per subject × area:
dS = g_HP − g_LP and its leave-one-run-out jackknife SE (which allows for temporal and spatial correlation of the
residuals); target gain and distractor gain alike, for comparison. Calibration: median |odd − even| / 2 / SE
(0.67 expected when the SE is right).

Tests, each at three levels (all nine areas; three pathways, BH over three; nine areas, BH over nine):

- ``wilcoxon``: the current test (per participant the median over areas, Wilcoxon), on these estimates;
- ``re``: random-effects meta-analysis (DerSimonian-Laird) over participants; for pathways and pooled, per participant
  first the inverse-variance-weighted mean over the areas;
- ``bayes``: the hierarchical model of Fig. 7 (``trialwise_meta_bayes.fit``): y ~ N(θ, sqrt(SE² + τ²)),
  θ = pathway mean + area + participant; partial pooling over areas, so no correction across areas is needed.
  Normal and Student-t likelihood, and without SEs as a check.

    python -m retsupp.visualize.paper.dS_meta --root /data/ds-retsupp/derivatives/field_widths_zs_lm_2026-11-14
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import fdr, median_ci
from retsupp.visualize.paper.field_widths_reliable import jackknife, solve
from retsupp.visualize.paper.paper_style import ROI_GROUPS, ROI_ORDER

GROUPS = dict(ROI_GROUPS)
TERMS = {'dS': lambda g, n: np.array([g[n.index('g_HP')] - g[n.index('g_LP')]]),
         'target': lambda g, n: np.array([g[n.index('g_T')]]),
         'distractor': lambda g, n: np.array([(g[n.index('g_D_HP')] + g[n.index('g_D_LP')]) / 2])}


def collect(root, exclude=(1, 2)):
    rows = []
    for f in sorted(glob.glob(f'{root}/sub-*/*_widthprof.npz')):
        z = np.load(f, allow_pickle=True)
        s, roi = int(z['subject']), str(z['roi'])
        if s in exclude or roi not in ROI_ORDER:
            continue
        names = list(z['gain_names'])
        A, b = z['A'], z['b']
        odd = z['runs'][:, 1] % 2 == 1
        for term, fun in TERMS.items():
            full, _, se = jackknife(A, b, lambda g: fun(g, names))
            go, ge = fun(solve(A[odd].sum(0), b[odd].sum(0)), names), fun(solve(A[~odd].sum(0), b[~odd].sum(0)), names)
            fit = fun(np.asarray(z['fitted_gains'], float), names)
            rows.append(dict(subject=s, roi=roi, term=term, y=100 * full[0], se=100 * se[0], fit=100 * fit[0],
                             halfdiff=100 * abs(go[0] - ge[0]) / 2))
    return pd.DataFrame(rows)


def dersimonian_laird(y, se):
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    w = 1 / v
    mf = (w * y).sum() / w.sum()
    q = (w * (y - mf) ** 2).sum()
    tau2 = max(0., (q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    m = (ws * y).sum() / ws.sum()
    sem = np.sqrt(1 / ws.sum())
    return m, sem, 2 * stats.norm.sf(abs(m / sem)), np.sqrt(tau2)


def ivw(d):
    """Per subject, the inverse-variance-weighted mean over the given rows and its SE."""
    w = 1 / d.se ** 2
    return pd.Series(dict(y=(w * d.y).sum() / w.sum(), se=np.sqrt(1 / w.sum())))


def frequentist(t):
    rows = []
    units = [('all', 'pooled', ROI_ORDER)] + [('pathway', g, list(r)) for g, r in ROI_GROUPS] + \
            [('area', a, [a]) for a in ROI_ORDER]
    for level, unit, rois in units:
        d = t[t.roi.isin(rois)]
        per = d.groupby('subject').apply(ivw, include_groups=False)
        m, sem, p, tau = dersimonian_laird(per.y, per.se)
        med = d.groupby('subject').y.median()
        mm, lo, hi = median_ci(med)
        rows.append(dict(level=level, unit=unit, n=len(per), re_mean=m, re_lo=m - 1.96 * sem, re_hi=m + 1.96 * sem,
                         re_p=p, re_tau=tau, wil_median=mm, wil_lo=lo, wil_hi=hi, k_neg=int((med < 0).sum()),
                         wil_p=stats.wilcoxon(med).pvalue))
    r = pd.DataFrame(rows)
    for level in ('pathway', 'area'):
        k = r.level == level
        r.loc[k, 're_q'] = fdr(r.loc[k, 're_p'])
        r.loc[k, 'wil_q'] = fdr(r.loc[k, 'wil_p'])
    return r


def bayes(t, term):
    from retsupp.visualize.paper import trialwise_meta_bayes as tmb
    tmb.GROUPS = {'early': list(GROUPS['Early']), 'dorsal': list(GROUPS['Dorsal']),
                  'occipitotemporal': list(GROUPS['Occipitotemporal'])}
    tmb.AREAS = [a for g in tmb.GROUPS.values() for a in g]
    tmb.GROUP_OF = {a: g for g, aa in tmb.GROUPS.items() for a in aa}
    rows, diags = [], []
    for variant, kw in [('normal', {}), ('t', dict(likelihood='t')), ('no_se', dict(use_se=False))]:
        idata, areas, groups = tmb.fit(t[['subject', 'roi', 'y', 'se']].reset_index(drop=True), **kw)
        s, diag = tmb.summarize(idata, t.reset_index(drop=True), areas, groups, term, variant)
        rows.append(s)
        diags.append(diag)
    return pd.concat(rows), pd.DataFrame(diags)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = collect(a.root)
    d.to_csv(out / 'estimates.tsv', sep='\t', index=False)
    cal = d.groupby('term').apply(lambda x: np.median(x.halfdiff / x.se), include_groups=False)
    print('calibration (0.67 expected):', cal.round(2).to_dict())
    fr, by, dg = [], [], []
    for term in a.terms:
        t = d[d.term == term]
        fr.append(frequentist(t).assign(term=term))
        if not a.no_bayes:
            b, g = bayes(t, term)
            by.append(b)
            dg.append(g)
    fr = pd.concat(fr)
    fr.to_csv(out / 'frequentist.tsv', sep='\t', index=False)
    if by:
        pd.concat(by).to_csv(out / 'bayes.tsv', sep='\t', index=False)
        pd.concat(dg).to_csv(out / 'bayes_diagnostics.tsv', sep='\t', index=False)
    print(fr.round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/ds-retsupp/derivatives/field_widths_zs_lm_2026-11-14')
    ap.add_argument('--out', default='notes/data/dS_meta_zs_lm_2026-11-25')
    ap.add_argument('--terms', nargs='+', default=['dS', 'target', 'distractor'])
    ap.add_argument('--no-bayes', action='store_true')
    main(ap.parse_args())
