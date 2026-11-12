"""Clamped (pRFs frozen at the m0 fit) vs polished (pRFs refitted together with the gain fields): group statistics.

Inputs: ``collect_polished_fits`` tables (joint m7_full and m4_TDS shared terms, pRF shifts) and two runs of
``collect_conditional_estimates`` (estimated alone on the clamped and on the polished base). Per area: median over
subjects, Wilcoxon against 0 with BH over the nine areas; pooled = per-subject median over the areas. Also the
paired polished − clamped difference (Wilcoxon). Gains in %.

    python -m retsupp.visualize.paper.compare_polished_fits --prefix notes/data/prf_refit_2026-11-12 \\
        --cond-clamped notes/data/conditional_estimates_clampedcheck_2026-11-12.tsv \\
        --cond-polished notes/data/conditional_estimates_polished_2026-11-12.tsv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']


def wil(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.all(x == 0):
        return np.nan
    return wilcoxon(x).pvalue


def group(long, value, label):
    """long: subject, roi, fit, <value>. One row per fit x area (+ pooled) with median, % <0, p, q."""
    rows = []
    for fit, d in long.groupby('fit'):
        for roi in ROIS + ['pooled']:
            x = d.groupby('subject')[value].median() if roi == 'pooled' else d[d.roi == roi].set_index('subject')[value]
            rows.append(dict(term=label, fit=fit, roi=roi, n=len(x), median_pct=100 * x.median(),
                             q25_pct=100 * x.quantile(.25), q75_pct=100 * x.quantile(.75),
                             frac_neg=(x < 0).mean(), p=wil(x)))
    out = pd.DataFrame(rows)
    for fit in out.fit.unique():
        m = (out.fit == fit) & (out.roi != 'pooled') & out.p.notna()
        out.loc[m, 'q_bh'] = multipletests(out.loc[m, 'p'], method='fdr_bh')[1]
    # paired difference polished - clamped
    w = long.pivot_table(index=['subject', 'roi'], columns='fit', values=value).dropna()
    if {'clamped', 'polished'} <= set(w.columns):
        diff = (w.polished - w.clamped).rename('d').reset_index()
        for roi in ROIS + ['pooled']:
            x = diff.groupby('subject').d.median() if roi == 'pooled' else diff[diff.roi == roi].d
            out.loc[(out.fit == 'polished') & (out.roi == roi), 'diff_median_pct'] = 100 * x.median()
            out.loc[(out.fit == 'polished') & (out.roi == roi), 'p_diff'] = wil(x)
    return out


def main(prefix, cond_clamped, cond_polished, out):
    sh = pd.read_csv(f'{prefix}_shared.tsv', sep='\t')
    sh = sh[sh.subject.between(3, 30)]
    sh['T'] = sh.g_T_dyn
    sh['D'] = (sh.g_HP_dyn + 3 * sh.g_LP_dyn) / 4
    res = []
    j = sh[sh.model == 'm7_full']
    for col, lab in [('dS', 'joint dS'), ('g_LP', 'joint g_LP'), ('g_HP', 'joint g_HP'), ('T', 'joint target'),
                     ('D', 'joint distractor'), ('dD', 'joint dD'),
                     ('item_scale', 'joint items')]:
        res.append(group(j[['subject', 'roi', 'fit', col]], col, lab))
    c = pd.concat([pd.read_csv(cond_clamped, sep='\t').assign(fit='clamped'),
                   pd.read_csv(cond_polished, sep='\t').assign(fit='polished')])
    c = c[c.subject.between(3, 30)]
    # Only the terms on m4_TDS (the polished base); items, T, D, S are reported from the joint fit (m7_full).
    for mech in ['dS', 'dD', 'dT']:
        res.append(group(c[c.mech == mech][['subject', 'roi', 'fit', 'estimate']], 'estimate', f'alone {mech}'))
    res = pd.concat(res)
    res.to_csv(out, sep='\t', index=False)
    shift = pd.read_csv(f'{prefix}_prf_shift.tsv', sep='\t')
    shift = shift[shift.subject.between(3, 30)]
    print(shift.groupby(['model', 'voxels'])[['shift_median', 'shift_p90', 'dsd_median', 'damp_median']].median()
          .round(4).to_string())
    sse = sh[sh.fit == 'polished']
    rel = (sse.sse_final - sse.sse_clamped) / sse.sse_clamped
    print('training SSE change, polished: median %.2e%%, max %.2e%%, n above clamped %d'
          % (100 * rel.median(), 100 * rel.max(), (rel > 0).sum()))
    pd.set_option('display.width', 250)
    show = res[res.roi.isin(['pooled', 'V3', 'hV4', 'TO'])]
    print(show[['term', 'fit', 'roi', 'n', 'median_pct', 'frac_neg', 'p', 'q_bh', 'diff_median_pct', 'p_diff']]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--cond-clamped', required=True)
    ap.add_argument('--cond-polished', required=True)
    ap.add_argument('--out', default='notes/data/prf_refit_group_2026-11-12.tsv')
    a = ap.parse_args()
    main(a.prefix, a.cond_clamped, a.cond_polished, a.out)
