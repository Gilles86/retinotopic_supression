"""Collect the multiplicative vs additive HP item-gain models (m8_full_hpmult, m8_full_hpadd) and test them.

Held-out prediction (``--cv-root``, a ``--prf-source-root`` run3 root): per subject × area, held-out R² averaged
over voxels within each fold, then over the three folds, for m5_TDS_dS (the base: no HP term on the item gains),
m7_full (HP − LP distractor gain dD only), m8_full_hpmult and m8_full_hpadd. Differences mult − add, mult − base,
add − base, dD − base.

Statistics as in key_numbers.md: per area the median over participants with a bootstrapped 95% CI (10,000 resamples),
k of n above 0, two-sided Wilcoxon, BH across the nine areas; pooled: per participant the median over the nine areas;
per pathway: per participant the median over the pathway's areas, BH across the three pathways.

Fitted terms: h (mult) and a (add), in %, per subject × area as the mean over folds (``--cv-root``) and from the
all-runs fit (``--full-root``, same models with ``--cv-scheme full``) with the same summaries, plus the target and
distractor gains of the same fits.

Usage::

    python -m retsupp.modeling.collect_hp_item_gain --cv-root <run3 _polprf root> --full-root <full _polprf root> \\
        --out notes/data/hp_item_gain_mult_vs_add_2026-11-13
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

MODELS = ('m5_TDS_dS', 'm7_full', 'm8_full_hpmult', 'm8_full_hpadd')
AREAS = ('V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS')
PATHWAY = {'Early': ('V1', 'V2', 'V3'), 'Dorsal': ('V3AB', 'IPS'), 'Occipitotemporal': ('hV4', 'VO', 'LO', 'TO')}
CONTRASTS = {'mult_minus_add': ('m8_full_hpmult', 'm8_full_hpadd'),
             'mult_minus_base': ('m8_full_hpmult', 'm5_TDS_dS'),
             'add_minus_base': ('m8_full_hpadd', 'm5_TDS_dS'),
             'dD_minus_base': ('m7_full', 'm5_TDS_dS')}
EXCLUDE = (1, 2)


def summarise(x, rng):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if x.size == 0:
        return dict(n=0, median=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_pos=0, n_neg=0, p=np.nan)
    boot = np.median(rng.choice(x, size=(10000, x.size), replace=True), axis=1)
    p = wilcoxon(x).pvalue if x.size >= 5 and np.any(x != 0) else np.nan
    return dict(n=x.size, median=np.median(x), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                n_pos=int((x > 0).sum()), n_neg=int((x < 0).sum()), p=p)


def stats_table(df, cols):
    """``df``: one row per subject × roi; summaries per area, pooled and per pathway for each column."""
    rng = np.random.default_rng(0)
    out = []
    for col in cols:
        rows = [dict(level=a, kind='area', **summarise(df.loc[df.roi.eq(a), col], rng)) for a in AREAS]
        p = np.array([r['p'] for r in rows])
        q = np.full(p.size, np.nan)
        q[~np.isnan(p)] = multipletests(p[~np.isnan(p)], method='fdr_bh')[1]
        for r, qq in zip(rows, q):
            r['q_bh'] = qq
        wide = df.pivot(index='subject', columns='roi', values=col).reindex(columns=list(AREAS))
        rows.append(dict(level='pooled', kind='pooled', q_bh=np.nan,
                         **summarise(wide[list(AREAS)].median(axis=1, skipna=True), rng)))
        prow = [dict(level=pw, kind='pathway', **summarise(wide[list(areas)].median(axis=1),
                                                             rng)) for pw, areas in PATHWAY.items()]
        p = np.array([r['p'] for r in prow])
        qp = np.full(p.size, np.nan)
        qp[~np.isnan(p)] = multipletests(p[~np.isnan(p)], method='fdr_bh')[1]
        for r, qq in zip(prow, qp):
            r['q_bh'] = qq
        out += [dict(variable=col, **r) for r in rows + prow]
    return pd.DataFrame(out)


def shared_rows(root, scheme):
    rows = []
    for model in MODELS:
        for pkl in sorted((Path(root) / model).glob('sub-*/sub-*_roi-*_cv-fits.pkl')):
            sub, roi = re.match(r'sub-(\d+)_roi-(.+)_cv-fits.pkl', pkl.name).groups()
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            for k, (p, info) in enumerate(zip(d['fit_pars_per_fold'], d['optimizer_info'])):
                if p is None:
                    continue
                r = dict(subject=int(sub), roi=roi, model=model, fold=k, scheme=scheme,
                         sse_start=info.get('sse_parent_start', np.nan), sse_final=info.get('sse_final', np.nan))
                for c in ('sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale',
                          'hp_scale', 'hp_offset'):
                    if c in p:
                        r[c] = float(p[c].iloc[0])
                rows.append(r)
    return pd.DataFrame(rows)


def main(cv_root, full_root, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Held-out R².
    rows = []
    for model in MODELS:
        for f in sorted((Path(cv_root) / model).glob('sub-*/sub-*_roi-*_cv-r2.tsv')):
            d = pd.read_csv(f, sep='\t')
            per_fold = d.groupby('fold').agg(cv_r2=('cv_r2', 'mean'), train_r2=('train_r2', 'mean'))
            rows.append(dict(subject=int(d.subject.iloc[0]), roi=d.roi.iloc[0], model=model,
                             cv_r2=per_fold.cv_r2.mean(), train_r2=per_fold.train_r2.mean(),
                             n_voxels=d.voxel_id.nunique()))
    cv = pd.DataFrame(rows)
    cv = cv[~cv.subject.isin(EXCLUDE) & cv.roi.isin(AREAS)]
    wide = cv.pivot_table(index=['subject', 'roi'], columns='model', values='cv_r2')
    wtr = cv.pivot_table(index=['subject', 'roi'], columns='model', values='train_r2')
    complete = wide[list(MODELS)].notna().all(axis=1)
    print(f'{complete.sum()} subject × area with all four models ({(~complete).sum()} incomplete dropped)')
    wide, wtr = wide[complete], wtr[complete]
    delta = pd.DataFrame({c: (wide[a] - wide[b]) * 100 for c, (a, b) in CONTRASTS.items()})   # % of variance
    for c, (a, b) in CONTRASTS.items():
        delta[f'train_{c}'] = (wtr[a] - wtr[b]) * 100
    delta = delta.reset_index()
    delta.to_csv(f'{out}_cv_delta.tsv', sep='\t', index=False)
    cv.to_csv(f'{out}_cv_r2.tsv', sep='\t', index=False)
    st = stats_table(delta, list(CONTRASTS) + [f'train_{c}' for c in CONTRASTS])
    st.to_csv(f'{out}_cv_stats.tsv', sep='\t', index=False)
    pd.set_option('display.width', 220)
    print(st.round(5).to_string())
    # Fitted terms.
    sh = [shared_rows(cv_root, 'run3')]
    if full_root:
        sh.append(shared_rows(full_root, 'full'))
    sh = pd.concat(sh, ignore_index=True)
    sh = sh[~sh.subject.isin(EXCLUDE) & sh.roi.isin(AREAS)]
    sh.to_csv(f'{out}_shared_terms.tsv', sep='\t', index=False)
    for scheme, d in sh.groupby('scheme'):
        m = d.groupby(['subject', 'roi', 'model']).mean(numeric_only=True).reset_index()
        terms = pd.DataFrame({
            'h_pct': m[m.model.eq('m8_full_hpmult')].set_index(['subject', 'roi']).hp_scale * 100,
            'a_pct': m[m.model.eq('m8_full_hpadd')].set_index(['subject', 'roi']).hp_offset * 100,
            'gT_base_pct': m[m.model.eq('m5_TDS_dS')].set_index(['subject', 'roi']).g_T_dyn * 100,
            'gD_base_pct': m[m.model.eq('m5_TDS_dS')].set_index(['subject', 'roi']).g_LP_dyn * 100,
            'dD_m7_pct': (m[m.model.eq('m7_full')].set_index(['subject', 'roi']).eval('g_HP_dyn - g_LP_dyn')) * 100,
        }).reset_index()
        g = m[m.model.eq('m8_full_hpmult')].set_index(['subject', 'roi'])
        # The HP change the multiplicative fit implies for each item: h·g_T and h·g_D (percentage points).
        terms = terms.set_index(['subject', 'roi'])
        terms['mult_dT_pct'] = g.hp_scale * g.g_T_dyn * 100
        terms['mult_dD_pct'] = g.hp_scale * g.g_LP_dyn * 100
        # A fit with a shared term at its box (|gain| 10, |h|, |a| 10, σ 0.5 or 8°) in any of the four models.
        lim = m.set_index(['subject', 'roi'])
        gcols = [c for c in ('g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'hp_scale', 'hp_offset') if c in lim]
        hit = ((lim[gcols].abs() >= 9.99).any(axis=1) | (lim[['sigma_AF', 'sigma_dyn']] <= 0.501).any(axis=1)
               | (lim[['sigma_AF', 'sigma_dyn']] >= 7.99).any(axis=1))
        terms['at_bound'] = hit.groupby(level=[0, 1]).any().reindex(terms.index).fillna(False)
        terms = terms.reset_index()
        terms.to_csv(f'{out}_terms_{scheme}.tsv', sep='\t', index=False)
        cols = [c for c in terms.columns if c not in ('subject', 'roi', 'at_bound')]
        tt = pd.concat([stats_table(terms, cols).assign(fits='all'),
                        stats_table(terms[~terms.at_bound], cols).assign(fits='none at a bound')], ignore_index=True)
        tt.to_csv(f'{out}_terms_{scheme}_stats.tsv', sep='\t', index=False)
        print(f'--- fitted terms ({scheme}); {int(terms.at_bound.sum())} of {len(terms)} subject × area with a term '
              f'at a bound')
        print(tt[tt.kind.ne('area')].round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cv-root', required=True)
    ap.add_argument('--full-root', default=None)
    ap.add_argument('--out', required=True, help='Output prefix (…_cv_delta.tsv, …_cv_stats.tsv, …_terms_*.tsv).')
    a = ap.parse_args()
    main(a.cv_root, a.full_root, a.out)
