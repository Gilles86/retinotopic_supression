"""Permutation test of the HP-specific gains against relabelled HP locations.

Each subject x ROI is fitted four times: with the true HP labels and with every run's
HP moved 1, 2 or 3 ring steps. If no ring position relative to HP is special, the four
labellings of a subject are exchangeable, so the null distribution assigns each subject
one labelling at random, the same for all their ROIs. The statistic is the median over
subjects of each subject's median over ROIs (or the median over subjects within a ROI).

``collect`` reads the fits into one table, on the cluster. ``test`` works from that table.

A problem specific to the true HP location is not covered: the distractor is most frequent
there, and relabelling breaks that link in the fits it enters as the null.

Usage::

    python -m retsupp.modeling.hp_rotation_null collect \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
    python -m retsupp.modeling.hp_rotation_null test --table notes/data/hp_rotation_null.tsv
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

SHARED = ['sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale']
MODELS = ['m5_TDS_dS', 'm6_TDS_dD', 'm7_full']
# Real fits per family; the first root holding a fit wins, so the outlier-free refits go first.
REAL = {'dog': ['af_prf_cv_ladder_nested_full_bounded_maxsd20', 'af_prf_cv_ladder_nested_full_bounded'],
        'dn': ['af_prf_cv_ladder_nested_full_bounded_maxsd20_dn']}
NULL = {'dog': 'af_prf_cv_ladder_nested_full_bounded_maxsd20_hprot{k}',
        'dn': 'af_prf_cv_ladder_nested_full_bounded_maxsd20_dn_hprot{k}'}
TERMS = [('dS', 'm5_TDS_dS', 'g_HP', 'g_LP'), ('dS', 'm7_full', 'g_HP', 'g_LP'),
         ('dD', 'm6_TDS_dD', 'g_HP_dyn', 'g_LP_dyn'), ('dD', 'm7_full', 'g_HP_dyn', 'g_LP_dyn')]


def collect(bids_folder, out):
    deriv = Path(bids_folder) / 'derivatives'
    rows = []
    for family in REAL:
        for k in range(4):
            roots = REAL[family] if k == 0 else [NULL[family].format(k=k)]
            seen = set()
            for root in roots:
                for model in MODELS:
                    for pkl in sorted((deriv / root / model).glob('sub-*/*_cv-fits.pkl')):
                        parts = pkl.name.split('_')
                        subject, roi = int(parts[0][4:]), parts[1][4:]
                        if (subject, roi, model) in seen:
                            continue
                        seen.add((subject, roi, model))
                        with open(pkl, 'rb') as f:
                            pars = pickle.load(f)['fit_pars_per_fold'][0]
                        rows.append(dict(family=family, rotation=k, model=model, subject=subject, roi=roi,
                                         **pars[SHARED].iloc[0].to_dict()))
    d = pd.DataFrame(rows)
    d.to_csv(out, sep='\t', index=False)
    print(d.groupby(['family', 'rotation', 'model']).size().unstack())


def perm_p(values, n_perm, rng):
    """values: subjects x labellings, labelling 0 the real one. Returns the observed median and p values."""
    obs = np.median(values[:, 0])
    pick = rng.integers(0, values.shape[1], size=(n_perm, values.shape[0]))
    null = np.median(values[np.arange(values.shape[0])[None], pick], axis=1)
    return (obs, null.mean(), null.std(), (1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (n_perm + 1),
            (1 + (null <= obs).sum()) / (n_perm + 1))


def test(table, out, n_perm=100_000, seed=1):
    d = pd.read_csv(table, sep='\t')
    rng = np.random.default_rng(seed)
    rows = []
    for family in d.family.unique():
        for term, model, hi, lo in TERMS:
            x = d[(d.family == family) & (d.model == model)].copy()
            x['v'] = x[hi] - x[lo]
            # Only subject x ROI cells present in all four labellings enter.
            x = x[x.groupby(['subject', 'roi']).v.transform('count') == 4]
            per_roi = []
            for scope, g in [('all ROIs', x)] + list(x.groupby('roi')):
                s = g.pivot_table(index=['subject', 'roi'], columns='rotation', values='v') \
                    .groupby('subject').median().dropna()
                obs, m, sd, p2, p1 = perm_p(s.values, n_perm, rng)
                row = dict(family=family, term=term, model=model, scope=scope, n=len(s), observed=obs,
                           pct_negative=(s[0] < 0).mean() * 100, null_mean=m, null_sd=sd, p_two=p2, p_negative=p1,
                           **{f'rot{k}_median': s[k].median() for k in s.columns if k})
                (per_roi.append(row) if scope != 'all ROIs' else rows.append(row))
            q = false_discovery_control([r['p_two'] for r in per_roi])
            rows += [dict(r, q_two=v) for r, v in zip(per_roi, q)]
    res = pd.DataFrame(rows)
    res.to_csv(out, sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.max_rows', 500):
        print(res[res.scope == 'all ROIs'].drop(columns=['scope', 'q_two']).round(4).to_string(index=False))
        print(res[res.q_two < .05].round(4).to_string(index=False) if (res.q_two < .05).any() else 'No ROI at q < .05')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['collect', 'test'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--table', default='notes/data/hp_rotation_null.tsv')
    ap.add_argument('--out', default='notes/data/hp_rotation_null_stats.tsv')
    a = ap.parse_args()
    if a.action == 'collect':
        collect(a.bids_folder, a.table)
    else:
        test(a.table, a.out)
