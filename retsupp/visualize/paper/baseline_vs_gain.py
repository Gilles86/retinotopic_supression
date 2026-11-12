"""Baseline shift vs response gain at the HP location (2026-11-12), step 1: slope of the bar response by location.

A sustained gain at HP scales the response to the bar when it crosses pRFs on the HP location; a baseline shift adds
an offset that does not depend on the bar. A sustained (block-constant) offset is not identifiable here: the cleaned
BOLD is percent signal change of each voxel's run mean (``preprocess/clean.py``, nilearn ``standardize='psc'``, plus
fMRIPrep cosine regressors), and HP is constant within a run, so each run's mean, and any constant offset with it, is
removed. What remains is the slope.

Per subject x ROI x ring location (the Fig. 3 location series, ``export_location_timeseries``: voxels with cross-run
reliability > 0 and a pRF centre within 2 deg of the location; ``raw`` = their mean BOLD, ``bar`` = the bar-only pRF
prediction from the final pRFs and per-ROI HRF), one OLS over all runs:

    raw = c_run + beta_rel(run) * bar + FIR(search onsets at this location, by item class x HP/LP) + e

with a free intercept per run (the run means are removed anyway), one bar slope per relation of the location to the run's
HP location (HP, next to HP, opposite HP), and a 10-bin FIR (0-16 s) per item class (target, distractor, neutral at
HP or LP) so the item responses do not load on the bar. Per subject x ROI the slopes are averaged over the four
locations (each location is HP, next and opposite in different runs), and the relative bar gain at HP is
100 * (beta_HP / mean(beta_next, beta_opp) - 1) [%]; also HP vs next and next vs opposite. Pooled per subject as the
median over the areas of a group. n = 28 (sub-01, sub-02 excluded).

    python -m retsupp.visualize.paper.baseline_vs_gain step1 --ts-dir /data/ds-retsupp/derivatives/paper_2026-11/location_timeseries_final2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
GROUPS = {'Early (V1–V3)': ['V1', 'V2', 'V3'], 'Dorsal (V3AB, IPS)': ['V3AB', 'IPS'],
          'Occipitotemporal (hV4, VO, LO, TO)': ['hV4', 'VO', 'LO', 'TO'], 'All areas': ROIS}
CLASSES = [f'{w}_{h}' for w in ('target', 'distractor', 'neutral') for h in ('hp', 'lp')]
N_FIR = 10


def relation(loc, hp_loc):
    d = (loc - hp_loc) % 4
    return 'hp' if d == 0 else 'opp' if d == 2 else 'next'


def fit_location(ts, trials, loc):
    """OLS of the location's raw series on run intercepts, bar x relation, and FIR per item class."""
    runs = ts[['session', 'run']].drop_duplicates().itertuples(index=False)
    blocks, ys = [], []
    for ses, run in runs:
        t = ts[(ts.session == ses) & (ts.run == run)].sort_values('tr_index')
        tri = trials[(trials.session == ses) & (trials.run == run)]
        hp_loc = tri[tri.cls.str.endswith('_hp')]['loc'].iloc[0]
        rel = relation(loc, hp_loc)
        tr = float(np.median(np.diff(t.time.to_numpy())))
        n = len(t)
        X = {}
        for r_ in ('hp', 'next', 'opp'):
            X[f'bar_{r_}'] = t.bar.to_numpy() * (rel == r_)
        for c in CLASSES:
            on = tri[(tri['loc'] == loc) & (tri.cls == c)].onset.to_numpy()
            idx = np.round(on / tr).astype(int)
            for k in range(N_FIR):
                col = np.zeros(n)
                j = idx + k
                j = j[(j >= 0) & (j < n)]
                np.add.at(col, j, 1.)
                X[f'{c}_{k}'] = col
        X[f'run_{ses}_{run}'] = np.ones(n)
        blocks.append(pd.DataFrame(X))
        ys.append(t.raw.to_numpy())
    X = pd.concat(blocks, ignore_index=True).fillna(0.)
    y = np.concatenate(ys)
    X = X.loc[:, X.abs().sum() > 0]
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    b = dict(zip(X.columns, beta))
    return {r_: b.get(f'bar_{r_}', np.nan) for r_ in ('hp', 'next', 'opp')}


def step1(ts_dir, out, exclude=(1, 2)):
    ts_dir = Path(ts_dir)
    rows = []
    for f in sorted(ts_dir.glob('sub-*_timeseries.tsv.gz')):
        s = int(f.name[4:6])
        if s in exclude:
            continue
        ts = pd.read_csv(f, sep='\t')
        trials = pd.read_csv(ts_dir / f'sub-{s:02d}_trials.tsv', sep='\t')
        for (roi, loc), g in ts.groupby(['roi', 'loc']):
            b = fit_location(g, trials, loc)
            rows.append(dict(subject=s, roi=roi, loc=loc, n_voxels=int(g.n_voxels.iloc[0]), **{f'beta_{k}': v for k, v in b.items()}))
        print(f'sub-{s:02d} done', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out.with_name(out.stem + '_per_location.tsv'), sep='\t', index=False)
    # per subject x ROI: mean slope over the four locations, then relative gains
    a = d.groupby(['subject', 'roi'])[['beta_hp', 'beta_next', 'beta_opp']].mean().reset_index()
    lp = (a.beta_next + a.beta_opp) / 2
    a['hp_vs_lp'] = 100 * (a.beta_hp / lp - 1)
    a['hp_vs_next'] = 100 * (a.beta_hp / a.beta_next - 1)
    a['next_vs_opp'] = 100 * (a.beta_next / a.beta_opp - 1)
    a = a[(a.beta_next > 0) & (a.beta_opp > 0)]          # a ratio needs a positive bar response at LP
    a.to_csv(out.with_name(out.stem + '_per_roi.tsv'), sep='\t', index=False)
    rng = np.random.default_rng(0)
    res = []
    for name, rois in GROUPS.items():
        for col in ('hp_vs_lp', 'hp_vs_next', 'next_vs_opp'):
            s = a[a.roi.isin(rois)].groupby('subject')[col].median().dropna()
            boot = np.percentile([np.median(rng.choice(s.to_numpy(), len(s))) for _ in range(10000)], [2.5, 97.5])
            res.append(dict(group=name, contrast=col, n=len(s), median_pct=s.median(), ci_lo=boot[0], ci_hi=boot[1],
                            k_neg=int((s < 0).sum()), p=wilcoxon(s).pvalue))
    res = pd.DataFrame(res)
    res.to_csv(out, sep='\t', index=False)
    print(res.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p1 = sp.add_parser('step1')
    p1.add_argument('--ts-dir', default='/data/ds-retsupp/derivatives/paper_2026-11/location_timeseries_final2')
    p1.add_argument('--out', default='notes/data/baseline_vs_gain_step1_n28_2026-11-12.tsv')
    a = ap.parse_args()
    if a.cmd == 'step1':
        step1(a.ts_dir, Path(a.out))
