"""Side project (hippocampus): group tests and figure for the pRF-free HP test (``deconvolve_region_events.py``).

Readout: mean deconvolved response 3.2–8 s after search onset (z units of the run's region-mean time series).
Per subject × region:

- amplitude: mean over the five event types;
- dD = distractor at HP − at LP (target at LP in both: dHP_tLP − dLP_tLP);
- dT = target at HP − at LP (mean over distractor at LP and absent);
- capture = distractor present − absent (target at LP);
- learning: dD and dT in session 2 − session 1, and in the third − first run of each HP block.

Wilcoxon signed-rank, exact, two-sided (no directional prediction for the MTL), n = 28; Benjamini-Hochberg across the
four MTL regions per measure; controls uncorrected. ``rt`` = with the trial-wise RT regressor. Behaviour: RT for
distractor at HP vs at LP.

Usage::

    python -m retsupp.side_hippocampus.stats_region_events --deconv notes/side_hippocampus/data/region_event_deconv.tsv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from retsupp.side_hippocampus.plot_prf_cv import COLOR, LABEL, MTL, REGIONS, style

WINDOW = (3.2, 8.0)


def measures(d):
    w = d[d.t.between(*WINDOW)].groupby(['subject', 'region', 'split', 'rt', 'cls']).response.mean().unstack('cls')
    m = pd.DataFrame(index=w.index)
    m['amplitude'] = w.mean(1)
    m['dD'] = w.dHP_tLP - w.dLP_tLP
    m['dT'] = (w.dLP_tHP + w.dabs_tHP) / 2 - (w.dLP_tLP + w.dabs_tLP) / 2
    m['capture'] = (w.dHP_tLP + w.dLP_tLP) / 2 - w.dabs_tLP
    return m.reset_index()


def wtest(v):
    v = v.dropna()
    return stats.wilcoxon(v, method='exact').pvalue if len(v) else np.nan


def tests(m):
    rows = []
    full = m[m.split == 'all']
    for rt in (False, True):
        for meas in ('amplitude', 'dD', 'dT', 'capture'):
            for r in REGIONS:
                v = full[(full.rt == rt) & (full.region == r)][meas]
                rows.append(dict(test=meas, rt=rt, region=r, n=len(v), mean=v.mean(), sem=v.sem(), p=wtest(v)))
        for meas in ('dD', 'dT'):
            for a, b, name in (('ses2', 'ses1', 'session 2 − 1'), ('pos3', 'pos1', 'run 3 − 1 of block')):
                w = m[m.rt == rt].pivot_table(index=['subject', 'region'], columns='split', values=meas)
                for r in REGIONS:
                    v = (w[b] - w[a]).xs(r, level='region') * -1     # a − b
                    rows.append(dict(test=f'{meas}: {name}', rt=rt, region=r, n=len(v), mean=v.mean(), sem=v.sem(),
                                     p=wtest(v)))
    t = pd.DataFrame(rows)
    t['p_bh_mtl'] = np.nan
    for (test, rt), g in t.groupby(['test', 'rt']):
        ix = g.index[g.region.isin(MTL)]
        t.loc[ix, 'p_bh_mtl'] = multipletests(t.loc[ix, 'p'], method='fdr_bh')[1]
    return t


def behaviour(trials):
    r = trials.dropna(subset=['rt']).groupby(['subject', 'cls']).rt.mean().unstack()
    d = (r.dHP_tLP - r.dLP_tLP) * 1000
    return dict(rt_dHP_ms=r.dHP_tLP.mean() * 1000, rt_dLP_ms=r.dLP_tLP.mean() * 1000, diff_ms=d.mean(),
                sem_ms=d.sem(), p=wtest(d), n=len(d))


def figure(d, m, t, out):
    style()
    fig, axes = plt.subplots(1, 4, figsize=(180 / 25.4, 55 / 25.4), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1, 1]))
    ax = axes[0]
    h = d[(d.region == 'hippocampus') & (d.split == 'all') & (~d.rt)]
    for cls, col, lab in (('dHP_tLP', '#D62828', 'Distractor at HP'), ('dLP_tLP', '#0077B6', 'Distractor at LP'),
                          ('dabs_tLP', '0.5', 'No distractor')):
        g = h[h.cls == cls].groupby('t').response
        mu, se = g.mean(), g.sem()
        ax.fill_between(mu.index, mu - se, mu + se, color=col, alpha=.2, lw=0)
        ax.plot(mu.index, mu, color=col, lw=1.2)
        ax.text(.97, {'dHP_tLP': .25, 'dLP_tLP': .16, 'dabs_tLP': .07}[cls], lab, color=col, transform=ax.transAxes,
                ha='right', va='top', fontsize=6.5)
    ax.axhline(0, color='0.6', lw=.6, ls='--', zorder=0)
    ax.axvspan(*WINDOW, color='0.92', zorder=-1, lw=0)
    ax.set_xlabel('Time from search onset (s)')
    ax.set_ylabel('Hippocampus response (z)')
    ax.set_xticks([0, 5, 10, 15, 20])
    sns.despine(ax=ax, offset=4)
    rng = np.random.default_rng(0)
    full = m[(m.split == 'all') & (~m.rt)]
    w = m[~m.rt].pivot_table(index=['subject', 'region'], columns='split', values='dD')
    panels = [(axes[1], full.set_index('region')['dD'], 'dD', 'Distractor at HP − LP (z)'),
              (axes[2], full.set_index('region')['dT'], 'dT', 'Target at HP − LP (z)'),
              (axes[3], (w.ses2 - w.ses1).reset_index(level='subject', drop=True), 'dD: session 2 − 1',
               'Distractor HP − LP,\nsession 2 − 1 (z)')]
    for ax, series, test, ylab in panels:
        for i, r in enumerate(REGIONS):
            v = series.loc[r].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=5, color=COLOR[r], alpha=.35, lw=0)
            ax.errorbar(i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='D', ms=4, color=COLOR[r], mec='0.15',
                        mew=.8, elinewidth=1, capsize=0, zorder=3)
            row = t[(t.test == test) & (~t.rt) & (t.region == r)].iloc[0]
            if (row.p_bh_mtl if r in MTL else row.p) < .05:
                ax.text(i, 1.0, '*', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
        ax.axhline(0, color='0.6', lw=.6, ls='--', zorder=0)
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels([LABEL[r] for r in REGIONS])
        ax.set_ylabel(ylab)
        sns.despine(ax=ax, offset=4)
        for lab_ in ax.get_xticklabels():
            lab_.set(rotation=40, ha='right', rotation_mode='anchor')
    for ax, letter in zip(axes, 'abcd'):
        ax.text(-.3, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', bbox_inches='tight', pad_inches=.02)


def main(deconv, trials, out_dir):
    out_dir = Path(out_dir)
    d = pd.read_csv(deconv, sep='\t')
    m = measures(d)
    m.to_csv(out_dir / 'data' / 'region_event_measures.tsv', sep='\t', index=False, float_format='%.5g')
    t = tests(m)
    t.to_csv(out_dir / 'data' / 'region_event_tests.tsv', sep='\t', index=False, float_format='%.5g')
    pd.set_option('display.width', 200)
    print(t[~t.rt].round(4).to_string(index=False))
    print(t[t.rt & t.test.isin(['dD', 'dT', 'amplitude'])].round(4).to_string(index=False))
    b = behaviour(pd.read_csv(trials, sep='\t'))
    print('RT', {k: round(v, 4) for k, v in b.items()})
    pd.Series(b).to_csv(out_dir / 'data' / 'region_event_rt.tsv', sep='\t', header=False)
    figure(d, m, t, out_dir / 'fig_region_hp_test')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--deconv', default='notes/side_hippocampus/data/region_event_deconv.tsv.gz')
    ap.add_argument('--trials', default='notes/side_hippocampus/data/region_event_trials.tsv.gz')
    ap.add_argument('--out-dir', default='notes/side_hippocampus')
    a = ap.parse_args()
    main(a.deconv, a.trials, a.out_dir)
