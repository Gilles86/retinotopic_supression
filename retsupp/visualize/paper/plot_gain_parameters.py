"""Fitted gain-field parameters across the hierarchy, DoG and DN pRFs (m7_full).

Each subject x ROI has one value per shared parameter. dS = g_HP - g_LP and
dD = g_HP_dyn - g_LP_dyn. Wilcoxon against 0 across subjects, per ROI, FDR over ROIs;
the ``all`` row takes each subject's median over ROIs first.

Usage::

    python -m retsupp.visualize.paper.plot_gain_parameters
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import false_discovery_control, spearmanr, wilcoxon

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'axes.linewidth': .8, 'xtick.labelsize': 6, 'ytick.labelsize': 7, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': .8,
    'ytick.major.width': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})
ROIS = ['V1', 'V2', 'V3', 'hV4', 'VO', 'LO', 'TO', 'V3AB', 'IPS', 'SPL1', 'FEF']
FG, MUTED, HP, TARGET, AF = '#1F2933', '#9AA5B1', '#D62828', '#F77F00', '#0077B6'
FAMILIES = {'dog': 'DoG', 'dn': 'DN'}
BOUNDS = {'g_HP': (-10, 10), 'g_LP': (-10, 10), 'g_HP_dyn': (-10, 10), 'g_LP_dyn': (-10, 10),
          'g_T_dyn': (-10, 10), 'sigma_AF': (.5, 8), 'sigma_dyn': (.5, 8), 'item_scale': (1e-3, 50)}
# title, parameters with colour and label
PANELS = [('Sustained gain', [('g_HP', HP, 'HP'), ('g_LP', MUTED, 'LP')]),
          ('dS, HP − LP', [('dS', HP, None)]),
          ('Distractor gain', [('g_HP_dyn', AF, 'HP'), ('g_LP_dyn', MUTED, 'LP')]),
          ('dD, HP − LP', [('dD', AF, None)]),
          ('Target gain', [('g_T_dyn', TARGET, None)]),
          ('Width σ (°)', [('sigma_AF', HP, 'Sustained'), ('sigma_dyn', AF, 'Phasic')])]
TESTED = ['g_HP', 'g_LP', 'dS', 'g_HP_dyn', 'g_LP_dyn', 'dD', 'g_T_dyn']


def load(table, model):
    d = pd.read_csv(table, sep='\t')
    d = d[(d.model == model) & d.roi.isin(ROIS)].copy()
    d['dS'] = d.g_HP - d.g_LP
    d['dD'] = d.g_HP_dyn - d.g_LP_dyn
    return d


def test(x):
    x = x.dropna()
    return dict(median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75),
                pct_negative=(x < 0).mean() * 100, p=wilcoxon(x).pvalue, n=len(x))


def statistics(d):
    rows = []
    for fam, df in d.groupby('family'):
        for par in TESTED:
            by_roi = [dict(roi=r, **test(g[par])) for r, g in df.groupby('roi')]
            q = false_discovery_control([r['p'] for r in by_roi])
            rows += [dict(family=fam, par=par, p_fdr=qi, **r) for r, qi in zip(by_roi, q)]
            rows.append(dict(family=fam, par=par, roi='all', p_fdr=np.nan,
                             **test(df.groupby('subject')[par].median())))
    return pd.DataFrame(rows)


def main(table, model, out, stats_out):
    d = load(table, model)
    s = statistics(d)
    s.to_csv(stats_out, sep='\t', index=False, float_format='%.4g')
    fig, axes = plt.subplots(2, len(PANELS), figsize=(7.25, 3.5), sharex=True, constrained_layout=True)
    for i, (fam, name) in enumerate(FAMILIES.items()):
        df = d[d.family == fam]
        for j, (title, pars) in enumerate(PANELS):
            ax = axes[i, j]
            if j < 5:
                ax.axhline(0, color=FG, lw=.8)
            for k, (par, col, lab) in enumerate(pars):
                sns.pointplot(df, x='roi', y=par, order=ROIS, estimator='median', errorbar=('pi', 50),
                              color=col, ax=ax, markersize=2.2, linewidth=.9, linestyle='none',
                              dodge=False, native_scale=False)
                if lab and i == 0:
                    ax.text(.04 + .55 * k if j == 5 else .04, .02 if j == 5 else .02 + .1 * k, lab,
                            color=col, transform=ax.transAxes, va='bottom', fontsize=6.5)
                if par in ('dS', 'dD'):
                    t = s[(s.family == fam) & (s.par == par)].set_index('roi')
                    top = df.groupby('roi')[par].quantile(.75)
                    for x, r in enumerate(ROIS):
                        if t.loc[r, 'p_fdr'] < .05:
                            ax.text(x, top[r], '*', ha='center', va='bottom', color=FG, fontsize=8)
            if j == 5:
                ax.set_ylim(-1.2, 8.5)
                ax.set_yticks([2, 4, 6, 8])
            ax.set_xlabel('')
            ax.set_ylabel(name if j == 0 else '', color=FG)
            if i == 0:
                ax.set_title(title, loc='left', color=FG)
            ax.tick_params(axis='x', labelrotation=90)
            sns.despine(ax=ax, offset=3)
    fig.savefig(out)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=220)

    print(s[s.roi == 'all'].round(4).to_string(index=False))
    print(s[s.par.isin(['dS', 'dD'])].pivot_table(index=['par', 'roi'], columns='family',
                                                  values=['median', 'pct_negative', 'p_fdr']).round(3).to_string())
    x = d.pivot_table(index=['subject', 'roi'], columns='family', values=['dS', 'dD', 'g_T_dyn', 'g_HP', 'sigma_AF', 'sigma_dyn', 'item_scale'])
    for par in x.columns.levels[0]:
        u = x[par].dropna()
        slope = (u.dn.abs().median() / u.dog.abs().median())
        print(f'{par:10s} DoG vs DN across subject x ROI: rho={spearmanr(u.dog, u.dn).statistic:+.2f}  median |DN|/|DoG|={slope:.2f}')
    for par, (lo, hi) in BOUNDS.items():
        for fam, df in d.groupby('family'):
            print(f'{par:10s} {fam:4s} at bounds {((df[par] <= lo * 1.001) | (df[par] >= hi * .999)).mean() * 100:5.1f}%')
    print('Saved', out, stats_out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--model', default='m7_full')
    ap.add_argument('--out', default='notes/figures/gain_parameters_m7.pdf')
    ap.add_argument('--stats', default='notes/data/gain_field_parameter_stats.tsv')
    a = ap.parse_args()
    main(a.table, a.model, a.out, a.stats)
