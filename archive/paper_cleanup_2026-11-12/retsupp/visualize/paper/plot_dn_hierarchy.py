"""DN pRF parameters across the visual hierarchy, against Aqil et al. (2021).

Input: the per-voxel table of ``modeling.cv.collect_dn_params``. Voxels: training-run
reliability > 0, positive amplitude, eccentricity within the bar aperture, SD <= 20%.
Per subject × fold × ROI (>= 20 voxels) the median, averaged over folds. The trend is
each subject's Spearman ρ over the ROIs in Aqil's order, tested across subjects.

Note a, b and d share one free scale: multiplying all three leaves the prediction
unchanged, so raw b and d are only defined relative to each other. b/d and φ are not
affected. ``log_ecc_`` values are residuals of the log after regressing out
eccentricity within subject.

Usage::

    python -m retsupp.visualize.paper.plot_dn_hierarchy \\
        --table /data/ds-retsupp/derivatives/prf_cv_roi_dn_params.tsv.gz \\
        --out notes/figures/dn_hierarchy_aqil.pdf --stats notes/data/dn_hierarchy_stats.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 7, 'axes.linewidth': .8,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3,
    'ytick.major.size': 3, 'xtick.major.width': .8, 'ytick.major.width': .8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})
FG, MUTED, PRF = '#1F2933', '#9AA5B1', '#2A9D8F'
ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS']
APERTURE = 3.17
PIXEL_AREA = (10 / 49) ** 2          # resolution 50 on ±5°
MIN_VOXELS = 20
PARS = ['sd', 'srf_size', 'neural_baseline', 'surround_baseline', 'b_over_d', 'a_over_d', 'd_over_c']
PANELS = [('sd', 'σ centre (°)', 'Reported: increase', False),
          ('srf_size', 'Size ratio σ2/σ1', 'Reported: decrease', False),
          ('neural_baseline', 'Activation constant b', 'Reported: decrease', False),
          ('surround_baseline', 'Normalization constant d', 'Reported: decrease', False),
          ('b_over_d', 'b / d', 'Free of the common scale', True)]


def load(table):
    d = pd.read_csv(table, sep='\t')
    d = d[d.roi.isin(ROIS) & (d.sd_train <= 20) & (d.rf_amplitude > 0)].copy()
    d['ecc'] = np.hypot(d.x, d.y)
    gauss = PIXEL_AREA / (d.sd * np.sqrt(2 * np.pi))
    d['a_peak'] = d.rf_amplitude * gauss
    d['b_over_d'] = d.neural_baseline / d.surround_baseline
    d['a_over_d'] = d.a_peak / d.surround_baseline
    d['d_over_c'] = d.surround_baseline / (d.rf_amplitude * d.srf_amplitude * gauss / d.srf_size)
    return d[d.ecc <= APERTURE]


def by_roi(x, par):
    k = x.groupby(['subject', 'fold', 'roi'])[par].agg(['median', 'size'])
    k = k[k['size'] >= MIN_VOXELS]['median']
    return k.groupby(['subject', 'roi']).mean().unstack().reindex(columns=ROIS)


def trend(t):
    rank = np.arange(len(ROIS))
    rho = t.apply(lambda r: spearmanr(rank[r.notna()], r.dropna()).statistic
                  if r.notna().sum() >= 5 else np.nan, axis=1).dropna()
    return dict(rho=rho.median(), pct_negative=(rho < 0).mean() * 100,
                p=wilcoxon(rho).pvalue, n=len(rho))


def ecc_residual(x, par):
    y = np.log(x[par])
    z = x.loc[np.isfinite(y), ['subject', 'fold', 'roi', 'ecc']].copy()
    z['res'] = y[np.isfinite(y)]
    for _, g in z.groupby('subject'):
        X = np.c_[np.ones(len(g)), g.ecc, g.ecc ** 2]
        z.loc[g.index, 'res'] = g.res - X @ np.linalg.lstsq(X, g.res, rcond=None)[0]
    return z


def main(table, out, stats):
    d = load(table)
    selections = {'reliable': lambda x: x[x.reliability > 0],
                  'r2_train_0.3': lambda x: x[x.r2_train > .3]}
    rows = []
    for model in (5, 6):
        for sel, f in selections.items():
            x = f(d[d.model == model])
            for par in PARS:
                t = by_roi(x, par)
                rows.append(dict(model=model, selection=sel, par=par, control='none',
                                 **trend(t), **t.median().to_dict()))
                t = by_roi(ecc_residual(x, par), 'res')
                rows.append(dict(model=model, selection=sel, par=par, control='ecc',
                                 **trend(t), **t.median().to_dict()))
    s = pd.DataFrame(rows)
    if not Path(stats).exists():
        s.to_csv(stats, sep='\t', index=False, float_format='%.4g')

    x = d[(d.model == 6) & (d.reliability > 0)]
    fig, axes = plt.subplots(1, 5, figsize=(7.25, 1.9), constrained_layout=True)
    for ax, (par, label, note, logy) in zip(axes, PANELS):
        t = by_roi(x, par)
        pos = np.arange(len(ROIS))
        for _, r in t.iterrows():
            ax.plot(pos, r.values, color=MUTED, lw=.3, alpha=.5)
        ax.plot(pos, t.median().values, color=PRF, lw=1.6, marker='o', ms=2.5)
        if logy:
            ax.set_yscale('log')
        ax.set_ylim(np.nanpercentile(t.values, 1) * .8, np.nanpercentile(t.values, 99) * 1.2)
        tr = s[(s.model == 6) & (s.selection == 'reliable') & (s.par == par)
               & (s.control == 'none')].iloc[0]
        ax.set_title(f'{note}\nρ = {tr.rho:+.2f}', loc='left', fontsize=6.5)
        ax.set_ylabel(label, labelpad=2)
        ax.set_xticks(pos, ROIS)
        ax.tick_params(axis='x', labelrotation=90)
    sns.despine(fig=fig, offset=3)
    fig.savefig(out)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=220)
    print('Saved', out, stats)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='/data/ds-retsupp/derivatives/prf_cv_roi_dn_params.tsv.gz')
    ap.add_argument('--out', default='notes/figures/dn_hierarchy_aqil.pdf')
    ap.add_argument('--stats', default='notes/data/dn_hierarchy_stats.tsv')
    a = ap.parse_args()
    main(a.table, a.out, a.stats)
