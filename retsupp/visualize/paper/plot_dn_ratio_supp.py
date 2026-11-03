"""Supplementary figure: DN parameters and their scale-free ratios across the hierarchy.

a, b and d share one free scale, so the top row (as fitted) depends on it for b and d;
the bottom row does not. a and c are in peak units (a Gaussian of height 1), so the ratios
do not depend on the grid or on σ. Selection and tests: ``plot_dn_hierarchy``.

Usage::

    python -m retsupp.visualize.paper.plot_dn_ratio_supp \\
        --out notes/figures/dn_hierarchy_ratios_supp.pdf \\
        --stats notes/data/dn_hierarchy_ratios_stats.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon

from retsupp.visualize.paper.plot_dn_hierarchy import (
    FG, MUTED, PRF, ROIS, by_roi, ecc_residual, load,
)

HP_RED = '#D62828'
ROWS = [[('sd', 'Centre σ (°)', False), ('srf_size', 'Size ratio σ2/σ1', False),
         ('neural_baseline', 'b, as fitted', False), ('surround_baseline', 'd, as fitted', False)],
        [('b_over_d', 'b/d', True), ('d_over_c', 'd/c', True),
         ('b_over_a', 'b/a', True), ('a_over_d', 'a/d', True)]]
FOREST = [('sd', 'Centre σ'), ('srf_size', 'σ2/σ1'), ('neural_baseline', 'b'),
          ('surround_baseline', 'd'), ('b_over_d', 'b/d'), ('d_over_c', 'd/c'),
          ('b_over_a', 'b/a'), ('a_over_d', 'a/d')]


def subject_rho(t):
    rank = np.arange(len(ROIS))
    return t.apply(lambda r: spearmanr(rank[r.notna()], r.dropna()).statistic
                   if r.notna().sum() >= 5 else np.nan, axis=1).dropna()


def main(table, out, stats):
    d = load(table)
    d['b_over_a'] = d.neural_baseline / d.a_peak
    x = {m: d[(d.model == m) & (d.reliability > 0)] for m in (5, 6)}
    rows, rhos = [], {}
    for m, xm in x.items():
        for par, _ in FOREST:
            for control in ('none', 'ecc'):
                t = by_roi(xm, par) if control == 'none' else by_roi(ecc_residual(xm, par), 'res')
                r = subject_rho(t)
                rhos[m, par, control] = r
                rows.append(dict(model=m, par=par, control=control, rho=r.median(),
                                 q25=r.quantile(.25), q75=r.quantile(.75),
                                 pct_negative=(r < 0).mean() * 100, p=wilcoxon(r).pvalue,
                                 n=len(r), **t.median().to_dict()))
    s = pd.DataFrame(rows)
    if not Path(stats).exists():
        s.to_csv(stats, sep='\t', index=False, float_format='%.4g')

    fig = plt.figure(figsize=(7.25, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.15])
    pos = np.arange(len(ROIS))
    for i, row in enumerate(ROWS):
        for j, (par, label, logy) in enumerate(row):
            ax = fig.add_subplot(gs[i, j])
            t = by_roi(x[6], par)
            for _, r in t.iterrows():
                ax.plot(pos, r.values, color=MUTED, lw=.3, alpha=.5)
            ax.plot(pos, t.median().values, color=PRF, lw=1.6, marker='o', ms=2.5)
            if logy:
                ax.set_yscale('log')
            lo, hi = np.nanpercentile(t.values, [1, 99])
            ax.set_ylim(lo * .8, hi * 1.2)
            ax.set_title(label, loc='left', color=FG)
            ax.set_xticks(pos, ROIS if i == 1 else [])
            ax.tick_params(axis='x', labelrotation=90)
            sns.despine(ax=ax, offset=3)

    ax = fig.add_subplot(gs[:, 4])
    ys = np.arange(len(FOREST))[::-1]
    ax.plot([0, 0], [-.5, len(FOREST) - .5], color=FG, lw=.8)
    for y, (par, _) in zip(ys, FOREST):
        for k, (m, control, col, mfc, dy) in enumerate(
                [(6, 'none', PRF, PRF, .2), (6, 'ecc', PRF, 'white', 0), (5, 'none', MUTED, MUTED, -.2)]):
            r = rhos[m, par, control]
            ax.plot([r.quantile(.25), r.quantile(.75)], [y + dy] * 2, color=col, lw=.9)
            ax.plot(r.median(), y + dy, 'o', ms=3, color=col, mfc=mfc, mew=.8)
    ax.set_yticks(ys, [lab for _, lab in FOREST])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel('Trend across hierarchy (ρ)')
    ax.set_title('Up or down the hierarchy', loc='left', color=FG)
    for k, (lab, col, mfc) in enumerate([('m6', PRF, PRF), ('m6, eccentricity controlled', PRF, 'white'),
                                         ('m5', MUTED, MUTED)]):
        ax.plot(-.95, -1.0 - .42 * k, 'o', ms=3, color=col, mfc=mfc, mew=.8, clip_on=False)
        ax.text(-.82, -1.0 - .42 * k, lab, color=col, va='center', fontsize=6.5)
    ax.set_ylim(-2.2, len(FOREST) - .4)
    sns.despine(ax=ax, offset=3)
    ax.spines['left'].set_bounds(0, len(FOREST) - 1)
    fig.savefig(out)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=220)
    print('Saved', out, stats)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='/data/ds-retsupp/derivatives/prf_cv_roi_dn_params.tsv.gz')
    ap.add_argument('--out', default='notes/figures/dn_hierarchy_ratios_supp.pdf')
    ap.add_argument('--stats', default='notes/data/dn_hierarchy_ratios_stats.tsv')
    a = ap.parse_args()
    main(a.table, a.out, a.stats)
