"""DN pRF parameters by where the pRF lies relative to the HP distractor location (exploratory).

Input: ``notes/data/dn_by_hp_2026-11.tsv.gz`` (``modeling.collect_dn_by_hp``; fits of ``modeling.fit_dn_by_hp``,
one DN fit per HP condition, HRF fixed per subject × ROI, no cross-validation).

Voxels: positive rf_amplitude in the all-runs fit and all four condition fits, all-runs pRF centre within the
bar aperture (3.17°) and at least ``MERIDIAN``° from both meridians, so its quadrant is unambiguous. For each
voxel and HP condition the relation of the voxel's quadrant (all-runs fit) to the HP quadrant is ``HP``,
``Next`` (sharing one side, two conditions, averaged) or ``Opposite``. Each parameter is expressed within
voxel: log ratio to the voxel's geometric mean over the four conditions for positive parameters, the
difference from its mean for bold_baseline and R². Per subject × ROI (>= ``MIN_VOXELS`` voxels): median over
voxels per relation. Tests across subjects per ROI × parameter: Friedman over the three relations and
Wilcoxon HP vs Opposite, BH over ROIs within each parameter.

Second figure (``_distance``): the same within-voxel values against the distance from the all-runs pRF centre
to the HP item (all voxels within the aperture), binned; mean ± SEM over subjects.

Usage::

    python -m retsupp.visualize.paper.figS_dn_params_by_hp --table notes/data/dn_by_hp_2026-11.tsv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from retsupp.visualize.paper.paper_style import (DOUBLE, FG, HP, MUTED, OUT, ROI_GROUPS, ROI_ORDER, ROI_X,
                                                 letter)

APERTURE, MERIDIAN, MIN_VOXELS = 3.17, .25, 10
ECC_HP = 4.0
PIXEL_AREA = (10 / 49) ** 2          # resolution 50 on ±5°
QUAD = {'upper_right': (1, 1), 'upper_left': (-1, 1), 'lower_left': (-1, -1), 'lower_right': (1, -1)}
RELATIONS = ['HP', 'Next', 'Opposite']
REL_COLORS = {'HP': HP, 'Next': '#E9A0A0', 'Opposite': MUTED}
# (column, label, log): log -> within-voxel log ratio, else difference
PARAMS = [('sd', 'σ centre', True), ('a_peak', 'Peak gain a', True), ('srf_amplitude', 'Surround amplitude c', True),
          ('srf_size', 'Size ratio σ2/σ1', True), ('neural_baseline', 'Activation constant b', True),
          ('surround_baseline', 'Normalization constant d', True), ('b_over_d', 'b / d', True),
          ('a_over_d', 'a / d', True), ('bold_baseline', 'BOLD baseline', False), ('r2', 'R² (fitted runs)', False)]
DIST_BINS = [0, 1.5, 3, 4.5, 6, 8]


def load(table):
    d = pd.read_csv(table, sep='\t')
    d = d[d.roi.isin(ROI_ORDER)].copy()
    gauss = PIXEL_AREA / (d.sd * np.sqrt(2 * np.pi))
    d['a_peak'] = d.rf_amplitude * gauss
    d['b_over_d'] = d.neural_baseline / d.surround_baseline
    d['a_over_d'] = d.a_peak / d.surround_baseline
    ref = d[d.condition == 'all'].set_index(['subject', 'voxel_index'])[['x', 'y', 'rf_amplitude']]
    c = d[d.condition != 'all'].join(ref, on=['subject', 'voxel_index'], rsuffix='_all')
    ok = c.groupby(['subject', 'voxel_index']).rf_amplitude.transform(lambda a: (a > 0).all() and len(a) == 4)
    c = c[ok.astype(bool) & (c.rf_amplitude_all > 0) & (np.hypot(c.x_all, c.y_all) <= APERTURE)].copy()
    for col, _, log in PARAMS:
        v = np.log(c[col].where(c[col] > 0)) if log else c[col]
        c[col] = v - v.groupby([c.subject, c.voxel_index]).transform('mean')
    hx = c.condition.map(lambda h: QUAD[h][0])
    hy = c.condition.map(lambda h: QUAD[h][1])
    same = (np.sign(c.x_all) == hx).astype(int) + (np.sign(c.y_all) == hy).astype(int)
    c['relation'] = same.map({2: 'HP', 1: 'Next', 0: 'Opposite'})
    c['clear'] = (c.x_all.abs() >= MERIDIAN) & (c.y_all.abs() >= MERIDIAN)
    c['dist_hp'] = np.hypot(c.x_all - hx * ECC_HP / np.sqrt(2), c.y_all - hy * ECC_HP / np.sqrt(2))
    return c


def per_subject(c, by):
    cols = [p for p, _, _ in PARAMS]
    g = c.groupby(['subject', 'roi', 'voxel_index', by])[cols].mean().reset_index()   # Next: mean of two
    n = g.groupby(['subject', 'roi']).voxel_index.nunique().rename('n').reset_index()
    s = g.groupby(['subject', 'roi', by])[cols].median().reset_index().merge(n, on=['subject', 'roi'])
    return s[s.n >= MIN_VOXELS]


def stats(s):
    rows = []
    for col, _, _ in PARAMS:
        for roi in ROI_ORDER:
            w = s[s.roi == roi].pivot(index='subject', columns='relation', values=col)
            if not set(RELATIONS) <= set(w.columns):
                continue
            w = w[RELATIONS].dropna()
            if len(w) < 6:
                continue
            rows.append(dict(param=col, roi=roi, n=len(w), mean_hp_minus_opp=(w.HP - w.Opposite).mean(),
                             p_friedman=friedmanchisquare(*[w[r] for r in RELATIONS]).pvalue,
                             p_hp_vs_opp=wilcoxon(w.HP, w.Opposite).pvalue,
                             p_hp_vs_next=wilcoxon(w.HP, w.Next).pvalue))
    st = pd.DataFrame(rows)
    for p in ('p_friedman', 'p_hp_vs_opp'):
        st[p + '_bh'] = st.groupby('param')[p].transform(lambda x: multipletests(x, method='fdr_bh')[1])
    return st


def roi_axis(ax):
    ax.set_xticks([ROI_X[r] for r in ROI_ORDER])
    ax.set_xticklabels(ROI_ORDER, rotation=90)
    ax.axhline(0, color=FG, lw=.6, zorder=0)
    ax.set_xlim(-.7, max(ROI_X.values()) + .7)


def plot_relations(s, st, out):
    fig, axes = plt.subplots(4, 3, figsize=(DOUBLE, DOUBLE * 1.15))
    off = {'HP': -.22, 'Next': 0, 'Opposite': .22}
    for i, (ax, (col, label, log)) in enumerate(zip(axes.flat, PARAMS)):
        for rel in RELATIONS:
            m = s[s.relation == rel].groupby('roi')[col].agg(['mean', 'sem'])
            m = m.reindex([r for r in ROI_ORDER if r in m.index])
            x = [ROI_X[r] + off[rel] for r in m.index]
            ax.errorbar(x, m['mean'], m['sem'], fmt='o', ms=2.5, lw=.8, color=REL_COLORS[rel], label=rel)
        roi_axis(ax)
        ax.set_title(label, fontsize=7, color=FG)
        ax.set_ylabel('Log ratio to voxel mean' if log else 'Difference from voxel mean', fontsize=6)
        for _, r in st[(st.param == col) & (st.p_friedman_bh < .05)].iterrows():
            ax.text(ROI_X[r.roi], 1, '*', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
        letter(ax, 'abcdefghijkl'[i])
    for ax in axes.flat[len(PARAMS):]:
        ax.set_axis_off()
    axes.flat[0].legend(frameon=False, fontsize=6, loc='lower left', handletextpad=.2)
    ax = axes.flat[len(PARAMS)]
    ax.text(0, .9, 'Exploratory: one DN fit per HP condition (about 3 runs each), no cross-validation.\n'
            'Pathways: ' + '; '.join(f'{g}: {", ".join(r)}' for g, r in ROI_GROUPS) + '.\n'
            '* Friedman over HP / Next / Opposite, BH over ROIs, p < .05.\n'
            'Mean ± SEM over subjects of per-subject medians over voxels.',
            transform=ax.transAxes, fontsize=6, va='top', color=FG, wrap=True)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out)
    print(f'wrote {out}')


def plot_distance(c, out):
    c = c.copy()
    c['dist_bin'] = pd.cut(c.dist_hp, DIST_BINS, labels=[(a + b) / 2 for a, b in zip(DIST_BINS, DIST_BINS[1:])])
    cols = [p for p, _, _ in PARAMS]
    s = c.groupby(['subject', 'roi', 'dist_bin'], observed=True)[cols].median().reset_index()
    fig, axes = plt.subplots(len(PARAMS), len(ROI_ORDER), figsize=(DOUBLE, DOUBLE * 1.3), sharex=True, sharey='row')
    for i, (col, label, log) in enumerate(PARAMS):
        for j, roi in enumerate(ROI_ORDER):
            ax = axes[i, j]
            m = s[s.roi == roi].groupby('dist_bin', observed=True)[col].agg(['mean', 'sem'])
            ax.errorbar(m.index.astype(float), m['mean'], m['sem'], fmt='o-', ms=2, lw=.8, color=HP)
            ax.axhline(0, color=FG, lw=.5, zorder=0)
            if i == 0:
                ax.set_title(roi, fontsize=7)
            if j == 0:
                ax.set_ylabel(label, fontsize=5.5)
            ax.tick_params(labelsize=5)
    for ax in axes[-1]:
        ax.set_xlabel('Distance to HP (°)', fontsize=6)
    fig.tight_layout(h_pad=.3, w_pad=.2)
    fig.savefig(out)
    print(f'wrote {out}')


def main(table, out, stats_out):
    c = load(table)
    s = per_subject(c[c.clear], 'relation')
    st = stats(s)
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
    st.to_csv(stats_out, sep='\t', index=False, float_format='%.4g')
    print(st[st.p_friedman_bh < .05].to_string(index=False) if (st.p_friedman_bh < .05).any()
          else 'No ROI × parameter with Friedman p_BH < .05')
    plot_relations(s, st, out)
    plot_distance(c, str(out).replace('.pdf', '_distance.pdf'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='notes/data/dn_by_hp_2026-11.tsv.gz')
    ap.add_argument('--out', default=str(OUT / 'figS_dn_params_by_hp.pdf'))
    ap.add_argument('--stats', default='notes/data/dn_by_hp_stats_2026-11.tsv')
    a = ap.parse_args()
    main(a.table, a.out, a.stats)
