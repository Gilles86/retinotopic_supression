"""Side project (hippocampus): group tests and figure for the cross-validated region pRFs.

Input: the per-(subject, region, model) summary of ``collect_prf_cv.py``. Tests, n = 28, Wilcoxon signed-rank
(exact), one-sided in the direction a pRF predicts, Benjamini-Hochberg across the four medial-temporal regions:

- cvR² true − scramble (median over the region's voxels) > 0;
- fraction of voxels above the scramble 95th percentile > 0.05;
- contralateral fraction among those voxels > 0.5;
- fold-to-fold agreement of the pRF x-coordinate among those voxels > 0.

Usage::

    python -m retsupp.side_hippocampus.plot_prf_cv --summary notes/side_hippocampus/data/prf_cv_summary.tsv \\
        --out-dir notes/side_hippocampus
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

REGIONS = ['hippocampus', 'entorhinal', 'parahippocampal', 'amygdala', 'precentral', 'lateral_occipital']
MTL = REGIONS[:4]
LABEL = {'hippocampus': 'Hippocampus', 'entorhinal': 'Entorhinal', 'parahippocampal': 'Parahippocampal',
         'amygdala': 'Amygdala', 'precentral': 'Precentral', 'lateral_occipital': 'Lateral occipital'}
COLOR = {r: '#3B5BA5' for r in MTL} | {'precentral': '#7F7F7F', 'lateral_occipital': '#2A9D8F'}
TESTS = [('d_cvr2', 0, 'cvR² true − scramble'), ('frac_above_thr', .05, 'Fraction above scramble 95th pct'),
         ('contra_prf', .5, 'Contralateral fraction'), ('x_agree', 0, 'Fold-to-fold r of x')]


def style():
    mpl.rcParams.update({
        'font.family': 'Helvetica', 'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'], 'font.size': 7,
        'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': .8,
        'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': .8, 'ytick.major.width': .8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300})


def run_tests(s):
    rows = []
    for col, null, lab in TESTS:
        for r in REGIONS:
            v = (s.loc[s.region == r, col] - null).dropna()
            p = stats.wilcoxon(v, alternative='greater', method='exact' if len(v) <= 50 else 'auto').pvalue if len(v) else np.nan
            rows.append(dict(measure=col, label=lab, region=r, n=len(v), mean=v.mean() + null, sem=v.sem(),
                             median=v.median() + null, n_above_null=int((v > 0).sum()), p_one_sided=p))
    t = pd.DataFrame(rows)
    t['p_bh_mtl'] = np.nan
    for col in t.measure.unique():
        ix = t.index[(t.measure == col) & t.region.isin(MTL)]
        t.loc[ix, 'p_bh_mtl'] = multipletests(t.loc[ix, 'p_one_sided'], method='fdr_bh')[1]
    return t


def main(summary, out_dir, model='dog_roihrf'):
    s = pd.read_csv(summary, sep='\t')
    s = s[s.model == model].copy()
    s['d_cvr2'] = s.median_cvr2_true - s.median_cvr2_scramble
    out_dir = Path(out_dir)
    (out_dir / 'data').mkdir(parents=True, exist_ok=True)
    t = run_tests(s)
    t.to_csv(out_dir / 'data' / f'prf_cv_tests_{model}.tsv', sep='\t', index=False)
    print(t.round(4).to_string(index=False))

    style()
    fig, axes = plt.subplots(1, 4, figsize=(180 / 25.4, 52 / 25.4), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (col, null, lab) in zip(axes, TESTS):
        for i, r in enumerate(REGIONS):
            v = s.loc[s.region == r, col].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=5, color=COLOR[r], alpha=.35, lw=0)
            ax.errorbar(i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='D', ms=4, color=COLOR[r], mec='0.15',
                        mew=.8, elinewidth=1, capsize=0, zorder=3)
            p = t.loc[(t.measure == col) & (t.region == r), 'p_bh_mtl' if r in MTL else 'p_one_sided'].iloc[0]
            if p < .05:
                ax.text(i, 1.0, '*', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
        ax.axhline(null, color='0.6', lw=.6, ls='--', zorder=0)
        ax.set_xticks(range(len(REGIONS)), [LABEL[r] for r in REGIONS], rotation=40, ha='right')
        ax.set_ylabel(lab)
        sns.despine(ax=ax, offset=4, trim=False)
        if col == 'd_cvr2':
            ax.set_yscale('symlog', linthresh=.002)
    axes[0].text(.03, .97, 'MTL', color=COLOR['hippocampus'], transform=axes[0].transAxes, va='top', fontsize=6.5)
    axes[0].text(.03, .89, 'Negative control', color=COLOR['precentral'], transform=axes[0].transAxes, va='top',
                 fontsize=6.5)
    axes[0].text(.03, .81, 'Visual control', color=COLOR['lateral_occipital'], transform=axes[0].transAxes, va='top',
                 fontsize=6.5)
    for ax, letter in zip(axes, 'abcd'):
        ax.text(-.32, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'fig_prf_evidence_{model}.{ext}', bbox_inches='tight', pad_inches=.02)
    print('wrote', out_dir / f'fig_prf_evidence_{model}.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out-dir', default='notes/side_hippocampus')
    ap.add_argument('--model', default='dog_roihrf')
    a = ap.parse_args()
    main(a.summary, a.out_dir, a.model)
