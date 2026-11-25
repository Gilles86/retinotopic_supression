"""Overview of dS (sustained gain at HP minus LP) across every fit version and variant we have.

One row per fit; left, the pooled effect (per participant the median over the nine areas; median and bootstrapped 95% CI,
Wilcoxon p); right, per area the median over participants, coloured, with BH q over the nine areas (Wilcoxon); per pathway
(per participant the median over its areas), BH over the three. n = 28
(sub-01 and sub-02 left out) everywhere. Reads only the existing parameter tables in notes/data.

Usage::

    python -m retsupp.visualize.paper.figS_dS_overview
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from retsupp.visualize.paper.fig45_common import fdr, median_ci
from retsupp.visualize.paper.paper_style import ROI_GROUPS, ROI_ORDER

PW = [g for g, _ in ROI_GROUPS]

D = Path('notes/data')
OUT = Path('notes/figures/paper_2026-11')

# (group, label, kind, file, family, model or mech)
VERSIONS = [
    ('Estimated alone (pRFs held)', 'Prototype fits (6 Nov)', 'cond', 'conditional_estimates_dog_2026-11.tsv', None, 'dS'),
    ('Estimated alone (pRFs held)', 'Final DoG fits (10 Nov, update 8)', 'cond', 'conditional_estimates_dog_final2_2026-11.tsv', None, 'dS'),
    ('Estimated alone (pRFs held)', 'Refitted pRFs (12 Nov, updates 9-11; manuscript)', 'cond', 'conditional_estimates_polished_2026-11-12.tsv', None, 'dS'),
    ('Estimated alone (pRFs held)', 'Refitted, polished further (13 Nov, not converged)', 'cond', 'conditional_estimates_polished_conv_2026-11-13.tsv', None, 'dS'),
    ('Estimated alone (pRFs held)', 'Refitted, voxels r > 0.2', 'cond', 'conditional_estimates_dS_relnull_r0.2_polished_2026-11-13.tsv', None, 'dS'),
    ('Estimated alone (pRFs held)', 'Refitted, voxels r > 0.1', 'cond', 'conditional_estimates_dS_relnull_r0.1_polished_2026-11-13.tsv', None, 'dS'),
    ('Estimated alone (pRFs held)', 'Zero-sum (16 Nov, update 13)', 'cond', 'conditional_estimates_zs_lm_2026-11-13.tsv', None, 'dS'),
    ('Estimated alone (pRFs held)', 'Gaussian pRFs (update 8 fits; manuscript)', 'exact', 'conditional_exact_final2_allfam_2026-11.tsv', 'gauss', 'dS'),
    ('Estimated alone (pRFs held)', 'DN pRFs (update 8 fits; manuscript)', 'exact', 'conditional_exact_final2_allfam_2026-11.tsv', 'dn', 'dS'),
    ('Joint fit', 'Final DoG fits, with dD (update 8)', 'par', 'gain_field_parameters_final2_2026-11.tsv', 'dog', 'm7_full'),
    ('Joint fit', 'Refitted pRFs, with dD (updates 9-11; manuscript)', 'par', 'gain_field_parameters_polished_2026-11-12.tsv', 'dog', 'm7_full'),
    ('Joint fit', 'Refitted pRFs, no dD', 'par', 'gain_field_parameters_polished_2026-11-12.tsv', 'dog', 'm5_TDS_dS'),
    ('Joint fit', 'Refitted, with dD, voxels r > 0.2', 'vox', 'gain_field_parameters_voxsel_relnull_2026-11-13.tsv', 'r0.2', 'm7_full'),
    ('Joint fit', 'Refitted, with dD, voxels r > 0.1', 'vox', 'gain_field_parameters_voxsel_relnull_2026-11-13.tsv', 'r0.1', 'm7_full'),
    ('Joint fit', 'LP split next/opposite (update 8 fits)', 'par', 'gain_field_parameters_zs_lm_2026-11-13.tsv', 'dog', 'm8_full_adjopp'),
    ('Joint fit', 'Zero-sum, no dD (update 13 main model)', 'par', 'gain_field_parameters_zs_lm_2026-11-13.tsv', 'dog', 'm5_TDS_dS_zs'),
    ('Joint fit', 'Zero-sum, with dD', 'par', 'gain_field_parameters_zs_lm_2026-11-13.tsv', 'dog', 'm7_full_zs'),
    ('Joint fit', 'Gaussian pRFs, with dD', 'par', 'gain_field_parameters_zs_lm_2026-11-13.tsv', 'gauss', 'm7_full'),
    ('Joint fit', 'DN pRFs, with dD', 'par', 'gain_field_parameters_zs_lm_2026-11-13.tsv', 'dn', 'm7_full'),
]


def load(kind, file, family, key):
    d = pd.read_csv(D / file, sep='\t')
    if kind == 'cond':
        d = d[(d.mech == key) & d.valid].rename(columns={'estimate': 'dS'})
    elif kind == 'exact':   # exact ΔSSE minimum, areas with >= 20 voxels (as in the key numbers)
        d = d[(d.family == family) & (d.mech == key) & (d.nvox >= 20)].rename(columns={'own': 'dS'})
    else:
        d = d[(d.family == family) & (d.model == key)].copy()
        d['dS'] = d.g_HP - d.g_LP
    d = d[d.roi.isin(ROI_ORDER) & (d.subject > 2)]
    return d[['subject', 'roi', 'dS']].dropna().assign(dS=lambda x: x.dS * 100)


def summarise():
    rows = []
    for group, label, kind, file, family, key in VERSIONS:
        d = load(kind, file, family, key)
        pooled = d.groupby('subject').dS.median()
        m, lo, hi = median_ci(pooled)
        row = dict(group=group, label=label, n=len(pooled), pooled=m, lo=lo, hi=hi, k_neg=int((pooled < 0).sum()),
                   p=stats.wilcoxon(pooled).pvalue)
        ps = []
        for r in ROI_ORDER:
            v = d[d.roi == r].dS
            row[r] = v.median()
            ps.append(stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan)
        for r, q in zip(ROI_ORDER, fdr(ps)):
            row[f'q_{r}'] = q
        pps = []
        for g, rois in ROI_GROUPS:
            v = d[d.roi.isin(rois)].groupby('subject').dS.median()
            row[g] = v.median()
            pps.append(stats.wilcoxon(v).pvalue)
        for g, q in zip(PW, fdr(pps)):
            row[f'q_{g}'] = q
        row['n_sig_neg'] = sum((row[f'q_{r}'] < .05) and (row[r] < 0) for r in ROI_ORDER)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_p(p):
    if p < .001:
        e = int(np.floor(np.log10(p)))
        return f'{p / 10 ** e:.1f}e{e}'
    return f'{p:.3f}'.lstrip('0') if p < .01 else f'{p:.2f}'.lstrip('0')


def plot(t, out):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'], 'font.size': 6.5})
    n = len(t)
    groups = t.group.tolist()
    # y positions with a gap between the two groups
    y, cur = [], 0.
    for i, g in enumerate(groups):
        if i and g != groups[i - 1]:
            cur += .8
        y.append(cur)
        cur += 1
    y = np.array(y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 0.21 * n + 1.3), gridspec_kw=dict(width_ratios=[1, 1.9], wspace=.05))

    # Left: pooled median and CI
    for yi, (_, r) in zip(y, t.iterrows()):
        sig = r.p < .05
        col = '#5B2A86' if sig else '#9AA5B1'
        ax1.plot([r.lo, r.hi], [yi, yi], color=col, lw=1.2, solid_capstyle='round')
        ax1.plot(r.pooled, yi, 'o', ms=4, color=col, mec='w', mew=.5)
        ax1.text(1.2, yi, f'{r.pooled:+.1f}%, {r.k_neg}/{r.n}, p = {fmt_p(r.p)}'.replace('-', '−'),
                 va='center', ha='left', fontsize=5.5, color=col)
    ax1.axvline(0, color='#1F2933', lw=.6)
    ax1.set_xlim(-12, 10.5)
    ax1.spines['bottom'].set_bounds(-12, 0)
    ax1.set_xticks([-10, -5, 0])
    ax1.set_yticks(y)
    ax1.set_yticklabels(t.label, fontsize=6)
    ax1.set_ylim(y.max() + .7, -2.4)
    ax1.set_xlabel('Pooled dS (% of drive), median and 95% CI')
    for s in ('top', 'right', 'left'):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(axis='y', length=0)
    for g in dict.fromkeys(groups):
        yi = y[[i for i, x in enumerate(groups) if x == g][0]] - .9
        ax1.annotate(g, (0, yi), xycoords=('axes fraction', 'data'), xytext=(-2, 0), textcoords='offset points',
                     ha='right', va='center', fontsize=6.5, fontweight='bold')

    # Right: per-area medians, q < .05 outlined and starred
    cols = ROI_ORDER + PW
    vals = t[cols].to_numpy(float)
    lim = np.nanpercentile(np.abs(vals), 95)
    norm = TwoSlopeNorm(0, -lim, lim)
    cmap = plt.get_cmap('PuOr_r')  # purple = lower at HP
    xs = np.r_[np.arange(len(ROI_ORDER)), np.arange(len(PW)) + len(ROI_ORDER) + .5] + 1.6
    for yi, (_, r) in zip(y, t.iterrows()):
        for xi, roi in zip(xs, cols):
            v, q = r[roi], r[f'q_{roi}']
            ax2.add_patch(plt.Rectangle((xi - .45, yi - .42), .9, .84, color=cmap(norm(v)), lw=0))
            tc = 'w' if abs(norm(v) - .5) > .33 else '#1F2933'
            ax2.text(xi, yi + (-.1 if q < .05 else 0), f'{v:+.1f}'.replace('-', '−'), ha='center', va='center',
                     fontsize=5, color=tc, fontweight='bold' if q < .05 else 'normal')
            if q < .05:
                ax2.add_patch(plt.Rectangle((xi - .45, yi - .42), .9, .84, fill=False, ec='#1F2933', lw=.9))
                ax2.text(xi, yi + .3, '*' if q >= .01 else '**' if q >= .001 else '***', ha='center', va='center',
                         fontsize=5, color=tc)
        ax2.text(xs[-1] + 1.0, yi, f"{int(r.n_sig_neg)}", ha='center', va='center', fontsize=6,
                 fontweight='bold' if r.n_sig_neg else 'normal', color='#5B2A86' if r.n_sig_neg else '#9AA5B1')
    ax2.text(xs[-1] + 1.1, -2.0, 'Areas\nq < .05', ha='center', va='center', fontsize=5.5)
    for xi, roi in zip(xs, ROI_ORDER + ['Early', 'Dorsal', 'Occ.-\ntemporal']):
        ax2.text(xi, -1.0, roi, ha='center', va='center', fontsize=6 if len(roi) < 6 else 5)
    ax2.text(xs[len(ROI_ORDER) + 1], -2.0, 'Pathways (BH over three)', ha='center', va='center', fontsize=5.5)
    ax2.text(xs[4], -2.0, 'Areas (BH over nine)', ha='center', va='center', fontsize=5.5)
    ax2.set_xlim(xs[0] - .6, xs[-1] + 1.6)
    ax2.set_ylim(y.max() + .7, -2.4)
    ax2.axis('off')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([.62, .035, .2, .012])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Median dS (%); outlined, BH q < .05', fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    fig.subplots_adjust(left=.27, right=.99, top=.97, bottom=.1)
    for ext in ('pdf', 'png'):
        fig.savefig(out.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)


def main():
    t = summarise()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / 'figS_dS_overview_n28'
    t.to_csv(D / 'dS_overview_n28.tsv', sep='\t', index=False)
    plot(t, out)
    print(t[['label', 'pooled', 'p', 'n_sig_neg'] + PW + [f'q_{g}' for g in PW]].round(4).to_string())
    print(out)


if __name__ == '__main__':
    main()
