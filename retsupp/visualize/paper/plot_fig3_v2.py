"""Figure 3, second version: does the model fit, and what does each step add?

``--loso-table`` and ``--selection-table`` are the only inputs that depend on gain fits or voxel
selection, so the final version takes one command. Panels whose data do not exist yet are grey placeholders.

Usage::

    python -m retsupp.visualize.paper.plot_fig3_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_COLORS, ROI_ORDER, STIM, TARGET,
                                                 letter, offset_spines, pending, provisional, save)

LEGEND = """
**Figure 3. Model fits.** a, Polar angle and eccentricity of one subject on a flattened cortical surface. b, Share of pRF centres in the contralateral field per ROI, for voxels with cross-run reliability > 0; dots are subjects, bars are medians, the number under each ROI is the number of subjects with at least {min_vox} such voxels (n = 30 in total); dashed line, chance. c, One voxel over {win:.0f} s of one run: BOLD and the m0 and m7 predictions, with target and singleton onsets at the voxel's ring location and at any location as ticks, and the bar position as a strip. d, Held-out R² per ROI for Gaussian, DoG and DN pRFs with a fixed or a per-voxel HRF; voxels selected per fold on training-run reliability; dots are subjects (n = 30), bars are medians, triangles are subjects above the axis. e, Leave-one-subject-out test per ladder step: share of subjects whose held-out prediction improves with the effect size from the other subjects; n = 30, Wilcoxon across subjects on ΔR², median over ROIs. ΔR² (×10⁻⁵; DoG, DN): {dr2}.
"""
FAM_COL = {'Gaussian': '#8D99A6', 'DoG': '#1F6F8B', 'DN': '#A23B72'}
TERM_ORDER = [('items', 'Items'), ('T', 'T'), ('D', 'D'), ('DonT', 'D on T'), ('S', 'S'), ('dS', 'dS'), ('dD', 'dD')]
MIN_VOX = 10


def panel_contra(ax, table, candidate):
    d = pd.read_csv(table, sep='\t')
    d = d[(d.candidate == candidate) & d.roi.isin(ROI_ORDER) & (d.n >= MIN_VOX)]
    ns = []
    for i, roi in enumerate(ROI_ORDER):
        v = d[d.roi == roi].contra.to_numpy()
        ns.append(len(v))
        rng = np.random.default_rng(i)
        ax.scatter(i + rng.uniform(-.2, .2, len(v)), v, s=5, color=ROI_COLORS[roi], alpha=.55, lw=0)
        ax.plot([i - .3, i + .3], [np.median(v)] * 2, color=FG, lw=1.3, solid_capstyle='butt')
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels([f'{r}\n{n}' for r, n in zip(ROI_ORDER, ns)], fontsize=6)
    ax.set_yticks([50, 75, 100])
    ax.set_ylim(45, 102)
    ax.set_ylabel('Contralateral (%)')
    ax.text(-.02, -.21, 'n', transform=ax.transAxes, ha='right', fontsize=6, color=FG)
    offset_spines(ax)
    return ns


def panel_voxel(ax_main, ax_ev, ax_bar, path):
    d = pd.read_csv(path, sep='\t')
    t = d.t.to_numpy()
    ax_main.plot(t, d.bold, color='.55', lw=.7, marker='o', ms=1.6, mew=0, zorder=1)
    ax_main.plot(t, d.m0, color=STIM, lw=1.1, zorder=2)
    ax_main.plot(t, d.m7, color=HP, lw=1.1, zorder=3)
    ax_main.set_xlim(t[0], t[-1])
    ax_main.set_ylabel('BOLD (% signal)')
    ax_main.spines['bottom'].set_visible(False)
    ax_main.set_xticks([])
    y = ax_main.get_ylim()[1]
    for x, lab, col in ((.0, 'Data', '.45'), (.06, 'm0: pRF + items', STIM), (.2, 'm7: + gain fields', HP)):
        ax_main.text(x, 1.0, lab, transform=ax_main.transAxes, fontsize=6.5, color=col, va='bottom')
    r = d.iloc[0]
    ax_main.text(1., 1.0, f'sub-{int(r.subject):02d}, {r.roi}, pRF at ({r.x:.1f}°, {r.y:.1f}°), R² = {r.r2_m7:.2f}', transform=ax_main.transAxes,
                 fontsize=6, color=FG, ha='right', va='bottom')
    rows = [('target_here', TARGET, 'Target here', 3), ('singleton_here', DISTRACTOR, 'Singleton here', 2), ('target_any', '#F5C89A', 'Target anywhere', 1), ('singleton_any', '#A6CFE3', 'Singleton anywhere', 0)]
    for c, col, lab, row in rows:
        on = t[(d[c].to_numpy() > 0)]
        ax_ev.vlines(on, row + .1, row + .9, color=col, lw=1.1)
        ax_ev.text(-.005, (row + .5) / 4, lab, transform=ax_ev.transAxes, ha='right', va='center', fontsize=5.8, color=col if 'here' in c else MUTED)
    ax_ev.set_xlim(t[0], t[-1])
    ax_ev.set_ylim(0, 4)
    ax_ev.set_axis_off()
    pos = d.bar_x.where(np.abs(d.bar_x) >= np.abs(d.bar_y), d.bar_y).to_numpy()
    im = np.ma.masked_invalid(pos)[None]
    ax_bar.imshow(im, aspect='auto', cmap='coolwarm', vmin=-3.2, vmax=3.2, extent=(t[0], t[-1], 0, 1), interpolation='nearest')
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Time in run (s)')
    ax_bar.set_ylabel('Bar', rotation=0, fontsize=6, ha='right', va='center')
    ax_bar.spines['left'].set_visible(False)
    return d


def panel_cv(ax, table):
    d = pd.read_csv(table, sep='\t')
    d = d[(d.selection == 'reliable') & d.roi.isin(ROI_ORDER) & d.model.isin([1, 2, 3, 4, 5, 6])]
    s = d.groupby(['roi', 'subject', 'model']).median_r2.mean().unstack('model')
    groups = [(1, 'Gaussian', False), (3, 'Gaussian', True), (2, 'DoG', False), (4, 'DoG', True), (5, 'DN', False), (6, 'DN', True)]
    w = .135
    lo, hi = 0, .5
    for i, roi in enumerate(ROI_ORDER):
        g = s.loc[roi]
        for k, (m, fam, flex) in enumerate(groups):
            v = g[m].dropna().to_numpy()
            x = i + (k - 2.5) * w
            rng = np.random.default_rng(100 * i + k)
            col = FAM_COL[fam]
            xx = x + rng.uniform(-.035, .035, len(v))
            keep = v <= hi
            ax.scatter(xx[keep], v[keep], s=3, facecolors=col if flex else 'none', edgecolors=col, linewidths=.3, alpha=.45, zorder=1)
            ax.scatter(xx[~keep], np.full((~keep).sum(), hi), s=6, marker='^', color=col, lw=0, alpha=.6, zorder=1, clip_on=False)
            ax.plot([x - .055, x + .055], [np.median(v)] * 2, color=col, lw=1.5, solid_capstyle='butt', zorder=3)
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, fontsize=6)
    ax.set_xlim(-.55, len(ROI_ORDER) - .45)
    ax.set_ylim(lo, hi)
    ax.set_yticks([0, .25, .5])
    ax.set_ylabel('Held-out R²')
    x0 = .01
    for lab, col, filled in (('Gaussian', FAM_COL['Gaussian'], True), ('DoG', FAM_COL['DoG'], True), ('DN', FAM_COL['DN'], True), ('Fixed HRF', FG, False), ('Per-voxel HRF', FG, True)):
        ax.scatter([x0 + .012], [1.05], s=11, facecolors=col if filled else 'none', edgecolors=col, linewidths=.6, transform=ax.transAxes, clip_on=False)
        ax.text(x0 + .03, 1.05, lab, transform=ax.transAxes, fontsize=6.5, color=col, va='center')
        x0 += .055 + .0125 * len(lab)
    ax.text(1., 1.05, 'Pending: HRF per ROI', transform=ax.transAxes, fontsize=6, color=MUTED, ha='right', va='center')
    offset_spines(ax)


def panel_loso(ax, table):
    d = pd.read_csv(table, sep='\t')
    fam = {'dog': ('DoG', -.17), 'dn': ('DN', .17)}
    x = np.arange(len(TERM_ORDER))
    for f, (name, off) in fam.items():
        g = d[d.family == f].set_index('term')
        for i, (t, _) in enumerate(TERM_ORDER):
            if t not in g.index:
                continue
            v = g.loc[t, 'pct_improved']
            ax.bar(i + off, v - 50, bottom=50, width=.32, color=FAM_COL[name], lw=0)
    i = [t for t, _ in TERM_ORDER].index('DonT')
    ax.add_patch(plt.Rectangle((i - .36, 50), .72, 22, facecolor='#ECEFF2', lw=0))
    ax.text(i, 61, 'Pending', ha='center', va='center', fontsize=5.5, color='#6B7785', rotation=90)
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in TERM_ORDER], fontsize=6.5)
    ax.set_ylim(40, 102)
    ax.set_yticks([50, 75, 100])
    ax.set_ylabel('Subjects improved (%)')
    for k, name in enumerate(('DoG', 'DN')):
        col = FAM_COL[name]
        ax.text(.0 + k * .13, 1.05, name, transform=ax.transAxes, fontsize=6.5, color=col, va='center')
    provisional(ax, y=1.02)
    offset_spines(ax)


def main(selection_table, candidate, cv_table, loso_table, voxel_table, flat_png, name):
    fig = plt.figure(figsize=(DOUBLE, 5.3))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.5, 1.45], hspace=.42, left=.075, right=.99, top=.965, bottom=.08)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.75, 1], wspace=.2)
    ax = fig.add_subplot(top[0])
    if flat_png and Path(flat_png).exists():
        ax.imshow(plt.imread(flat_png))
        ax.set_axis_off()
    else:
        pending(ax, 'flatmap of polar angle and eccentricity, needs cortical flattening')
    letter(ax, 'a', x=-.02, y=1.)
    ax = fig.add_subplot(top[1])
    ns = panel_contra(ax, selection_table, candidate)
    letter(ax, 'b', x=-.17, y=.98)
    mid = outer[1].subgridspec(3, 1, height_ratios=[1, .28, .1], hspace=.08)
    d = None
    win = 75
    if voxel_table and Path(voxel_table).exists():
        a1, a2, a3 = (fig.add_subplot(mid[i]) for i in range(3))
        d = panel_voxel(a1, a2, a3, voxel_table)
        provisional(a1, y=1.12)
        win = d.t.max() - d.t.min() + 1.6
        letter(a1, 'c', x=-.05, y=1.)
    else:
        ax = fig.add_subplot(outer[1])
        pending(ax, 'example voxel, running on the cluster')
        letter(ax, 'c', x=-.02, y=1.)
    low = outer[2].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=.22)
    ax = fig.add_subplot(low[0])
    panel_cv(ax, cv_table)
    letter(ax, 'd', x=-.09, y=1.)
    ax = fig.add_subplot(low[1])
    panel_loso(ax, loso_table)
    letter(ax, 'e', x=-.17, y=1.)
    l = pd.read_csv(loso_table, sep='\t').pivot(index='term', columns='family', values='median_dr2_e5')
    dr2 = '; '.join(f'{t} {l.loc[t, "dog"]:.2f}, {l.loc[t, "dn"]:.2f}' for t, _ in TERM_ORDER if t in l.index)
    save(fig, name, LEGEND.format(min_vox=MIN_VOX, win=win, dr2=dr2) + f"\nSubjects per ROI in b ({', '.join(ROI_ORDER)}): {', '.join(map(str, ns))}.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection-table', default='notes/data/voxel_selection_candidates_2026-11.tsv')
    ap.add_argument('--candidate', default='reliable')
    ap.add_argument('--cv-table', default='notes/data/prf_cv_roi_summary.tsv')
    ap.add_argument('--loso-table', default='notes/data/loso_per_step_2026-11.tsv')
    ap.add_argument('--voxel-table', default=None)
    ap.add_argument('--flat-png', default=None)
    ap.add_argument('--name', default='fig3_model_fits_v2')
    a = ap.parse_args()
    main(a.selection_table, a.candidate, a.cv_table, a.loso_table, a.voxel_table, a.flat_png, a.name)
