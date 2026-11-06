"""Figure 3: pRF maps, held-out R², sanity checks, an example voxel and LOSO per ladder step.

Panel b reads ``--cv-table``, c the ``fig1_prf_*`` tables of ``plot_fig1 collect``. Panels d and e
depend on gain fits and are placeholders until ``--example-voxel`` and ``--loso-table`` are given.

Usage::

    python -m retsupp.visualize.paper.fig3_model_fits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from retsupp.visualize.paper.paper_style import (DOUBLE, FG, MUTED, ROI_COLORS, ROI_ORDER, ROI_X, letter, offset_spines, pending, provisional,
                                                 roi_axis, roi_positions, save)
from retsupp.visualize.paper.plot_fig3_v2 import panel_voxel
from retsupp.visualize.paper.plot_fig1 import COVER_TSV, SIGMA_TSV, contralateral

LEGEND = """
**Figure 3. Model fits.** a, Visual areas, polar angle, eccentricity and R² of the pRF fit on the inflated surface for sub-18, sub-19 and sub-23. b, Held-out R² per ROI of the DoG pRF with the canonical HRF, the model under the gain fields; voxels selected per fold on training-run reliability; dots are subjects (n = 30), bars are medians. Gaussian and DN pRFs and the HRF choices: supplementary figure. c, pRF size against eccentricity (mean ± SEM across subjects), and the share of pRF centres in the contralateral field, voxels with R² ≥ .1 inside the bar aperture. d, One voxel over {win:.0f} s of one run: BOLD and the m0 and m7 predictions, with target and distractor onsets at the voxel's ring location and anywhere, and the bar position as a strip. e, Leave-one-subject-out test per ladder step (DoG pRFs): share of subjects whose held-out prediction improves with the effect size estimated from the other subjects; n = 30, median ΔR² over ROIs; Wilcoxon across subjects. ΔR² (×10⁻⁵): {dr2}. Provisional: current inputs.
"""
TERM_ORDER = [('items', 'Items'), ('T', 'Target'), ('D', 'Distractor'), ('DonT', 'Distractor\ngiven target'), ('S', 'Sustained'),
              ('dS', 'dS'), ('dD', 'dD')]
CV_MODEL = 2   # DoG, canonical HRF: the pRF model of the gain-field ladder

MODEL_STYLE = {1: ('Gaussian', '#9AA5B1', ':'), 2: ('DoG', '#9AA5B1', '--'), 5: ('DN', '#9AA5B1', '-'),
               3: ('Gaussian', '#1F2933', ':'), 4: ('DoG', '#1F2933', '--'), 6: ('DN', '#1F2933', '-')}


def map_cell(ax, path, ref_paths):
    ims = [Image.open(p).convert('RGBA') for p in ref_paths]
    boxes = np.array([im.getbbox() for im in ims])
    box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    im = Image.open(path).convert('RGBA').crop(box)
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    ax.imshow(np.asarray(bg.convert('RGB')), interpolation='none')
    ax.set_axis_off()


def panel_maps(fig, gs, in_dir, subjects, view):
    pars = [('areas', 'Visual areas'), ('angle', 'Polar angle'), ('ecc', 'Eccentricity (°)'), ('r2', 'R²')]
    sub = gs.subgridspec(len(subjects), 4, wspace=.02, hspace=.02)
    first = None
    for i, s in enumerate(subjects):
        refs = [Path(in_dir) / f'sub-{s:02d}_{p}_{view}.png' for p, _ in pars]
        for j, (p, title) in enumerate(pars):
            ax = fig.add_subplot(sub[i, j])
            first = first or ax
            map_cell(ax, refs[j], refs)
            if i == 0:
                ax.set_title(title, fontsize=7, color=FG, pad=2)
            if j == 0:
                ax.text(-.02, .5, f'sub-{s:02d}', transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6.5, color=FG)
    return first


def panel_cv(ax, cv_table, hi=.6):
    d = pd.read_csv(cv_table, sep='\t')
    d = d[(d.selection == 'reliable') & d.roi.isin(ROI_ORDER) & (d.model == CV_MODEL)]
    s = d.groupby(['roi', 'subject']).median_r2.mean()
    for i, roi in enumerate(ROI_ORDER):
        v = s.loc[roi].to_numpy()
        rng = np.random.default_rng(i)
        xx = ROI_X[roi] + rng.uniform(-.2, .2, len(v))
        ax.scatter(xx[v <= hi], v[v <= hi], s=4, color=ROI_COLORS[roi], alpha=.55, lw=0)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=7, marker='^', color=ROI_COLORS[roi], lw=0, clip_on=False)
        ax.plot([ROI_X[roi] - .3, ROI_X[roi] + .3], [np.median(v)] * 2, color=FG, lw=1.3, solid_capstyle='butt')
    ax.set_ylabel('Held-out R²')
    ax.set_ylim(0, hi)
    ax.set_yticks([0, .2, .4, .6])
    offset_spines(ax)
    roi_axis(ax)


def panel_sigma(ax):
    d = pd.read_csv(SIGMA_TSV, sep='\t')
    d = d[d.n >= 5]
    for roi in ['V1', 'V2', 'V3', 'hV4', 'V3AB']:
        g = d[d.roi == roi].pivot(index='subject', columns='ecc', values='sd')
        g = g.loc[:, g.notna().sum() >= 10]
        m, se = g.mean(), g.sem()
        ax.fill_between(m.index, m - se, m + se, color=ROI_COLORS[roi], alpha=.25, lw=0)
        ax.plot(m.index, m.values, color=ROI_COLORS[roi], lw=1.1)
        ax.text(m.index[-1] + .05, m.values[-1], roi, color=ROI_COLORS[roi], fontsize=6, va='center')
    ax.set_xlim(0, 3.7)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel('Eccentricity (°)')
    ax.set_ylabel('pRF size σ (°)')
    offset_spines(ax)


def panel_contra(ax):
    d = pd.read_csv(COVER_TSV, sep='\t')
    for i, roi in enumerate(ROI_ORDER):
        p = contralateral(d[d.roi == roi]).values
        rng = np.random.default_rng(i)
        ax.scatter(ROI_X[roi] + rng.uniform(-.2, .2, len(p)), p, s=4, color=ROI_COLORS[roi], alpha=.55, lw=0)
        ax.plot([ROI_X[roi] - .3, ROI_X[roi] + .3], [np.median(p)] * 2, color=FG, lw=1.3, solid_capstyle='butt')
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.text(ROI_X['TO'] + .6, 51, 'Chance', fontsize=6, color='.45', ha='right', va='bottom')
    ax.set_yticks([50, 75, 100])
    ax.set_ylim(45, 102)
    ax.set_ylabel('Contralateral (%)')
    offset_spines(ax)
    roi_axis(ax)


def panel_loso(ax, table):
    d = pd.read_csv(table, sep='\t')
    g = d[d.family == 'dog'].set_index('term')
    for i, (t, _) in enumerate(TERM_ORDER):
        if t not in g.index:
            ax.add_patch(plt.Rectangle((i - .3, 0), .6, 50, facecolor='#ECEFF2', lw=0))
            ax.text(i, 25, 'Pending', ha='center', va='center', fontsize=5.5, color='#6B7785', rotation=90)
            continue
        r = g.loc[t]
        ax.bar(i, r.pct_improved, bottom=0, width=.6, color=FG if r.p < .05 else MUTED, lw=0)
        if r.p < .05:
            ax.text(i, r.pct_improved + 1.5, '***' if r.p < .001 else '**' if r.p < .01 else '*', ha='center', va='bottom',
                    fontsize=6.5, color=FG)
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.text(len(TERM_ORDER) - .35, 50, 'Chance', fontsize=6, color='.45', ha='left', va='center', clip_on=False)
    ax.set_xticks(range(len(TERM_ORDER)))
    ax.set_xticklabels([l for _, l in TERM_ORDER], fontsize=6)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlim(-.6, len(TERM_ORDER) - .4)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel('Subjects improved (%)')
    offset_spines(ax)


def main(in_dir, subjects, view, cv_table, voxel_table, loso_table, name='fig3_model_fits'):
    fig = plt.figure(figsize=(DOUBLE, 7.6))
    top = fig.add_gridspec(2, 1, height_ratios=[2.75, 3.6], hspace=.1, left=.075, right=.99, top=.975, bottom=.1)
    outer = [top[0]] + list(top[1].subgridspec(2, 1, height_ratios=[1.45, 1.5], hspace=.75))
    first = panel_maps(fig, outer[0], in_dir, subjects, view)
    letter(first, 'a', x=-.07, y=1.02)
    mid = outer[1].subgridspec(1, 3, width_ratios=[1.1, 1, 1.1], wspace=.42)
    ax = fig.add_subplot(mid[0])
    panel_cv(ax, cv_table)
    letter(ax, 'b', x=-.2, y=.98)
    ax = fig.add_subplot(mid[1])
    panel_sigma(ax)
    letter(ax, 'c', x=-.2, y=.98)
    ax = fig.add_subplot(mid[2])
    panel_contra(ax)
    bot = outer[2].subgridspec(1, 3, width_ratios=[.14, 2.1, 1], wspace=.35)
    vox = bot[1].subgridspec(3, 1, height_ratios=[1, .3, .1], hspace=.1)
    a1, a2, a3 = (fig.add_subplot(vox[i]) for i in range(3))
    d = panel_voxel(a1, a2, a3, voxel_table)
    win = d.t.max() - d.t.min() + 1.6
    letter(a1, 'd', x=-.12, y=1.05)
    provisional(a1, y=1.15)
    ax = fig.add_subplot(bot[2])
    panel_loso(ax, loso_table)
    letter(ax, 'e', x=-.25, y=.98)
    provisional(ax, y=1.02)
    l = pd.read_csv(loso_table, sep='\t').query('family == "dog"').set_index('term')
    dr2 = '; '.join(f'{lab.replace(chr(10), " ")} {l.loc[t, "median_dr2_e5"]:.2f}' for t, lab in TERM_ORDER if t in l.index)
    save(fig, name, LEGEND.format(win=win, dr2=dr2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map-dir', default='notes/figures/fig1_r2low')
    ap.add_argument('--subjects', type=int, nargs='+', default=[18, 19, 23])
    ap.add_argument('--view', default='pole')
    ap.add_argument('--cv-table', default='notes/data/prf_cv_roi_summary.tsv')
    ap.add_argument('--voxel-table', default='notes/data/fig3_example_voxel_sub-23_V3_2026-11.tsv')
    ap.add_argument('--loso-table', default='notes/data/loso_per_step_2026-11.tsv')
    a = ap.parse_args()
    main(a.map_dir, a.subjects, a.view, a.cv_table, a.voxel_table, a.loso_table)
