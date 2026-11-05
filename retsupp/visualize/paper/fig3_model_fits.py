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

from retsupp.visualize.paper.paper_style import DOUBLE, FG, MUTED, ROI_COLORS, ROI_ORDER, letter, offset_spines, pending, save
from retsupp.visualize.paper.plot_fig1 import COVER_TSV, SIGMA_TSV, contralateral

LEGEND = """
**Figure 3. Model fits.** a, Visual areas, polar angle, eccentricity and R² of the m4 pRF fit on the inflated surface for sub-18, sub-19 and sub-23. b, Held-out R² per ROI in reliable voxels, for Gaussian, DoG and DN pRFs with a fixed or a per-voxel flexible HRF; mean across subjects (n = 30), voxels selected per fold on training-run reliability. c, pRF size against eccentricity (mean ± SEM across subjects), and the share of pRF centres in the contralateral field, voxels with R² ≥ .1 inside the bar aperture. d, Example voxel time course. e, Leave-one-subject-out test per ladder step.
"""
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


def panel_cv(ax, cv_table):
    d = pd.read_csv(cv_table, sep='\t')
    d = d[(d.selection == 'reliable') & d.roi.isin(ROI_ORDER) & d.model.isin(MODEL_STYLE)]
    s = d.groupby(['roi', 'subject', 'model']).median_r2.mean().reset_index()
    m = s.groupby(['roi', 'model']).median_r2.mean().unstack().reindex(ROI_ORDER)
    x = np.arange(len(ROI_ORDER))
    for model, (lab, col, ls) in MODEL_STYLE.items():
        ax.plot(x, m[model].values, color=col, ls=ls, lw=1, label=f'{lab}, {"flexible" if model in (3, 4, 6) else "fixed"} HRF')
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Held-out R²')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc='lower left', bbox_to_anchor=(-.02, 1.0), handlelength=2, columnspacing=.8, borderaxespad=0)
    ax.text(1., .86, 'Pending: one HRF\nper subject × ROI', transform=ax.transAxes, ha='right', fontsize=5.5, color=MUTED)
    offset_spines(ax)


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
        ax.scatter(i + rng.uniform(-.2, .2, len(p)), p, s=4, color=ROI_COLORS[roi], alpha=.55, lw=0)
        ax.plot([i - .3, i + .3], [np.median(p)] * 2, color=FG, lw=1.3, solid_capstyle='butt')
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_yticks([50, 75, 100])
    ax.set_ylim(45, 102)
    ax.set_ylabel('Contralateral (%)')
    offset_spines(ax)


def main(in_dir, subjects, view, cv_table, name='fig3_model_fits'):
    fig = plt.figure(figsize=(DOUBLE, 6.3))
    outer = fig.add_gridspec(3, 1, height_ratios=[3.1, 1.3, 1.], hspace=.45, left=.075, right=.99, top=.975, bottom=.075)
    first = panel_maps(fig, outer[0], in_dir, subjects, view)
    letter(first, 'a', x=-.07, y=1.02)
    mid = outer[1].subgridspec(1, 3, width_ratios=[1.3, 1, 1.1], wspace=.42)
    ax = fig.add_subplot(mid[0])
    panel_cv(ax, cv_table)
    letter(ax, 'b', x=-.14, y=1.2)
    ax = fig.add_subplot(mid[1])
    panel_sigma(ax)
    letter(ax, 'c', x=-.2, y=.98)
    ax = fig.add_subplot(mid[2])
    panel_contra(ax)
    bot = outer[2].subgridspec(1, 2, width_ratios=[1.3, 1], wspace=.2)
    ax = fig.add_subplot(bot[0])
    pending(ax, 'example voxel, data with m0 and m7 predictions, target and distractor onsets')
    letter(ax, 'd', x=-.02, y=.98)
    ax = fig.add_subplot(bot[1])
    pending(ax, 'LOSO per ladder step, % improved and ΔR², DoG and DN, with D on top of T')
    letter(ax, 'e', x=-.02, y=.98)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map-dir', default='notes/figures/fig1_r2low')
    ap.add_argument('--subjects', type=int, nargs='+', default=[18, 19, 23])
    ap.add_argument('--view', default='pole')
    ap.add_argument('--cv-table', default='notes/data/prf_cv_roi_summary.tsv')
    a = ap.parse_args()
    main(a.map_dir, a.subjects, a.view, a.cv_table)
