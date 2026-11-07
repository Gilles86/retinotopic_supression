"""Figure 3 (redesign): the pRF fits are retinotopic, predict held-out runs, and carry the gain terms to new subjects.

a, One subject's visual areas, polar angle and eccentricity on the inflated surface (pole view), with keys.
b, pRF centres in the contralateral field per ROI (subject dots, n per ROI).
c, Held-out R² gained over the simplest pRF (Gaussian, canonical HRF) by the model used for the gain fits.
d, Leave-one-subject-out test of the gain terms.

Usage::

    python -m retsupp.visualize.paper.fig3_prf_fits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from retsupp.visualize.paper.paper_style import (DOUBLE, FG, MUTED, ROI_COLORS, ROI_GROUPS, ROI_ORDER, ROI_X, letter, offset_spines,
                                                 provisional, roi_axis, save)
from retsupp.visualize.paper.plot_fig1 import COVER_TSV, contralateral

MAP_DIR = Path('notes/figures/fig1_r2low')
TERMS = [('T', 'Target'), ('D', 'Distractor'), ('S', 'Sustained'), ('dS', 'Sustained HP − LP'), ('dD', 'Distractor HP − LP')]
LEGEND = """
**Figure 3. The pRF fits.** a, Visual areas (neuropythy atlas), polar angle and eccentricity of the pRF fit on the inflated
surface of one subject (sub-23, posterior view; voxels with R² above the R² mixture threshold). Polar-angle key: the visual
field, coloured as mapped on the contralateral hemisphere. b, Share of pRF centres in the contralateral visual field per ROI,
voxels with R² ≥ .1 inside the bar aperture; dots are subjects, lines medians; numbers give the subjects per ROI. c, Held-out
R² gained by the pRF model used for the gain fits (difference-of-Gaussians pRF, one HRF per subject × ROI) over a Gaussian pRF
with the canonical HRF, per ROI: three folds of runs, voxels with cross-run reliability > 0 in the training runs, median over
voxels, mean over folds; dots are subjects (n = 30), lines medians. d, Leave-one-subject-out test of each gain term: share of
subjects whose held-out prediction improves when the term is added with the size estimated from the other 29 subjects (median
over ROIs); Wilcoxon signed-rank test on ΔR²: * p < .05, ** p < .01, *** p < .001. Provisional: c uses the per-ROI HRF from the
median of the voxels' own HRF fits until the jointly fitted HRF is in; d uses the current gain fits.
"""


def angle_key(ax):
    """Disk keyed like the polar-angle map: right field as on the left hemisphere, left field as on the right."""
    n = 201
    g = np.linspace(-1, 1, n)
    x, y = np.meshgrid(g, g)
    th = np.arctan2(y, x)
    v = np.where(x >= 0, 1. - np.clip((np.mod(th + np.pi, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi), 0, 1),
                 np.clip((np.mod(th, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi), 0, 1))
    rgba = mpl.colormaps['hsv'](v)
    rgba[..., 3] = (np.hypot(x, y) <= 1).astype(float)
    ax.imshow(rgba, extent=(-1, 1, -1, 1), origin='lower')
    ax.plot([-1, 1], [0, 0], color='w', lw=.5)
    ax.plot([0, 0], [-1, 1], color='w', lw=.5)
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ax.text(0, -1.15, 'Visual field', ha='center', va='top', fontsize=6, color=FG)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_axis_off()


def show_map(ax, path):
    im = Image.open(path).convert('RGBA')
    w, h = im.size
    im = im.crop((int(.02 * w), int(.0 * h), int(.98 * w), int(.82 * h)))
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    ax.imshow(np.asarray(bg.convert('RGB')), interpolation='antialiased')
    ax.set_axis_off()


def panel_maps(fig, gs, subject=23, view='pole'):
    sub = gs.subgridspec(1, 3, wspace=.03)
    axes = []
    for j, (p, title) in enumerate([('areas', 'Visual areas'), ('angle', 'Polar angle'), ('ecc', 'Eccentricity')]):
        ax = fig.add_subplot(sub[j])
        show_map(ax, MAP_DIR / f'sub-{subject:02d}_{p}_{view}.png')
        ax.set_title(title, fontsize=7, color=FG, pad=2)
        axes.append(ax)
    # Keys under the maps.
    ax = axes[0]
    names = [r for _, rois in ROI_GROUPS for r in rois]
    for k, roi in enumerate(names):
        ax.text(.02 + (k % 5) * .2, -.06 - (k // 5) * .12, roi, transform=ax.transAxes, color=ROI_COLORS[roi], fontsize=6.5,
                fontweight='bold', va='top', ha='left')
    angle_key(axes[1].inset_axes([.4, -.33, .2, .27]))
    ax = axes[2]
    cax = ax.inset_axes([.2, -.12, .6, .05])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['nipy_spectral'], norm=mpl.colors.Normalize(0, 4), orientation='horizontal')
    cb.set_ticks([0, 2, 4])
    cb.ax.tick_params(labelsize=6, length=2, width=.6)
    cb.outline.set_visible(False)
    cax.set_xlabel('Eccentricity (°)', fontsize=6, labelpad=1)
    return axes[0]


def panel_contra(ax):
    d = pd.read_csv(COVER_TSV, sep='\t')
    for i, roi in enumerate(ROI_ORDER):
        p = contralateral(d[d.roi == roi]).values
        rng = np.random.default_rng(i)
        ax.scatter(ROI_X[roi] + rng.uniform(-.2, .2, len(p)), p, s=5, color=ROI_COLORS[roi], alpha=.6, lw=0)
        ax.plot([ROI_X[roi] - .3, ROI_X[roi] + .3], [np.median(p)] * 2, color=FG, lw=1.4, solid_capstyle='butt')
        ax.text(ROI_X[roi], 39, f'{len(p)}', ha='center', va='bottom', fontsize=5.5, color='.45')
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.text(ROI_X['V2'], 51.5, 'Chance', fontsize=6, color='.45', ha='center', va='bottom')
    ax.set_yticks([50, 75, 100])
    ax.set_ylim(38, 102)
    ax.set_ylabel('Contralateral (%)')
    offset_spines(ax)
    roi_axis(ax)


def gain_r2(cv_table, hrf_table):
    """Per subject × ROI: held-out R² of DoG with the per-ROI HRF (model 14) minus Gaussian with the canonical HRF (model 1)."""
    a = pd.read_csv(cv_table, sep='\t')
    a = a[(a.selection == 'reliable') & (a.model == 1) & a.roi.isin(ROI_ORDER)].groupby(['subject', 'roi']).median_r2.mean()
    b = pd.read_csv(hrf_table, sep='\t')
    b = b[(b.model == 14) & b.roi.isin(ROI_ORDER)].groupby(['subject', 'roi']).median_r2.mean()
    return (b - a).dropna().rename('d').reset_index(), a


def panel_r2(ax, d):
    for i, roi in enumerate(ROI_ORDER):
        v = d[d.roi == roi].d.to_numpy()
        rng = np.random.default_rng(i)
        ax.scatter(ROI_X[roi] + rng.uniform(-.2, .2, len(v)), v, s=5, color=ROI_COLORS[roi], alpha=.6, lw=0)
        ax.plot([ROI_X[roi] - .3, ROI_X[roi] + .3], [np.median(v)] * 2, color=FG, lw=1.4, solid_capstyle='butt')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_yticks([0, .05, .1])
    ax.set_ylabel('Held-out R² gained')
    offset_spines(ax)
    roi_axis(ax)


def panel_loso(ax, table):
    d = pd.read_csv(table, sep='\t')
    g = d[d.family == 'dog'].set_index('term')
    for i, (t, lab) in enumerate(TERMS):
        r = g.loc[t]
        ax.bar(i, r.pct_improved, width=.62, color=FG if r.p < .05 else MUTED, lw=0)
        st = '***' if r.p < .001 else '**' if r.p < .01 else '*' if r.p < .05 else ''
        ax.text(i, r.pct_improved + 1.5, st, ha='center', va='bottom', fontsize=7, color=FG)
    ax.axhline(50, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_xticks(range(len(TERMS)))
    ax.set_xticklabels([l for _, l in TERMS], fontsize=6.5)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlim(-.6, len(TERMS) - .4)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel('Subjects improved (%)')
    offset_spines(ax)


def main(cv_table, hrf_table, loso_table, name='fig3_prf_fits'):
    d, base = gain_r2(cv_table, hrf_table)
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    top = fig.add_gridspec(1, 1, left=.01, right=.99, top=.95, bottom=.6)
    first = panel_maps(fig, top[0])
    letter(first, 'a', x=.0, y=1.05)
    bot = fig.add_gridspec(1, 3, width_ratios=[1, 1, .85], left=.075, right=.99, top=.4, bottom=.17, wspace=.42)
    ax = fig.add_subplot(bot[0])
    panel_contra(ax)
    ax.text(0, 1.16, 'pRFs sit in the contralateral field', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    letter(ax, 'b', x=-.22, y=1.12)
    ax = fig.add_subplot(bot[1])
    panel_r2(ax, d)
    ax.text(0, 1.16, 'Better than the simplest pRF', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    letter(ax, 'c', x=-.22, y=1.12)
    ax = fig.add_subplot(bot[2])
    panel_loso(ax, loso_table)
    ax.text(0, 1.16, 'Gain terms predict new subjects', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    letter(ax, 'd', x=-.3, y=1.12)
    provisional(ax, y=1.3)
    save(fig, name, LEGEND)
    print('median gain per ROI:', d.groupby('roi').d.median().round(4).to_dict())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cv-table', default='notes/data/prf_cv_roi_summary.tsv')
    ap.add_argument('--hrf-table', default='notes/data/hrf_comparison_2026-11.tsv')
    ap.add_argument('--loso-table', default='notes/data/loso_per_step_2026-11.tsv')
    a = ap.parse_args()
    main(a.cv_table, a.hrf_table, a.loso_table)
