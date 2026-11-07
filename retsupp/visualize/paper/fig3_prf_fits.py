"""Figure 3 (redesign): the pRF fits are retinotopic, predict held-out runs, and carry the gain terms to new subjects.

a, One subject's visual areas, polar angle and eccentricity on the flattened occipital cortex, with keys; insets show
where the flattened window lies on the inflated surface.
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
from retsupp.visualize.paper.plot_fig1 import COVER_TSV, SIGMA_TSV, contralateral

MAP_DIR = Path('notes/figures/fig1_r2low')
TERMS = [('T', 'Target'), ('D', 'Distractor'), ('S', 'Sustained'), ('dS', 'Sustained, HP − LP'), ('dD', 'Distractor, HP − LP')]
LEGEND = """
**Figure 3. The pRF fits.** a, Visual areas (neuropythy atlas), polar angle and eccentricity of the pRF fit on the flattened
cortex of one subject (sub-23; left hemisphere on the left; colour fades in from R² = .05 to .20). Left: the inflated
surface turned to show its lateral and medial walls, visual areas in their colours; black lines mark the edge of the
flattened window and, in the medial view, the cut along the calcarine sulcus made to flatten the surface. Polar-angle key: the visual
field, coloured as mapped on the contralateral hemisphere. b, Share of pRF centres in the contralateral visual field per ROI,
voxels with R² ≥ .1 inside the bar aperture; dots are subjects, lines medians; numbers give the subjects per ROI. c, Held-out
R² gained per ROI over a Gaussian pRF with the canonical HRF by a difference-of-Gaussians pRF (teal), by one HRF per subject
× ROI on top of the DoG pRF (grey), and by both together, the model used for the gain fits (black): three folds of runs,
voxels with cross-run reliability > 0 in the training runs, median over voxels, mean over folds; points and bars, median
over subjects (n = 30) and bootstrapped 95% CI. d, Response to a neutral item (at an LP location) against the distance of the pRF centre from that item: deconvolved
response (nideconv), mean over 3.2-6.4 s minus −4 to 0 s, reliable voxels binned by distance (1° bins), pooled per pathway
weighted by voxel count; lines and bands, mean ± SEM across subjects. Items appear at all eight ring positions at once,
so pRFs far from this item lie near the opposite ones and the response rises again; between the items, near fixation, it
is close to 0 in early visual cortex. Provisional: c uses the per-ROI HRF from the median of the voxels' own HRF fits until the jointly fitted HRF is in.
"""


def angle_key(ax_l, ax_r):
    """Two 270° wedges keyed like the polar-angle map: each hemisphere's contralateral field plus 45° past the meridians."""
    n = 201
    g = np.linspace(-1, 1, n)
    x, y = np.meshgrid(g, g)
    th = np.arctan2(y, x)
    for ax, hemi, lab in ((ax_l, 'L', 'Left hemisphere'), (ax_r, 'R', 'Right hemisphere')):
        if hemi == 'R':
            u = (np.mod(th, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi)
            v = u
        else:
            u = (np.mod(th + np.pi, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi)
            v = 1. - u
        rgba = mpl.colormaps['hsv'](np.clip(v, 0, 1))
        rgba[..., 3] = ((np.hypot(x, y) <= 1) & (u >= 0) & (u <= 1)).astype(float)
        ax.imshow(rgba, extent=(-1, 1, -1, 1), origin='lower')
        ax.plot([-1, 1], [0, 0], color='.6', lw=.4)
        ax.plot([0, 0], [-1, 1], color='.6', lw=.4)
        ax.text(0, -1.2, lab, ha='center', va='top', fontsize=5.5, color=FG)
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


def show_flat(ax, path):
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    ax.imshow(np.asarray(bg.convert('RGB')), interpolation='antialiased')
    ax.set_axis_off()


def panel_flat_window(fig, gs, subject=23):
    """Where the flat maps come from: the inflated surface with the flat-map window outlined, lateral and medial."""
    sub = gs.subgridspec(2, 1, hspace=.25)
    axes = []
    for j, (side, lab) in enumerate([('lateral', 'Lateral'), ('medial', 'Medial')]):
        ax = fig.add_subplot(sub[j])
        show_flat(ax, MAP_DIR / f'sub-{subject:02d}_flatwindow_{side}.png')
        ax.set_title(lab, fontsize=6, color=FG, pad=1)
        axes.append(ax)
    return axes[0]


def panel_maps(fig, gs, subject=23, view='flat'):
    sub = gs.subgridspec(1, 3, wspace=.03)
    axes = []
    for j, (p, title) in enumerate([('areas', 'Visual areas'), ('angle', 'Polar angle'), ('ecc', 'Eccentricity')]):
        ax = fig.add_subplot(sub[j])
        fn = MAP_DIR / f'sub-{subject:02d}_{p}_{view}.png'
        (show_flat if view == 'flat' else show_map)(ax, fn)
        ax.set_title(title, fontsize=7, color=FG, pad=2)
        axes.append(ax)
    # Keys under the maps.
    ax = axes[0]
    names = [r for _, rois in ROI_GROUPS for r in rois]
    for k, roi in enumerate(names):
        ax.text(.02 + (k % 5) * .2, -.06 - (k // 5) * .12, roi, transform=ax.transAxes, color=ROI_COLORS[roi], fontsize=6.5,
                fontweight='bold', va='top', ha='left')
    angle_key(axes[1].inset_axes([.2, -.33, .2, .27]), axes[1].inset_axes([.6, -.33, .2, .27]))
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


PRF_TEAL = '#2A9D8F'
STEPS = [('DoG pRF', 'dog', PRF_TEAL), ('HRF per ROI', 'hrf', '#8D99A6'), ('Both', 'both', FG)]


def gain_r2(cv_table, hrf_table):
    """Per subject × ROI, held-out R² gained over a Gaussian pRF with the canonical HRF (model 1) by: the DoG pRF (model 2),
    the per-ROI HRF on top of the DoG pRF (model 14 − model 2), and both (model 14 − model 1)."""
    a = pd.read_csv(cv_table, sep='\t')
    a = a[(a.selection == 'reliable') & a.roi.isin(ROI_ORDER)].groupby(['subject', 'roi', 'model']).median_r2.mean().unstack()
    b = pd.read_csv(hrf_table, sep='\t')
    b = b[(b.model == 14) & b.roi.isin(ROI_ORDER)].groupby(['subject', 'roi']).median_r2.mean()
    out = pd.DataFrame(dict(dog=a[2] - a[1], hrf=b - a[2], both=b - a[1])).dropna().reset_index()
    return out, a[1]


def panel_r2(ax, d):
    """Median over subjects per ROI and its bootstrapped 95% CI, one line per comparison."""
    from retsupp.visualize.paper.fig45_common import median_ci
    offs = dict(zip([k for _, k, _ in STEPS], (-.22, 0., .22)))
    for lab, key, col in STEPS:
        for grp, rois in ROI_GROUPS:
            xs, ms = [], []
            for roi in rois:
                m, lo, hi = median_ci(d[d.roi == roi][key].to_numpy())
                x = ROI_X[roi] + offs[key]
                ax.plot([x, x], [lo, hi], color=col, lw=1.1, zorder=2)
                xs.append(x)
                ms.append(m)
            ax.plot(xs, ms, 'o-', color=col, ms=3.5, lw=1.1, mec='w', mew=.5, zorder=3)
    for k, (lab, _, col) in enumerate(STEPS):
        ax.text(.99, .97 - k * .1, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_ylabel('Held-out R² gained')
    offset_spines(ax)
    roi_axis(ax)


def panel_loso(ax, table):
    """Per ROI and gain term: share of subjects whose held-out prediction improves (leave-one-subject-out)."""
    d = pd.read_csv(table, sep='\t')
    d = d[(d.family == 'dog') & d.roi.isin(ROI_ORDER)]
    xs = [ROI_X[r] for r in ROI_ORDER]
    cmap = mpl.colormaps['RdBu_r']
    for i, (t, lab) in enumerate(TERMS):
        x = d[d.mech == t]
        pct = [(x[x.roi == r].dr2_sign.dropna() > 0).mean() * 100 for r in ROI_ORDER]
        ps = [stats.wilcoxon(x[x.roi == r].dr2_sign.dropna()).pvalue for r in ROI_ORDER]
        qs = stats.false_discovery_control(ps)
        for xx, v, q in zip(xs, pct, qs):
            ax.add_patch(plt.Rectangle((xx - .46, i - .46), .92, .92, color=cmap(.5 + (v - 50) / 100), lw=0))
            ax.text(xx, i, f'{v:.0f}', ha='center', va='center', fontsize=5.5,
                    color='w' if abs(v - 50) > 30 else FG, fontweight='bold' if q < .05 else 'normal')
    ax.set_yticks(range(len(TERMS)))
    ax.set_yticklabels([l for _, l in TERMS], fontsize=6.5)
    ax.set_ylim(len(TERMS) - .5, -.5)
    for sp in ('left',):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)
    roi_axis(ax)


DIST_DIR = 'notes/data/event_deconv_distance_2026-11'
PATHWAY = {'Early': ('V1', 'V2', 'V3'), 'Dorsal': ('V3AB', 'IPS'), 'Occipitotemporal': ('hV4', 'VO', 'LO', 'TO')}
PATHWAY_COL = {'Early': ROI_COLORS['V1'], 'Dorsal': ROI_COLORS['V3AB'], 'Occipitotemporal': ROI_COLORS['LO']}


def panel_distance(ax, folder=DIST_DIR, cls='neutral_lp'):
    """Deconvolved response to an item against the distance of the pRF centre from that item, per pathway."""
    import glob
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(f'{folder}/sub-*_2026-11.tsv'))])
    grp = {r: g for g, rois in PATHWAY.items() for r in rois}
    d = d[d.roi.isin(grp)].assign(grp=lambda x: x.roi.map(grp), t=lambda x: x.t.round(2))
    d['w'] = d.response * d.n_voxels
    def pooled(x):
        g = x.groupby(['subject', 'grp', 'dbin'])
        return g.w.sum() / g.n_voxels.sum()
    x = d[d.cls == cls]
    r = (pooled(x[x.t.between(3.2, 6.4)]) - pooled(x[x.t < 0])).rename('r').reset_index()
    for grp_name, col in PATHWAY_COL.items():
        g = r[r.grp == grp_name].groupby('dbin').r
        m, se = g.mean(), g.sem()
        xs = m.index + .5
        ax.fill_between(xs, m - se, m + se, color=col, alpha=.2, lw=0)
        ax.plot(xs, m, 'o-', color=col, ms=3, lw=1.3)
        dy = {'Early': 0, 'Dorsal': -.06, 'Occipitotemporal': .07}[grp_name]
        ax.text(xs[-1] + .2, m.iloc[-1] + dy, grp_name if grp_name != 'Occipitotemporal' else 'Occ.-temp.', color=col,
                fontsize=6, va='center', ha='left')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_xlim(0, 8.6)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel('pRF distance from the item (°)')
    ax.set_ylabel('Response (% signal)')
    offset_spines(ax)


def panel_size(ax, rois=('V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO')):
    """pRF size against eccentricity per ROI: mean ± SEM across subjects of each subject's voxel-weighted mean."""
    d = pd.read_csv(SIGMA_TSV, sep='\t')
    d = d[d.n >= 5]
    ends = []
    for roi in rois:
        g = d[d.roi == roi].pivot(index='subject', columns='ecc', values='sd')
        g = g.loc[:, g.notna().sum() >= 10]
        m, se = g.mean(), g.sem()
        ax.fill_between(m.index, m - se, m + se, color=ROI_COLORS[roi], alpha=.22, lw=0)
        ax.plot(m.index, m.values, color=ROI_COLORS[roi], lw=1.3)
        ends.append([m.values[-1], m.index[-1], roi])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least 0.12 deg apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + .12)
    for y, x, roi in ends:
        ax.text(x + .07, y, roi, color=ROI_COLORS[roi], fontsize=6, va='center')
    ax.set_xlim(0, 3.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel('Eccentricity (°)')
    ax.set_ylabel('pRF size σ (°)')
    offset_spines(ax)


NEAREST_DIR = 'notes/data/event_deconv_nearest_2026-11'


def panel_distance_tc(fig, gs, folder=NEAREST_DIR):
    """Deconvolved response to the search display per pathway, one line per distance of the pRF from the nearest item."""
    import glob
    files = sorted(glob.glob(f'{folder}/sub-*_2026-11.tsv'))
    sub = gs.subgridspec(1, 3, wspace=.18)
    if not files:
        return None
    d = pd.concat([pd.read_csv(f, sep='\t') for f in files])
    grp = {r: g for g, rois in PATHWAY.items() for r in rois}
    d = d[d.roi.isin(grp)].assign(grp=lambda x: x.roi.map(grp), t=lambda x: x.t.round(2))
    d['w'] = d.response * d.n_voxels
    g = d.groupby(['subject', 'grp', 'dbin', 't'])
    p = (g.w.sum() / g.n_voxels.sum()).rename('r').reset_index()
    base = p[p.t < 0].groupby(['subject', 'grp', 'dbin']).r.mean()
    p['r'] -= p.set_index(['subject', 'grp', 'dbin']).index.map(base)
    bins = sorted(p.dbin.unique())
    cmap = mpl.colormaps['viridis']
    cols = {b: cmap(.9 * i / max(len(bins) - 1, 1)) for i, b in enumerate(bins)}
    axes = []
    for j, grp_name in enumerate(PATHWAY):
        ax = fig.add_subplot(sub[j], sharey=axes[0] if axes else None)
        for b in bins:
            m = p[(p.grp == grp_name) & (p.dbin == b)].groupby('t').r.mean()
            ax.plot(m.index, m.values, color=cols[b], lw=1.2)
        ax.axhline(0, color='.75', lw=.6, zorder=0)
        ax.axvline(0, color='.75', lw=.6, zorder=0)
        ax.set_title(grp_name, fontsize=7, color=FG, pad=2)
        ax.set_xticks([0, 10, 20])
        ax.set_xlabel('Time from search onset (s)')
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel('Response (% signal)')
        offset_spines(ax)
        axes.append(ax)
    # Colour key: distance of the pRF centre from the nearest item.
    cax = axes[-1].inset_axes([1.06, .1, .05, .8])
    edges = list(bins) + [bins[-1] + (bins[-1] - bins[-2])]
    norm = mpl.colors.BoundaryNorm(edges, cmap.N)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colors.ListedColormap([cols[b] for b in bins]),
                                   norm=mpl.colors.BoundaryNorm(edges, len(bins)), orientation='vertical')
    cb.set_ticks([edges[0], edges[len(edges) // 2], edges[-1]])
    cb.ax.tick_params(labelsize=6, length=2, width=.6)
    cb.outline.set_visible(False)
    cax.set_ylabel('Distance from\nnearest item (°)', fontsize=6, labelpad=2)
    return axes[0]


def distance_icon(fig, x0, y0, size, n_bins=8):
    """The display's eight items with pRFs (filled circles) stepping from an item to fixation, coloured as in panel e."""
    from matplotlib.patches import Circle
    w, h = fig.get_size_inches()
    ins = fig.add_axes([x0, y0, size / w, size / h])
    ins.add_patch(Circle((0, 0), 4., fill=False, ec=MUTED, lw=.4, ls=(0, (1.5, 1.5))))
    for a in np.deg2rad(np.arange(8) * 45.):
        ins.add_patch(Circle((4 * np.cos(a), 4 * np.sin(a)), .55, facecolor='w', ec=FG, lw=.6, zorder=3))
    cmap = mpl.colormaps['viridis']
    u = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    for k, dist in enumerate((0., 1.35, 2.7, 4.)):
        ins.add_patch(Circle((4 - dist) * u, 1.05, facecolor=cmap(.9 * k / 3), ec='none', alpha=.75, zorder=2 if k else 4))
    ins.plot(0, 0, '+', color=FG, ms=3, mew=.5, zorder=5)
    ins.set_xlim(-5, 5)
    ins.set_ylim(-5, 5)
    ins.set_aspect('equal')
    ins.axis('off')
    return ins


def main(cv_table, hrf_table, loso_table, name='fig3_prf_fits'):
    d, base = gain_r2(cv_table, hrf_table)
    fig = plt.figure(figsize=(DOUBLE, 7.1))
    top = fig.add_gridspec(1, 2, width_ratios=[.27, 1], left=.01, right=.99, top=0.965, bottom=0.665, wspace=.03)
    first = panel_flat_window(fig, top[0])
    letter(first, 'a', x=.04, y=1.12)
    panel_maps(fig, top[1])
    mid = fig.add_gridspec(1, 3, width_ratios=[1, .7, 1], left=.075, right=.99, top=0.535, bottom=0.385, wspace=.38)
    ax = fig.add_subplot(mid[0])
    panel_contra(ax)
    ax.text(0, 1.16, 'pRFs sit in the contralateral field', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    letter(ax, 'b', x=-.22, y=1.12)
    ax = fig.add_subplot(mid[1])
    panel_size(ax)
    ax.text(0, 1.16, 'pRFs grow with eccentricity', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    letter(ax, 'c', x=-.3, y=1.12)
    ax = fig.add_subplot(mid[2])
    panel_r2(ax, d)
    ax.text(0, 1.2, 'Held-out R² gained over a Gaussian pRF\nwith the canonical HRF', transform=ax.transAxes, fontsize=6.5,
            color=FG, va='center', linespacing=1.2)
    letter(ax, 'd', x=-.22, y=1.12)
    fig.text(.99, .995, 'Provisional: current inputs', fontsize=5.5, color=MUTED, ha='right', va='top')
    low = fig.add_gridspec(1, 1, left=.075, right=.9, top=0.17, bottom=0.055)
    first = panel_distance_tc(fig, low[0])
    if first is not None:
        bb = first.get_position()
        size = .55
        distance_icon(fig, bb.x0 - .01, bb.y1 + .02, size)
        fig.text(bb.x0 + (size + .06) / fig.get_size_inches()[0], bb.y1 + .02 + size / 2 / fig.get_size_inches()[1],
                 'Responses to the search display fall off with distance from the nearest item', fontsize=7, color=FG, va='center')
        letter(first, 'e', x=-.2, y=1.5)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cv-table', default='notes/data/prf_cv_roi_summary.tsv')
    ap.add_argument('--hrf-table', default='notes/data/hrf_comparison_2026-11.tsv')
    ap.add_argument('--loso-table', default='notes/data/loso_per_roi_2026-11.tsv')
    a = ap.parse_args()
    main(a.cv_table, a.hrf_table, a.loso_table)
