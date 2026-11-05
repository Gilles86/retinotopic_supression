"""Figure 1: the paradigm and the checks that the pRF fits are sound.

``collect`` reduces the per-voxel m4 fits of all subjects to three small tables under
``notes/data``. ``plot`` draws the figure from those tables, the held-out comparison
(``notes/data/prf_cv_roi_summary.tsv``) and the pycortex renders of ``render_fig1_prf``.

Mapped voxels have at least half of the pRF inside the bar aperture and sigma .3-4°. Signal
voxels are mapped voxels with R² above the subject's whole-brain tail-FDR threshold (alpha .05).
Sigma against eccentricity and coverage use signal voxels; panel e is their share of mapped voxels.

Usage::

    python -m retsupp.visualize.paper.plot_fig1 collect
    python -m retsupp.visualize.paper.plot_fig1 plot --subject 19
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8, 'lines.linewidth': 1.2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.pad_inches': .02,
})

FG, MUTED, HP = '#1F2933', '#9AA5B1', '#D62828'
SCREEN, ITEM_A, ITEM_B = '#808080', '#E39100', '#00A900'   # as displayed: psychopy (.78, .14, -1), (-1, .33, -1)
AREAS = {'V1': '#3D5A80', 'V2': '#2A9D8F', 'V3': '#8AB17D', 'hV4': '#E9C46A', 'V3AB': '#7209B7',
         'VO': '#F4A261', 'LO': '#E76F51', 'TO': '#B5179E', 'IPS': '#0077B6', 'FEF': '#D62828'}
ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
LINE_ROIS = ['V1', 'V2', 'V3', 'hV4', 'V3AB', 'IPS']
STRIP_ROIS = ['V1', 'V2', 'V3', 'hV4', 'VO', 'LO', 'TO', 'V3AB', 'IPS']
COVER_ROIS = ['V1', 'V3', 'V3AB', 'IPS']
APERTURE, RING = 3.17, 4.0
ECC_BINS = np.arange(0, 3.51, .5)
XY_BINS = np.linspace(-APERTURE, APERTURE, 41)
DATA = Path('notes/data')
SIGMA_TSV, R2_TSV, COVER_TSV = (DATA / f'fig1_prf_{x}.tsv' for x in ('sigma', 'r2', 'coverage'))
CV_TSV = DATA / 'prf_cv_roi_summary.tsv'


def collect(bids_folder):
    from retsupp.utils.data import Subject, select_well_fit_voxels

    sigma, r2s, cover = [], [], []
    for s in range(1, 31):
        sub = Subject(s, bids_folder)
        d = []
        for roi in ROIS:
            for hemi in 'LR':
                x = sub.get_prf_roi_pars(roi=f'{roi}_{hemi}', model=4,
                                         params=['x', 'y', 'sd', 'amplitude', 'r2'])
                d.append(x.assign(roi=roi, hemi=hemi))
        d = pd.concat(d, ignore_index=True)
        thr = sub.get_r2_fdr_threshold(model=4, alpha=.05)
        good, _ = select_well_fit_voxels(d, r2_threshold=thr, aperture_radius=APERTURE)
        mapped, _ = select_well_fit_voxels(d, r2_threshold=-np.inf, aperture_radius=APERTURE)
        print(f'sub-{s:02d}: threshold {thr:.3f}, {len(good)} of {len(d)} voxels', flush=True)
        good = good.assign(ecc=np.hypot(good.x, good.y))
        for roi, g in good.groupby('roi'):
            b = pd.cut(g.ecc, ECC_BINS, include_lowest=True)
            m = g.groupby(b, observed=True).sd.agg(['median', 'size']).reset_index()
            sigma.append(pd.DataFrame(dict(subject=s, roi=roi, ecc=[i.mid for i in m.ecc], sd=m['median'], n=m['size'])))
            h, _, _ = np.histogram2d(g.x, g.y, bins=[XY_BINS, XY_BINS])
            ix, iy = np.nonzero(h)
            cover.append(pd.DataFrame(dict(subject=s, roi=roi, ix=ix, iy=iy, n=h[ix, iy].astype(int))))
        r2s.append(mapped.groupby('roi').agg(n_mapped=('r2', 'size'), median_r2=('r2', 'median'))
                   .join(good.groupby('roi').size().rename('n_signal')).fillna(0).reset_index().assign(subject=s))
    pd.concat(sigma).to_csv(SIGMA_TSV, sep='\t', index=False)
    pd.concat(r2s).to_csv(R2_TSV, sep='\t', index=False)
    pd.concat(cover).to_csv(COVER_TSV, sep='\t', index=False)


def letter(ax, s, x=-.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right', color='k')


def item_positions():
    # Eight items every 45°; odd indices are the diagonals where target and singleton appear.
    a = np.deg2rad(np.arange(8) * 45.)
    return np.c_[RING * np.cos(a), RING * np.sin(a)]


def draw_item(ax, xy, col, vertical, dot, scale=1.):
    w, h = 1.5 * scale, .375 * scale
    if vertical:
        w, h = h, w
    ax.add_patch(Rectangle((xy[0] - w / 2, xy[1] - h / 2), w, h, facecolor=col, lw=0, zorder=3))
    if dot:
        ax.add_patch(Circle(xy, .1 * scale, facecolor='#BDBDBD', lw=0, zorder=4))


def draw_bar(ax, x0=-.9):
    """Checkerboard bar in the aperture."""
    w = 2 * APERTURE / 8
    clip = Circle((0, 0), APERTURE, transform=ax.transData)
    n = 16
    s = 2 * APERTURE / n
    for i in range(n):
        for j in range(2):
            r = Rectangle((x0 + j * w / 2, -APERTURE + i * s), w / 2, s, lw=0, zorder=2,
                          facecolor='k' if (i + j) % 2 else 'w')
            ax.add_patch(r)
            r.set_clip_path(clip)
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='#6E6E6E', lw=.4, zorder=2))


def display(ax, search=True, fix='w', lim=5.4, scale=1., bar=True, hp=None):
    ax.add_patch(Rectangle((-lim, -lim), 2 * lim, 2 * lim, facecolor=SCREEN, lw=0, zorder=0))
    if bar:
        draw_bar(ax)
    if search:
        pos = item_positions()
        # Target upper left among horizontal bars, singleton lower right.
        dots = {0, 2, 3, 7}
        for i, xy in enumerate(pos):
            draw_item(ax, xy, ITEM_B if i == 7 else ITEM_A, i == 3, i in dots, scale)
    ax.add_patch(Circle((0, 0), .2 if lim < 6 else .3, facecolor=fix, lw=0, zorder=5))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_axis_off()


def panel_display(ax):
    display(ax)
    pos = item_positions()
    ax.add_patch(Circle(pos[7], 1.12, fill=False, ec=HP, lw=1.1, ls=(0, (2.5, 1.5)), zorder=6, clip_on=False))
    kw = dict(fontsize=7, color=FG, annotation_clip=False,
              arrowprops=dict(arrowstyle='-|>', color=FG, lw=.8, mutation_scale=6, shrinkA=1, shrinkB=2))
    ax.annotate('Target: odd orientation,\nreport the dot inside', xy=pos[3] + [0, .85], xytext=(-9.3, 7.3),
                ha='left', va='center', **kw)
    ax.annotate('Singleton distractor', xy=pos[7] + [0, -1.2], xytext=(5.4, -7.0), ha='right', va='center', **kw)
    ax.annotate('Bar in 3.17°\naperture', xy=(-.5, 3.2), xytext=(9.3, 7.3), ha='right', va='center', **kw)
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-8, 8.6)
    ax.text(pos[7][0] + 1.3, pos[7][1] + 1.2, 'HP', color=HP, fontsize=7, fontweight='bold', va='center', zorder=7)


def panel_trial(fig, ax):
    """Frames of one trial, offset along a time arrow."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    frames = [('Fixation', '1 s', dict(search=False)), ('Search', '1.5 s', dict(search=True)),
              ('Feedback', '1 s', dict(search=False)), ('Interval', '1.3-4.5 s', dict(search=False))]
    bb = ax.get_position()
    fw_in = bb.width * fig.get_figwidth()
    size = .2                        # frame width, as fraction of this axes
    h = size * fw_in / (bb.height * fig.get_figheight())
    y0 = .93 - h
    for i, (name, dur, kw) in enumerate(frames):
        x = .01 + i * (size + .06)
        inset = ax.inset_axes([x, y0, size, h])
        display(inset, scale=1., **kw)
        ax.text(x + size / 2, y0 + h + .012, name, ha='center', va='bottom', fontsize=7, color=FG)
        ax.text(x + size / 2, y0 - .02, dur, ha='center', va='top', fontsize=7, color=FG)
    y = y0 - .085
    ax.add_patch(FancyArrowPatch((.01, y), (.985, y), arrowstyle='-|>', mutation_scale=7, lw=.8, color=FG,
                                 transform=ax.transAxes))
    ax.text(.5, y - .015, 'The bar keeps sweeping, independent of the trials', ha='center', va='top', fontsize=7, color=FG)
    # The HP location per block of runs.
    y1 = .02
    icon = .1 * fw_in / (bb.height * fig.get_figheight())
    ax.text(.01, y1 + icon + .015, 'HP location moves between blocks of 3 runs, 12 runs in 2 sessions',
            ha='left', va='bottom', fontsize=7, color=FG)
    pos = item_positions()
    for i, k in enumerate([1, 5, 7, 3]):
        x = .01 + i * .125
        inset = ax.inset_axes([x, y1, .1, icon])
        inset.add_patch(Circle((0, 0), RING, fill=False, ec=MUTED, lw=.5))
        for j in (1, 3, 5, 7):
            inset.add_patch(Circle(pos[j], 1.2, facecolor=HP if j == k else 'w', ec=HP if j == k else MUTED, lw=.5))
        inset.set_xlim(-5.4, 5.4)
        inset.set_ylim(-5.4, 5.4)
        inset.set_aspect('equal')
        inset.set_axis_off()
    x = .56
    for yy, s, c in ((y1 + icon * .82, 'HP: 50% of trials', HP), (y1 + icon * .5, 'Each other location: 10%', FG),
                     (y1 + icon * .18, 'No singleton: 20%', FG)):
        ax.text(x, yy, s, fontsize=7, color=c, va='center')


def theta_value(theta, hemi):
    if hemi == 'R':
        v = np.mod(theta, 2 * np.pi)
        return np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)
    v = np.mod(theta + np.pi, 2 * np.pi)
    return 1. - np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)


def angle_key(ax):
    g = np.linspace(-1, 1, 301)
    x, y = np.meshgrid(g, g)
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    for cx, hemi, wedge in ((-1.25, 'L', x >= -np.abs(y)), (1.25, 'R', x <= np.abs(y))):
        rgba = mpl.colormaps['hsv'](theta_value(th, hemi))
        rgba[..., 3] = (r <= 1) & wedge
        ax.imshow(rgba, extent=(cx - 1, cx + 1, -1, 1), origin='lower', interpolation='bilinear')
        ax.plot([cx, cx], [-1, 1], color='white', lw=.8)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-1.05, 1.05)
    ax.text(-2.35, 0, 'RVF', ha='right', va='center', color=FG, fontsize=6.5)
    ax.text(2.35, 0, 'LVF', ha='left', va='center', color=FG, fontsize=6.5)
    ax.set_axis_off()


def show_render(ax, images, par):
    boxes = np.array([im.getbbox() for im in images.values()])
    box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    im = images[par].crop(box)
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    ax.imshow(np.asarray(bg.convert('RGB')), interpolation='none')
    ax.set_axis_off()


def panel_maps(fig, sub, subject, in_dir):
    """All maps in the ventral view, plus areas and polar angle in the dorsal view for IPS and FEF.

    ``sub`` is a grid of 3 rows and 5 columns; columns 2-4 of the last row are left to the caller."""
    panels = [('areas', 'Visual areas'), ('angle', 'Polar angle'), ('ecc', 'Eccentricity (°)'),
              ('sd', 'pRF size σ (°)'), ('r2', 'R²')]
    bars = {'ecc': ('nipy_spectral', 0, 4, [0, 2, 4]), 'sd': ('nipy_spectral', 0, 3, [0, 1.5, 3]),
            'r2': ('inferno', 0, .6, [0, .3, .6])}

    def load(view, pars):
        return {p: Image.open(Path(in_dir) / f'sub-{subject:02d}_{p}_{view}.png').convert('RGBA') for p in pars}

    ventral = load('ventral', [p for p, _ in panels])
    first = None
    for j, (par, title) in enumerate(panels):
        ax = fig.add_subplot(sub[0, j])
        first = first or ax
        show_render(ax, ventral, par)
        ax.set_title(title, color=FG, pad=3)
    dorsal = load('dorsal', ['areas', 'angle'])
    for j, par in enumerate(dorsal):
        ax = fig.add_subplot(sub[2, j])
        show_render(ax, dorsal, par)
    for r, name in ((0, 'Ventral'), (2, 'Dorsal')):
        ax = fig.axes[-2] if r == 2 else first
        ax.text(-.02, .5, name, transform=ax.transAxes, rotation=90, ha='right', va='center', color=FG, fontsize=7)
    ax = fig.add_subplot(sub[1, 0])
    for i, (name, col) in enumerate(AREAS.items()):
        ax.text(.02 + .2 * (i % 5), .72 - .5 * (i // 5), name, color=col, fontsize=6, fontweight='bold',
                va='center', transform=ax.transAxes)
    ax.set_axis_off()
    angle_key(fig.add_subplot(sub[1, 1]))
    for j, (par, _) in enumerate(panels[2:], start=2):
        ax = fig.add_subplot(sub[1, j])
        ax.set_axis_off()
        cmap, lo, hi, ticks = bars[par]
        cax = ax.inset_axes([.18, .62, .64, .2])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(lo, hi), cmap), cax=cax,
                          orientation='horizontal', ticks=ticks)
        cb.outline.set_visible(False)
        cb.ax.tick_params(pad=1.5, labelsize=6.5)
    return first


def panel_sigma(ax):
    d = pd.read_csv(SIGMA_TSV, sep='\t')
    d = d[d.n >= 5]
    ends = []
    for roi in LINE_ROIS:
        g = d[d.roi == roi].pivot(index='subject', columns='ecc', values='sd')
        g = g.loc[:, g.notna().sum() >= 10]
        m, se = g.mean(), g.sem()
        ax.fill_between(m.index, m - se, m + se, color=AREAS[roi], alpha=.25, lw=0)
        ax.plot(m.index, m.values, color=AREAS[roi], lw=1.2)
        ends.append((roi, m.index[-1], m.values[-1]))
    ends.sort(key=lambda e: e[2])
    y = -np.inf
    for roi, x, v in ends:
        y = max(v, y + .2)
        ax.text(3.3, y, roi, color=AREAS[roi], fontsize=6.5, va='center', fontweight='bold')
    ax.set_xlim(0, 3.3)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel('pRF eccentricity (°)')
    ax.set_ylabel('pRF size σ (°)')


def panel_share(ax):
    d = pd.read_csv(R2_TSV, sep='\t')
    d['share'] = 100 * d.n_signal / d.n_mapped
    rng = np.random.default_rng(1)
    for i, roi in enumerate(STRIP_ROIS):
        v = d[d.roi == roi].share.values
        ax.scatter(i + rng.uniform(-.2, .2, len(v)), v, s=5, color=AREAS[roi], alpha=.55, lw=0, zorder=2)
        ax.plot([i - .3, i + .3], [np.median(v)] * 2, color=FG, lw=1.4, zorder=3, solid_capstyle='butt')
    ax.set_xticks(range(len(STRIP_ROIS)))
    ax.set_xticklabels(STRIP_ROIS)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlim(-.6, len(STRIP_ROIS) - .4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel('Significant voxels (%)')


def panel_coverage(fig, gs):
    d = pd.read_csv(COVER_TSV, sep='\t')
    sub = gs.subgridspec(1, 4, wspace=.05)
    first = None
    centres = .5 * (XY_BINS[:-1] + XY_BINS[1:])
    from scipy.ndimage import gaussian_filter
    for k, roi in enumerate(COVER_ROIS):
        ax = fig.add_subplot(sub[k])
        first = first or ax
        h = np.zeros((len(centres),) * 2)
        g = d[d.roi == roi]
        # Each subject counts once: proportion of their voxels per bin, then the average.
        for _, x in g.groupby('subject'):
            one = np.zeros_like(h)
            one[x.ix, x.iy] = x.n
            h += one / one.sum()
        h = gaussian_filter(h / g.subject.nunique(), 1.)
        im = ax.imshow(h.T, origin='lower', extent=(-APERTURE, APERTURE, -APERTURE, APERTURE), cmap='viridis',
                       vmin=0, vmax=h.max(), interpolation='bilinear')
        rim = Circle((0, 0), APERTURE, fill=False, ec=MUTED, lw=.5)
        ax.add_patch(rim)
        im.set_clip_path(Circle((0, 0), APERTURE, transform=ax.transData))
        ax.plot(0, 0, '+', color='w', ms=4, mew=.7)
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-3.4, 3.4)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(roi, color=AREAS[roi], fontsize=7, fontweight='bold', pad=1)
    return first


def panel_cv(ax):
    d = pd.read_csv(CV_TSV, sep='\t')
    d = d[(d.selection == 'reliable') & (d.roi == 'all') & d.model.isin(range(1, 7))]
    s = d.groupby(['subject', 'model']).median_r2.mean().reset_index()
    family = {1: 0, 3: 0, 2: 1, 4: 1, 5: 2, 6: 2}
    flexible = {1: False, 2: False, 5: False, 3: True, 4: True, 6: True}
    s['x'] = s.model.map(family)
    s['flex'] = s.model.map(flexible)
    colors = {False: MUTED, True: FG}
    for flex, g in s.groupby('flex'):
        m = g.groupby('x').median_r2.mean()
        se = g.groupby('x').median_r2.sem()
        ax.errorbar(m.index, m.values, yerr=se.values, color=colors[flex], marker='o', ms=3.5, lw=1.1,
                    elinewidth=.8, capsize=0)
        ax.text(1, m.values[1] + (.0075 if flex else -.0075), 'Flexible HRF' if flex else 'Fixed HRF', color=colors[flex],
                fontsize=6.5, ha='right' if flex else 'left', va='center')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Gaussian', 'DoG', 'DN'])
    ax.set_xlim(-.25, 2.25)
    ax.set_ylabel('Held-out R²')
    ax.set_ylim(.19, .26)
    ax.set_yticks([.2, .22, .24])


def plot(subject, in_dir, out):
    fig = plt.figure(figsize=(7.25, 8.1))
    outer = fig.add_gridspec(3, 1, height_ratios=[2.1, 3.45, 1.15], hspace=.27,
                             left=.065, right=.985, top=.985, bottom=.055)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.95], wspace=.06)
    ax = fig.add_subplot(top[0])
    panel_display(ax)
    letter(ax, 'a', x=.03, y=.97)
    ax = fig.add_subplot(top[1])
    panel_trial(fig, ax)
    letter(ax, 'b', y=.97)
    maps = outer[1].subgridspec(3, 5, height_ratios=[1.4, .3, 1.4], wspace=.03, hspace=.04)
    first = panel_maps(fig, maps, subject, in_dir)
    letter(first, 'c', x=-.09, y=1.04)
    lower = maps[2, 2:].subgridspec(1, 3, width_ratios=[.02, 1.2, 1], wspace=.9)
    ax = fig.add_subplot(lower[1])
    panel_sigma(ax)
    letter(ax, 'd', x=-.2, y=.98)
    ax = fig.add_subplot(lower[2])
    panel_share(ax)
    letter(ax, 'e', x=-.3, y=.98)
    bottom = outer[2].subgridspec(1, 2, width_ratios=[2.2, 1], wspace=.22)
    first = panel_coverage(fig, bottom[0])
    letter(first, 'f', x=-.03, y=1.02)
    ax = fig.add_subplot(bottom[1])
    panel_cv(ax)
    letter(ax, 'g', x=-.2, y=.98)
    for ax in fig.axes:
        if ax.spines['bottom'].get_visible() and ax.get_xlabel() + ax.get_ylabel():
            ax.spines['left'].set_position(('outward', 3))
            ax.spines['bottom'].set_position(('outward', 3))
    fig.savefig(out, dpi=400)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=200)
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['collect', 'plot'])
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    ap.add_argument('--subject', type=int, default=19)
    ap.add_argument('--in-dir', default='notes/figures/fig1_r2low')
    ap.add_argument('--out', default='notes/figures/fig1_paradigm_prf.pdf')
    a = ap.parse_args()
    if a.action == 'collect':
        collect(a.bids_folder)
    else:
        plot(a.subject, a.in_dir, a.out)
