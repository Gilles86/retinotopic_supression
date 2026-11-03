"""Figure 1 maps: pycortex renders of ``render_fig1_prf`` in one row, with their keys.

Usage (pycortex2 env)::

    ~/mambaforge/envs/pycortex2/bin/python retsupp/visualize/pycortex/compose_fig1_prf.py 23
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.titlesize': 8, 'axes.linewidth': .8,
    'xtick.direction': 'out', 'xtick.major.size': 2.5, 'xtick.major.width': .8,
    'xtick.labelsize': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})
FG = '#1F2933'
AREAS = {'V1': '#3D5A80', 'V2': '#2A9D8F', 'V3': '#8AB17D', 'hV4': '#E9C46A', 'V3AB': '#7209B7',
         'VO': '#F4A261', 'LO': '#E76F51', 'TO': '#B5179E', 'IPS': '#0077B6', 'FEF': '#D62828'}
PANELS = [('areas', 'Visual areas'), ('angle', 'Polar angle'), ('ecc', 'Eccentricity (°)'),
          ('sd', 'pRF size σ (°)'), ('r2', 'R²')]
VIEW_NAMES = {'pole': 'Posterior', 'dorsal': 'Dorsal', 'ventral': 'Ventral'}
BARS = {'ecc': ('nipy_spectral', 0, 4, [0, 2, 4]), 'sd': ('nipy_spectral', 0, 3, [0, 1.5, 3]),
        'r2': ('inferno', 0, .6, [0, .3, .6])}


def theta_value(theta, hemi):
    if hemi == 'R':
        v = np.mod(theta, 2 * np.pi)
        return np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)
    v = np.mod(theta + np.pi, 2 * np.pi)
    return 1. - np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)


def angle_key(ax):
    """One wheel per visual field, as coded: the colour scale spans 270°, so it runs 45°
    past each vertical meridian (white line) into the other field. Further than that,
    ``theta_value`` clips to the red at the ends of the scale."""
    g = np.linspace(-1, 1, 401)
    x, y = np.meshgrid(g, g)
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    for cx, hemi, side, wedge in ((-1.25, 'R', 'L', x <= np.abs(y)), (1.25, 'L', 'R', x >= -np.abs(y))):
        rgba = mpl.colormaps['hsv'](theta_value(th, hemi))
        rgba[..., 3] = (r <= 1) & wedge
        ax.imshow(rgba, extent=(cx - 1, cx + 1, -1, 1), origin='lower', interpolation='bilinear')
        ax.plot([cx, cx], [-1, 1], color='white', lw=.8)
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-1.05, 1.05)
    ax.text(-2.35, 0, 'L', ha='right', va='center', color=FG, fontsize=6.5)
    ax.text(2.35, 0, 'R', ha='left', va='center', color=FG, fontsize=6.5)
    ax.set_axis_off()


def area_key(ax):
    for i, (name, col) in enumerate(AREAS.items()):
        ax.text(.02 + .2 * (i % 5), .68 - .5 * (i // 5), name, color=col,
                fontsize=6, fontweight='bold', va='center', transform=ax.transAxes)
    ax.set_axis_off()


def bar_key(fig, ax, cmap, lo, hi, ticks):
    pos = ax.get_position()
    ax.set_axis_off()
    cax = fig.add_axes([pos.x0 + .18 * pos.width, pos.y0 + .55 * pos.height,
                        .64 * pos.width, .22 * pos.height])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(lo, hi), cmap),
                      cax=cax, orientation='horizontal', ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(pad=1.5)


def main(subject, in_dir, out, views):
    in_dir = Path(in_dir)
    n = len(views)
    fig, axes = plt.subplots(n + 1, 5, figsize=(7.25, .2 + 1.3 * n + .42),
                             height_ratios=[1] * n + [.32],
                             gridspec_kw=dict(wspace=.03, hspace=.02))
    for i, view in enumerate(views):
        images = {p: Image.open(in_dir / f'sub-{subject:02d}_{p}_{view}.png').convert('RGBA')
                  for p, _ in PANELS}
        boxes = np.array([im.getbbox() for im in images.values()])
        box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
        for j, (par, title) in enumerate(PANELS):
            im = images[par].crop(box)
            bg = Image.new('RGBA', im.size, 'white')
            bg.alpha_composite(im)
            axes[i, j].imshow(np.asarray(bg.convert('RGB')), interpolation='none')
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(title, color=FG, pad=3)
        axes[i, 0].text(-.02, .5, VIEW_NAMES[view], transform=axes[i, 0].transAxes, rotation=90,
                        ha='right', va='center', color=FG, fontsize=8)
    area_key(axes[n, 0])
    angle_key(axes[n, 1])
    for j, (par, _) in enumerate(PANELS[2:], start=2):
        bar_key(fig, axes[n, j], *BARS[par])
    fig.savefig(out, dpi=600)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=220)
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--in-dir', default='notes/figures/fig1_inflated')
    ap.add_argument('--out', default=None)
    ap.add_argument('--views', nargs='+', default=['pole', 'dorsal', 'ventral'])
    a = ap.parse_args()
    main(a.subject, a.in_dir, a.out or f'{a.in_dir}/fig1_prf_maps_sub-{a.subject:02d}.pdf', a.views)
