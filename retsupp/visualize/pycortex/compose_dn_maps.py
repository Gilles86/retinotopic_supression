"""DN maps: pycortex renders of ``render_fig1_prf --prf-dir prf.hrffix/model6`` in one row.

Usage (pycortex2 env)::

    ~/mambaforge/envs/pycortex2/bin/python retsupp/visualize/pycortex/compose_dn_maps.py 19 \
        --in-dir notes/figures/dn_maps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from compose_fig1_prf import FG, VIEW_NAMES

PANELS = [('sd', 'pRF size σ1 (°)', 'nipy_spectral', mpl.colors.Normalize(0, 3), [0, 1.5, 3]),
          ('srf_size', 'Size ratio σ2/σ1', 'nipy_spectral', mpl.colors.Normalize(1, 8), [1, 4, 8]),
          ('b_over_d', 'b/d', 'nipy_spectral', mpl.colors.LogNorm(.5, 50), [.5, 5, 50]),
          ('d_over_c', 'd/c', 'nipy_spectral', mpl.colors.LogNorm(5, 1000), [5, 50, 500]),
          ('r2', 'R²', 'inferno', mpl.colors.Normalize(0, .6), [0, .3, .6])]


def log_or_linear_key(fig, ax, cmap, norm, ticks):
    pos = ax.get_position()
    ax.set_axis_off()
    cax = fig.add_axes([pos.x0 + .18 * pos.width, pos.y0 + .55 * pos.height,
                        .64 * pos.width, .22 * pos.height])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax, orientation='horizontal')
    cb.set_ticks(ticks, labels=[f'{t:g}' for t in ticks])
    cb.ax.minorticks_off()
    cb.outline.set_visible(False)
    cb.ax.tick_params(pad=1.5)


def main(subject, in_dir, out, views):
    in_dir = Path(in_dir)
    n = len(views)
    fig, axes = plt.subplots(n + 1, len(PANELS), figsize=(7.25, .2 + 1.3 * n + .42),
                             height_ratios=[1] * n + [.32],
                             gridspec_kw=dict(wspace=.03, hspace=.02))
    for i, view in enumerate(views):
        images = {p[0]: Image.open(in_dir / f'sub-{subject:02d}_{p[0]}_{view}.png').convert('RGBA')
                  for p in PANELS}
        boxes = np.array([im.getbbox() for im in images.values()])
        box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
        for j, (par, title, *_rest) in enumerate(PANELS):
            im = images[par].crop(box)
            bg = Image.new('RGBA', im.size, 'white')
            bg.alpha_composite(im)
            axes[i, j].imshow(np.asarray(bg.convert('RGB')), interpolation='none')
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(title, color=FG, pad=3)
        axes[i, 0].text(-.02, .5, VIEW_NAMES[view], transform=axes[i, 0].transAxes, rotation=90,
                        ha='right', va='center', color=FG, fontsize=8)
    for j, (_par, _title, cmap, norm, ticks) in enumerate(PANELS):
        log_or_linear_key(fig, axes[n, j], cmap, norm, ticks)
    fig.savefig(out, dpi=600)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=220)
    print('Saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--in-dir', default='notes/figures/dn_maps')
    ap.add_argument('--out', default=None)
    ap.add_argument('--views', nargs='+', default=['pole', 'dorsal', 'ventral'])
    a = ap.parse_args()
    main(a.subject, a.in_dir, a.out or f'{a.in_dir}/dn_maps_sub-{a.subject:02d}.pdf', a.views)
