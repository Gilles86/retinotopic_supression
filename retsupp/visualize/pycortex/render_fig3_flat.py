"""Flatmap renders of one subject's visual areas, polar angle and eccentricity, for Figure 3a.

Same per-vertex colours as ``render_fig1_prf.py`` (curvature, R² ramp, white area borders), shown on the subject's
autoflattened surfaces (``autoflatten`` on the cluster, imported with ``cortex.freesurfer.import_flat``) and cropped to
the occipital patch of both hemispheres. No browser needed (``quickflat``). Runs in pycortex2::

    PYTHONPATH=. ~/mambaforge/envs/pycortex2/bin/python retsupp/visualize/pycortex/render_fig3_flat.py 23 \\
        --out-dir notes/figures/fig1_r2low
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib.pyplot as plt
import numpy as np

from retsupp.visualize.pycortex.render_fig1_prf import area_rgb, border_mask, read_varea, vertex_rgb


def flat_border(subject, cx_subject):
    """Area borders on the flat surface's triangles (the flat patch drops the cut vertices)."""
    return border_mask(subject, cx_subject)


def crop_box(subject, cx, pad=12.):
    """Flat-map window around the visual areas (V1 to IPS), both hemispheres."""
    va = read_varea(subject)
    pts, _ = cortex.db.get_surf(cx, 'flat', merge=True, nudge=True)
    keep = np.isin(va, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 101])
    x, y = pts[keep, 0], pts[keep, 1]
    xlim = (np.percentile(x, .5) - pad, np.percentile(x, 99.5) + pad)
    ylim = (np.percentile(y, .5) - pad, np.percentile(y, 99.5) + pad)
    return xlim, ylim


def inset(subject, out_dir, side='medial', turn=50., gap=8.):
    """The inflated surface, seen from behind, with the part shown on the flat maps in the area colours and outlined.

    Each hemisphere is turned about the vertical axis by ``turn`` degrees so that its medial wall (``side='medial'``,
    which shows the cut along the calcarine sulcus) or its lateral wall faces the viewer. Drawn with matplotlib (painter's algorithm, Lambert shading); the
    webgl viewer would show the flat surface once one is in the store.
    """
    from matplotlib.collections import PolyCollection
    cx = f'retsupp.sub-{subject:02d}'
    xlim, ylim = crop_box(subject, cx)
    shift = cortex.db.get_surf(cx, 'flat', merge=True, nudge=True)[0]
    rgb_all = area_rgb(subject, cx, np.zeros(len(shift), bool)).astype(float) / 255
    polys, cols, depth = [], [], []
    offset = 0
    for h, sign in (('lh', -1), ('rh', 1)):
        pts, tri = cortex.db.get_surf(cx, 'inflated', hemisphere=h)
        flat = cortex.db.get_surf(cx, 'flat', hemisphere=h)[0]
        n = len(pts)
        sh = shift[offset:offset + n]
        rgb = rgb_all[offset:offset + n].copy()
        offset += n
        inside = ((np.abs(flat[:, :2]).sum(1) > 0) & (sh[:, 0] > xlim[0]) & (sh[:, 0] < xlim[1]) & (sh[:, 1] > ylim[0])
                  & (sh[:, 1] < ylim[1]))
        rgb[~inside] = rgb[~inside] * .5 + .5
        m = inside[tri].sum(1)
        edge = np.zeros(n, bool)
        edge[np.unique(tri[(m > 0) & (m < 3)])] = True
        rgb[edge] = .12
        p = pts - pts.mean(0)
        a = np.deg2rad(sign * turn * (1 if side == 'medial' else -1))
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        p = p @ rot.T
        p[:, 0] += sign * gap / 2 - (p[:, 0].max() if sign < 0 else p[:, 0].min())
        v = p[tri]
        nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        front = nrm[:, 1] < 0  # faces the viewer at -y
        shade = .35 + .65 * np.clip(-nrm[:, 1], 0, 1)
        c = rgb[tri].mean(1) * shade[:, None]
        polys.append(v[front][:, :, [0, 2]])
        cols.append(c[front])
        depth.append(v[front][:, :, 1].mean(1))
    polys, cols, depth = np.concatenate(polys), np.concatenate(cols), np.concatenate(depth)
    order = np.argsort(-depth)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.add_collection(PolyCollection(polys[order], facecolors=np.clip(cols[order], 0, 1), edgecolors='none', antialiased=False))
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_axis_off()
    fn = Path(out_dir) / f'sub-{subject:02d}_flatwindow_{side}.png'
    fig.savefig(fn, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print('wrote', fn)


def main(subject, out_dir, pad=12., height=2400):
    cx = f'retsupp.sub-{subject:02d}'
    out_dir = Path(out_dir)
    border = flat_border(subject, cx)
    xlim, ylim = crop_box(subject, cx, pad)
    print('crop', xlim, ylim)
    for par in ('areas', 'angle', 'ecc'):
        fn = out_dir / f'sub-{subject:02d}_{par}_flat.png'
        if fn.exists():
            print('exists', fn)
            continue
        rgb = area_rgb(subject, cx, border) if par == 'areas' else vertex_rgb(subject, cx, par, border)
        ds = cortex.VertexRGB(rgb[:, 0], rgb[:, 1], rgb[:, 2], cx)
        fig = cortex.quickflat.make_figure(ds, with_rois=False, with_sulci=False, with_labels=False, with_curvature=False,
                                           with_colorbar=False, height=height, nanmean=True)
        ax = fig.axes[0]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        fig.savefig(fn, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print('wrote', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--out-dir', default='notes/figures/fig1_r2low')
    ap.add_argument('--r2-ramp', type=float, nargs=2, default=None,
                    help='R² at which the colour starts and reaches full opacity (default: render_fig1_prf.R2_RAMP).')
    a = ap.parse_args()
    if a.r2_ramp:
        import retsupp.visualize.pycortex.render_fig1_prf as r1
        r1.R2_RAMP = tuple(a.r2_ramp)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    main(a.subject, a.out_dir)
    for side in ('lateral', 'medial'):
        if not (Path(a.out_dir) / f'sub-{a.subject:02d}_flatwindow_{side}.png').exists():
            inset(a.subject, a.out_dir, side)
