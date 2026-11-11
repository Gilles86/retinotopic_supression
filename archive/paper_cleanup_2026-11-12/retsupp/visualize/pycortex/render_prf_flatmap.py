"""Static pycortex flatmap rendering of PRF parameters for one subject.

Saves a multi-panel PDF / PNG showing R² (alpha-thresholded), polar
angle, eccentricity, and σ on a flattened cortex.

Usage:
    python render_prf_flatmap.py 2 --model 4 \
        --out notes/figures/prf_flatmap_sub-02_m4.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib.pyplot as plt
import numpy as np

from retsupp.utils.data import Subject


def main(subject: int, model: int, bids_folder: str,
         r2_thr: float, out: Path, with_label: str = ''):
    pc_subject = f'retsupp.sub-{subject:02d}'
    sub = Subject(subject, bids_folder=Path(bids_folder))

    pars = sub.get_prf_parameters_surface(model=model)
    alpha = (pars['r2'] > r2_thr).values.astype(np.float32)

    pars['theta'] = np.arctan2(pars['y'], pars['x'])
    pars['ecc']   = np.hypot(pars['x'], pars['y'])

    panels = [
        ('R²', pars['r2'].values, 'hot', 0.0, 0.8),
        ('polar angle', pars['theta'].values, 'hsv', -np.pi, np.pi),
        ('eccentricity (deg)', pars['ecc'].values, 'nipy_spectral', 0.0, 4.0),
        ('σ (deg)', pars['sd'].values, 'nipy_spectral', 0.0, 4.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    for ax, (name, data, cmap, vmin, vmax) in zip(axes, panels):
        v = cortex.Vertex2D(
            data, alpha,
            subject=pc_subject,
            cmap=cmap, vmin=vmin, vmax=vmax, vmin2=0, vmax2=1,
        )
        cortex.quickflat.make_figure(
            v, with_curvature=True, with_colorbar=False,
            with_rois=True, with_labels=False,
            with_borders=False, fig=ax.figure, recache=False,
        )
        # Replace last axis with our subplot. Easier: render to PNG then
        # imshow into our subplot.
        # Workaround: save then reload.
        ax.set_axis_off()

    # Cleaner approach: render each as its own quickflat figure, save
    # PNGs to a temp dir, then composite.
    plt.close(fig)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        png_paths = []
        for name, data, cmap, vmin, vmax in panels:
            v = cortex.Vertex2D(
                data, alpha,
                subject=pc_subject,
                cmap=cmap, vmin=vmin, vmax=vmax, vmin2=0, vmax2=1,
            )
            png = os.path.join(tmp, f'{name}.png')
            cortex.quickflat.make_png(
                png, v, with_curvature=True, with_colorbar=True,
                with_rois=True, with_labels=False, with_borders=False,
                recache=False,
            )
            png_paths.append((name, png, cmap, vmin, vmax))

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        axes = axes.ravel()
        from matplotlib.image import imread
        for ax, (name, png, cmap, vmin, vmax) in zip(axes, png_paths):
            ax.imshow(imread(png))
            ax.set_axis_off()
            ax.set_title(f'{name}  [{vmin}, {vmax}]', fontsize=11,
                          weight='bold')
        title = (f'sub-{subject:02d}  model {model}  '
                 f'(R² > {r2_thr}; alpha = R²-thresholded)')
        if with_label:
            title += f'    [{with_label}]'
        fig.suptitle(title, fontsize=13, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight', dpi=110)
        plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--r2-thr', type=float, default=0.05)
    p.add_argument('--label', default='')
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/prf_flatmap.pdf'))
    args = p.parse_args()
    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    main(args.subject, args.model, args.bids_folder, args.r2_thr, out,
         args.label)
