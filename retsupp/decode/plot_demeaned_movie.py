"""Render a decoded npz as a *demeaned* movie / frame strip.

For each pixel, subtracts the time-mean before plotting, so the constant
(static) component is removed and the time-varying signal (i.e. the bar
tracking) pops out. Uses a diverging colormap centred on zero.

    python -m retsupp.decode.plot_demeaned_movie \\
        --npz notes/data/decode_sweep/sub-02_V1_ses-1_run-1/l2-0.1_lr-0.05.npz \\
        --out notes/figures/demeaned_l2-0.1_lr-0.05.pdf

Also writes a one-line stats summary to stdout:
    mean decoded(t, pixel-on-bar) - mean decoded(t, pixel-never-on)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-frames', type=int, default=12)
    p.add_argument('--gif', type=Path, default=None,
                   help='Optional GIF output (full 258-frame movie).')
    p.add_argument('--fps', type=int, default=8)
    args = p.parse_args()

    with np.load(args.npz) as f:
        decoded = f['decoded']         # (T, R, R)
        par = f['paradigm']            # (T, R, R)  full bar + distractors
        grid = f['grid']               # (R*R, 2)

    T, R, _ = decoded.shape
    demeaned = decoded - decoded.mean(axis=0, keepdims=True)
    vmax = float(np.quantile(np.abs(demeaned), 0.99))

    extent = [float(grid[:, 0].min()), float(grid[:, 0].max()),
              float(grid[:, 1].min()), float(grid[:, 1].max())]

    # Stats: bar-vs-never temporal contrast in demeaned space.
    bar_ever = (par > 0.5).any(axis=0)
    bar_now_mask = par > 0.5
    never_mask = ~bar_ever[None, :, :].repeat(T, axis=0)
    on_bar = demeaned[bar_now_mask].mean() if bar_now_mask.sum() else np.nan
    on_never = demeaned[never_mask].mean() if never_mask.sum() else np.nan
    print(f'Demeaned signal at active-stim pixels: {on_bar:.4f}')
    print(f'Demeaned signal at never-stim pixels:  {on_never:.4f}')
    print(f'Temporal contrast (on - never):        {on_bar - on_never:.4f}')

    # Frame-strip PDF.
    pick = np.linspace(0, T - 1, args.n_frames).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5),
                              sharex=True, sharey=True)
    for ax, t in zip(axes.flat, pick):
        ax.imshow(demeaned[t], extent=extent, origin='lower',
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if par[t].max() > 0:
            ax.contour(par[t], levels=[0.5], colors='k',
                       linewidths=0.8, origin='lower', extent=extent)
        ax.set_title(f'TR {t}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'Demeaned decoded stimulus  ({args.npz.stem})  '
                  f'+/-vmax={vmax:.3f}',
                  fontsize=12, weight='bold')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote: {args.out}')

    if args.gif is not None:
        import matplotlib.animation as animation
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        im = ax2.imshow(demeaned[0], extent=extent, origin='lower',
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        cont = [None]
        ax2.set_xticks([])
        ax2.set_yticks([])
        ttl = ax2.set_title('TR 0', fontsize=11)

        def update(t):
            im.set_data(demeaned[t])
            if cont[0] is not None:
                try:
                    cont[0].remove()
                except (AttributeError, ValueError):
                    for c in getattr(cont[0], 'collections', []):
                        c.remove()
            if par[t].max() > 0:
                cont[0] = ax2.contour(par[t], levels=[0.5], colors='k',
                                       linewidths=0.8, origin='lower',
                                       extent=extent)
            else:
                cont[0] = None
            ttl.set_text(f'TR {t}')
            return [im]

        anim = animation.FuncAnimation(fig2, update, frames=T,
                                        interval=1000 / args.fps, blit=False)
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        anim.save(args.gif, writer='pillow', fps=args.fps)
        plt.close(fig2)
        print(f'Wrote: {args.gif}')


if __name__ == '__main__':
    main()
