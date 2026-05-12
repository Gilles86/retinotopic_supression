"""Render a decoded npz as a *demeaned* movie / frame strip.

For each pixel, subtracts the time-mean before plotting, so the constant
(static) component is removed and the time-varying signal (i.e. the bar
tracking) pops out.

Adds an optional HP-distractor overlay: pass ``--hp <corner>`` to mark
the HP location with a thick magenta ring and the 3 LP locations with
white rings. With ``--corner-traces``, also writes a companion PDF of
the demeaned decoded value sampled inside each corner's disk over time
(HP highlighted), plus a one-line summary HP vs LP-mean.

    python -m retsupp.decode.plot_demeaned_movie \\
        --npz notes/data/decode_sweep/m4/sub-02_V1_ses-1_run-1/l2-0.01_lr-0.1.npz \\
        --out notes/figures/decode_sweep/m4/demeaned_l2-0.01_lr-0.1.pdf \\
        --gif notes/figures/decode_sweep/m4/demeaned_l2-0.01_lr-0.1.gif \\
        --hp upper_left --corner-traces
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


CORNERS = {
    'upper_right': (1, 1),
    'upper_left':  (-1, 1),
    'lower_left':  (-1, -1),
    'lower_right': (1, -1),
}


def corner_xy(corner: str, ecc: float = 4.0) -> tuple[float, float]:
    sx, sy = CORNERS[corner]
    return sx * ecc / np.sqrt(2), sy * ecc / np.sqrt(2)


def sample_disk(decoded: np.ndarray, grid: np.ndarray,
                cx: float, cy: float, radius: float = 0.4) -> np.ndarray:
    """decoded: (T, R, R), grid: (R*R, 2). Returns (T,) mean inside disk."""
    R = decoded.shape[-1]
    flat = decoded.reshape(decoded.shape[0], -1)  # (T, R*R)
    d2 = (grid[:, 0] - cx) ** 2 + (grid[:, 1] - cy) ** 2
    mask = d2 <= radius ** 2
    if mask.sum() == 0:
        mask = np.zeros(grid.shape[0], dtype=bool)
        mask[d2.argmin()] = True
    return flat[:, mask].mean(axis=1)


def add_corner_markers(ax, hp: str | None, ecc: float = 4.0,
                       ring_radius: float = 0.45):
    """Draw thick rings at the 4 corner positions; HP magenta, LP white."""
    from matplotlib.patches import Circle
    for name in CORNERS:
        cx, cy = corner_xy(name, ecc)
        is_hp = (name == hp)
        ax.add_patch(Circle((cx, cy), ring_radius, fill=False,
                            edgecolor='magenta' if is_hp else 'white',
                            linewidth=2.0 if is_hp else 1.2,
                            zorder=10))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-frames', type=int, default=12)
    p.add_argument('--gif', type=Path, default=None,
                   help='Optional GIF output (full 258-frame movie).')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--cmap', default='viridis',
                   help='Matplotlib colormap (default viridis; try magma, '
                        'plasma, inferno, rocket, mako).')
    p.add_argument('--vmax-quantile', type=float, default=0.99,
                   help='Quantile of |demeaned| used for vmax (default 0.99; '
                        'larger = less tight / less saturation).')
    p.add_argument('--vmin', default='sym',
                   help="Lower colorbar limit. 'sym' = -vmax (default, "
                        'diverging); pass 0 to clip negatives (sequential).')
    p.add_argument('--no-demean', action='store_true',
                   help='Show raw decoded values instead of demeaned. '
                        'Auto-sets vmin=0 unless explicitly overridden.')
    p.add_argument('--bar-color', default='white',
                   help='Bar/paradigm outline color (default white).')
    p.add_argument('--bar-lw', type=float, default=2.0,
                   help='Bar outline linewidth (default 2.0).')
    p.add_argument('--hp', default=None,
                   choices=list(CORNERS.keys()),
                   help='HP-distractor corner (one of upper_right etc.). '
                        'If given, marks HP with magenta ring, LPs white.')
    p.add_argument('--corner-traces', action='store_true',
                   help='Also write a per-corner timecourse PDF '
                        '(<out>_corners.pdf) and print HP vs LP summary.')
    p.add_argument('--corner-radius', type=float, default=0.4,
                   help='Radius (deg) of disk used to sample at each corner.')
    args = p.parse_args()

    with np.load(args.npz) as f:
        decoded = f['decoded']         # (T, R, R)
        par = f['paradigm']            # (T, R, R)  bar + distractors
        grid = f['grid']               # (R*R, 2)

    T, R, _ = decoded.shape
    if args.no_demean:
        demeaned = decoded
        vmax = float(np.quantile(decoded, args.vmax_quantile))
        vmin = 0.0 if args.vmin == 'sym' else float(args.vmin)
    else:
        demeaned = decoded - decoded.mean(axis=0, keepdims=True)
        vmax = float(np.quantile(np.abs(demeaned), args.vmax_quantile))
        vmin = -vmax if args.vmin == 'sym' else float(args.vmin)

    extent = [float(grid[:, 0].min()), float(grid[:, 0].max()),
              float(grid[:, 1].min()), float(grid[:, 1].max())]

    # Frame-strip PDF.
    pick = np.linspace(0, T - 1, args.n_frames).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5),
                              sharex=True, sharey=True)
    for ax, t in zip(axes.flat, pick):
        ax.imshow(demeaned[t], extent=extent, origin='lower',
                   cmap=args.cmap, vmin=vmin, vmax=vmax)
        if par[t].max() > 0:
            ax.contour(par[t], levels=[0.5],
                       colors=args.bar_color,
                       linewidths=args.bar_lw,
                       origin='lower', extent=extent)
        add_corner_markers(ax, args.hp)
        ax.set_title(f'TR {t}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    mode = 'Raw' if args.no_demean else 'Demeaned'
    title = (f'{mode} decoded stimulus  ({args.npz.stem})  '
             f'vmin={vmin:+.3f} vmax={vmax:+.3f}  cmap={args.cmap}')
    if args.hp:
        title += f'  |  HP={args.hp} (magenta), LP=white'
    fig.suptitle(title, fontsize=13, weight='bold')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote: {args.out}')

    if args.gif is not None:
        import matplotlib.animation as animation
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        im = ax2.imshow(demeaned[0], extent=extent, origin='lower',
                        cmap=args.cmap, vmin=vmin, vmax=vmax)
        cont = [None]
        ax2.set_xticks([])
        ax2.set_yticks([])
        add_corner_markers(ax2, args.hp)
        ttl = ax2.set_title('TR 0', fontsize=12)

        def update(t):
            im.set_data(demeaned[t])
            if cont[0] is not None:
                try:
                    cont[0].remove()
                except (AttributeError, ValueError):
                    for c in getattr(cont[0], 'collections', []):
                        c.remove()
            if par[t].max() > 0:
                cont[0] = ax2.contour(par[t], levels=[0.5],
                                       colors=args.bar_color,
                                       linewidths=args.bar_lw,
                                       origin='lower', extent=extent)
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

    if args.corner_traces:
        traces = {}
        for name in CORNERS:
            cx, cy = corner_xy(name)
            traces[name] = sample_disk(demeaned, grid, cx, cy,
                                       radius=args.corner_radius)

        print('\nHP/LP demeaned-decoded summary '
              f'(radius={args.corner_radius} deg disk):')
        for name, trace in traces.items():
            tag = 'HP' if name == args.hp else 'LP'
            print(f'  [{tag}] {name:13s}  mean={trace.mean():+.4f}  '
                  f'std={trace.std():.4f}')
        if args.hp:
            lp_means = [traces[n].mean() for n in CORNERS if n != args.hp]
            hp_mean = traces[args.hp].mean()
            print(f'\n  HP mean = {hp_mean:+.4f}')
            print(f'  LP mean of means = {np.mean(lp_means):+.4f}')
            print(f'  HP - LP_mean      = {hp_mean - np.mean(lp_means):+.4f}')

        fig3, ax3 = plt.subplots(figsize=(11, 5))
        for name in CORNERS:
            is_hp = (name == args.hp)
            ax3.plot(traces[name],
                     color='magenta' if is_hp else 'tab:gray',
                     lw=2.0 if is_hp else 1.0,
                     alpha=1.0 if is_hp else 0.7,
                     label=f'{"HP " if is_hp else "LP "}{name}',
                     zorder=3 if is_hp else 2)
        ax3.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax3.set_xlabel('TR')
        ax3.set_ylabel('Demeaned decoded (disk-mean)')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.set_title('Demeaned decoded drive at the 4 search-array '
                      f'corners  ({args.npz.stem})',
                      fontsize=13, weight='bold')
        fig3.tight_layout()
        out_corners = args.out.with_name(args.out.stem + '_corners.pdf')
        fig3.savefig(out_corners, bbox_inches='tight')
        plt.close(fig3)
        print(f'Wrote: {out_corners}')


if __name__ == '__main__':
    main()
