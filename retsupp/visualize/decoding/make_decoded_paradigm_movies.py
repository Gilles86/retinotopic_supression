"""MP4 movies of decoded paradigm tensors.

Two entry points:

1. **Single-run smoke movie** -- render a ``(T, R, R)`` decoded tensor
   from the smoke test (or any single ``decode_runwise`` output) with
   the true paradigm contour overlaid in cyan and the 4 ring positions
   marked. HP location, if known, gets a special marker.

       python -m retsupp.visualize.decoding.make_decoded_paradigm_movies \\
           --npz notes/data/decoded_smoke_sub-02_V1.npz \\
           --out notes/figures/decoding/decoded_smoke_sub-02_V1.mp4

2. **Mean / event-locked group movie** -- render a ``(T, R, R)`` group
   tensor produced by ``retsupp.decode.aggregate_decoded`` (plan §2E, §2F).
   Same plumbing; ``--paradigm`` overlay is optional. With ``--grid 2x4``
   the renderer expects a list of 8 npz paths and tiles them into one
   movie (the 2F distractor×target × 4-location grid).

       python -m retsupp.visualize.decoding.make_decoded_paradigm_movies \\
           --npz notes/data/decoded_group_roi-V1_mean.npz \\
           --out notes/figures/decoding/mean_decoded_roi-V1.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from retsupp.utils.data import distractor_locations as _DISTRACTOR_LOCS
# Convention: keys in distractor_locations are space-form ('upper right'),
# whereas Subject.get_hpd_locations returns underscore form
# ('upper_right'). Build an underscore-keyed copy to bridge.
RING_LOCS = {
    k.replace(' ', '_'): v for k, v in _DISTRACTOR_LOCS.items()
}


def _decoded_extent(grid: np.ndarray) -> list[float]:
    """imshow extent [xmin, xmax, ymin, ymax] from a (G, 2) grid."""
    return [float(grid[:, 0].min()), float(grid[:, 0].max()),
            float(grid[:, 1].min()), float(grid[:, 1].max())]


def render_decoded_movie(decoded: np.ndarray,
                          grid: np.ndarray | None = None,
                          extent: list[float] | None = None,
                          paradigm: np.ndarray | None = None,
                          hp_location: str | None = None,
                          ring_locations: bool = True,
                          title: str = '',
                          fps: int = 6,
                          out_path: Path | str | None = None,
                          vmax_quantile: float = 0.99,
                          cmap: str = 'magma',
                          ax_size: float = 3.5,
                          tr: float = 1.6):
    """Render ``decoded`` (T, R, R) to MP4.

    Parameters
    ----------
    decoded : (T, R, R) array
        Decoded intensity per TR. ``imshow(origin='lower')`` is used,
        so the array is indexed ``[t, y_row, x_col]`` with row 0 at
        the bottom of the plot (positive y).
    grid : (G, 2) array, optional
        Grid coordinates in degrees. If omitted, ``extent`` must be
        provided. ``G == R*R``.
    extent : [xmin, xmax, ymin, ymax], optional
        Alternative to ``grid``; passed straight to imshow.
    paradigm : (T, R, R) array, optional
        True stimulus to overlay as a cyan contour at level 0.5.
    hp_location : str, optional
        ``'upper_right'`` / ``'upper_left'`` / ``'lower_left'`` /
        ``'lower_right'``. If set, that ring corner is drawn in
        magenta and labelled "HP".
    ring_locations : bool
        If True, mark the 4 ring positions with white circles.
    title : str
        Suptitle.
    fps : int
        Movie frame rate. TR = 1.6 s, so fps=6 plays at ~10x real-time.
    out_path : Path or None
        If given, save MP4; else return the FuncAnimation object.
    vmax_quantile : float
        Quantile of ``decoded`` used to set ``vmax``.
    """
    if grid is None and extent is None:
        raise ValueError('Provide either `grid` or `extent`.')
    if extent is None:
        extent = _decoded_extent(grid)

    n_frames, R, R2 = decoded.shape
    assert R == R2, f'decoded must be (T, R, R); got {decoded.shape}'

    vmax = float(np.quantile(np.abs(decoded), vmax_quantile))
    # Symmetric color limits when the data crosses zero (e.g. group-mean
    # decoded paradigms with baseline subtraction); else positive only.
    has_negative = float(decoded.min()) < -0.05 * vmax
    vmin = -vmax if has_negative else 0.0

    fig, ax = plt.subplots(figsize=(ax_size, ax_size))
    im = ax.imshow(decoded[0], extent=extent, origin='lower',
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation='bilinear')
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')
    ax.set_aspect('equal')

    contour_artists: list = []

    if ring_locations:
        for name, (cx, cy) in RING_LOCS.items():
            is_hp = (name == hp_location)
            ax.plot(cx, cy, 'o',
                    mfc='none',
                    mec='magenta' if is_hp else 'white',
                    ms=12 if is_hp else 9,
                    mew=1.5 if is_hp else 1.0)
            if is_hp:
                ax.text(cx, cy + 0.6, 'HP', color='magenta',
                        ha='center', va='bottom', fontsize=8,
                        fontweight='bold')

    tr_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                      color='white', fontsize=10, va='top', ha='left',
                      bbox=dict(boxstyle='round', facecolor='black',
                                edgecolor='none', alpha=0.6))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Decoded')

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    def update(t: int):
        im.set_array(decoded[t])
        tr_text.set_text(f'TR {t}  ({t * tr:.1f} s)')
        # ContourSet API differs across matplotlib versions; on >=3.8
        # the whole set is a single artist with .remove(); on <3.8 it
        # is `.collections`. Just try both.
        while contour_artists:
            c = contour_artists.pop()
            try:
                c.remove()
            except Exception:
                pass
        if paradigm is not None and paradigm[t].max() > 0:
            cs = ax.contour(paradigm[t], levels=[0.5],
                            colors='cyan', linewidths=1.2,
                            origin='lower', extent=extent)
            contour_artists.append(cs)
        return [im, tr_text, *contour_artists]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=1000 / fps, blit=False)
    if out_path is None:
        return anim, fig

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400,
                                      codec='libx264',
                                      extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    return out_path


def _load_npz(npz_path: Path) -> dict:
    """Load decoded npz with flexible key set.

    Supported keys
    --------------
    Required: ``decoded`` (T, R, R) or ``paradigm`` (T, R, R).
    Optional: ``grid`` (G, 2), ``extent`` (4,), ``hp_location`` (str),
    ``paradigm`` (T, R, R), ``subject``, ``roi``, ``session``, ``run``.
    """
    with np.load(npz_path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True, type=Path,
                   help='Decoded .npz (must contain key "decoded").')
    p.add_argument('--out', required=True, type=Path,
                   help='Output MP4.')
    p.add_argument('--fps', type=int, default=6)
    p.add_argument('--title', default=None,
                   help='Title; default constructed from npz metadata.')
    p.add_argument('--no-paradigm-overlay', action='store_true',
                   help='Skip cyan stimulus contour even if paradigm '
                        'is present in the npz.')
    p.add_argument('--vmax-quantile', type=float, default=0.99)
    p.add_argument('--cmap', default='magma')
    p.add_argument('--hp-location', default=None,
                   help='Override HP location ("upper_left" etc.) when '
                        'the npz lacks an hp_location field (older smoke '
                        'tests). Run-level decodes save it automatically.')
    args = p.parse_args()

    d = _load_npz(args.npz)
    decoded = d['decoded']
    if decoded.ndim != 3:
        raise ValueError(f'`decoded` must be (T, R, R); got {decoded.shape}')

    grid = d.get('grid')
    extent = d.get('extent')
    paradigm = None if args.no_paradigm_overlay else d.get('paradigm')

    # Reconstruct an HP location and title from npz metadata when possible.
    sub = d.get('subject')
    roi = d.get('roi')
    ses = d.get('session')
    run = d.get('run')
    hp = d.get('hp_location')
    hp_str = str(hp) if hp is not None and hp.size > 0 else None
    if args.hp_location is not None:
        hp_str = args.hp_location

    if args.title is None:
        bits = []
        if sub is not None and sub.size > 0:
            bits.append(f'sub-{int(sub):02d}')
        if roi is not None and roi.size > 0:
            bits.append(str(roi))
        if ses is not None and ses.size > 0:
            bits.append(f'ses-{int(ses)}')
        if run is not None and run.size > 0:
            bits.append(f'run-{int(run)}')
        title = '  '.join(bits) or args.npz.name
        if hp_str:
            title += f'    HP = {hp_str}'
    else:
        title = args.title

    out_path = render_decoded_movie(
        decoded=decoded,
        grid=grid.reshape(-1, 2) if (grid is not None and grid.ndim == 2 and grid.shape[1] == 2) else None,
        extent=extent.tolist() if extent is not None else None,
        paradigm=paradigm,
        hp_location=hp_str,
        title=title,
        fps=args.fps,
        out_path=args.out,
        vmax_quantile=args.vmax_quantile,
        cmap=args.cmap,
    )
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
