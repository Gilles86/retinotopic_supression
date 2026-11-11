"""Animate the retsupp PRF-bar + rectangle-distractor paradigm.

Renders sub-03, ses-1, run-1 (representative — 48 distractor trials,
multiple orientations, all 4 ring locations) on a 200x200 grid spanning
[-5, +5] deg in both axes. Distractor footprints are oriented rectangles
matching the actual PsychoPy ``visual.Rect`` rendering.

Output:
    notes/figures/rectangle_paradigm.gif

The full 258-TR run (~7 min wall time) is compressed to a ~30 s GIF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from retsupp.utils.data import Subject

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'notes' / 'figures' / 'rectangle_paradigm.gif'

SUBJECT_ID = 3
SESSION = 1
RUN = 1
RESOLUTION = 200
GRID_RADIUS = 5.0
APERTURE_R = 3.17           # bar aperture radius (deg)
RING_R = 4.0                 # ring eccentricity (deg)

# Target GIF length in seconds; choose fps so n_frames / fps ~= TARGET_SECS.
TARGET_SECS = 30.0


def main():
    bids_folder = '/data/ds-retsupp'
    sub = Subject(SUBJECT_ID, bids_folder=bids_folder)

    print(f"Building stimulus for sub-{SUBJECT_ID:02d} ses-{SESSION} run-{RUN}...")
    stim = sub.get_stimulus_with_distractors(
        session=SESSION, run=RUN,
        resolution=RESOLUTION, grid_radius=GRID_RADIUS,
        distractor_shape='rectangle',
        distractor_long_side=1.5, distractor_short_side=0.5,
    )
    n_frames = stim.shape[0]
    tr = sub.get_tr(SESSION, RUN)
    print(f"Stimulus shape: {stim.shape}, TR={tr:.3f}s, "
          f"total duration={n_frames * tr:.1f}s")

    # Pick fps so n_frames / fps == TARGET_SECS (rounded).
    fps = max(1, int(round(n_frames / TARGET_SECS)))
    print(f"GIF: fps={fps} -> duration ~ {n_frames / fps:.1f}s")

    # Ring-position centres (deg) — annotation only.
    ring_xy = np.array([
        [+RING_R / np.sqrt(2), +RING_R / np.sqrt(2)],   # UR
        [-RING_R / np.sqrt(2), +RING_R / np.sqrt(2)],   # UL
        [-RING_R / np.sqrt(2), -RING_R / np.sqrt(2)],   # LL
        [+RING_R / np.sqrt(2), -RING_R / np.sqrt(2)],   # LR
    ])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.set_xlim(-GRID_RADIUS, GRID_RADIUS)
    ax.set_ylim(-GRID_RADIUS, GRID_RADIUS)
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')

    # Imshow extent: pixels go from -GRID_RADIUS to +GRID_RADIUS on both axes.
    # origin='lower' so y increases upward (matches grid_y meshgrid).
    im = ax.imshow(
        stim[0],
        extent=(-GRID_RADIUS, GRID_RADIUS, -GRID_RADIUS, GRID_RADIUS),
        origin='lower', cmap='hot', vmin=0, vmax=1, interpolation='nearest',
    )

    # Bar aperture: dashed circle.
    ax.add_patch(Circle(
        (0, 0), APERTURE_R, fill=False, ec='white', lw=1.0,
        ls='--', alpha=0.5,
    ))
    # 4 ring positions: small open circles for annotation only.
    for (cx, cy) in ring_xy:
        ax.add_patch(Circle(
            (cx, cy), 0.2, fill=False, ec='cyan', lw=1.0, alpha=0.7,
        ))

    title = ax.set_title('', fontsize=10)

    def update(i):
        im.set_data(stim[i])
        t_sec = (i + 0.5) * tr
        title.set_text(
            f'sub-{SUBJECT_ID:02d} ses-{SESSION} run-{RUN}   '
            f'TR {i + 1}/{n_frames}   t = {t_sec:6.1f}s'
        )
        return [im, title]

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing GIF -> {OUT}")
    writer = PillowWriter(fps=fps)
    anim.save(str(OUT), writer=writer, dpi=80)
    plt.close(fig)

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Done: {OUT}  ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()
