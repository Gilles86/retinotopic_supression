"""Quick visual sanity check for ``Subject.get_stimulus_with_distractors``.

Renders a grid of frames from one subject/session/run that include both
bar-sweep frames and distractor-on frames, plus a profile of distractor
"on-fraction" across the whole run for each ring location.  Output is a
single PNG saved in ``notes/``.

Run example::

    python retsupp/modeling/check_extended_stimulus.py 5 \
        --bids_folder /shares/zne.uzh/gdehol/ds-retsupp \
        --output notes/extended_stimulus_check.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from retsupp.utils.data import Subject, distractor_locations


def main(subject, session=1, run=1,
         bids_folder='/data/ds-retsupp',
         output='notes/extended_stimulus_check.png',
         resolution=120, grid_radius=5.0, distractor_radius=0.4):
    sub = Subject(subject, bids_folder)

    print(f"Building extended stimulus for sub-{int(subject):02d} ses-{session} run-{run}")
    stim = sub.get_stimulus_with_distractors(
        session=session, run=run,
        resolution=resolution, grid_radius=grid_radius,
        distractor_radius=distractor_radius,
    )
    grid_x, grid_y = sub.get_extended_grid_coordinates(
        resolution=resolution, session=session, run=run,
        grid_radius=grid_radius,
    )

    tr = sub.get_tr(session=session, run=run)
    n_volumes = stim.shape[0]
    frametimes = (np.arange(n_volumes) + 0.5) * tr

    # --- Pick a few representative frames. ---
    # 1. Frames where total stimulus > 0 → some content.
    nonzero = np.where(stim.sum(axis=(1, 2)) > 0)[0]
    # 2. Frames where ring pixels light up → distractor active.
    settings = sub.get_experimental_settings(session, run)
    radius_bar = settings['radius_bar_aperture']
    out_of_bar = np.sqrt(grid_x ** 2 + grid_y ** 2) > radius_bar + 0.05
    distractor_signal = (stim * out_of_bar[None]).sum(axis=(1, 2))
    bar_signal = (stim * (~out_of_bar)[None]).sum(axis=(1, 2))
    distractor_frames = np.where(distractor_signal > 0)[0]
    # Pick 4 frames showing bar in different places + 4 frames with distractor.
    if len(nonzero) >= 8:
        bar_only = nonzero[bar_signal[nonzero] > 0]
        bar_only = bar_only[~np.isin(bar_only, distractor_frames)]
        bar_pick = (np.linspace(0, len(bar_only) - 1, 4).astype(int)
                    if len(bar_only) >= 4 else bar_only)
        bar_pick = bar_only[bar_pick] if len(bar_only) >= 4 else bar_only
    else:
        bar_pick = nonzero[: min(4, len(nonzero))]
    distractor_pick = (distractor_frames[
        np.linspace(0, len(distractor_frames) - 1, 4).astype(int)
    ] if len(distractor_frames) >= 4 else distractor_frames)

    chosen = np.concatenate([bar_pick, distractor_pick])
    chosen = np.unique(chosen)[: 8]

    # --- Build the figure. ---
    n_panels = len(chosen)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols)) + 1  # last row = time courses
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    # Frame thumbnails.
    extent = [-grid_radius, grid_radius, -grid_radius, grid_radius]
    for k, fi in enumerate(chosen):
        ax = fig.add_subplot(n_rows, n_cols, k + 1)
        ax.imshow(stim[fi], extent=extent, origin='lower',
                  vmin=0, vmax=1, cmap='magma')
        # Ring + bar-aperture circles.
        circ_bar = plt.Circle((0, 0), radius_bar, edgecolor='cyan',
                              facecolor='none', linestyle='--', linewidth=1)
        ax.add_patch(circ_bar)
        for label, (cx, cy) in distractor_locations.items():
            ax.plot(cx, cy, 'o', mfc='none', mec='lime',
                    markersize=10, mew=1)
        ax.set_title(f"frame {fi}, t={frametimes[fi]:.1f}s")
        ax.set_xlim(-grid_radius, grid_radius)
        ax.set_ylim(-grid_radius, grid_radius)
        ax.set_xticks([]); ax.set_yticks([])

    # Final row: per-distractor on-fraction over time.
    ax = fig.add_subplot(n_rows, 1, n_rows)
    pix = 2 * grid_radius / resolution
    disk_area = np.pi * distractor_radius ** 2  # approx
    # for each distractor location, average pixel value over a small disk
    # at that location → effective "on fraction" timecourse.
    for label, (cx, cy) in distractor_locations.items():
        disk_mask = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2) < distractor_radius ** 2
        if disk_mask.sum() == 0:
            continue
        timecourse = stim[:, disk_mask].mean(axis=1)
        ax.plot(frametimes, timecourse, label=label)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('mean disk intensity\n(= fraction of TR active)')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Per-distractor activation across the run')

    fig.suptitle(
        f"Extended stimulus check — sub-{int(subject):02d}, "
        f"ses-{session}, run-{run}\n"
        f"resolution={resolution}, grid_radius={grid_radius}°, "
        f"distractor_radius={distractor_radius}°",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    print(f"Saved figure -> {output}")
    print(
        f"Distractor frames: {len(distractor_frames)} / {n_volumes} "
        f"(unique frames with non-zero distractor pixel intensity)"
    )
    print(f"Mean non-zero distractor TR-fraction: "
          f"{distractor_signal[distractor_signal > 0].mean() if (distractor_signal > 0).any() else 0:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('subject', type=int)
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--run', type=int, default=1)
    p.add_argument('--bids_folder', default='/data/ds-retsupp')
    p.add_argument('--output', default='notes/extended_stimulus_check.png')
    p.add_argument('--resolution', type=int, default=120)
    p.add_argument('--grid_radius', type=float, default=5.0)
    p.add_argument('--distractor_radius', type=float, default=0.4)
    a = p.parse_args()
    main(
        a.subject, session=a.session, run=a.run,
        bids_folder=a.bids_folder, output=a.output,
        resolution=a.resolution, grid_radius=a.grid_radius,
        distractor_radius=a.distractor_radius,
    )
