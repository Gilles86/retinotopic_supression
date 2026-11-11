"""Synthetic, data-free validation of the fractional-overlap math used in
``Subject.get_stimulus_with_distractors``.

We build a tiny mock onsets DataFrame with three trials at known target
times and known distractor locations, run the same TR-fraction overlap
math the real method uses, and plot the resulting time courses + a
handful of frames.

The test is checking three things:

  1. A trial whose entire on-window falls inside one TR produces an
     intensity equal to (t_off - t_on) / TR in that TR (and 0
     elsewhere).
  2. A trial whose on-window straddles a TR boundary distributes the
     intensity proportionally to overlap.
  3. The cap ``max_distractor_duration`` truncates at the right place.

Output PNG is written to ``notes/extended_stimulus_check.png`` so it
can serve as the "validation figure" requested in the meeting plan
even when the BIDS dataset isn't accessible locally.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fractional_overlap_stimulus(
    targets, frametimes, tr, grid_x, grid_y,
    distractor_radius=0.4, max_distractor_duration=1.5,
):
    """Replicates the distractor pass of get_stimulus_with_distractors."""
    loc_xy = {
        1.0: ( 4 / np.sqrt(2),  4 / np.sqrt(2)),
        3.0: (-4 / np.sqrt(2),  4 / np.sqrt(2)),
        5.0: (-4 / np.sqrt(2), -4 / np.sqrt(2)),
        7.0: ( 4 / np.sqrt(2), -4 / np.sqrt(2)),
    }
    tr_starts = frametimes - tr / 2.
    tr_ends = frametimes + tr / 2.
    stim = np.zeros((len(frametimes), *grid_x.shape), dtype=np.float32)

    for _, trial in targets.iterrows():
        loc_code = trial['distractor_location']
        if pd.isna(loc_code) or loc_code == 10.0:
            continue
        cx, cy = loc_xy[loc_code]
        t_on = trial['onset']
        t_off = trial.get('feedback_time', t_on + max_distractor_duration)
        t_off = min(t_off, t_on + max_distractor_duration)

        overlap = np.clip(
            np.minimum(tr_ends, t_off) - np.maximum(tr_starts, t_on),
            0.0, None,
        ) / tr
        disk = (((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                < distractor_radius ** 2).astype(np.float32)
        active = overlap > 0
        if active.any():
            contrib = overlap[active, None, None] * disk[None, :, :]
            stim[active] = np.maximum(stim[active], contrib)
    return stim


def main(output='notes/extended_stimulus_check.png'):
    tr = 1.6
    n_volumes = 30
    frametimes = (np.arange(n_volumes) + 0.5) * tr  # centers

    # --- Synthetic trials. ---
    # Frametime centres are (0.5+i)*1.6 = 0.8, 2.4, 4.0, 5.6, ...
    # so TR window i = [i*1.6, (i+1)*1.6].
    #
    # Trial 1: upper_right (1.0), on [5.0, 5.5] — entirely inside TR 3
    #          ([4.8, 6.4]). Expected intensity = 0.5/1.6.
    # Trial 2: upper_left  (3.0), on [7.5, 8.5] straddles boundary 8.0
    #          → 0.5s in TR 4 ([6.4,8.0]) and 0.5s in TR 5 ([8.0,9.6]).
    # Trial 3: lower_left  (5.0), feedback far away so cap kicks in.
    #          on-window [15.0, 16.5]. TR 9 covers [14.4, 16.0] → 1.0s,
    #          TR 10 covers [16.0, 17.6] → 0.5s.
    # Trial 4: lower_right (7.0), distractor_location == 10 → ignored.
    # Trial 5: lower_right (7.0), 1.5s on-window straddling 2 TRs, sanity.
    targets = pd.DataFrame([
        dict(onset=5.0, distractor_location=1.0, feedback_time=5.5),
        dict(onset=7.5, distractor_location=3.0, feedback_time=8.5),
        dict(onset=15.0, distractor_location=5.0, feedback_time=20.0),  # cap @1.5
        dict(onset=22.0, distractor_location=10.0, feedback_time=22.5),
        dict(onset=27.0, distractor_location=7.0, feedback_time=28.6),
    ])

    # Grid.
    grid_radius = 5.0
    resolution = 100
    g = np.linspace(-grid_radius, grid_radius, resolution)
    grid_x, grid_y = np.meshgrid(g, g)

    stim = fractional_overlap_stimulus(
        targets, frametimes, tr, grid_x, grid_y,
        distractor_radius=0.4, max_distractor_duration=1.5,
    )

    # --- Time courses per ring location. ---
    distractor_locations = {
        'upper_right': ( 4 / np.sqrt(2),  4 / np.sqrt(2)),
        'upper_left':  (-4 / np.sqrt(2),  4 / np.sqrt(2)),
        'lower_left':  (-4 / np.sqrt(2), -4 / np.sqrt(2)),
        'lower_right': ( 4 / np.sqrt(2), -4 / np.sqrt(2)),
    }
    timecourses = {}
    for label, (cx, cy) in distractor_locations.items():
        m = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2) < 0.4 ** 2
        timecourses[label] = stim[:, m].mean(axis=1)

    # --- Build the figure. ---
    n_panels = 4
    fig = plt.figure(figsize=(16, 9))

    # Pick the four most active frames.
    frame_signal = stim.sum(axis=(1, 2))
    chosen = np.argsort(frame_signal)[-n_panels:][::-1]

    extent = [-grid_radius, grid_radius, -grid_radius, grid_radius]
    for k, fi in enumerate(np.sort(chosen)):
        ax = fig.add_subplot(2, n_panels, k + 1)
        ax.imshow(stim[fi], extent=extent, origin='lower',
                  vmin=0, vmax=1, cmap='magma')
        for label, (cx, cy) in distractor_locations.items():
            ax.plot(cx, cy, 'o', mfc='none', mec='lime',
                    markersize=14, mew=1)
        ax.set_title(
            f"frame {fi}, t={frametimes[fi]:.2f}s\n"
            f"sum={frame_signal[fi]:.2f}"
        )
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(2, 1, 2)
    for label, tc in timecourses.items():
        ax.plot(frametimes, tc, '-o', label=label, markersize=3)
    # annotate trial windows
    for _, t in targets.iterrows():
        if pd.isna(t['distractor_location']) or t['distractor_location'] == 10.0:
            continue
        t_on, t_off = t['onset'], min(t['feedback_time'], t['onset'] + 1.5)
        ax.axvspan(t_on, t_off, color='grey', alpha=0.15)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('mean disk intensity\n(= TR-fraction the distractor was on)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Per-distractor activation across the synthetic run')
    ax.set_ylim(-0.02, 1.05)

    fig.suptitle(
        'Synthetic validation of fractional-overlap distractor stimulus\n'
        '(grey bands = on-windows; values along curves = (overlap / TR))',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved figure -> {out}")

    # --- Print test summary. ---
    print("\n=== Math check ===")
    expected_t1 = 0.5 / tr
    actual_t1 = timecourses['upper_right'].max()
    print(f"Trial 1 (UR, on-window 5.0-5.5 entirely inside TR 3): "
          f"expected intensity {expected_t1:.3f}, got {actual_t1:.3f}")
    assert np.isclose(actual_t1, expected_t1, atol=1e-3), "Trial 1 mismatch"

    # Trial 2 straddles boundary 8.0.  on-window [7.5, 8.5].
    # TR4 covers [6.4, 8.0] → overlap 0.5  → 0.5/1.6 = 0.3125
    # TR5 covers [8.0, 9.6] → overlap 0.5  → 0.3125
    expected_t2 = 0.5 / tr
    tc_ul = timecourses['upper_left']
    print(f"Trial 2 (UL, straddle): expected each-TR intensity {expected_t2:.3f}, "
          f"got {tc_ul[(tc_ul > 0)][:2]}")
    assert np.isclose(tc_ul[(tc_ul > 0)][0], expected_t2, atol=1e-3)

    # Trial 3 capped at 1.5s, on-window [15.0, 16.5].
    # TR9 [14.4, 16.0] overlap 1.0 → 1.0/1.6 = 0.625
    # TR10 [16.0, 17.6] overlap 0.5 → 0.5/1.6 = 0.3125
    tc_ll = timecourses['lower_left']
    nonzero = np.where(tc_ll > 0)[0]
    print(f"Trial 3 (LL, capped 1.5s): TR{nonzero[0]} intensity "
          f"{tc_ll[nonzero[0]]:.3f} (expect ~0.625), TR{nonzero[1]} "
          f"{tc_ll[nonzero[1]]:.3f} (expect ~0.3125)")
    assert np.isclose(tc_ll[nonzero[0]], 1.0 / tr, atol=1e-3)
    assert np.isclose(tc_ll[nonzero[1]], 0.5 / tr, atol=1e-3)

    print("All assertions passed.\n")


if __name__ == '__main__':
    main()
