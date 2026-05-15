"""Per-ROI rotated decoded-paradigm heatmaps (plan §5a).

Reads ``derivatives/decoded_paradigm/group/model{M}/`` group npzs and
draws one panel per ROI in a side-by-side row:

* ``group_roi-{ROI}_desc-meanDecoded.npz`` -- TIME-AVERAGED rotated
  decoded paradigm (HP at top), one ``(R, R)`` heatmap per ROI.
* Optional ring markers + HP marker overlaid.

Output: ``notes/figures/decoding/rotated_heatmaps_model{M}.pdf``.

The time-averaging is on the *un-rotated* mean decoded paradigm
(§2E output): we re-rotate each subject-run's mean to HP-at-top here.
That's a separate concept from §2F's event-locked rotated grid. The
group npz currently holds the un-rotated mean, so we average each
(sub, run) decoded -> rotate per HP -> group-average across (sub, run).
This keeps the rotation correct for each run.

(For the heatmap, we operate on per-run decoded npzs directly rather
than the §2E group mean, because the rotation has to be per-run.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as nd_rotate

from retsupp.utils.data import Subject, location_angles, distractor_locations
from retsupp.decode.aggregate_decoded import _rot_for_hp, _run_npz_path


def per_roi_mean_rotated(bids: Path, roi: str, model: int,
                          subjects: list,
                          drop_pre_tr: int = 4) -> np.ndarray | None:
    """Mean decoded paradigm across (sub, run), each run rotated so HP -> top.

    ``drop_pre_tr``: skip the first N TRs at run start (dummy-equivalent
    pre-stim). Plan §2A defaults to 4.
    """
    accum = None
    n = 0
    for subject in subjects:
        sub = Subject(subject, bids_folder=str(bids))
        hp_by = sub.get_hpd_locations()
        for (ses, run), hp_loc in hp_by.items():
            p = _run_npz_path(bids, subject, model, roi, ses, run)
            if not p.exists():
                continue
            with np.load(p) as d:
                decoded = d['decoded'].astype(np.float32)
            decoded = decoded[drop_pre_tr:]
            mean_map = decoded.mean(axis=0)
            if hp_loc not in (None, 'no distractor', 'no_distractor', 'None'):
                deg = np.degrees(_rot_for_hp(hp_loc))
                mean_map = nd_rotate(mean_map, deg,
                                       reshape=False, order=1,
                                       mode='constant', cval=0.0)
            if accum is None:
                accum = mean_map.astype(np.float64)
            else:
                accum += mean_map
            n += 1
    if accum is None or n == 0:
        return None
    return (accum / n).astype(np.float32)


def plot_rotated_heatmaps(bids_folder: str, rois: list, model: int = 4,
                           subjects: list | None = None,
                           drop_pre_tr: int = 4,
                           out_path: Path | str = 'notes/figures/decoding/rotated_heatmaps_model4.pdf'):
    """Side-by-side rotated heatmaps, one per ROI."""
    bids = Path(bids_folder)
    if subjects is None:
        from retsupp.data import load_subjects
        subjects = sorted(int(s) for s in load_subjects())

    maps = []
    for roi in rois:
        m = per_roi_mean_rotated(bids, roi, model, subjects,
                                   drop_pre_tr=drop_pre_tr)
        maps.append((roi, m))

    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 1, 2.6),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    extent = [-5.0, 5.0, -5.0, 5.0]
    # Symmetric color scale across ROIs.
    all_vals = np.concatenate([
        m.ravel() for _, m in maps if m is not None])
    if all_vals.size == 0:
        raise RuntimeError('No rotated heatmaps to plot (no decoded npzs '
                            'found for any ROI).')
    vmax = float(np.quantile(all_vals, 0.99))

    for ax, (roi, m) in zip(axes, maps):
        if m is None:
            ax.set_title(f'{roi} (no data)', fontsize=9)
            ax.axis('off')
            continue
        im = ax.imshow(m, extent=extent, origin='lower',
                        cmap='magma', vmin=0, vmax=vmax,
                        interpolation='bilinear')
        # Ring markers; HP highlighted (it's at the top).
        for name, (cx, cy) in distractor_locations.items():
            # In rotated frame, the "HP" location is wherever the
            # un-rotated coords map to angle pi/2 — for the rotated
            # frame, HP is canonically at +y.
            pass
        # Generic top-marker:
        ax.plot(0.0, 4.0 / np.sqrt(2), 'o',
                mfc='none', mec='cyan', ms=10, mew=1.2)
        for cx, cy in [(4 / np.sqrt(2), 0), (-4 / np.sqrt(2), 0),
                         (0, -4 / np.sqrt(2))]:
            ax.plot(cx, cy, 'o', mfc='none', mec='white',
                    ms=8, mew=1.0)
        ax.set_title(roi, fontsize=10)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        ax.set_aspect('equal')

    axes[0].set_ylabel('y (deg)')
    for ax in axes:
        ax.set_xlabel('x (deg)')

    fig.suptitle(
        f'Group mean decoded paradigm (HP -> top), m{model}, '
        f'{len(subjects)} subjects; cyan = HP, white = LPs',
        fontsize=10)
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75,
                          fraction=0.025, pad=0.02, label='Decoded')
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--rois', nargs='+',
                   default=['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO'])
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--subjects', nargs='*', type=int, default=None)
    p.add_argument('--drop-pre-tr', type=int, default=4)
    p.add_argument('--out',
                   default='notes/figures/decoding/rotated_heatmaps_model4.pdf')
    a = p.parse_args()
    plot_rotated_heatmaps(
        bids_folder=a.bids_folder, rois=a.rois, model=a.model,
        subjects=a.subjects, drop_pre_tr=a.drop_pre_tr,
        out_path=a.out,
    )


if __name__ == '__main__':
    main()
