"""Aggregate per-run decoded paradigms into group-level tensors.

Reads the ``(T, R, R)`` decoded NPZs written by
:mod:`retsupp.decode.decode` (or any per-run decoder writing the
same ``(decoded, paradigm, grid)`` npz schema) and produces:

* **§2E mean decoded paradigm per ROI** -- ``(T, R, R)`` group mean of
  the un-rotated decoded paradigm across all subjects × runs. One
  outfile per ROI.

* **§2F event-locked rotated grid per ROI** -- 2 event types
  (distractor, target) × 4 rotated-frame ring positions (HP, right,
  bottom, left in the HP-at-top rotated frame). Each cell is a
  ``(T_event, R, R)`` group-mean event-locked tensor. 8 outfiles per
  ROI.

Output layout (all relative to ``<bids>/derivatives/decoded_paradigm/group/model{M}/``)::

    group_roi-{ROI}_desc-meanDecoded.npz
    group_roi-{ROI}_event-{distractor|target}_loc-{HP|right|bottom|left}_desc-eventLocked.npz

After HP-to-top rotation, the 4 ring corners always land on the
cardinal axes -- top (HP), right, bottom, left -- regardless of which
of the 4 HP conditions a given run was in. We label by these cardinal
positions because they are unambiguous; the plan's "upper-right /
lower-left / lower-right" naming was approximate.

Usage::

    python -m retsupp.decode.aggregate_decoded \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
        --rois V1 V2 V3 V3AB hV4 LO TO VO \\
        --model 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import rotate as nd_rotate

from retsupp.utils.data import Subject, location_angles

# Channel order used by Subject.get_dynamic_indicator / get_target_indicator.
EVENT_LOC_ORDER = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


# After HP-to-top rotation, every ring corner lands on a cardinal axis.
# The rotation angle = pi/2 - hp_angle, applied to all 4 corners.
# Result mapping (ring location in the run's original frame ->
# rotated-frame cardinal):
ROTATED_LOCS = ('HP', 'right', 'bottom', 'left')


def _rot_for_hp(hp_location: str) -> float:
    """Rotation (radians) that takes ``hp_location`` to the top (+y)."""
    angle_hp = location_angles[hp_location.replace('_', ' ')]
    return np.pi / 2 - angle_hp


def _rotated_label(ring_loc: str, hp_location: str) -> str:
    """For an event at ``ring_loc`` and run HP at ``hp_location``,
    return where it lands in the HP-at-top rotated frame.

    Returns one of 'HP' / 'right' / 'bottom' / 'left'.
    """
    rot = _rot_for_hp(hp_location)
    a = location_angles[ring_loc.replace('_', ' ')] + rot
    # Wrap to (-pi, pi].
    a = (a + np.pi) % (2 * np.pi) - np.pi
    # Cardinal bins centered at pi/2, 0, -pi/2, pi.
    candidates = {
        'HP': np.pi / 2,
        'right': 0.0,
        'bottom': -np.pi / 2,
        'left': np.pi,
    }
    def d(x, y):
        dd = abs(x - y)
        return min(dd, 2 * np.pi - dd)
    return min(candidates, key=lambda k: d(a, candidates[k]))


def _run_npz_path(bids: Path, subject: int, model: int, roi: str,
                  session: int, run: int) -> Path:
    return (bids / 'derivatives' / 'decoded_paradigm'
            / f'model{model}' / f'sub-{subject:02d}'
            / f'sub-{subject:02d}_ses-{session}_run-{run}_'
              f'roi-{roi}_desc-decoded.npz')


def _group_npz_path(bids: Path, model: int, roi: str, kind: str) -> Path:
    return (bids / 'derivatives' / 'decoded_paradigm' / 'group'
            / f'model{model}' / f'group_roi-{roi}_{kind}.npz')


def _rotate_tensor(decoded: np.ndarray, hp_location: str) -> np.ndarray:
    """Rotate ``(T, R, R)`` decoded so the run's HP is at the top.

    The decoded array is indexed ``[t, y_row, x_col]`` with row 0 at
    the bottom of the visual field (origin='lower'). ``scipy.ndimage.
    rotate`` rotates anticlockwise for positive ``angle`` (in degrees)
    when applied to that convention.
    """
    if hp_location in (None, 'no distractor', 'no_distractor', 'None'):
        return decoded.copy()
    deg = np.degrees(_rot_for_hp(hp_location))
    return nd_rotate(decoded, deg, axes=(1, 2),
                      reshape=False, order=1, mode='constant', cval=0.0).astype(np.float32)


def _event_tr_indices(indicator: np.ndarray, threshold: float = 0.5) -> list:
    """Onset TRs per channel.

    ``indicator`` is the ``(T, 4)`` per-TR on-fraction from
    Subject.get_{dynamic,target}_indicator. We define an "onset TR" as
    the first TR within a contiguous on-region where indicator crosses
    ``threshold`` (default 0.5 = half-TR coverage).
    """
    above = indicator > threshold
    # Onset = first True after a False.
    prev = np.concatenate([[False], above[:-1]], axis=0)
    onsets = np.where(above & ~prev)[0]  # 1D indices across the (T*4) flat? no.
    # Actually need to do per channel:
    out = [[] for _ in range(indicator.shape[1])]
    for ch in range(indicator.shape[1]):
        a = above[:, ch]
        p = np.concatenate([[False], a[:-1]])
        idx = np.where(a & ~p)[0]
        out[ch] = idx.tolist()
    return out


def _extract_event_windows(decoded_rot: np.ndarray,
                            onsets: list,
                            pre_tr: int, post_tr: int) -> np.ndarray:
    """Stack per-event windows of length ``pre_tr + post_tr + 1``.

    Drops events whose window would fall off the run boundaries.
    """
    T, R, _ = decoded_rot.shape
    win_len = pre_tr + post_tr + 1
    keep = []
    for t in onsets:
        if t - pre_tr < 0 or t + post_tr >= T:
            continue
        keep.append(decoded_rot[t - pre_tr:t + post_tr + 1])
    if not keep:
        return np.empty((0, win_len, R, R), dtype=np.float32)
    return np.stack(keep, axis=0).astype(np.float32)


def aggregate(bids_folder: str, rois: list, model: int = 4,
              subjects: list | None = None,
              pre_tr: int = 2, post_tr: int = 2,
              indicator_threshold: float = 0.5,
              verbose: bool = True):
    """Walk all (sub, roi, run) decoded npzs and write the §2E + §2F group files.

    Parameters
    ----------
    bids_folder : str or Path
    rois : list of ROI names ('V1', 'V2', ...).
    model : encoding-model index (default 4).
    subjects : optional whitelist; default = all subjects with at least
        one decoded run for the first ROI.
    pre_tr, post_tr : event window in TRs. Default 2/2 = 5 TRs = 8 s.
    indicator_threshold : channel on-fraction above which a TR is an onset.
    """
    bids = Path(bids_folder)
    from retsupp.data import load_subjects
    all_subjects = subjects if subjects is not None else load_subjects()
    all_subjects = sorted(int(s) for s in all_subjects)

    for roi in rois:
        t_roi = time.time()
        if verbose:
            print(f'[aggregate] === ROI {roi} ===', flush=True)

        # §2E -- accumulators for un-rotated mean.
        mean_sum: np.ndarray | None = None
        mean_count = 0

        # §2F -- accumulators per (event_type, rotated_loc).
        events = ('distractor', 'target')
        accum: dict = {
            (et, loc): {'sum': None, 'count': 0}
            for et in events for loc in ROTATED_LOCS
        }

        for subject in all_subjects:
            try:
                sub = Subject(subject, bids_folder=str(bids))
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f'  sub-{subject:02d}: Subject init failed: {e}',
                          flush=True)
                continue
            hp_by_run = sub.get_hpd_locations()
            n_run_used = 0
            for (ses, run), hp_loc in hp_by_run.items():
                p = _run_npz_path(bids, subject, model, roi, ses, run)
                if not p.exists():
                    continue
                try:
                    with np.load(p) as d:
                        decoded = d['decoded'].astype(np.float32)
                except Exception as e:  # noqa: BLE001
                    if verbose:
                        print(f'  sub-{subject:02d} ses-{ses} run-{run} '
                              f'{roi}: load failed: {e}', flush=True)
                    continue
                n_run_used += 1

                # §2E -- un-rotated mean.
                if mean_sum is None:
                    mean_sum = decoded.astype(np.float64)
                else:
                    mean_sum += decoded
                mean_count += 1

                # §2F -- HP-to-top rotated event-locked.
                decoded_rot = _rotate_tensor(decoded, hp_loc)

                for et in events:
                    ind = (sub.get_dynamic_indicator(session=ses, run=run)
                           if et == 'distractor'
                           else sub.get_target_indicator(session=ses, run=run))
                    # Crop indicator to decoded length (paradigm is 258 TR,
                    # decoded is 258 TR; indicators are n_volumes per run
                    # which may be 258).
                    ind = ind[:decoded_rot.shape[0]]
                    onsets_per_ch = _event_tr_indices(
                        ind, threshold=indicator_threshold)
                    for ch, loc in enumerate(EVENT_LOC_ORDER):
                        rot_loc = _rotated_label(loc, hp_loc)
                        wins = _extract_event_windows(
                            decoded_rot, onsets_per_ch[ch],
                            pre_tr=pre_tr, post_tr=post_tr)
                        if wins.shape[0] == 0:
                            continue
                        key = (et, rot_loc)
                        ag = accum[key]
                        s = wins.mean(axis=0).astype(np.float64)
                        if ag['sum'] is None:
                            ag['sum'] = s
                        else:
                            ag['sum'] += s
                        ag['count'] += 1

            if verbose and n_run_used:
                print(f'  sub-{subject:02d} {roi}: {n_run_used} runs used',
                      flush=True)

        # Write outputs.
        out_dir = (bids / 'derivatives' / 'decoded_paradigm' / 'group'
                   / f'model{model}')
        out_dir.mkdir(parents=True, exist_ok=True)

        if mean_sum is not None:
            mean_out = mean_sum / max(1, mean_count)
            out_p = _group_npz_path(bids, model, roi, 'desc-meanDecoded')
            np.savez_compressed(
                out_p,
                decoded=mean_out.astype(np.float32),
                n_runs=np.array(mean_count),
                roi=np.array(roi),
                model=np.array(model),
                pre_tr=np.array(pre_tr),
                post_tr=np.array(post_tr),
            )
            if verbose:
                print(f'  wrote {out_p}  (n_runs={mean_count})',
                      flush=True)

        for (et, rot_loc), ag in accum.items():
            if ag['sum'] is None:
                continue
            out = ag['sum'] / max(1, ag['count'])
            out_p = _group_npz_path(
                bids, model, roi,
                f'event-{et}_loc-{rot_loc}_desc-eventLocked')
            np.savez_compressed(
                out_p,
                decoded=out.astype(np.float32),
                n_events=np.array(ag['count']),
                event_type=np.array(et),
                rotated_loc=np.array(rot_loc),
                roi=np.array(roi),
                model=np.array(model),
                pre_tr=np.array(pre_tr),
                post_tr=np.array(post_tr),
            )
            if verbose:
                print(f'  wrote {out_p}  '
                      f'(n_run_avg={ag["count"]})', flush=True)

        if verbose:
            print(f'  ROI {roi} done in {time.time() - t_roi:.1f}s',
                  flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--rois', nargs='+',
                   default=['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO'])
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--subjects', nargs='*', type=int, default=None)
    p.add_argument('--pre-tr', type=int, default=2)
    p.add_argument('--post-tr', type=int, default=2)
    p.add_argument('--indicator-threshold', type=float, default=0.5)
    a = p.parse_args()
    aggregate(
        bids_folder=a.bids_folder, rois=a.rois, model=a.model,
        subjects=a.subjects, pre_tr=a.pre_tr, post_tr=a.post_tr,
        indicator_threshold=a.indicator_threshold,
    )


if __name__ == '__main__':
    main()
