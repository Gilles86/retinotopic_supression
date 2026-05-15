"""Per-HP-condition aggregation of per-run decoded NPZs (plan §2A/§2E).

For one (subject, ROI), reads all per-run decoded NPZs written by
:mod:`retsupp.decode.decode`, groups them by HP-distractor location,
rotates each run's decoded paradigm so the HP corner is at the top,
and averages within + across HP conditions.

Produces:
- per-HP rotated mean decoded paradigm ``(T, R, R)`` (4 of them)
- group-mean rotated decoded paradigm (averaged over the 4 HP
  conditions) — this is the "HP-at-top mean" used for §2A
- the sustained (time-averaged) rotated map — §2A target

Usage::

    python -m retsupp.decode.aggregate_per_hp \\
        --subject 23 --roi V1 --bids-folder /data/ds-retsupp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate as nd_rotate

from retsupp.utils.data import Subject, location_angles


def _rot_angle_deg(hp_location: str) -> float:
    """Rotation (degrees) to take ``hp_location`` to +y (top).

    scipy.ndimage.rotate uses a CW-in-math-coords convention: a positive
    angle rotates points clockwise in (x_right, y_up) space. So the
    rotation that maps HP at angle θ_HP to (0, +R) is angle_hp - π/2,
    NOT π/2 - angle_hp (which was the original bug).
    """
    key = hp_location.replace('_', ' ')
    angle_hp = location_angles[key]
    return np.degrees(angle_hp - np.pi / 2)


def _decoded_path(bids: Path, subject: int, model: int, roi: str,
                   session: int, run: int, vox: int) -> Path:
    return (bids / 'derivatives' / 'decoded' / f'model{model}'
            / f'sub-{subject:02d}'
            / f'decoded_ses-{session}_run-{run}_roi-{roi}_vox{vox}.npz')


def aggregate_subject_roi(bids: str, subject: int, roi: str, model: int = 4,
                           vox: int = 200, drop_pre_tr: int = 0):
    bids_p = Path(bids)
    sub = Subject(subject, bids_folder=bids)
    hp_by_run = sub.get_hpd_locations()

    # Collect per-run decoded; group by HP.
    by_hp: dict[str, list[np.ndarray]] = {}
    paradigm_ref: np.ndarray | None = None
    grid_ref: np.ndarray | None = None
    for (ses, run), hp in hp_by_run.items():
        p = _decoded_path(bids_p, subject, model, roi, ses, run, vox)
        if not p.exists():
            print(f'  skip (missing): {p.name}')
            continue
        with np.load(p) as d:
            dec = d['decoded'].astype(np.float32)
            if paradigm_ref is None:
                paradigm_ref = d['paradigm'].astype(np.float32)
                grid_ref = d['grid'].astype(np.float32)
        if drop_pre_tr:
            dec = dec[drop_pre_tr:]
        by_hp.setdefault(hp, []).append(dec)

    if not by_hp:
        raise RuntimeError(
            f'No decoded NPZs found for sub-{subject:02d} {roi} '
            f'(looking in {_decoded_path(bids_p, subject, model, roi, 1, 1, vox).parent}).')

    print(f'\nsub-{subject:02d} {roi}: HP-group sizes (after filtering):')
    for hp, runs in by_hp.items():
        print(f'  {hp:14s}  n_runs={len(runs)}  decoded shape={runs[0].shape}')

    # Per-HP mean + rotate so HP -> top.
    per_hp_rotated = {}
    for hp, decs in by_hp.items():
        avg = np.mean(np.stack(decs, axis=0), axis=0)  # (T, R, R)
        rot_deg = _rot_angle_deg(hp)
        # scipy rotates anticlockwise for positive deg w/ origin='lower'.
        avg_rot = nd_rotate(avg, rot_deg, axes=(1, 2),
                             reshape=False, order=1, mode='constant',
                             cval=0.0).astype(np.float32)
        per_hp_rotated[hp] = avg_rot
        print(f'  rotated {hp:14s} by {rot_deg:+.1f} deg')

    # Group mean (across 4 HP conditions).
    rotated_stack = np.stack(list(per_hp_rotated.values()), axis=0)
    group_mean = rotated_stack.mean(axis=0).astype(np.float32)
    # Sustained map = time-average across the run.
    sustained = group_mean.mean(axis=0).astype(np.float32)

    # Output.
    out_dir = (bids_p / 'derivatives' / 'decoded' / f'model{model}'
               / f'sub-{subject:02d}' / 'aggregated')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f'agg_roi-{roi}_vox{vox}_hp-rotated.npz'
    np.savez_compressed(
        out_p,
        group_mean=group_mean,
        sustained=sustained,
        per_hp_rotated_keys=np.array(list(per_hp_rotated.keys())),
        per_hp_rotated_data=np.stack(list(per_hp_rotated.values()), axis=0),
        paradigm_one_run=paradigm_ref,
        grid=grid_ref,
        roi=np.array(roi),
        subject=np.int32(subject),
        model=np.int32(model),
        n_voxels=np.int32(vox),
        n_runs_used=np.int32(sum(len(v) for v in by_hp.values())),
    )
    print(f'\nWrote {out_p}')
    print(f'  group_mean shape:   {group_mean.shape}')
    print(f'  sustained shape:    {sustained.shape}')
    return out_p


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--roi', default='V1')
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--vox', type=int, default=200,
                   help='max_voxels suffix in the per-run npz filename.')
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--drop-pre-tr', type=int, default=0,
                   help='Skip first N TRs of each run (dummy-equivalent '
                        'pre-stim). Default 0; plan §2A suggests 4.')
    a = p.parse_args()
    aggregate_subject_roi(bids=a.bids_folder, subject=a.subject, roi=a.roi,
                          model=a.model, vox=a.vox, drop_pre_tr=a.drop_pre_tr)


if __name__ == '__main__':
    main()
