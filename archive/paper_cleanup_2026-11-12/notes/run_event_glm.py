"""Group-level GLM of decoded paradigm against event regressors.

For each subject (excluding sub-01/02):
  1. Load per-run decoded paradigm
  2. Rotate to HP-at-top per run
  3. Build 6-column design matrix using INDICATOR VALUES (not thresholded
     onsets) — preserves sub-TR overlap weighting:
       HPD       = distractor at HP location
       LPD-orth  = distractor at one of the 2 orthogonal LP positions
       LPD-opp   = distractor at the diagonally opposite LP position
       HPT       = target at HP location
       LPT-orth  = target at orth LP positions
       LPT-opp   = target at opposite LP position
     plus an intercept
  4. Concat across runs (rotated decoded + regressors stacked)
  5. Per-pixel OLS fit: β_x = (X'X)^-1 X'y_x
  6. 6 β-maps in the rotated frame per subject

Group output: mean β-map across subjects, rendered as 2x3 static figure.

Usage::

    python notes/run_event_glm.py V3AB vox2000_psig0.5
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as nd_rotate
from retsupp.utils.data import Subject, location_angles


ROI = sys.argv[1] if len(sys.argv) > 1 else 'V3AB'
VOX_TAG = sys.argv[2] if len(sys.argv) > 2 else 'vox2000_psig0.5'
RING_R = 4.0
EXCLUDE = {1, 2}
# Channel order from Subject.get_dynamic_indicator
CHANNEL_LOCS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
# For each HP channel index → (HP, ortho channels, opposite channel)
RELATIONS = {
    0: {'HP': 0, 'orth': (1, 3), 'opp': 2},  # HP=UR  → ortho UL+LR, opp LL
    1: {'HP': 1, 'orth': (0, 2), 'opp': 3},  # HP=UL
    2: {'HP': 2, 'orth': (1, 3), 'opp': 0},  # HP=LL
    3: {'HP': 3, 'orth': (0, 2), 'opp': 1},  # HP=LR
}
CONDITIONS = ['HPD', 'LPD-orth', 'LPD-opp',
              'HPT', 'LPT-orth', 'LPT-opp']


def _rot_deg(pos: str) -> float:
    return float(np.degrees(location_angles[pos.replace('_', ' ')] - np.pi / 2))


def _hp_channel_idx(hp_loc: str) -> int:
    return CHANNEL_LOCS.index(hp_loc)


def build_run_regressors(sub: Subject, ses: int, run: int) -> np.ndarray | None:
    """6-column design matrix for one run + intercept."""
    try:
        dist = sub.get_dynamic_indicator(session=ses, run=run)
        targ = sub.get_target_indicator(session=ses, run=run)
    except Exception:
        return None
    hp = sub.get_hpd_locations()[(ses, run)]
    h = _hp_channel_idx(hp)
    rel = RELATIONS[h]
    T = min(dist.shape[0], targ.shape[0])
    cols = np.zeros((T, 7), dtype=np.float32)
    cols[:, 0] = dist[:T, rel['HP']]
    cols[:, 1] = dist[:T, rel['orth'][0]] + dist[:T, rel['orth'][1]]
    cols[:, 2] = dist[:T, rel['opp']]
    cols[:, 3] = targ[:T, rel['HP']]
    cols[:, 4] = targ[:T, rel['orth'][0]] + targ[:T, rel['orth'][1]]
    cols[:, 5] = targ[:T, rel['opp']]
    cols[:, 6] = 1.0   # intercept
    return cols


def fit_subject(subject: int) -> dict | None:
    base = Path(f'/data/ds-retsupp/derivatives/decoded/model4/sub-{subject:02d}')
    try:
        sub = Subject(subject, bids_folder='/data/ds-retsupp')
        hp_by_run = sub.get_hpd_locations()
    except Exception as e:
        print(f'  sub-{subject:02d}: subject init failed ({e})')
        return None

    X_list, Y_list = [], []
    grid = None
    for (ses, run), hp in hp_by_run.items():
        p = base / f'decoded_ses-{ses}_run-{run}_roi-{ROI}_{VOX_TAG}.npz'
        if not p.exists():
            continue
        with np.load(p) as d:
            dec = d['decoded'].astype(np.float32)
            if grid is None:
                grid = d['grid'].astype(np.float32)
        rot_deg = _rot_deg(hp)
        dec_rot = nd_rotate(dec, rot_deg, axes=(1, 2),
                              reshape=False, order=1,
                              mode='constant', cval=0.0).astype(np.float32)
        X = build_run_regressors(sub, ses, run)
        if X is None or X.shape[0] != dec_rot.shape[0]:
            # length mismatch -> skip
            T = min(X.shape[0], dec_rot.shape[0]) if X is not None else 0
            if T < 20:
                continue
            X = X[:T]; dec_rot = dec_rot[:T]
        T, R, _ = dec_rot.shape
        Y_list.append(dec_rot.reshape(T, R * R))   # (T, P)
        X_list.append(X)
    if not X_list:
        return None
    X = np.concatenate(X_list, axis=0)     # (T_total, 7)
    Y = np.concatenate(Y_list, axis=0)     # (T_total, P)
    # OLS: B = (X'X)^-1 X'Y → (7, P)
    # Use lstsq for numerical stability.
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    P = R * R
    R_side = int(np.sqrt(P))
    betas = {CONDITIONS[i]: B[i].reshape(R_side, R_side).astype(np.float32)
              for i in range(len(CONDITIONS))}
    return {'betas': betas, 'grid': grid, 'T_total': X.shape[0]}


def main():
    avail_subs = []
    base_all = Path('/data/ds-retsupp/derivatives/decoded/model4')
    for d in sorted(base_all.glob('sub-*')):
        try:
            n = int(d.name.split('-')[1])
        except (IndexError, ValueError):
            continue
        if n in EXCLUDE:
            continue
        if list(d.glob(f'decoded_*_roi-{ROI}_{VOX_TAG}.npz')):
            avail_subs.append(n)
    print(f'Subjects with {ROI} {VOX_TAG}: {len(avail_subs)} '
          f'(excluded {sorted(EXCLUDE)})')

    # Per-subject fits
    per_sub_betas = {c: [] for c in CONDITIONS}
    grid = None
    t0 = time.time()
    for s in avail_subs:
        r = fit_subject(s)
        if r is None:
            print(f'  sub-{s:02d}: no fit'); continue
        for c in CONDITIONS:
            per_sub_betas[c].append(r['betas'][c])
        if grid is None:
            grid = r['grid']
        print(f'  sub-{s:02d}: fitted ({r["T_total"]} TRs, '
              f'elapsed {time.time()-t0:.0f}s)')
    n_fitted = len(per_sub_betas['HPD'])
    print(f'\nFitted {n_fitted} subjects')

    # Group mean β per condition
    group_betas = {c: np.mean(np.stack(per_sub_betas[c], axis=0), axis=0)
                    for c in CONDITIONS}

    extent = [float(grid[:, 0].min()), float(grid[:, 0].max()),
              float(grid[:, 1].min()), float(grid[:, 1].max())]

    # Shared symmetric vmax for diverging cmap
    vmax = float(np.quantile(np.abs(np.stack(list(group_betas.values()))),
                              0.99))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    for ax, cond in zip(axes.flat, CONDITIONS):
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        im = ax.imshow(group_betas[cond], extent=extent, origin='lower',
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        interpolation='bilinear')
        ax.plot(0.0, RING_R, '*', mfc='gold', mec='black',
                 ms=22, mew=1.2, zorder=10)
        for cx, cy in [(RING_R, 0), (-RING_R, 0), (0, -RING_R)]:
            ax.plot(cx, cy, 'o', mfc='none', mec='black',
                     ms=7, mew=1.0, zorder=9)
        ax.set_title(cond, fontsize=12, weight='bold')
    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.03, right=0.92,
                         hspace=0.06, wspace=0.03)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='β (decoded a.u. per unit indicator)')
    fig.suptitle(
        f'GROUP {ROI} ({VOX_TAG}) — GLM β-maps (n={n_fitted} subjects, '
        f'HP-at-top rotated frame)', fontsize=12, weight='bold')
    out = Path('/Users/gdehol/git/retsupp/notes/figures/decode_sweep/m4/'
               f'group_{ROI}_{VOX_TAG}_event_glm_betas.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    # Also save the per-subject β arrays for downstream stats
    out_npz = Path('/data/ds-retsupp/derivatives/decoded/model4/group_aggregated')
    out_npz.mkdir(parents=True, exist_ok=True)
    out_npz = out_npz / f'glm_betas_{ROI}_{VOX_TAG}.npz'
    np.savez_compressed(
        out_npz,
        **{f'group_{c}': group_betas[c] for c in CONDITIONS},
        **{f'persub_{c}': np.stack(per_sub_betas[c], axis=0)
           for c in CONDITIONS},
        subjects=np.array(avail_subs[:n_fitted]),
        grid=grid,
        roi=np.array(ROI),
    )
    print(f'Wrote {out_npz}')


if __name__ == '__main__':
    main()
