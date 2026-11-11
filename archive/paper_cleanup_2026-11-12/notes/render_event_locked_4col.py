"""Group-level event-locked decoded movies: 2 rows × 4 columns.

Columns: event at rotated-frame position {HP=top, right, bottom, left}.
Rows:    distractor onsets / target onsets.

For each (sub, run), use Subject.get_dynamic_indicator and
get_target_indicator to find onset TRs per channel
({upper_right, upper_left, lower_left, lower_right}). Rotate the
decoded paradigm so HP is at top (anchor = the run's HP), classify
each event by its rotated-frame position, extract ±PRE TR..+POST TR,
average across all (sub, run, event).

Excludes sub-01 / sub-02 (counterbalancing bug).

Usage::

    python notes/render_event_locked_4col.py V1 vox2000_psig0.5
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import rotate as nd_rotate
from retsupp.utils.data import Subject, location_angles


ROI = sys.argv[1] if len(sys.argv) > 1 else 'V1'
VOX_TAG = sys.argv[2] if len(sys.argv) > 2 else 'vox2000_psig0.5'
PRE_TR = 2     # 2 TRs before onset = 3.2s
POST_TR = 2    # 2 TRs after = 3.2s
RING_R = 4.0
TR = 1.6
EXCLUDE = {1, 2}

# Channel order from Subject.get_dynamic_indicator
CHANNEL_LOCS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


def _rot_deg(pos: str) -> float:
    return float(np.degrees(location_angles[pos.replace('_', ' ')] - np.pi / 2))


def _rotated_label(event_loc: str, hp_loc: str) -> str:
    """Where does ``event_loc`` land after rotating so ``hp_loc``→top?
    Returns 'HP' / 'right' / 'bottom' / 'left'."""
    rot = np.radians(_rot_deg(hp_loc))  # scipy CW-in-math angle
    # Apply CW rotation to event_loc's angle (since rotate(arr, +deg) maps
    # points CW in math coords).
    event_angle = location_angles[event_loc.replace('_', ' ')]
    new_angle = event_angle - rot  # CW rotation subtracts the angle
    new_angle = (new_angle + np.pi) % (2 * np.pi) - np.pi
    candidates = {'HP': np.pi/2, 'right': 0.0,
                  'bottom': -np.pi/2, 'left': np.pi}
    def d(a, b):
        dd = abs(a - b)
        return min(dd, 2*np.pi - dd)
    return min(candidates, key=lambda k: d(new_angle, candidates[k]))


def _onsets_per_channel(indicator: np.ndarray, threshold: float = 0.5):
    out = []
    for ch in range(indicator.shape[1]):
        a = indicator[:, ch] > threshold
        p = np.concatenate([[False], a[:-1]])
        out.append(np.where(a & ~p)[0].tolist())
    return out


def gather_subject(subject: int, accumulators, grid_ref):
    """Accumulators: dict[(event_type, rot_loc)] → list of (T_win, R, R)."""
    try:
        sub = Subject(subject, bids_folder='/data/ds-retsupp')
        hp_by_run = sub.get_hpd_locations()
    except Exception as e:
        print(f'  sub-{subject:02d}: subject init failed ({e})')
        return 0
    base = Path(f'/data/ds-retsupp/derivatives/decoded/model4/sub-{subject:02d}')
    n_runs = 0
    for (ses, run), hp in hp_by_run.items():
        p = base / f'decoded_ses-{ses}_run-{run}_roi-{ROI}_{VOX_TAG}.npz'
        if not p.exists():
            continue
        with np.load(p) as d:
            dec = d['decoded'].astype(np.float32)
            if grid_ref[0] is None:
                grid_ref[0] = d['grid'].astype(np.float32)
        rot_deg = _rot_deg(hp)
        dec_rot = nd_rotate(dec, rot_deg, axes=(1, 2),
                              reshape=False, order=1,
                              mode='constant', cval=0.0).astype(np.float32)
        try:
            dist_ind = sub.get_dynamic_indicator(session=ses, run=run)
            targ_ind = sub.get_target_indicator(session=ses, run=run)
        except Exception:
            continue
        T = dec.shape[0]
        for et_name, ind in [('distractor', dist_ind), ('target', targ_ind)]:
            onsets = _onsets_per_channel(ind[:T])
            for ch, ev_loc in enumerate(CHANNEL_LOCS):
                rot_loc = _rotated_label(ev_loc, hp)
                for t in onsets[ch]:
                    t0, t1 = t - PRE_TR, t + POST_TR + 1
                    if t0 < 0 or t1 > T:
                        continue
                    accumulators[(et_name, rot_loc)].append(dec_rot[t0:t1])
        n_runs += 1
    return n_runs


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
    print(f'Subjects with {ROI} {VOX_TAG}: {len(avail_subs)} (excluded {sorted(EXCLUDE)})')

    accumulators = {(et, rl): [] for et in ('distractor', 'target')
                                for rl in ('HP', 'right', 'bottom', 'left')}
    grid_ref = [None]
    t0 = time.time()
    total_runs = 0
    for s in avail_subs:
        n = gather_subject(s, accumulators, grid_ref)
        total_runs += n
        print(f'  sub-{s:02d}: {n} runs  ({time.time()-t0:.0f}s)')
    print(f'\nTotal runs used: {total_runs}')
    for k, v in accumulators.items():
        print(f'  {k}: {len(v)} events')

    means = {k: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
              for k, v in accumulators.items()}
    grid = grid_ref[0]
    extent = [float(grid[:, 0].min()), float(grid[:, 0].max()),
              float(grid[:, 1].min()), float(grid[:, 1].max())]

    # Two rows × four columns
    rows = ['distractor', 'target']
    cols = ['HP', 'right', 'bottom', 'left']
    win_len = PRE_TR + POST_TR + 1
    # Shared vmax per row
    vmax_per_row = {
        et: float(np.quantile(np.stack([means[(et, c)] for c in cols]), 0.99))
        for et in rows
    }

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    ims = {}
    for r, et in enumerate(rows):
        for c, loc in enumerate(cols):
            ax = axes[r, c]
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ims[(r, c)] = ax.imshow(means[(et, loc)][0],
                                      extent=extent, origin='lower',
                                      cmap='magma', vmin=0,
                                      vmax=vmax_per_row[et],
                                      interpolation='bilinear')
            # HP star + 3 LP rings
            ax.plot(0.0, RING_R, '*', mfc='gold', mec='black',
                     ms=20, mew=1.0, zorder=10)
            for cx, cy in [(RING_R, 0), (-RING_R, 0), (0, -RING_R)]:
                ax.plot(cx, cy, 'o', mfc='none', mec='white',
                         ms=7, mew=1.0, zorder=9)
            # Highlight event location for this panel
            event_pos = {'HP': (0, RING_R), 'right': (RING_R, 0),
                         'bottom': (0, -RING_R), 'left': (-RING_R, 0)}[loc]
            ax.plot(*event_pos, 'o', mfc='none', mec='lime',
                     ms=18, mew=2.5, zorder=11)
            ax.set_title(f'{et} @ {loc}  '
                          f'(n={len(accumulators[(et, loc)])})',
                          fontsize=9, weight='bold')

    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.03, right=0.99,
                         hspace=0.10, wspace=0.04)
    t_text = fig.text(0.5, 0.97, '', fontsize=12, weight='bold', ha='center')

    def update(t):
        for (r, c), im in ims.items():
            et = rows[r]; loc = cols[c]
            im.set_array(means[(et, loc)][t])
        t_text.set_text(
            f'GROUP {ROI} ({VOX_TAG})  '
            f'event-locked, rotated to HP-at-top  '
            f't = {(t - PRE_TR) * TR:+.1f}s')
        return list(ims.values()) + [t_text]

    anim = animation.FuncAnimation(fig, update, frames=win_len,
                                     interval=1000 / 3, blit=False)
    out = Path('/Users/gdehol/git/retsupp/notes/figures/decode_sweep/m4/'
               f'group_{ROI}_{VOX_TAG}_event_locked_4col.mp4')
    writer = animation.FFMpegWriter(fps=3, bitrate=2400, codec='libx264',
                                      extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
