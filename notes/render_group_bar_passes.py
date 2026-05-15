"""Group-level rotated bar-pass concat movie + HP-vs-LP contrast.

Pools bar-pass windows across all subjects with available decoded data
for the chosen ROI. Excludes sub-01 / sub-02 (counterbalancing bug per
CLAUDE.md).

Usage::

    python notes/render_group_bar_passes.py V1 vox2000_psig0.5
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import rotate as nd_rotate
from retsupp.utils.data import Subject, location_angles


ROI = sys.argv[1] if len(sys.argv) > 1 else 'V1'
VOX_TAG = sys.argv[2] if len(sys.argv) > 2 else 'vox2000_psig0.5'
WIN_LEN = 27
RING_R = 4.0
RING_POSITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
EXCLUDE = {1, 2}   # known counterbalancing bug

ROT_DIR_TO_ORIG = {
    '↗': {'upper_right': 'right', 'upper_left': 'up',
          'lower_left': 'left', 'lower_right': 'down'},
    '↘': {'upper_right': 'down', 'upper_left': 'right',
          'lower_left': 'up', 'lower_right': 'left'},
    '↙': {'upper_right': 'left', 'upper_left': 'down',
          'lower_left': 'right', 'lower_right': 'up'},
    '↖': {'upper_right': 'up', 'upper_left': 'left',
          'lower_left': 'down', 'lower_right': 'right'},
}


def _rot_deg(pos: str) -> float:
    return float(np.degrees(location_angles[pos.replace('_', ' ')] - np.pi / 2))


def _bar_onset_trs(sub: Subject, session: int, run: int,
                    direction: str, tr: float = 1.6) -> list[int]:
    ev = sub.get_onsets(session=session, run=run)
    rows = ev[ev['event_type'] == f'bar_{direction}']
    return [int(round(o / tr)) for o in rows['onset'].values]


def gather_subject(subject: int, roi: str, vox_tag: str,
                    win_hp: dict, win_lp: dict, par_hp: dict,
                    grid_ref: list, par_ref: list) -> int:
    base = Path(f'/data/ds-retsupp/derivatives/decoded/model4/sub-{subject:02d}')
    try:
        sub = Subject(subject, bids_folder='/data/ds-retsupp')
        hp_by_run = sub.get_hpd_locations()
    except Exception as e:
        print(f'  sub-{subject:02d}: subject init failed ({e})')
        return 0
    rotated_dirs = list(ROT_DIR_TO_ORIG.keys())
    n_runs_used = 0
    for (ses, run), hp in hp_by_run.items():
        p = base / f'decoded_ses-{ses}_run-{run}_roi-{roi}_{vox_tag}.npz'
        if not p.exists():
            continue
        with np.load(p) as d:
            dec = d['decoded'].astype(np.float32)
            if not grid_ref:
                grid_ref.append(d['grid'].astype(np.float32))
        # Use bar+distractors as the "true paradigm" reference — the BOLD
        # was driven by both, so residuals should be against the full
        # stimulus to highlight things the encoder did NOT predict.
        par_full = sub.get_stimulus_with_distractors(
            session=ses, run=run, resolution=dec.shape[-1],
            grid_radius=5.0, distractor_shape='rectangle',
            distractor_long_side=1.5, distractor_short_side=0.375
        ).astype(np.float32)
        for anchor in RING_POSITIONS:
            rot_deg = _rot_deg(anchor)
            dec_rot = nd_rotate(dec, rot_deg, axes=(1, 2),
                                  reshape=False, order=1,
                                  mode='constant', cval=0.0).astype(np.float32)
            par_rot = nd_rotate(par_full, rot_deg, axes=(1, 2),
                                  reshape=False, order=0,
                                  mode='constant', cval=0.0).astype(np.float32)
            for rot_dir in rotated_dirs:
                orig = ROT_DIR_TO_ORIG[rot_dir][anchor]
                onsets = _bar_onset_trs(sub, ses, run, orig)
                for t0 in onsets:
                    t1 = t0 + WIN_LEN
                    if t1 > dec.shape[0]:
                        continue
                    if anchor == hp:
                        win_hp[rot_dir].append(dec_rot[t0:t1])
                        par_hp[rot_dir].append(par_rot[t0:t1])
                    else:
                        win_lp[rot_dir].append(dec_rot[t0:t1])
        n_runs_used += 1
    return n_runs_used


def main():
    # Discover available subjects (have at least one ROI decoded npz)
    avail_subs = []
    base_all = Path('/data/ds-retsupp/derivatives/decoded/model4')
    for d in sorted(base_all.glob('sub-*')):
        try:
            n = int(d.name.split('-')[1])
        except (IndexError, ValueError):
            continue
        if n in EXCLUDE:
            continue
        # Check if any file matches the (ROI, vox_tag) pattern
        if list(d.glob(f'decoded_*_roi-{ROI}_{VOX_TAG}.npz')):
            avail_subs.append(n)
    print(f'Subjects with {ROI} {VOX_TAG} available: {len(avail_subs)} '
          f'(excluded {sorted(EXCLUDE)})')
    print(f'  {avail_subs}')

    rotated_dirs = list(ROT_DIR_TO_ORIG.keys())
    win_hp = {d: [] for d in rotated_dirs}
    win_lp = {d: [] for d in rotated_dirs}
    par_hp = {d: [] for d in rotated_dirs}
    grid_ref: list = []
    par_ref: list = []

    t0 = time.time()
    total_runs = 0
    for s in avail_subs:
        n = gather_subject(s, ROI, VOX_TAG, win_hp, win_lp, par_hp,
                            grid_ref, par_ref)
        total_runs += n
        print(f'  sub-{s:02d}: {n} runs  '
              f'(elapsed {time.time()-t0:.0f}s)')
    print(f'\nTotal runs used: {total_runs}')

    n_hp = {d: len(v) for d, v in win_hp.items()}
    n_lp = {d: len(v) for d, v in win_lp.items()}
    print(f'HP-anchored windows per direction: {n_hp}')
    print(f'LP-anchored windows per direction: {n_lp}')

    mean_hp = {d: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
                for d, v in win_hp.items()}
    mean_lp = {d: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
                for d, v in win_lp.items()}
    contrast = {d: mean_hp[d] - mean_lp[d] for d in rotated_dirs}
    mean_par = {d: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
                for d, v in par_hp.items()}

    sequence = ['↖', '↗', '↘', '↙']
    concat_hp = np.concatenate([mean_hp[d] for d in sequence], axis=0)
    concat_par = np.concatenate([mean_par[d] for d in sequence], axis=0)
    # Residual = decoded − true paradigm, but normalised so the bar's
    # SCALE doesn't dominate. We z-score-paradigm-to-decoded match:
    # decoded magnitudes vary across pixels; we want to see where the
    # decoder placed mass beyond/below the bar's true position.
    # Simplest: scale paradigm to match decoded's per-frame peak.
    concat_par_scaled = concat_par * (
        concat_hp.max() / max(concat_par.max(), 1e-9))
    concat_resid = concat_hp - concat_par_scaled
    T_total = concat_hp.shape[0]
    direction_of_t = np.repeat(sequence, WIN_LEN)
    within_t = np.tile(np.arange(WIN_LEN), len(sequence))

    grid_arr = grid_ref[0]
    extent = [float(grid_arr[:, 0].min()), float(grid_arr[:, 0].max()),
              float(grid_arr[:, 1].min()), float(grid_arr[:, 1].max())]
    vmax_dec = float(np.quantile(concat_hp, 0.99))
    vmax_par = float(np.quantile(concat_par_scaled, 0.99))
    vmax_resid = float(np.quantile(np.abs(concat_resid), 0.98))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax in axes:
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.plot(0.0, RING_R, '*', mfc='gold', mec='black',
                 ms=22, mew=1.2, zorder=10)
        for cx, cy in [(RING_R, 0), (-RING_R, 0), (0, -RING_R)]:
            ax.plot(cx, cy, 'o', mfc='none', mec='white',
                     ms=7, mew=1.0, zorder=9)
    im_dec = axes[0].imshow(concat_hp[0], extent=extent, origin='lower',
                               cmap='magma', vmin=0, vmax=vmax_dec,
                               interpolation='bilinear')
    im_par = axes[1].imshow(concat_par_scaled[0], extent=extent,
                               origin='lower', cmap='magma', vmin=0,
                               vmax=vmax_par, interpolation='bilinear')
    im_resid = axes[2].imshow(concat_resid[0], extent=extent, origin='lower',
                                 cmap='RdBu_r', vmin=-vmax_resid,
                                 vmax=vmax_resid, interpolation='bilinear')
    axes[0].set_title(f'HP-anchored decoded  '
                        f'(n={len(avail_subs)} subjects)',
                        fontsize=11, weight='bold')
    axes[1].set_title('True paradigm (scaled)',
                       fontsize=11, weight='bold')
    axes[2].set_title('Residual = decoded − paradigm',
                       fontsize=11, weight='bold')
    suptitle = fig.suptitle('', fontsize=12, weight='bold')

    def update(t):
        im_dec.set_array(concat_hp[t])
        im_par.set_array(concat_par_scaled[t])
        im_resid.set_array(concat_resid[t])
        suptitle.set_text(
            f'GROUP {ROI} ({VOX_TAG})  '
            f'Bar → {direction_of_t[t]}  '
            f'(t={within_t[t]*1.6:+.1f}s within pass)')
        return [im_dec, im_par, im_resid, suptitle]

    anim = animation.FuncAnimation(fig, update, frames=T_total,
                                     interval=1000 / 6, blit=False)
    out = Path('/Users/gdehol/git/retsupp/notes/figures/decode_sweep/m4/'
               f'group_{ROI}_{VOX_TAG}_bar_passes_HP_vs_residual.mp4')
    writer = animation.FFMpegWriter(fps=6, bitrate=2400, codec='libx264',
                                      extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
