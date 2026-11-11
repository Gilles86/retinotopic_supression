"""Aligned bar-passes-through-HP movies in the HP-at-top rotated frame.

Each run has 8 bar passes (2 per cardinal direction). After rotating
the decoded paradigm so HP→top, the 4 cardinal directions become 4
diagonal directions in the rotated frame. The mapping from cardinal
→ rotated-diagonal depends on the HP rotation angle.

For each "rotated diagonal direction" (↗ ↘ ↙ ↖), this script averages
the bar pass windows across all HPs and runs that contribute to that
direction. Output: 2x2 movie grid, one panel per rotated diagonal.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import rotate as nd_rotate
from retsupp.utils.data import Subject, location_angles


import sys
SUBJECT = int(sys.argv[1]) if len(sys.argv) > 1 else 23
ROI = sys.argv[2] if len(sys.argv) > 2 else 'V1'
VOX_TAG = sys.argv[3] if len(sys.argv) > 3 else 'vox200'    # or 'vox2000_psig0.5'
MODEL = 4
WIN_LEN = 27   # TRs per bar pass
RING_R = 4.0   # corner ring eccentricity
RING_POSITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


# Mapping: rotated-direction (target) → {HP: original cardinal bar direction}.
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


def _rot_deg(hp: str) -> float:
    """scipy.ndimage.rotate uses CW-in-math (sigh) — so the rotation
    that takes HP→top is `angle_hp − π/2`, NOT π/2−angle_hp."""
    key = hp.replace('_', ' ')
    return float(np.degrees(location_angles[key] - np.pi / 2))


def _bar_onset_trs(sub: Subject, session: int, run: int,
                    direction: str, tr: float = 1.6) -> list[int]:
    """Return the TR indices of every bar_<direction> onset in this run."""
    ev = sub.get_onsets(session=session, run=run)
    rows = ev[ev['event_type'] == f'bar_{direction}']
    return [int(round(o / tr)) for o in rows['onset'].values]


def main():
    sub = Subject(SUBJECT, bids_folder='/data/ds-retsupp')
    hp_by_run = sub.get_hpd_locations()
    base = Path(f'/data/ds-retsupp/derivatives/decoded/model4/sub-{SUBJECT:02d}')

    # Pre-load all per-run decoded + paradigm.
    runs_data = {}
    grid_ref = None
    par_ref = None
    for (ses, run), hp in hp_by_run.items():
        p = base / f'decoded_ses-{ses}_run-{run}_roi-{ROI}_{VOX_TAG}.npz'
        if not p.exists():
            continue
        with np.load(p) as d:
            dec = d['decoded'].astype(np.float32)
            if grid_ref is None:
                grid_ref = d['grid'].astype(np.float32)
                par_ref = d['paradigm'].astype(np.float32)
        runs_data[(ses, run, hp)] = dec
    print(f'Loaded {len(runs_data)} runs from sub-{SUBJECT:02d} {ROI} ({VOX_TAG})')

    extent = [float(grid_ref[:, 0].min()), float(grid_ref[:, 0].max()),
              float(grid_ref[:, 1].min()), float(grid_ref[:, 1].max())]

    # For each rotated direction, gather bar-pass windows.
    # Two cases: anchor = HP (one per run) OR anchor = any of 3 LPs (per run).
    # The mapping table ROT_DIR_TO_ORIG works for any position at a diagonal.
    print('\nGathering bar-pass windows per rotated direction (HP + 3 LP anchors)...')
    rotated_dirs = list(ROT_DIR_TO_ORIG.keys())
    win_hp: dict[str, list[np.ndarray]] = {d: [] for d in rotated_dirs}
    win_lp: dict[str, list[np.ndarray]] = {d: [] for d in rotated_dirs}
    par_hp: dict[str, list[np.ndarray]] = {d: [] for d in rotated_dirs}

    for (ses, run, hp), dec in runs_data.items():
        for anchor in RING_POSITIONS:
            rot_deg = _rot_deg(anchor)
            dec_rot = nd_rotate(dec, rot_deg, axes=(1, 2),
                                  reshape=False, order=1,
                                  mode='constant', cval=0.0).astype(np.float32)
            par_rot = nd_rotate(par_ref, rot_deg, axes=(1, 2),
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

    n_hp = {d: len(v) for d, v in win_hp.items()}
    n_lp = {d: len(v) for d, v in win_lp.items()}
    print(f'  HP-anchored windows per direction: {n_hp}')
    print(f'  LP-anchored windows per direction: {n_lp}')

    mean_per = {d: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
                for d, v in win_hp.items()}
    mean_lp = {d: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
                for d, v in win_lp.items()}
    contrast = {d: mean_per[d] - mean_lp[d] for d in rotated_dirs}
    mean_par = {d: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
                for d, v in par_hp.items()}

    # Render two panels (HP-anchored + contrast) concatenated in time.
    sequence = ['↖', '↗', '↘', '↙']
    vmax_hp = float(np.quantile(np.stack(list(mean_per.values())), 0.99))
    vmax_c = float(np.quantile(np.abs(np.stack(list(contrast.values()))), 0.99))

    concat_dec = np.concatenate([mean_per[d] for d in sequence], axis=0)
    concat_con = np.concatenate([contrast[d] for d in sequence], axis=0)
    concat_par = np.concatenate([mean_par[d] for d in sequence], axis=0)
    T_total = concat_dec.shape[0]
    direction_of_t = np.repeat(sequence, WIN_LEN)
    within_t = np.tile(np.arange(WIN_LEN), len(sequence))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.plot(0.0, RING_R, '*', mfc='gold', mec='black',
                 ms=24, mew=1.2, zorder=10)
        for cx, cy in [(RING_R, 0), (-RING_R, 0), (0, -RING_R)]:
            ax.plot(cx, cy, 'o', mfc='none', mec='white',
                     ms=8, mew=1.2, zorder=9)
    im_hp = axes[0].imshow(concat_dec[0], extent=extent, origin='lower',
                              cmap='magma', vmin=0, vmax=vmax_hp,
                              interpolation='bilinear')
    im_c = axes[1].imshow(concat_con[0], extent=extent, origin='lower',
                             cmap='RdBu_r', vmin=-vmax_c, vmax=vmax_c,
                             interpolation='bilinear')
    axes[0].set_title('HP-anchored', fontsize=11, weight='bold')
    axes[1].set_title('HP − mean(LPs)', fontsize=11, weight='bold')
    cont_state = {0: [None], 1: [None]}
    suptitle = fig.suptitle('', fontsize=12, weight='bold')

    def update(t):
        im_hp.set_array(concat_dec[t])
        im_c.set_array(concat_con[t])
        for k, (ax, color) in enumerate([(axes[0], 'cyan'),
                                          (axes[1], 'black')]):
            while cont_state[k][0] is not None:
                try: cont_state[k][0].remove()
                except Exception: pass
                cont_state[k][0] = None
            if concat_par[t].max() > 0.1:
                cs = ax.contour(concat_par[t], levels=[0.3],
                                 colors=color, linewidths=2.5,
                                 origin='lower', extent=extent)
                cont_state[k][0] = cs
        suptitle.set_text(
            f'sub-{SUBJECT:02d} {ROI}  ({VOX_TAG})  '
            f'Bar → {direction_of_t[t]}  '
            f'(t={within_t[t]*1.6:+.1f}s within pass)')
        return [im_hp, im_c, suptitle]

    anim = animation.FuncAnimation(fig, update, frames=T_total,
                                     interval=1000 / 6, blit=False)
    out = Path('/Users/gdehol/git/retsupp/notes/figures/decode_sweep/m4/'
               f'sub-{SUBJECT:02d}_{ROI}_{VOX_TAG}_bar_passes_HP_vs_contrast.mp4')
    writer = animation.FFMpegWriter(fps=6, bitrate=2400, codec='libx264',
                                      extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
