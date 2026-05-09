"""3-panel movie of (paradigm + events, gain field, drive).

For one (subject, ROI, session, run) plays out the run TR-by-TR:

  Left   : paradigm + on-screen events
             - bar pass (greyscale mask)
             - aperture (dashed black circle, r = 3.17°)
             - 4 ring positions; HP gold star, LPs grey small dots
             - fixation '+' at centre
             - target = lime circle at target_location while target on
             - distractor = red dashed circle at distractor_location
  Middle : gain field (sustained + dynamic), with σ_AF circles around
           the four ring positions and σ_dyn circles drawn on currently-
           active distractor rings only.
  Right  : drive  =  paradigm × (1 + gain)  ("attended stimulus").

All three panels are forced to the same size; the colour scale for
gain/drive is symmetric and clamped to a moderate value so single-cell
outliers don't wash out the rest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
from scipy.signal import fftconvolve

from retsupp.utils.data import Subject

ECC = 4.0
APERTURE_R = 3.17
DISTRACTOR_R = 0.4
TARGET_R = 0.4
RING_KEY_TO_XY = {
    'upper_right': ( ECC / np.sqrt(2),  ECC / np.sqrt(2)),
    'upper_left':  (-ECC / np.sqrt(2),  ECC / np.sqrt(2)),
    'lower_left':  (-ECC / np.sqrt(2), -ECC / np.sqrt(2)),
    'lower_right': ( ECC / np.sqrt(2), -ECC / np.sqrt(2)),
}
CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
LOC_TO_KEY = {1.0: 'upper_right', 3.0: 'upper_left',
              5.0: 'lower_left',  7.0: 'lower_right'}
AF_DIR_DEFAULT = ('/data/ds-retsupp/derivatives/'
                   'af_prf_joint_dynamic_v3_dog')


def canonical_hrf(tr=1.6, length_s=32.0, peak=6.0, undershoot=16.0,
                   ratio=6.0):
    from scipy.stats import gamma
    t = np.arange(0.0, length_s + 1e-6, tr)
    h = (gamma.pdf(t, peak / 0.9) -
         gamma.pdf(t, undershoot / 0.9) / ratio)
    return h / h.max()


def gain_field_2d(grid_xy_x, grid_xy_y, sigma, g_HP, g_LP, hp_xy):
    out = g_HP * np.exp(-((grid_xy_x - hp_xy[0]) ** 2 +
                          (grid_xy_y - hp_xy[1]) ** 2)
                         / (2 * sigma ** 2))
    for k in CONDITIONS:
        if (abs(RING_KEY_TO_XY[k][0] - hp_xy[0]) < 1e-3 and
            abs(RING_KEY_TO_XY[k][1] - hp_xy[1]) < 1e-3):
            continue
        lx, ly = RING_KEY_TO_XY[k]
        out += g_LP * np.exp(-((grid_xy_x - lx) ** 2 +
                                (grid_xy_y - ly) ** 2)
                              / (2 * sigma ** 2))
    return out


def per_tr_dynamic_gain(grid_xy_x, grid_xy_y, dyn_indicator_T_C,
                         sigma_dyn, g_HP_dyn, g_LP_dyn, hp_xy, tr):
    T = dyn_indicator_T_C.shape[0]
    R = grid_xy_x.shape[0]
    is_hp = np.array([
        (abs(RING_KEY_TO_XY[k][0] - hp_xy[0]) < 1e-3 and
         abs(RING_KEY_TO_XY[k][1] - hp_xy[1]) < 1e-3)
        for k in CONDITIONS
    ])
    h = canonical_hrf(tr=tr)
    conv = np.zeros_like(dyn_indicator_T_C)
    for c in range(dyn_indicator_T_C.shape[1]):
        conv[:, c] = fftconvolve(dyn_indicator_T_C[:, c], h,
                                   mode='full')[:T]
    out = np.zeros((T, R, R), dtype=np.float32)
    per_ring = np.zeros((T, len(CONDITIONS)), dtype=np.float32)
    for c, k in enumerate(CONDITIONS):
        rx, ry = RING_KEY_TO_XY[k]
        gauss = np.exp(-((grid_xy_x - rx) ** 2 +
                         (grid_xy_y - ry) ** 2)
                       / (2 * sigma_dyn ** 2))
        weight = (g_HP_dyn if is_hp[c] else g_LP_dyn)
        out += np.einsum('t,xy->txy', weight * conv[:, c], gauss)
        per_ring[:, c] = weight * conv[:, c]
    return out, per_ring


def per_tr_event_state(events_df, T, tr):
    """Per-TR booleans: is target / distractor on at each ring."""
    target_evs = events_df[events_df['event_type'] == 'target']
    target_active   = np.zeros((T, 4), dtype=bool)
    distract_active = np.zeros((T, 4), dtype=bool)
    for _, ev in target_evs.iterrows():
        onset = ev['onset']
        dur = ev['duration'] if not pd.isna(ev['duration']) else 1.5
        t_lo = int(np.floor(onset / tr))
        t_hi = int(np.ceil((onset + dur) / tr))
        t_lo = max(t_lo, 0); t_hi = min(t_hi, T)
        if t_hi <= t_lo:
            continue
        if not pd.isna(ev.get('target_location')):
            tl = ev['target_location']
            for c, k in enumerate(CONDITIONS):
                if LOC_TO_KEY.get(tl) == k:
                    target_active[t_lo:t_hi, c] = True
        if not pd.isna(ev.get('distractor_location')):
            dl = ev['distractor_location']
            for c, k in enumerate(CONDITIONS):
                if LOC_TO_KEY.get(dl) == k:
                    distract_active[t_lo:t_hi, c] = True
    return target_active, distract_active


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, default=2)
    p.add_argument('--roi', default='V1')
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--run', type=int, default=1)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--af-dir', default=AF_DIR_DEFAULT)
    p.add_argument('--resolution', type=int, default=80)
    p.add_argument('--fps', type=int, default=3)
    p.add_argument('--gain-clip', type=float, default=2.0,
                   help='Hard cap on |gain| for the colour scale.')
    p.add_argument('--out', type=Path,
                   default=Path('notes/figures/paradigm_movie.mp4'))
    args = p.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path('/Users/gdehol/git/retsupp') / out
    out.parent.mkdir(parents=True, exist_ok=True)

    sub = Subject(args.subject, args.bids_folder)
    print(f'sub-{args.subject:02d}  ses-{args.session}  run-{args.run}  '
          f'roi-{args.roi}')

    paradigm = sub.get_stimulus(args.session, args.run,
                                  resolution=args.resolution)
    T = paradigm.shape[0]
    tr = sub.get_tr(args.session, args.run)
    print(f'  paradigm: {paradigm.shape} (TR={tr:.2f}s)')

    hp_loc = sub.get_hpd_locations()[(args.session, args.run)]
    hp_xy = RING_KEY_TO_XY[hp_loc]
    print(f'  HP @ {hp_loc} = ({hp_xy[0]:+.2f}, {hp_xy[1]:+.2f})')

    dyn = sub.get_dynamic_indicator(args.session, args.run,
                                     oversampling=1)[:T]

    events = sub.get_onsets(args.session, args.run)
    target_active, distract_active = per_tr_event_state(events, T, tr)
    print(f'  target on for {target_active.any(1).sum()}/{T} TRs, '
          f'distractor on for {distract_active.any(1).sum()}/{T} TRs')

    pars_fn = (Path(args.af_dir) / f'sub-{args.subject:02d}'
               / f'sub-{args.subject:02d}_roi-{args.roi}_mode-signed_'
                 'dog-dyn-v3-af-prf-pars.tsv')
    pars_df = pd.read_csv(pars_fn, sep='\t')
    p0 = pars_df.iloc[0]
    sigma_AF, g_HP, g_LP = float(p0.sigma_AF), float(p0.g_HP), float(p0.g_LP)
    sigma_dyn = float(p0.sigma_dyn)
    g_HP_dyn = float(p0.g_HP_dyn); g_LP_dyn = float(p0.g_LP_dyn)
    print(f'  AF: σ_AF={sigma_AF:.2f} g_HP={g_HP:+.2f} g_LP={g_LP:+.2f} | '
          f'σ_dyn={sigma_dyn:.2f} g_HP_dyn={g_HP_dyn:+.2f} '
          f'g_LP_dyn={g_LP_dyn:+.2f}')

    fov = 5.5
    grid = np.linspace(-fov, fov, args.resolution)
    XX, YY = np.meshgrid(grid, grid)
    sustained = gain_field_2d(XX, YY, sigma_AF, g_HP, g_LP, hp_xy)
    dyn_field, dyn_per_ring = per_tr_dynamic_gain(
        XX, YY, dyn, sigma_dyn, g_HP_dyn, g_LP_dyn, hp_xy, tr)

    if paradigm.shape[1] != args.resolution:
        from scipy.ndimage import zoom
        f = args.resolution / paradigm.shape[1]
        paradigm = zoom(paradigm, (1, f, f), order=1)
    paradigm = paradigm[:T]

    gain_total = sustained[None, :, :] + dyn_field
    drive = paradigm * (1.0 + gain_total)

    # Hard-cap colour scale.
    g_lim = args.gain_clip
    g_norm = TwoSlopeNorm(vmin=-g_lim, vcenter=0, vmax=g_lim)
    d_lim = max(g_lim, 1.5)
    d_norm = TwoSlopeNorm(vmin=-d_lim, vcenter=0, vmax=d_lim)

    # ---- Layout: 3 EQUAL-SIZED panels in 1 row ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8),
                              gridspec_kw=dict(wspace=0.06,
                                                left=0.02, right=0.98,
                                                top=0.86, bottom=0.05))
    titles = ['Paradigm + events',
              f'Gain field   (σ_AF={sigma_AF:.1f}°, σ_dyn={sigma_dyn:.1f}°)',
              'Drive   (paradigm × (1 + gain))']
    for ax, ttl in zip(axes, titles):
        ax.set_xlim(-fov, fov); ax.set_ylim(-fov, fov)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=12, weight='bold')

    extent = (-fov, fov, -fov, fov)
    im_p = axes[0].imshow(paradigm[0], extent=extent, origin='lower',
                            cmap='gray', vmin=0, vmax=1)
    im_g = axes[1].imshow(gain_total[0], extent=extent, origin='lower',
                            cmap='RdBu_r', norm=g_norm,
                            interpolation='bilinear')
    im_d = axes[2].imshow(drive[0], extent=extent, origin='lower',
                            cmap='RdBu_r', norm=d_norm,
                            interpolation='bilinear')

    # ----- Static decoration on every panel -----
    for ax in axes:
        # Aperture (dashed black, r=3.17°).
        ax.add_patch(Circle((0, 0), APERTURE_R, fill=False, ec='k',
                             lw=1.6, ls='--', zorder=4))
        # Fixation cross.
        ax.plot(0, 0, marker='+', color='k', ms=11, mew=1.4, zorder=5)
        # 4 LP rings as faint grey markers.
        for k in CONDITIONS:
            x, y = RING_KEY_TO_XY[k]
            if k == hp_loc:
                continue
            ax.plot(x, y, 'o', mfc='none', mec='0.5',
                    mew=0.7, ms=8, zorder=5)
        # HP star (gold).
        ax.plot(*hp_xy, marker='*', color='gold',
                 mec='k', mew=0.6, ms=22, zorder=6)

    # ----- Left panel: target lime circle + distractor red dashed
    target_patches, distract_patches = [], []
    for c, k in enumerate(CONDITIONS):
        x, y = RING_KEY_TO_XY[k]
        tcirc = Circle((x, y), TARGET_R, ec='black', fc='#7CFC00',
                       lw=1.0, alpha=0.0, zorder=7)
        dcirc = Circle((x, y), DISTRACTOR_R, ec='#d62728', fc='none',
                       lw=2.0, ls='--', alpha=0.0, zorder=7)
        axes[0].add_patch(tcirc); axes[0].add_patch(dcirc)
        target_patches.append(tcirc); distract_patches.append(dcirc)

    # ----- Middle panel: σ_AF and σ_dyn circles -----
    # Convention: draw Gaussian extents as FWHM circles (= 2.355·σ).
    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
    sus_circles = []
    for c, k in enumerate(CONDITIONS):
        x, y = RING_KEY_TO_XY[k]
        col = ('gold' if k == hp_loc else '0.45')
        circ = Circle((x, y), sigma_AF * SIGMA_TO_FWHM, ec=col, fc='none',
                      lw=1.6, ls='-', zorder=6, alpha=0.85)
        axes[1].add_patch(circ)
        sus_circles.append(circ)
    dyn_circles = []
    for c, k in enumerate(CONDITIONS):
        x, y = RING_KEY_TO_XY[k]
        circ = Circle((x, y), sigma_dyn * SIGMA_TO_FWHM, ec='k', fc='none',
                      lw=2.4, ls='--', zorder=7, alpha=0.0)
        axes[1].add_patch(circ)
        dyn_circles.append(circ)
    axes[1].text(-fov + 0.15, -fov + 0.55,
                  'thin solid = AF FWHM (sustained)\n'
                  'dashed = dyn FWHM (when active)',
                  fontsize=8, color='0.20', va='bottom')

    # ----- One shared, slim colour bar at the bottom of mid+right
    cbar_ax = fig.add_axes([0.36, 0.03, 0.30, 0.018])
    cb = fig.colorbar(im_g, cax=cbar_ax, orientation='horizontal')
    cb.set_label('gain  (added to 1; right panel uses the same scale)',
                  fontsize=8)
    cb.ax.tick_params(labelsize=7)

    suptitle = fig.suptitle(
        f'sub-{args.subject:02d}  ses-{args.session}  run-{args.run}  '
        f'roi-{args.roi}  HP={hp_loc}'
        f'   ★=HP   ○=LP   ⌀=aperture (3.17°)   ●=distractor   ●=target',
        fontsize=11)
    timestamp = fig.text(0.5, 0.92, f'TR 0/{T - 1}    t = 0.0 s',
                         ha='center', fontsize=11, weight='bold',
                         color='#444')

    def update(t):
        im_p.set_data(paradigm[t])
        im_g.set_data(gain_total[t])
        im_d.set_data(drive[t])
        # Toggle event markers on the left panel.
        for c, k in enumerate(CONDITIONS):
            target_patches[c].set_alpha(0.95 if target_active[t, c] else 0.0)
            distract_patches[c].set_alpha(0.95 if distract_active[t, c] else 0.0)
        # Pulse-strength alpha on the middle panel.
        for c in range(len(CONDITIONS)):
            mag = abs(dyn_per_ring[t, c])
            ref = max(abs(g_HP_dyn), abs(g_LP_dyn), 1e-3)
            alpha = float(np.clip(mag / (0.5 * ref), 0.0, 0.95))
            dyn_circles[c].set_alpha(alpha)
            sign_col = ('#d62728' if dyn_per_ring[t, c] > 0 else '#1f77b4')
            dyn_circles[c].set_edgecolor(sign_col)
        timestamp.set_text(f'TR {t}/{T - 1}    t = {t * tr:.1f} s')
        return [im_p, im_g, im_d, timestamp,
                *target_patches, *distract_patches, *dyn_circles]

    print(f'  writing {T} frames @ {args.fps} fps  '
          f'(duration = {T / args.fps:.1f} s)')
    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps,
                          blit=False)
    anim.save(out, writer='ffmpeg', fps=args.fps, dpi=120,
              savefig_kwargs={'facecolor': 'white'})
    print(f'\nDone: {out}')


if __name__ == '__main__':
    main()
