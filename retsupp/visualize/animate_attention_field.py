"""Animate the attention-field modulation M(g, t) across multiple trials.

Clean two-panel design:
  - Big heatmap of M(g, t) with quiver overlay (gradient = ∇M, shows
    where attention pulls the response).
  - Slim event timeline at the bottom.

Parameters: V3AB across-subject medians from the v3+target fit.
σ pinned at 2° (the principled half-radius value).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'notes' / 'data' / 'af_dog_v3_target_parameters.tsv'
OUT = REPO / 'notes' / 'figures' / 'attention_field_movie.gif'

ROI_FOR_DEMO = 'V3AB'
DEMO_SUBJECT = 28
# σ values get filled at runtime from the actual fit (see main()).
SIGMA_AF = None
SIGMA_DYN = None

RING_R = 4.0
RING_POS = np.array([
    [+RING_R / np.sqrt(2),  +RING_R / np.sqrt(2)],
    [-RING_R / np.sqrt(2),  +RING_R / np.sqrt(2)],
    [-RING_R / np.sqrt(2),  -RING_R / np.sqrt(2)],
    [+RING_R / np.sqrt(2),  -RING_R / np.sqrt(2)],
])
RING_LABEL = ['UR', 'UL', 'LL', 'LR']

# Two HP blocks: HP=UR for the first half, HP=UL for the second half.
# Each block contains its own trial sequence.
HP_BLOCKS = [
    {'t_start': 0.0,  't_end': 22.0, 'hp_idx': 0,  # UR
     'label': 'block 1: HP = UR'},
    {'t_start': 22.0, 't_end': 44.0, 'hp_idx': 1,  # UL
     'label': 'block 2: HP = UL'},
]

# Stimulus geometry — using values that the FITS use (which match
# get_stimulus_with_distractors's defaults). expsettings.yml has
# eccentricity_stimulus=4.0 and size_stimuli=1.5; the aperture formula
# `ecc - size_stimuli/1.8 = 3.17` is hard-coded. The fitting code uses
# distractor_radius=0.4 (NOT 0.75 = size_stimuli/2) — TBD if this
# should match the real rendered disk size.
APERTURE_R = 3.17          # bar aperture radius (deg)
DISTRACTOR_R = 0.4         # what the fits actually use
TARGET_R = 0.4

# Example PRFs INSIDE the bar aperture (3.17° radius) — chosen to be
# in three quadrants spanning HP-near and HP-far positions.
# (1.5°, 1.5°) is at ~2.12° eccentricity → inside the 3.17° aperture.
PRF_EXAMPLES = [
    ((+1.5, +1.5), 0.7),    # near UR ring direction (block-1 HP-side)
    ((-1.5, +1.5), 0.7),    # near UL ring direction (block-2 HP-side)
    ((+0.0, -1.5), 0.7),    # below fovea — far from any HP
]


def hp_at_time(t):
    """Return the HP ring index active at time t."""
    for blk in HP_BLOCKS:
        if blk['t_start'] <= t < blk['t_end']:
            return blk['hp_idx']
    return HP_BLOCKS[-1]['hp_idx']

EVENT_DURATION = 3.0
TRIALS = [
    # Block 1: HP=UR (idx 0)
    {'onset': 4.0,  'target': 1, 'distractor': 0},   # distractor at HP=UR
    {'onset': 14.0, 'target': 3, 'distractor': 2},   # distractor at LP
    # Block 2: HP=UL (idx 1) — sustained map jumps to UL at t=22
    {'onset': 26.0, 'target': 2, 'distractor': 1},   # distractor at NEW HP=UL
    {'onset': 36.0, 'target': 0, 'distractor': 3},   # distractor at LP
]
TOTAL_T = 44.0
DT = 0.2
TIME = np.arange(0, TOTAL_T, DT)

RES = 90
GRID_R = 5.5
g1d = np.linspace(-GRID_R, GRID_R, RES)
GX, GY = np.meshgrid(g1d, g1d)


def gaussian2d(mu, sigma):
    return np.exp(-((GX - mu[0]) ** 2 + (GY - mu[1]) ** 2)
                  / (2 * sigma ** 2)).astype(np.float32)


def load_real_params():
    df = pd.read_csv(TSV, sep='\t')
    sub = df[df['roi'] == ROI_FOR_DEMO]
    return (
        float(sub['g_HP'].median()),
        float(sub['g_LP'].median()),
        float(sub['g_HP_dyn'].median()),
        float(sub['g_LP_dyn'].median()),
        float(sub['g_T_dyn'].median()),
    )


def demo_params():
    """Real parameters from sub-28 V3AB — an 'extreme' but well-fitting
    subject. Exhibits opposite-sign sustained gains (HP suppressed,
    LP captured), strong target capture, and reasonable σ values so
    the AF Gaussians are spatially localized."""
    global SIGMA_AF, SIGMA_DYN
    df = pd.read_csv(TSV, sep='\t')
    row = df[(df.subject == DEMO_SUBJECT) & (df.roi == ROI_FOR_DEMO)].iloc[0]
    SIGMA_AF = float(row.sigma_AF)
    SIGMA_DYN = float(row.sigma_dyn)
    return (
        float(row.g_HP),
        float(row.g_LP),
        float(row.g_HP_dyn),
        float(row.g_LP_dyn),
        float(row.g_T_dyn),
    )


def build_indicators():
    dyn = np.zeros((len(TIME), 4), dtype=np.float32)
    tgt = np.zeros((len(TIME), 4), dtype=np.float32)
    for trial in TRIALS:
        s = int(trial['onset'] / DT)
        e = s + int(EVENT_DURATION / DT)
        if trial['target'] is not None:
            tgt[s:e, trial['target']] = 1.0
        if trial['distractor'] is not None:
            dyn[s:e, trial['distractor']] = 1.0
    return dyn, tgt


def main():
    g_HP, g_LP, g_HP_dyn, g_LP_dyn, g_T_dyn = demo_params()
    print(f'Demo params: g_HP={g_HP:+.2f}, g_LP={g_LP:+.2f}, '
          f'g_HP_dyn={g_HP_dyn:+.2f}, g_LP_dyn={g_LP_dyn:+.2f}, '
          f'g_T_dyn={g_T_dyn:+.2f}')

    dyn_raw, tgt_raw = build_indicators()

    # Pre-compute the per-block sustained field (one per HP_IDX value).
    SUS_PER_HP = {}
    for i in range(4):
        A_HP_i = gaussian2d(RING_POS[i], SIGMA_AF)
        A_LP_i = sum(gaussian2d(RING_POS[j], SIGMA_AF)
                     for j in range(4) if j != i)
        SUS_PER_HP[i] = g_HP * A_HP_i + g_LP * A_LP_i

    A_dyn_per_loc = np.stack([gaussian2d(RING_POS[i], SIGMA_DYN)
                              for i in range(4)], axis=0)

    def field_at(t_idx):
        # HP location can change between blocks
        t = TIME[t_idx]
        hp_idx = hp_at_time(t)
        M = np.ones_like(GX) + SUS_PER_HP[hp_idx]
        for i in range(4):
            g_i = g_HP_dyn if i == hp_idx else g_LP_dyn
            M = M + g_i * dyn_raw[t_idx, i] * A_dyn_per_loc[i]
            M = M + g_T_dyn * tgt_raw[t_idx, i] * A_dyn_per_loc[i]
        return M

    Ms = np.stack([field_at(t) for t in range(0, len(TIME), 5)], axis=0)
    abs_max = float(np.max(np.abs(Ms - 1.0)))
    vmin, vmax = 1.0 - abs_max, 1.0 + abs_max
    print(f'M range: 1 ± {abs_max:.2f}')

    # ---- two-panel layout ---------------------------------------------
    fig = plt.figure(figsize=(8.5, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 1.0], hspace=0.18)
    ax_field = fig.add_subplot(gs[0])
    ax_event = fig.add_subplot(gs[1])

    # main field
    M0 = field_at(0)
    im = ax_field.imshow(M0, extent=[-GRID_R, GRID_R, -GRID_R, GRID_R],
                          cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          origin='lower', aspect='equal')
    # Use make_axes_locatable to PIN the colorbar position so it doesn't
    # flicker when contours are added/removed each frame.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax_field)
    cax = div.append_axes('right', size='4%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('M(g, t)  ( >1 capture, <1 suppress )', fontsize=10)
    # crop view to just the aperture + small margin around distractors
    ax_field.set_xlim(-5.0, 5.0)
    ax_field.set_ylim(-5.0, 5.0)

    # aperture (PRF-mapping circle)
    aperture = plt.Circle((0, 0), APERTURE_R, fill=False,
                           edgecolor='gray', lw=1.0, ls='--',
                           zorder=3)
    ax_field.add_patch(aperture)

    # distractor disks at ring positions, drawn at TRUE scale (0.3°
    # radius). Faint outlines because not all are active at any t.
    for x, y in RING_POS:
        circ = plt.Circle((x, y), DISTRACTOR_R, fill=False,
                           edgecolor='dimgray', lw=0.7, alpha=0.5,
                           zorder=4)
        ax_field.add_patch(circ)

    # HP marker — gold star, will move when block changes
    hp_marker = ax_field.scatter([], [], s=400, marker='*',
                                   facecolor='gold', edgecolor='black',
                                   lw=1.2, zorder=6, label='HP location')

    # fixation
    ax_field.scatter([0], [0], s=80, color='black', marker='+',
                      zorder=5)

    # active stimulus markers (filled circles at true radius)
    # Pre-allocate one Circle per ring position; toggle visible per frame.
    target_patches = []
    for x, y in RING_POS:
        c = plt.Circle((x, y), TARGET_R, facecolor='lime',
                        edgecolor='black', lw=1.0, alpha=0.9,
                        zorder=8, visible=False)
        ax_field.add_patch(c)
        target_patches.append(c)
    distr_patches = []
    for x, y in RING_POS:
        c = plt.Circle((x, y), DISTRACTOR_R, facecolor='red',
                        edgecolor='black', lw=1.0, alpha=0.9,
                        zorder=8, visible=False)
        ax_field.add_patch(c)
        distr_patches.append(c)

    # example PRFs:
    #   - dashed circle = original PRF (1σ contour) without modulation
    #   - magenta contour = HALF-MAX isoline of (PRF × max(M, 0)) per frame
    #     (the actual asymmetric "effective RF" shape, not a moved circle)
    prf_orig_patches = []
    prf_contours = [None] * len(PRF_EXAMPLES)
    for (px, py), sigma_prf in PRF_EXAMPLES:
        orig = plt.Circle((px, py), sigma_prf, fill=False,
                           edgecolor='magenta', lw=1.0, ls=':',
                           alpha=0.7, zorder=9)
        ax_field.add_patch(orig)
        prf_orig_patches.append(orig)
        # mark center
        ax_field.scatter([px], [py], color='magenta', marker='+', s=60,
                          alpha=0.7, zorder=10)

    # quiver overlay — gradient of M, on a SYMMETRIC grid that includes
    # the ring eccentricity (±2.83°) and the fovea.
    quiv_offsets = np.array([-4.2, -2.83, -1.4, 0.0, 1.4, 2.83, 4.2])
    SX, SY = np.meshgrid(quiv_offsets, quiv_offsets)
    # Pre-compute nearest-index lookup for sampling np.gradient output
    QUIV_IX = np.array([np.argmin(np.abs(g1d - x)) for x in quiv_offsets])
    QUIV_IY = np.array([np.argmin(np.abs(g1d - y)) for y in quiv_offsets])
    quiv = ax_field.quiver(SX, SY,
                            np.zeros_like(SX), np.zeros_like(SY),
                            color='black', alpha=0.7, scale=2,
                            scale_units='inches', width=0.009,
                            headwidth=4, headlength=5,
                            zorder=5)

    # COM-shift arrows per PRF (will update each frame).
    # Pre-allocate as FancyArrowPatch we can rewire.
    from matplotlib.patches import FancyArrowPatch
    com_arrows = []
    for (px, py), _ in PRF_EXAMPLES:
        arr = FancyArrowPatch((px, py), (px, py),
                                arrowstyle='-|>',
                                mutation_scale=20, lw=2.5,
                                color='magenta', alpha=0.9, zorder=11)
        ax_field.add_patch(arr)
        com_arrows.append(arr)

    ax_field.set_xlabel('x (deg)')
    ax_field.set_ylabel('y (deg)')
    ax_field.set_title(
        f'Attention field M(g, t) — sub-28 V3AB (real fit)\n'
        f'g_HP={g_HP:+.2f}  g_LP={g_LP:+.2f}  '
        f'g_HP_dyn={g_HP_dyn:+.2f}  g_LP_dyn={g_LP_dyn:+.2f}  '
        f'g_T_dyn={g_T_dyn:+.2f}',
        fontsize=14, weight='bold')
    ax_field.tick_params(labelsize=12)
    ax_field.set_xlabel('x (deg)', fontsize=13)
    ax_field.set_ylabel('y (deg)', fontsize=13)

    # event timeline at bottom — minimal
    for trial in TRIALS:
        s = trial['onset']; e = s + EVENT_DURATION
        ax_event.axvspan(s, e, color='lime', alpha=0.30,
                          ymin=0.55, ymax=0.95, label='target on')
        if trial['distractor'] is not None:
            ax_event.axvspan(s, e, color='red', alpha=0.30,
                              ymin=0.05, ymax=0.45, label='distractor on')
    ax_event.set_xlim(0, TOTAL_T)
    ax_event.set_ylim(0, 1)
    ax_event.set_yticks([0.25, 0.75])
    ax_event.set_yticklabels(['distr.', 'target'], fontsize=9)
    ax_event.set_xlabel('time (s)')
    ax_event.grid(alpha=0.2, axis='x')

    cursor = ax_event.axvline(0, color='black', lw=1.5, alpha=0.85)

    title_text = ax_field.text(
        0.02, 0.98, '', transform=ax_field.transAxes, fontsize=11,
        va='top', ha='left', weight='bold',
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='gray', alpha=0.85))

    def find_active(t):
        tgt_xy, dst_xy = [], []
        for trial in TRIALS:
            s = trial['onset']; e = s + EVENT_DURATION
            if s <= t < e:
                if trial['target'] is not None:
                    tgt_xy.append(RING_POS[trial['target']])
                if trial['distractor'] is not None:
                    dst_xy.append(RING_POS[trial['distractor']])
        return (np.array(tgt_xy) if tgt_xy else np.empty((0, 2)),
                np.array(dst_xy) if dst_xy else np.empty((0, 2)))

    def find_active_idx(t):
        """Return (target_idx_or_None, distractor_idx_or_None)."""
        for trial in TRIALS:
            s = trial['onset']; e = s + EVENT_DURATION
            if s <= t < e:
                return trial['target'], trial['distractor']
        return None, None

    # Precompute static PRF kernels (we'll multiply by current M).
    prf_kernels = [
        np.exp(-((GX - px) ** 2 + (GY - py) ** 2) / (2 * s ** 2))
        .astype(np.float32)
        for (px, py), s in PRF_EXAMPLES
    ]

    def update(frame):
        t = TIME[frame]
        M = field_at(frame)
        im.set_data(M)
        gy, gx = np.gradient(M, g1d, g1d)
        u = gx[np.ix_(QUIV_IY, QUIV_IX)]
        v = gy[np.ix_(QUIV_IY, QUIV_IX)]
        quiv.set_UVC(u, v)
        cursor.set_xdata([t, t])

        # HP marker follows the current block's HP
        hp_idx = hp_at_time(t)
        hp_marker.set_offsets([RING_POS[hp_idx]])

        # active stimuli — toggle visibility of pre-allocated patches
        tgt_idx, dst_idx = find_active_idx(t)
        for k, p in enumerate(target_patches):
            p.set_visible(k == tgt_idx)
        for k, p in enumerate(distr_patches):
            p.set_visible(k == dst_idx)

        # effective RF contours + COM arrows per PRF.
        M_pos = np.maximum(M, 0.0)
        for k, ((px, py), _) in enumerate(PRF_EXAMPLES):
            old = prf_contours[k]
            if old is not None:
                try:
                    old.remove()
                except (AttributeError, ValueError):
                    if hasattr(old, 'collections'):
                        for coll in old.collections:
                            try: coll.remove()
                            except ValueError: pass
            eff = prf_kernels[k] * M_pos
            mx = eff.max()
            tot = eff.sum()
            if mx > 1e-6:
                cs = ax_field.contour(
                    GX, GY, eff, levels=[mx * 0.5],
                    colors='magenta', linewidths=2.5, alpha=0.95,
                    zorder=10,
                )
                prf_contours[k] = cs
                com_x = (GX * eff).sum() / tot
                com_y = (GY * eff).sum() / tot
            else:
                prf_contours[k] = None
                com_x, com_y = px, py
            # update arrow from original to current COM
            com_arrows[k].set_positions((px, py), (com_x, com_y))

        title_text.set_text(f't = {t:5.1f} s')
        return (im, hp_marker, title_text, quiv, cursor,
                *target_patches, *distr_patches,
                )

    anim = FuncAnimation(fig, update, frames=len(TIME), interval=100,
                         blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
