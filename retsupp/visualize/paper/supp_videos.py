"""Draft supplementary videos (MP4, H.264).

``model``: animated Fig. 4b (schematic, illustrative gains as in ``fig4_model.main_hp``): one pRF near a ring location in
a block with HP near it and a block with HP opposite; per block the effective drive on screen (input × gain field),
the stimulus reaching the pRF, the gain it sees and the predicted BOLD with and without gain, revealed over time.

``decoding``: true bar vs the bar decoded from V1 in one held-out run (the Fig. 6a example run) and the group mean,
frame by frame.

Usage::

    python -m retsupp.visualize.paper.supp_videos model
    python -m retsupp.visualize.paper.supp_videos decoding --tr-stop 130
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from retsupp.visualize.paper import fig4_model as f4
from retsupp.visualize.paper.paper_style import DISTRACTOR, FG, HP, IMAGE_CMAP, MUTED, STIM, TARGET

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'notes' / 'figures' / 'videos'
DATA = REPO / 'notes' / 'data'
TR = 1.6


def encode(frames_dir, out, fps):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(Path(frames_dir) / 'f%04d.png'),
           '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
           str(out)]
    subprocess.run(cmd, check=True)
    print(out)


def drive_image(t, trials, hp_ix):
    """Model input × gain field on the display grid at time t (as in ``fig4_model.simulate``)."""
    X, Y = f4.X, f4.Y
    in_ap = np.hypot(X, Y) < f4.APERTURE
    st = ((X >= f4.bar_left(t)) & (X < f4.bar_left(t) + f4.BAR_W) & in_ap).astype(float)
    field = 1 + sum((f4.GAINS['hp'] if i == hp_ix else f4.GAINS['lp']) * f4.gauss(*p, f4.SIGMA['sus'])
                    for i, p in enumerate(f4.POS))
    for onset, ti, di in trials:
        if onset <= t < onset + 1.5:
            items = np.zeros_like(X, bool)
            for a in np.deg2rad(np.arange(8) * 45):
                cx, cy = f4.RING * np.cos(a), f4.RING * np.sin(a)
                items |= (np.abs(X - cx) < .75) & (np.abs(Y - cy) < .19)
            st = st + items
            field = field + f4.GAINS['t'] * f4.gauss(*f4.POS[ti], f4.SIGMA['dyn'])
            if di is not None:
                field = field + f4.GAINS['d'] * f4.gauss(*f4.POS[di], f4.SIGMA['dyn'])
    return st * field


def video_model(name='suppvideo_fig4b_model_draft', fps=12):
    trials = [(5, 1, 3), (16, 0, 2), (27, 2, 1), (37, 3, 0)]
    blocks = {'HP near this pRF': 3, 'HP opposite this pRF': 1}
    prf = {'pRF': f4.RING_PRF}
    sim = {k: f4.simulate(trials=trials, hp_ix=hp, prfs=prf)['pRF'] for k, hp in blocks.items()}
    t = next(iter(sim.values()))['t']
    s_hi = max(d['stimulus'].max() for d in sim.values())
    g_hi = max(max(d['target'].max(), d['distractor'].max()) for d in sim.values())
    g_lo = min(d['base'] for d in sim.values())
    p_hi = max(max(d['predicted'].max(), d['without'].max()) for d in sim.values())
    p_lo = min(min(d['predicted'].min(), d['without'].min()) for d in sim.values())
    with tempfile.TemporaryDirectory() as tmp:
        for i, ti in enumerate(t):
            fig = plt.figure(figsize=(8, 3.6), dpi=150, facecolor='w')
            for k, (nm, hp) in enumerate(blocks.items()):
                d = sim[nm]
                x0 = .02 + k * .5
                ax = fig.add_axes([x0, .3, .2, .5])
                ax.imshow(drive_image(ti, trials, hp), origin='lower', extent=(-f4.LIM, f4.LIM, -f4.LIM, f4.LIM),
                          cmap=IMAGE_CMAP, vmin=0, vmax=2.6, interpolation='bilinear')
                ax.add_patch(Circle((0, 0), f4.APERTURE, fill=False, ec=MUTED, lw=.6))
                ax.add_patch(Circle(f4.POS[hp], 1.12, fill=False, ec=HP, lw=1., ls=(0, (2.5, 1.5))))
                x, y, sd = f4.RING_PRF
                ax.add_patch(Circle((x, y), 1.18 * sd, fill=False, ec='#2A9D8F', lw=1.2))
                ax.set_axis_off()
                ax.set_title(f'{nm}\nEffective drive', fontsize=8, color=FG)
                rows = [fig.add_axes([x0 + .22, .7 - r * .28, .25, .2]) for r in range(3)]
                for ax_, lab in zip(rows, ('Stimulus in pRF', 'Gain', 'Predicted BOLD')):
                    ax_.set_xlim(0, t[-1])
                    ax_.set_axis_off()
                    ax_.text(0, 1.0, lab, transform=ax_.transAxes, fontsize=7, color=FG, va='bottom')
                    ax_.axvline(ti, color='.6', lw=.6)
                m = t <= ti
                rows[0].fill_between(t[m], 0, d['stimulus'][m] - d['items'][m], color='.72', lw=0)
                rows[0].fill_between(t[m], d['stimulus'][m] - d['items'][m], d['stimulus'][m], color='.25', lw=0)
                rows[0].set_ylim(0, s_hi * 1.05)
                rows[1].axhline(0, color='#B8C0C8', lw=.5, ls=(0, (2, 2)))
                rows[1].plot(t[m], np.full(m.sum(), d['base']), color=HP, lw=1.2)
                rows[1].fill_between(t[m], d['base'], d['base'] + d['target'][m], color=TARGET, lw=0)
                rows[1].fill_between(t[m], d['base'], d['base'] + d['distractor'][m], color=DISTRACTOR, lw=0)
                rows[1].set_ylim(g_lo * 2, g_hi * 1.05)
                rows[2].plot(t[m], d['without'][m], color='.65', lw=1.1, label='Without gain')
                rows[2].plot(t[m], d['predicted'][m], color=FG, lw=1.3, label='With gain')
                rows[2].set_ylim(p_lo - .05 * (p_hi - p_lo), p_hi * 1.05)
                if k == 0:
                    rows[2].legend(loc='lower left', bbox_to_anchor=(0, -.45), ncol=2, frameon=False, fontsize=7)
            fig.text(.5, .02, f'Time {ti:4.1f} s. Schematic: illustrative, exaggerated gains (as Fig. 4b).',
                     ha='center', fontsize=8, color=FG)
            fig.savefig(Path(tmp) / f'f{i:04d}.png')
            plt.close(fig)
        encode(tmp, OUT / f'{name}.mp4', fps)


def disc(ax, img, cmap, vmax, title=None):
    c = np.linspace(-5, 5, img.shape[0])
    Xg, Yg = np.meshgrid(c, c)
    ax.imshow(np.where(np.hypot(Xg, Yg) <= f4.APERTURE + .1, np.clip(img, 0, None), 0), origin='lower',
              extent=[-5.1, 5.1, -5.1, 5.1], cmap=cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax.add_patch(Circle((0, 0), f4.APERTURE, fill=False, ec='.75', lw=.5))
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.6, 3.6)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, color=FG)


def video_decoding(name='suppvideo_decoded_bar_V1_draft', source='jointhrf', key='sub-03_ses-1_run-4_roi-V1',
                   tr_start=0, tr_stop=130, fps=8):
    import matplotlib.colors as mcolors
    fr = np.load(DATA / f'decoding_frames_{source}_2026-11.npz')
    par = fr['paradigm'].astype(float)
    group = fr['group'][list(fr['rois']).index('V1'), list(fr['decoders']).index('free')].astype(float)
    ex = np.load(DATA / f'decoding_example_run_{source}_2026-11.npz')[key].astype(float)
    cm = mcolors.LinearSegmentedColormap.from_list('stim', ['#FFFFFF', STIM])
    vx, vg = np.nanpercentile(ex, 99.7), np.nanpercentile(group, 99.7)
    sub, ses, run, roi = key.split('_')
    with tempfile.TemporaryDirectory() as tmp:
        for i, k in enumerate(range(tr_start, tr_stop)):
            fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), dpi=150, facecolor='k')
            fig.subplots_adjust(left=.01, right=.99, top=.82, bottom=.1, wspace=.05)
            for ax in axs:
                ax.set_facecolor('k')
            disc(axs[0], par[k], 'gray', 1., 'True bar')
            disc(axs[1], ex[k], IMAGE_CMAP, vx, f'Decoded, one held-out run\n({sub}, {roi.replace("roi-", "")})')
            disc(axs[2], group[k], IMAGE_CMAP, vg, 'Decoded, mean over\nparticipants (V1)')
            for ax in axs:
                ax.title.set_color('w')
            fig.text(.5, .03, f'{k * TR:5.1f} s (TR {k})', ha='center', color='w', fontsize=8)
            fig.savefig(Path(tmp) / f'f{i:04d}.png', facecolor='k')
            plt.close(fig)
        encode(tmp, OUT / f'{name}.mp4', fps)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('which', choices=['model', 'decoding'])
    ap.add_argument('--tr-start', type=int, default=0)
    ap.add_argument('--tr-stop', type=int, default=130)
    a = ap.parse_args()
    if a.which == 'model':
        video_model()
    else:
        video_decoding(tr_start=a.tr_start, tr_stop=a.tr_stop)
