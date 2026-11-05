"""Figure 2: the gain-field model and what it predicts at three pRF locations. No fitted values.

Usage::

    python -m retsupp.visualize.paper.fig2_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, STIM, TARGET, letter, save
from retsupp.visualize.paper.plot_fig1 import display

LEGEND = """
**Figure 2. Gain-field model.** Top, the screen is reduced to a model input (bar plus items), multiplied by a gain field: the sum of sustained fields at the four ring locations and transient fields at the target and singleton distractor. Bottom, what three pRFs receive over the first 200 s of one real run of sub-19, HP at the lower right, with illustrative gains: the bar in the pRF, the sustained, target and distractor gain the pRF sees, and the predicted response after the canonical HRF. The same events give different signals depending on pRF location.
"""
LIM = 5.2
RING = 4.
POS = RING / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])   # order of the event indicators
HP_IX, T_IX = 3, 1                                                         # as in Figure 1a
GAINS = dict(hp=-.22, lp=-.1, t=.35, d=.2)                                 # illustrative
SIGMA = dict(sus=1.3, dyn=1.6)
PRFS = {'pRF near HP': (2.1, -2.1, .9), 'pRF near LP': (-2.1, 2.1, .9), 'pRF near fixation': (.9, .3, .6)}
PRF_COL = {'pRF near HP': HP, 'pRF near LP': '#1F2933', 'pRF near fixation': '#8D99A6'}
g = np.linspace(-LIM, LIM, 261)
X, Y = np.meshgrid(g, g)


def gauss(cx, cy, sd):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sd ** 2))


def stimulus():
    s = np.zeros_like(X)
    s[(np.abs(X + .9) < .4) & (np.hypot(X, Y) < 3.17)] = 1
    for a in np.deg2rad(np.arange(8) * 45):
        cx, cy = RING * np.cos(a), RING * np.sin(a)
        s[(np.abs(X - cx) < .75) & (np.abs(Y - cy) < .19)] = 1
    return s


def show(ax, im, cmap='RdBu_r', lim=None, title=None, color=FG):
    lim = lim if lim is not None else np.abs(im).max()
    ax.imshow(im, origin='lower', extent=(-LIM, LIM, -LIM, LIM), cmap=cmap, vmin=-lim, vmax=lim, interpolation='bilinear')
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.4))
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, color=color, pad=2)


def build_events(path, bids, subject=19):
    """Each term as seen by each example pRF, from the real events of one run with HP at the lower right."""
    from scipy.stats import gamma
    from retsupp.utils.data import Subject
    sub = Subject(subject, bids)
    (ses, run), _ = next((k, v) for k, v in sub.get_hpd_locations().items() if v == 'lower_right')
    kw = dict(session=ses, run=run, resolution=50, grid_radius=5.0)
    full = np.asarray(sub.get_stimulus_with_distractors(distractor_shape='rectangle', distractor_long_side=1.5,
                                                        distractor_short_side=.375, **kw), dtype=float)
    bar = np.asarray(sub.get_bar_stimulus(**kw), dtype=float)
    gx, gy = sub.get_extended_grid_coordinates(resolution=50, session=ses, run=run, grid_radius=5.0)
    tgt, dyn = np.asarray(sub.get_target_indicator(session=ses, run=run)), np.asarray(sub.get_dynamic_indicator(session=ses, run=run))
    n = min(len(full), len(bar), len(tgt), len(dyn))
    g2 = lambda c, sd: np.exp(-((gx - c[0]) ** 2 + (gy - c[1]) ** 2) / (2 * sd ** 2))
    sus = np.array([GAINS['hp'] if i == HP_IX else GAINS['lp'] for i in range(4)])
    A_s = np.stack([g2(p, SIGMA['sus']) for p in POS])
    A_d = np.stack([g2(p, SIGMA['dyn']) for p in POS])
    field_s = (sus[:, None, None] * A_s).sum(0)
    t = np.arange(0, 30, 1.6)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    rows = []
    for name, (x, y, sd) in PRFS.items():
        w = g2((x, y), sd)
        w = w / w.sum()
        field_t = GAINS['t'] * tgt[:n] @ (A_d * w).sum((1, 2))
        field_d = GAINS['d'] * dyn[:n] @ (A_d * w).sum((1, 2))
        M = 1 + field_s[None] + GAINS['t'] * np.einsum('tl,lxy->txy', tgt[:n], A_d) + GAINS['d'] * np.einsum('tl,lxy->txy', dyn[:n], A_d)
        drive = (full[:n] * M * w).sum((1, 2))
        pred = np.convolve(drive - drive.mean(), h)[:n]
        rows.append(pd.DataFrame(dict(prf=name, t=np.arange(n) * 1.6, bar=(bar[:n] * w).sum((1, 2)),
                                      sustained=(field_s * w).sum(), target=field_t, distractor=field_d, predicted=pred)))
    pd.concat(rows).to_csv(path, sep='\t', index=False)


def main(events_table, bids, window=200, name='fig2_model'):
    if not Path(events_table).exists():
        build_events(events_table, bids)
    ev = pd.read_csv(events_table, sep='\t')
    ev = ev[ev.t <= window]
    n_r = len(PRFS)
    fig = plt.figure(figsize=(DOUBLE, 3.15))
    a = fig.add_gridspec(1 + n_r, 6, height_ratios=[1.75] + [.46] * n_r, width_ratios=[1] * 6, wspace=.34, hspace=.1,
                         left=.012, right=.995, top=.925, bottom=.045)
    ax = fig.add_subplot(a[0, 0])
    display(ax)
    ax.set_title('On screen', fontsize=7, color=FG, pad=2)
    sus = sum((GAINS['hp'] if i == HP_IX else GAINS['lp']) * gauss(*p, SIGMA['sus']) for i, p in enumerate(POS))
    tgt = GAINS['t'] * gauss(*POS[T_IX], SIGMA['dyn'])
    dis = GAINS['d'] * gauss(*POS[HP_IX], SIGMA['dyn'])
    maps = [('Model input', -stimulus(), 'gray', 1, FG), ('Sustained:\nless gain at HP', sus, 'RdBu_r', .3, HP),
            ('Target', tgt, 'RdBu_r', .4, TARGET), ('Distractor', dis, 'RdBu_r', .4, DISTRACTOR)]
    for j, (title, im, cmap, lim, col) in enumerate(maps, start=1):
        ax = fig.add_subplot(a[0, j])
        show(ax, im, cmap, lim, title, col)
        if j == 1:
            for nm, (x, y, sd) in PRFS.items():
                ax.contour(X, Y, gauss(x, y, sd), levels=[.5], colors=[PRF_COL[nm]], linewidths=1.)
    ax = fig.add_subplot(a[0, 5])
    ax.imshow(stimulus() * (1 + sus + tgt + dis), origin='lower', extent=(-LIM, LIM, -LIM, LIM), cmap='magma', vmin=0, vmax=1.35)
    ax.set_axis_off()
    ax.set_title('Effective drive', fontsize=7, color=FG, pad=2)
    for i, sign in ((1, None), (2, '× (1 +'), (3, '+'), (4, '+'), (5, ') =')):
        pos = fig.add_subplot(a[0, i]).get_position()
        fig.axes[-1].remove()
        x, y = pos.x0 - .024, pos.y0 + pos.height / 2
        if sign is None:
            fig.add_artist(mpl.patches.FancyArrowPatch((x - .012, y), (x + .012, y), arrowstyle='-|>', mutation_scale=7, lw=.9, color=FG, transform=fig.transFigure))
        else:
            fig.text(x, y, sign, fontsize=8, color=FG, ha='center', va='center')
    cols = [('bar', STIM, 'Bar in the pRF'), ('sustained', HP, 'Sustained gain'), ('target', TARGET, 'Target gain'),
            ('distractor', DISTRACTOR, 'Distractor gain'), ('predicted', FG, 'Predicted response')]
    lims = {c: (min(ev[c].min(), 0), max(ev[c].max(), 0)) for c, _, _ in cols}
    for i, nm in enumerate(PRFS):
        d = ev[ev.prf == nm]
        lab = fig.add_subplot(a[1 + i, 0])
        lab.set_axis_off()
        lab.plot([.02, .14], [.5, .5], color=PRF_COL[nm], lw=2.2, transform=lab.transAxes, solid_capstyle='butt')
        lab.text(.19, .5, nm, transform=lab.transAxes, va='center', fontsize=7, color=PRF_COL[nm])
        for j, (c, col, title) in enumerate(cols, start=1):
            ax = fig.add_subplot(a[1 + i, j])
            ax.set_axis_off()
            t, y = d.t.to_numpy(), d[c].to_numpy()
            if c == 'predicted':
                ax.plot(t, y, color=col, lw=.8)
            elif c == 'sustained':
                ax.plot(t, y, color=col, lw=1.4)
            else:
                ax.fill_between(t, 0, y, step='mid', color=col, lw=.25, edgecolor=col)
            ax.axhline(0, color='#D5DAE0', lw=.5, zorder=0)
            lo, hi = lims[c]
            ax.set_ylim(lo - .03 * (hi - lo), hi + .05 * (hi - lo))
            ax.set_xlim(0, window)
            if i == 0:
                ax.text(.5, 1.0, title, transform=ax.transAxes, fontsize=6.5, ha='center', va='bottom', color=col)
            if i == n_r - 1 and j == 5:
                ax.plot([window - 50, window], [lo - .0 * hi] * 2, color=FG, lw=1.1, clip_on=False, solid_capstyle='butt')
                ax.text(window - 25, lo - .08 * (hi - lo), '50 s', ha='center', va='top', fontsize=6, color=FG)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events-table', default='notes/data/fig2_example_events_2026-11.tsv')
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    a = ap.parse_args()
    main(a.events_table, a.bids_folder)
