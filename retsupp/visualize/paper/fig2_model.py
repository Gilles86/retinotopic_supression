"""Figure 2: the gain-field model and what it predicts at three pRF locations. No fitted values.

Usage::

    python -m retsupp.visualize.paper.fig2_model
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, STIM, TARGET, letter, save
from retsupp.visualize.paper.plot_fig1 import APERTURE, ITEM_A, ITEM_B, SCREEN, display, draw_bar, draw_item, item_positions

LEGEND = """
**Figure 2. Gain-field model.** Top, the screen is reduced to a model input (bar plus items), multiplied by a gain field: the sum of sustained fields at the four ring locations and transient fields at the target and singleton distractor. Bottom, a simulated stretch of a run with illustrative gains, HP at the lower right: one bar sweep from left to right and six trials, shown as icons above their onsets, with the singleton in green and the odd-orientation target among them. For three pRFs (coloured contour), the rows show the bar in the pRF, the sustained, target and distractor gain the pRF sees, and the predicted response after the HRF. Rows share their scale across columns.
"""
LIM = 5.2
RING = 4.
POS = RING / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])   # order of the event indicators
HP_IX, T_IX = 3, 1                                                         # as in Figure 1a
GAINS = dict(hp=-.22, lp=-.1, t=.35, d=.2)                                 # illustrative
SIGMA = dict(sus=1.3, dyn=1.6)
PRFS = {'pRF near HP': (2.1, -2.1, .9), 'pRF near LP': (-2.1, 2.1, .9), 'pRF near fixation': (.9, .3, .6)}
DURATION = 40.
BAR_W = 2 * APERTURE / 8
# Onset, target location, singleton location (order of POS; None: no singleton).
TRIALS = [(4, 1, 3), (10, 0, 3), (16, 3, 1), (22, 2, 3), (28, 1, None), (34, 0, 3)]
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


def bar_left(t):
    return -APERTURE + .168 * t - BAR_W


def simulate(dt=.4):
    """Every term as one pRF sees it, for one left-to-right sweep and the trials of TRIALS."""
    from scipy.stats import gamma
    t = np.arange(0, DURATION, dt)
    in_ap = np.hypot(X, Y) < APERTURE
    items = np.zeros_like(X, bool)
    for a in np.deg2rad(np.arange(8) * 45):
        cx, cy = RING * np.cos(a), RING * np.sin(a)
        items |= (np.abs(X - cx) < .75) & (np.abs(Y - cy) < .19)
    tgt, dis = np.zeros((len(t), 4)), np.zeros((len(t), 4))
    for onset, ti, di in TRIALS:
        on = (t >= onset) & (t < onset + 1.5)
        tgt[on, ti] = 1
        if di is not None:
            dis[on, di] = 1
    A_s = np.stack([gauss(*p, SIGMA['sus']) for p in POS])
    A_d = np.stack([gauss(*p, SIGMA['dyn']) for p in POS])
    gains = np.array([GAINS['hp'] if i == HP_IX else GAINS['lp'] for i in range(4)])
    field_s = (gains[:, None, None] * A_s).sum(0)
    h = gamma.pdf(np.arange(0, 30, dt), 6) - gamma.pdf(np.arange(0, 30, dt), 16) / 6
    out = {}
    for name, (x, y, sd) in PRFS.items():
        w = gauss(x, y, sd)
        w = w / w.sum()
        at = (A_d * w).sum((1, 2))
        bar, drive = np.zeros_like(t), np.zeros_like(t)
        for i, ti in enumerate(t):
            b = (X >= bar_left(ti)) & (X < bar_left(ti) + BAR_W) & in_ap
            st = b | (items if tgt[i].any() else False)
            field = 1 + field_s + GAINS['t'] * (tgt[i, :, None, None] * A_d).sum(0) + GAINS['d'] * (dis[i, :, None, None] * A_d).sum(0)
            bar[i] = (b * w).sum()
            drive[i] = (st * field * w).sum()
        pred = np.convolve(drive, h)[:len(t)]
        out[name] = dict(t=t, bar=bar, sustained=np.full_like(t, (field_s * w).sum()), target=GAINS['t'] * tgt @ at,
                         distractor=GAINS['d'] * dis @ at, predicted=pred)
    return out


def icon(ax, t, ti, di, contour):
    """The screen at one trial onset: bar, items, the target and singleton, this pRF."""
    ax.add_patch(plt.Rectangle((-5.4, -5.4), 10.8, 10.8, facecolor=SCREEN, lw=0, zorder=0))
    draw_bar(ax, bar_left(t))
    pos = item_positions()
    idx = {0: 1, 1: 3, 2: 5, 3: 7}
    for i, xy in enumerate(pos):
        draw_item(ax, xy, ITEM_B if di is not None and i == idx[di] else ITEM_A, i == idx[ti], False, 1.25)
    x, y, sd = contour[1]
    ax.add_patch(Circle((x, y), 1.18 * sd, fill=False, ec=contour[0], lw=.9, zorder=7))
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-5.4, 5.4)
    ax.set_aspect('equal')
    ax.set_axis_off()


def main(name='fig2_model'):
    sim = simulate()
    fig = plt.figure(figsize=(DOUBLE, 3.85))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.6, 2.2], hspace=.1, left=.012, right=.995, top=.935, bottom=.035)
    a = outer[0].subgridspec(1, 6, wspace=.34)
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
    rows = [('bar', STIM, 'Bar in the pRF'), ('sustained', HP, 'Sustained gain'), ('target', TARGET, 'Target gain'),
            ('distractor', DISTRACTOR, 'Distractor gain'), ('predicted', FG, 'Predicted response')]
    low = outer[1].subgridspec(1, 4, width_ratios=[.42, 1, 1, 1], wspace=.06)
    lims = {c: (min(0, min(sim[n][c].min() for n in sim)), max(0, max(sim[n][c].max() for n in sim))) for c, _, _ in rows}
    lab = low[0].subgridspec(len(rows) + 1, 1, height_ratios=[.62] + [.62] * len(rows), hspace=.1)
    for r, (c, col, title) in enumerate(rows, start=1):
        ax = fig.add_subplot(lab[r])
        ax.set_axis_off()
        ax.text(1., .5, title, transform=ax.transAxes, ha='right', va='center', fontsize=6.5, color=col)
    for k, (nm, p) in enumerate(PRFS.items()):
        col = low[k + 1].subgridspec(len(rows) + 1, 1, height_ratios=[.62] + [.62] * len(rows), hspace=.1)
        top = fig.add_subplot(col[0])
        top.set_xlim(0, DURATION)
        top.set_ylim(0, 1)
        top.set_axis_off()
        top.text(.5, 1.03, nm, transform=top.transAxes, ha='center', va='bottom', fontsize=7, color=PRF_COL[nm])
        bb = top.get_position()
        w_ax = .165
        h_ax = w_ax * bb.width * fig.get_figwidth() / (bb.height * fig.get_figheight())
        d = sim[nm]
        for onset, ti, di in TRIALS:
            inset = top.inset_axes([(onset + .75) / DURATION - w_ax / 2, .0, w_ax, h_ax])
            icon(inset, onset, ti, di, (PRF_COL[nm], p))
        for r, (c, color, _) in enumerate(rows, start=1):
            ax = fig.add_subplot(col[r])
            ax.set_axis_off()
            t, y = d['t'], d[c]
            for onset, _, _ in TRIALS:
                ax.axvspan(onset, onset + 1.5, color='#ECEFF2', lw=0, zorder=0)
            if c == 'predicted':
                ax.plot(t, y, color=color, lw=1.)
            elif c == 'sustained':
                ax.plot(t, y, color=color, lw=1.4)
                ax.axhline(0, color='#B8C0C8', lw=.5, zorder=0.5)
            else:
                ax.fill_between(t, 0, y, color=color, lw=0)
            lo, hi = lims[c]
            ax.set_ylim(lo - .05 * (hi - lo), hi + .05 * (hi - lo))
            ax.set_xlim(0, DURATION)
            if k == 0 and r == len(rows):
                ax.plot([0, 10], [lo - .22 * (hi - lo)] * 2, color=FG, lw=1.1, clip_on=False, solid_capstyle='butt')
                ax.text(5, lo - .28 * (hi - lo), '10 s', ha='center', va='top', fontsize=6, color=FG, clip_on=False)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    main()
