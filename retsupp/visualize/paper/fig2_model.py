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
from retsupp.visualize.paper.plot_fig1 import APERTURE, display

COMMON = ("A search display on screen is reduced to the model input: the bar and the eight items as patches of the "
          "same size and orientation (their colour, and so which item is the singleton, is not in the input). The weight of the "
          "items relative to the bar is fitted per ROI. Each voxel's pRF, fixed from the mapping fit, reads this input "
          "multiplied by a gain field: 1 plus a sustained field at the four ring locations, which differs between the HP "
          "location (g_HP) and the three LP locations (g_LP), plus transient fields at the target (g_T) and at the singleton, "
          "whose gain can also differ between HP and LP. Shown here is less sustained gain at HP than at LP, relative to LP. "
          "HP is at the lower right (red dashed ring). All gains are illustrative and exaggerated for clarity.")
LEGEND = f"""
**Figure 2. Gain-field model.** a, {COMMON} b, A schematic stretch of a run with illustrative trials: one bar sweep from left to right and four trials, with symbols above their onsets (target: diamond; singleton: dot; HP: red ring; pRF: teal circle). For three pRFs, the rows show the stimulus reaching the pRF (bar and items), the gain it sees, and the predicted BOLD response after the HRF, with and without gain. Rows share their scale across columns.
"""
LEGEND_HP = f"""
**Figure 2. Gain-field model.** a, {COMMON} b, A schematic stretch of a run for one pRF (teal) near a ring location, in a block with HP near it and in a block with HP at the opposite location. One bar sweep from left to right and four illustrative trials, the same in both blocks, so that only the sustained gain differs (in the experiment the singleton is also more frequent at HP). Symbols above the onsets show the screen (target: diamond; singleton: dot; HP: red ring). Rows show the stimulus reaching the pRF (bar and items), the gain it sees, and the predicted BOLD response after the HRF, with and without gain. The dotted line marks the peak response in the block with HP not near, in both blocks; with HP near, the response peaks lower (arrow). Rows share their scale across columns.
"""
LIM = 5.2
RING = 4.
POS = RING / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])   # order of the event indicators
HP_IX, T_IX = 3, 1                                                         # as in Figure 1a
GAINS = dict(hp=-.75, lp=0., t=1.6, d=1.1)                                 # illustrative, relative to LP
SIGMA = dict(sus=1.3, dyn=1.6)
PRFS = {'pRF near HP': (2.1, -2.1, .9), 'pRF near LP': (-2.1, 2.1, .9), 'pRF near fixation': (.9, .3, .6)}
DURATION = 38.
ITEM_SCALE = 8.
BAR_W = 2 * APERTURE / 8
# Onset, target location, singleton location (order of POS; None: no singleton).
TRIALS = [(5, 1, 3), (14, 0, 3), (23, 3, 1), (31, 2, 3)]
RING_PRF = (1.75, -1.75, .9)
PRF_TEAL = '#2A9D8F'
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
        vertical = np.isclose(a, np.deg2rad(135))                            # the target, as in Figure 1a
        w, h = (.19, .75) if vertical else (.75, .19)
        s[(np.abs(X - cx) < w) & (np.abs(Y - cy) < h)] = 1
    return s


def show(ax, im, cmap='RdBu_r', lim=None, title=None, color=FG, signed=True):
    lim = lim if lim is not None else np.abs(im).max()
    ax.imshow(im, origin='lower', extent=(-LIM, LIM, -LIM, LIM), cmap=cmap, vmin=-lim if signed else 0, vmax=lim, interpolation='bilinear')
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.4))
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7, color=color, pad=2)


BAR_START = 12.   # s of the sweep already done at t = 0, so the bar crosses the ring pRF mid-window


def bar_left(t):
    return -APERTURE + .168 * (t + BAR_START) - BAR_W


def simulate(dt=.4, trials=None, hp_ix=None, prfs=None):
    """Each term as one pRF sees it, for one left-to-right sweep and the given trials."""
    trials = TRIALS if trials is None else trials
    hp_ix = HP_IX if hp_ix is None else hp_ix
    prfs = PRFS if prfs is None else prfs
    from scipy.stats import gamma
    t = np.arange(0, DURATION, dt)
    in_ap = np.hypot(X, Y) < APERTURE
    items = np.zeros_like(X, bool)
    for a in np.deg2rad(np.arange(8) * 45):
        cx, cy = RING * np.cos(a), RING * np.sin(a)
        items |= (np.abs(X - cx) < .75) & (np.abs(Y - cy) < .19)
    tgt, dis = np.zeros((len(t), 4)), np.zeros((len(t), 4))
    for onset, ti, di in trials:
        on = (t >= onset) & (t < onset + 1.5)
        tgt[on, ti] = 1
        if di is not None:
            dis[on, di] = 1
    A_s = np.stack([gauss(*p, SIGMA['sus']) for p in POS])
    A_d = np.stack([gauss(*p, SIGMA['dyn']) for p in POS])
    gains = np.array([GAINS['hp'] if i == hp_ix else GAINS['lp'] for i in range(4)])
    field_s = (gains[:, None, None] * A_s).sum(0)
    h = gamma.pdf(np.arange(0, 30, dt), 6) - gamma.pdf(np.arange(0, 30, dt), 16) / 6
    out = {}
    for name, (x, y, sd) in prfs.items():
        w = gauss(x, y, sd)
        w = w / w.sum()
        at = (A_d * w).sum((1, 2))
        stim, stim_items, drive = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
        for i, ti in enumerate(t):
            st = ((X >= bar_left(ti)) & (X < bar_left(ti) + BAR_W) & in_ap).astype(float)
            if tgt[i].any():
                stim_items[i] = (ITEM_SCALE * items * w).sum()
                st = st + ITEM_SCALE * items
            field = 1 + field_s + GAINS['t'] * (tgt[i, :, None, None] * A_d).sum(0) + GAINS['d'] * (dis[i, :, None, None] * A_d).sum(0)
            stim[i] = (st * w).sum()
            drive[i] = (st * field * w).sum()
        base = (field_s * w).sum()
        out[name] = dict(t=t, stimulus=stim, items=stim_items, base=base, target=GAINS['t'] * tgt @ at, distractor=GAINS['d'] * dis @ at,
                         without=np.convolve(stim, h)[:len(t)], predicted=np.convolve(drive, h)[:len(t)])
    return out


def icon(ax, t, ti, di, contour, hp_ix=None):
    """Symbol of the screen at one onset: bar, the four locations, target (diamond), singleton (dot), this pRF."""
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec=MUTED, lw=.5))
    r = plt.Rectangle((bar_left(t), -APERTURE), BAR_W, 2 * APERTURE, facecolor=FG, lw=0, zorder=2)
    ax.add_patch(r)
    r.set_clip_path(Circle((0, 0), APERTURE, transform=ax.transData))
    for a in np.deg2rad(np.arange(8) * 45):
        cx, cy = RING * np.cos(a), RING * np.sin(a)
        if not any(np.hypot(cx - p[0], cy - p[1]) < .1 for p in POS):
            ax.add_patch(plt.Rectangle((cx - .45, cy - .12), .9, .24, facecolor='.25', lw=0, zorder=3))
    for i, p in enumerate(POS):
        ax.add_patch(Circle(p, .55, facecolor='w', ec=MUTED, lw=.5, zorder=3))
    if hp_ix is not None:
        ax.add_patch(Circle(POS[hp_ix], 1.05, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2)), zorder=6))
    ax.scatter(*POS[ti], marker='D', s=9, color=FG, lw=0, zorder=4)
    if di is not None:
        ax.scatter(*POS[di], marker='o', s=12, color=FG, lw=0, zorder=4)
    x, y, sd = contour
    ax.add_patch(Circle((x, y), 1.18 * sd, fill=False, ec=PRF_TEAL, lw=.9, zorder=5))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_axis_off()


def main(name='fig2_model'):
    sim = simulate()
    fig = plt.figure(figsize=(DOUBLE, 3.75))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.95], hspace=.1, left=.012, right=.995, top=.935, bottom=.035)
    top_row(fig, outer[0], [v for v in PRFS.values()])
    low = outer[1].subgridspec(1, 4, width_ratios=[.55, 1, 1, 1], wspace=.07)
    ratios = [1.25, .8, .8, 1.15]
    lab = low[0].subgridspec(4, 1, height_ratios=ratios, hspace=.12)
    row_labels(fig, lab)
    s_hi = max(d['stimulus'].max() for d in sim.values())
    g_hi = max((d['target'].max(), d['distractor'].max()) for d in sim.values())
    g_hi = max(max(d['target'].max(), d['distractor'].max()) for d in sim.values())
    g_lo = min(d['base'] for d in sim.values())
    p_hi = max(max(d['predicted'].max(), d['without'].max()) for d in sim.values())
    p_lo = min(min(d['predicted'].min(), d['without'].min()) for d in sim.values())
    for k, (nm, p) in enumerate(PRFS.items()):
        col = low[k + 1].subgridspec(4, 1, height_ratios=ratios, hspace=.12)
        d, t = sim[nm], sim[nm]['t']
        top = fig.add_subplot(col[0])
        top.set_xlim(0, DURATION)
        top.set_ylim(0, 1)
        top.set_axis_off()
        top.text(.5, 1.02, nm, transform=top.transAxes, ha='center', va='bottom', fontsize=7, color=FG)
        bb = top.get_position()
        w_ax = .22
        h_ax = w_ax * bb.width * fig.get_figwidth() / (bb.height * fig.get_figheight())
        for onset, ti, di in TRIALS:
            inset = top.inset_axes([(onset + .75) / DURATION - w_ax / 2, (1 - h_ax) / 2, w_ax, h_ax])
            icon(inset, onset, ti, di, p, HP_IX)
        rows = [fig.add_subplot(col[r]) for r in (1, 2, 3)]
        for onset in [o for o, _, _ in (tr if 'tr' in dir() else TRIALS)]:
            top.plot([onset + .75] * 2, [0, (1 - h_ax) / 2], color='#C9CED4', lw=.5, zorder=0)
        for ax in rows:
            ax.set_axis_off()
            ax.set_xlim(0, DURATION)
            for onset, _, _ in TRIALS:
                ax.axvspan(onset, onset + 1.5, color='#F1F3F5', lw=0, zorder=0)
        rows[0].fill_between(t, 0, d['stimulus'] - d['items'], color='.72', lw=0)
        rows[0].fill_between(t, d['stimulus'] - d['items'], d['stimulus'], color='.25', lw=0)
        rows[0].set_ylim(0, s_hi * 1.05)
        g = rows[1]
        g.axhline(0, color='#B8C0C8', lw=.5, ls=(0, (2, 2)))
        if k == 0:
            g.text(0, 0, 'No change', fontsize=5.5, color='.6', ha='left', va='bottom')
        g.plot(t, np.full_like(t, d['base']), color=HP, lw=1.2)
        g.fill_between(t, d['base'], d['base'] + d['target'], color=TARGET, lw=0)
        g.fill_between(t, d['base'], d['base'] + d['distractor'], color=DISTRACTOR, lw=0)
        g.set_ylim(g_lo * 1.2, g_hi * 1.05)
        r3 = rows[2]
        r3.plot(t, d['without'], color='.65', lw=1.1)
        r3.plot(t, d['predicted'], color=FG, lw=1.2)
        r3.set_ylim(p_lo - .05 * (p_hi - p_lo), p_hi * 1.05)
        if k == 0:
            r3.plot([0, 10], [p_lo - .05 * (p_hi - p_lo)] * 2, color=FG, lw=1.1, clip_on=False, solid_capstyle='butt')
            r3.text(5, p_lo - .12 * (p_hi - p_lo), '10 s', ha='center', va='top', fontsize=6, color=FG)
    save(fig, name, LEGEND)


def top_row(fig, gs, prfs=None):
    prfs = prfs or [RING_PRF]
    a = gs.subgridspec(1, 6, wspace=.34)
    ax = fig.add_subplot(a[0, 0])
    display(ax)
    ax.add_patch(Circle(POS[HP_IX], 1.12, fill=False, ec=HP, lw=1., ls=(0, (2.5, 1.5)), zorder=8))
    ax.set_title('On screen', fontsize=7, color=FG, pad=2)
    letter(ax, 'a', x=-.02, y=1.08)
    sus = sum((GAINS['hp'] if i == HP_IX else GAINS['lp']) * gauss(*p, SIGMA['sus']) for i, p in enumerate(POS))
    tgt = GAINS['t'] * gauss(*POS[T_IX], SIGMA['dyn'])
    dis = GAINS['d'] * gauss(*POS[HP_IX], SIGMA['dyn'])
    maps = [('Model input', -stimulus(), 'gray', 1, FG, True, 'Fixed pRF (teal)'),
            ('Sustained:\nless gain at HP', -sus, 'Reds', .9, HP, False, 'Whole block · g_HP vs g_LP'),
            ('Target', tgt, 'Oranges', 1.8, TARGET, False, 'During search · g_T'),
            ('Singleton', dis, 'Blues', 1.8, DISTRACTOR, False, 'During search · HP vs LP')]
    for j, (title, im, cmap, lim, col, signed, sub) in enumerate(maps, start=1):
        ax = fig.add_subplot(a[0, j])
        show(ax, im, cmap, lim, title, col, signed)
        ax.text(.5, -.03, sub, transform=ax.transAxes, ha='center', va='top', fontsize=6, color=FG if j == 1 else col)
        if j == 1:
            for x, y, sd in prfs:
                ax.contour(X, Y, gauss(x, y, sd), levels=[.5], colors=[PRF_TEAL], linewidths=.9)
        if j == 2:
            ax.add_patch(Circle(POS[HP_IX], 1.12, fill=False, ec=HP, lw=.9, ls=(0, (2.5, 1.5))))
    ax = fig.add_subplot(a[0, 5])
    ax.imshow(stimulus() * (1 + sus + tgt + dis), origin='lower', extent=(-LIM, LIM, -LIM, LIM), cmap='magma', vmin=0, vmax=2.6)
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


def row_labels(fig, lab, key_hp=False):
    """Row names in the left column, each with its own key, so the data area stays clean."""
    keys = {1: [('Items', '.25'), ('Bar', '.6')], 2: [('Target', TARGET), ('Distractor', DISTRACTOR), ('Sustained', HP)],
            3: [('With gain', FG), ('without', '.55')]}
    names = {1: 'Stimulus in pRF', 2: 'Gain', 3: 'Predicted BOLD'}
    for r in (1, 2, 3):
        ax = fig.add_subplot(lab[r])
        ax.set_axis_off()
        ax.text(1., .62, names[r], transform=ax.transAxes, ha='right', va='center', fontsize=7, color=FG)
        x = 1.
        for txt, c in reversed(keys[r]):
            t = ax.text(x, .12, txt, transform=ax.transAxes, ha='right', va='center', fontsize=5.6, color=c)
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transAxes.inverted())
            x = bb.x0 - .05
    key = fig.add_subplot(lab[0])
    key.set_axis_off()
    items = [('D', 'Target'), ('o', 'Singleton')] + ([('ring', 'HP location')] if key_hp else [])
    for k, (m, txt) in enumerate(items):
        y = .8 - k * .3
        if m == 'ring':
            key.add_patch(Circle((.42, y), .05, transform=key.transAxes, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2))))
        else:
            key.scatter([.42], [y], marker=m, s=10, color=FG, lw=0, transform=key.transAxes)
        key.text(.5, y, txt, transform=key.transAxes, fontsize=6, va='center', color=FG)
    letter(key, 'b', x=.1, y=1.0)


def main_hp(name='fig2_model_hp_blocks'):
    """One pRF in two blocks: HP at this pRF, or at the opposite location."""
    prf = {'pRF': RING_PRF}
    # The same trials in both blocks, so that only the sustained gain differs.
    trials = [(5, 1, 3), (14, 0, 2), (23, 2, 1), (31, 3, 0)]
    blocks = {'HP near this pRF': (3, trials), 'HP not near this pRF': (1, trials)}
    sim = {k: simulate(trials=tr, hp_ix=hp, prfs=prf)['pRF'] for k, (hp, tr) in blocks.items()}
    fig = plt.figure(figsize=(DOUBLE, 3.75))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.95], hspace=.1, left=.012, right=.995, top=.935, bottom=.035)
    top_row(fig, outer[0])
    low = outer[1].subgridspec(1, 3, width_ratios=[.55, 1, 1], wspace=.08)
    ratios = [1.25, .8, .8, 1.15]
    lab = low[0].subgridspec(4, 1, height_ratios=ratios, hspace=.12)
    row_labels(fig, lab, key_hp=True)
    s_hi = max(d['stimulus'].max() for d in sim.values())
    g_hi = max(max(d['target'].max(), d['distractor'].max()) for d in sim.values())
    g_lo = min(d['base'] for d in sim.values())
    p_hi = max(max(d['predicted'].max(), d['without'].max()) for d in sim.values())
    p_lo = min(min(d['predicted'].min(), d['without'].min()) for d in sim.values())
    anchor = sim['HP not near this pRF']['predicted'].max()
    for k, (nm, (hp, tr)) in enumerate(blocks.items()):
        col = low[k + 1].subgridspec(4, 1, height_ratios=ratios, hspace=.12)
        d, t = sim[nm], sim[nm]['t']
        top = fig.add_subplot(col[0])
        top.set_xlim(0, DURATION)
        top.set_ylim(0, 1)
        top.set_axis_off()
        top.text(.5, 1.02, nm, transform=top.transAxes, ha='center', va='bottom', fontsize=7, color=FG)
        bb = top.get_position()
        w_ax = .22
        h_ax = w_ax * bb.width * fig.get_figwidth() / (bb.height * fig.get_figheight())
        for onset, ti, di in tr:
            inset = top.inset_axes([(onset + .75) / DURATION - w_ax / 2, (1 - h_ax) / 2, w_ax, h_ax])
            icon(inset, onset, ti, di, RING_PRF, hp)
        rows = [fig.add_subplot(col[r]) for r in (1, 2, 3)]
        for onset in [o for o, _, _ in (tr if 'tr' in dir() else TRIALS)]:
            top.plot([onset + .75] * 2, [0, (1 - h_ax) / 2], color='#C9CED4', lw=.5, zorder=0)
        for ax in rows:
            ax.set_axis_off()
            ax.set_xlim(0, DURATION)
            for onset, _, _ in tr:
                ax.axvspan(onset, onset + 1.5, color='#F1F3F5', lw=0, zorder=0)
        rows[0].fill_between(t, 0, d['stimulus'] - d['items'], color='.72', lw=0)
        rows[0].fill_between(t, d['stimulus'] - d['items'], d['stimulus'], color='.25', lw=0)
        rows[0].set_ylim(0, s_hi * 1.05)
        g = rows[1]
        g.axhline(0, color='#B8C0C8', lw=.5, ls=(0, (2, 2)))
        if k == 0:
            g.text(0, 0, 'No change', fontsize=5.5, color='.6', ha='left', va='bottom')
        g.plot(t, np.full_like(t, d['base']), color=HP, lw=1.2)
        g.fill_between(t, d['base'], d['base'] + d['target'], color=TARGET, lw=0)
        g.fill_between(t, d['base'], d['base'] + d['distractor'], color=DISTRACTOR, lw=0)
        g.set_ylim(g_lo * 1.2, g_hi * 1.05)
        r3 = rows[2]
        r3.plot(t, d['without'], color='.65', lw=1.1)
        r3.plot(t, d['predicted'], color=HP if k == 0 else FG, lw=1.3)
        r3.set_ylim(p_lo - .05 * (p_hi - p_lo), p_hi * 1.05)
        # Anchor: the peak response without HP near, drawn in both blocks.
        r3.axhline(anchor, color=FG, lw=.6, ls=(0, (1.5, 1.5)), zorder=0)
        if k == 0:
            r3.plot([0, 10], [p_lo - .05 * (p_hi - p_lo)] * 2, color=FG, lw=1.1, clip_on=False, solid_capstyle='butt')
            r3.text(5, p_lo - .12 * (p_hi - p_lo), '10 s', ha='center', va='top', fontsize=6, color=FG)
            r3.annotate('', xy=(t[np.argmax(d['predicted'])], d['predicted'].max()), xytext=(t[np.argmax(d['predicted'])], anchor),
                        arrowprops=dict(arrowstyle='-|>', color=HP, lw=.9, mutation_scale=6, shrinkA=0, shrinkB=0))
        else:
            r3.text(DURATION, anchor, 'Peak, HP not near', fontsize=5.6, color=FG, ha='right', va='bottom')
    save(fig, name, LEGEND_HP)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version', choices=['locations', 'blocks'], default='blocks')
    a = ap.parse_args()
    main_hp() if a.version == 'blocks' else main()
