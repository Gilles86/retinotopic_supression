"""Figure 2: the gain-field model, its effective pRF shifts, and the ladder.

Panels a and b use illustrative values. Panel a' uses the group medians of ``--gain-table``, so the
final version takes one command once the final fits are collected into a table of the same form.

Usage::

    python -m retsupp.visualize.paper.fig2_model --gain-table notes/data/gain_field_parameters.tsv
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, MUTED, STIM, TARGET, letter, offset_spines,
                                                 provisional, save)

LEGEND = """
**Figure 2. Gain-field model.** a, Forward model: a fixed pRF reads the stimulus (bar plus items) multiplied by a gain field, the sum of sustained fields at the four ring locations and transient fields at the target and singleton distractor. Maps use illustrative values; time courses show when each term applies, convolved with the HRF. a′, Group-median sustained field of m7_full for an early and a late ROI, with the shift of the pRF centre of mass it implies in the HP condition (vectors enlarged). Computed from the model, n = 30, DoG fits; approximation by centre of mass of pRF × field, without refitting. b, The nested ladder, each step labelled with its contrast.
"""
LIM = 5.2
RING = 4.
POS = RING / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
HP_IX = 0
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


def hrf(t):
    from scipy.stats import gamma
    return gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6


def timecourse(ax, onsets, color, sustained=False):
    t = np.arange(0, 60, .1)
    if sustained:
        y = np.ones_like(t)
    else:
        box = np.zeros_like(t)
        for o in onsets:
            box[(t >= o) & (t < o + 1.5)] = 1
        y = np.convolve(box, hrf(np.arange(0, 30, .1)))[:len(t)]
        y = y / y.max()
    ax.plot(t, y, color=color, lw=1)
    if not sustained:
        ax.plot(onsets, np.full(len(onsets), -.12), '|', color=color, ms=3, mew=.7)
    ax.set_ylim(-.25, 1.15)
    ax.set_xlim(0, 60)
    ax.set_axis_off()


def effective_shift(ax, pars, sd, scale=25):
    """Centre of mass of pRF x sustained plus mean transient field, HP at the upper right."""
    field = 1 + sum((pars.g_HP if i == HP_IX else pars.g_LP) * gauss(*p, pars.sigma_AF) for i, p in enumerate(POS))
    ax.imshow(field - 1, origin='lower', extent=(-LIM, LIM, -LIM, LIM), cmap='RdBu_r',
              vmin=-.15, vmax=.15, interpolation='bilinear')
    pts = [(r * np.cos(a), r * np.sin(a)) for r in (.8, 1.6, 2.4) for a in np.deg2rad(np.arange(0, 360, 30))]
    for cx, cy in pts:
        w = gauss(cx, cy, sd) * field
        dx, dy = (w * X).sum() / w.sum() - cx, (w * Y).sum() / w.sum() - cy
        ax.annotate('', xy=(cx + scale * dx, cy + scale * dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='-|>', color=FG, lw=.5, mutation_scale=3.5, shrinkA=0, shrinkB=0))
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.4))
    ax.add_patch(Circle(POS[HP_IX], .9, fill=False, ec=HP, lw=.7))
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect('equal')
    ax.set_axis_off()


def ladder(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    nodes = {'m00_bar': (.6, 1.5), 'm0_prf': (2.2, 1.5), 'm1_T': (3.9, 2.45), 'm2_D': (3.9, 1.5), 'm3_S': (3.9, .55),
             'm3b_TD': (5.3, 2.45), 'm4_TDS': (6.7, 1.5), 'm5_TDS_dS': (8.1, 2.3), 'm6_TDS_dD': (8.1, .7), 'm7_full': (9.45, 1.5)}
    labels = {'m00_bar': 'm00\nBar pRF', 'm0_prf': 'm0\n+ items', 'm1_T': 'm1\n+ target', 'm2_D': 'm2\n+ distractor', 'm3_S': 'm3\n+ sustained',
              'm3b_TD': 'm3b\nT + D', 'm4_TDS': 'm4\nT + D + S', 'm5_TDS_dS': 'm5\n+ HP ≠ LP\nsustained', 'm6_TDS_dD': 'm6\n+ HP ≠ LP\ndistractor', 'm7_full': 'm7\ndS + dD'}
    edges = [('m00_bar', 'm0_prf', 'Items'), ('m0_prf', 'm1_T', 'T'), ('m0_prf', 'm2_D', 'D'), ('m0_prf', 'm3_S', 'S'), ('m1_T', 'm3b_TD', ''),
             ('m3b_TD', 'm4_TDS', ''), ('m4_TDS', 'm5_TDS_dS', 'dS'), ('m4_TDS', 'm6_TDS_dD', 'dD'), ('m5_TDS_dS', 'm7_full', '')]
    tested = {'dS': HP, 'dD': DISTRACTOR, 'T': TARGET}
    for a, b, lab in edges:
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ax.annotate('', xy=(x1 - .5, y1), xytext=(x0 + .5, y0), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.8, mutation_scale=6, shrinkA=0, shrinkB=0))
        if lab:
            ax.text((x0 + x1) / 2 - .02, (y0 + y1) / 2 + .13, lab, fontsize=6.5, fontweight='bold', ha='center', color=tested.get(lab, FG))
    for k, (x, y) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - .5, y - .36), 1., .72, boxstyle='round,pad=0,rounding_size=.08', facecolor='#F1F3F5', edgecolor=MUTED, lw=.5))
        ax.text(x, y, labels[k], ha='center', va='center', fontsize=5.8, color=FG, linespacing=1.05)


def main(gain_table, family='dog', early='V2', late='V3AB', name='fig2_model'):
    fig = plt.figure(figsize=(DOUBLE, 3.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.45], hspace=.06, left=.015, right=.995, top=.93, bottom=.01)
    top = outer[0].subgridspec(1, 2, width_ratios=[2.15, 1], wspace=.08)
    a = top[0].subgridspec(2, 5, height_ratios=[1.35, .5], wspace=.32, hspace=-.1)
    ex = dict(sus=1.3, trans=1.6)
    t_ix, d_ix = 1, 3
    maps = [('Stimulus', stimulus(), 'Greys', 1, STIM), ('Sustained\nHP weaker', -.12 * gauss(*POS[HP_IX], ex['sus']) - .06 * sum(gauss(*p, ex['sus']) for p in POS[1:]), 'RdBu_r', .2, HP),
            ('Target', .3 * gauss(*POS[t_ix], ex['trans']), 'RdBu_r', .3, TARGET), ('Distractor', .15 * gauss(*POS[d_ix], ex['trans']), 'RdBu_r', .3, DISTRACTOR)]
    onsets = np.array([5, 14, 21, 33, 40, 52])
    prf = gauss(1.8, 1.0, .9)
    for j, (title, im, cmap, lim, col) in enumerate(maps):
        ax = fig.add_subplot(a[0, j])
        show(ax, im if cmap != 'Greys' else -im, 'RdBu_r' if cmap != 'Greys' else 'gray', lim, title, col)
        if j == 0:
            letter(ax, 'a', x=-.02, y=1.12)
            ax.contour(X, Y, prf, levels=[.5], colors=['#2A9D8F'], linewidths=.8)
        ax2 = fig.add_subplot(a[1, j])
        timecourse(ax2, onsets, col, sustained=j == 1) if j else ax2.set_axis_off()
        if j == 0:
            ax2.text(.5, .5, 'Fixed pRF\n(teal contour) reads\nstimulus × gain field', transform=ax2.transAxes, fontsize=6, ha='center', va='center', color=FG)
    ax = fig.add_subplot(a[0, 4])
    field = 1 + maps[1][1] + maps[2][1] + maps[3][1]
    ax.imshow(stimulus() * field, origin='lower', extent=(-LIM, LIM, -LIM, LIM), cmap='magma', vmin=0, vmax=1.35)
    ax.contour(X, Y, prf, levels=[.5], colors=['#2A9D8F'], linewidths=.8)
    ax.set_axis_off()
    ax.set_title('Effective drive', fontsize=7, color=FG, pad=2)
    fig.add_subplot(a[1, 4]).set_axis_off()
    for i, sign in ((1, '×'), (2, '+'), (3, '+'), (4, '=')):
        pos = fig.add_subplot(a[0, i]).get_position()
        fig.axes[-1].remove()
        fig.text(pos.x0 - .012, pos.y0 + pos.height / 2, sign, fontsize=11, color=FG, ha='center', va='center')

    t = pd.read_csv(gain_table, sep='\t')
    t = t[(t.family == family) & (t.model == 'm7_full')]
    b = top[1].subgridspec(1, 2, wspace=.05)
    sizes = {early: .65, late: 1.2}
    for j, roi in enumerate((early, late)):
        ax = fig.add_subplot(b[j])
        effective_shift(ax, t[t.roi == roi][['g_HP', 'g_LP', 'sigma_AF']].median(), sizes[roi])
        ax.set_title(roi, fontsize=7, color=FG, pad=2)
        if j == 0:
            letter(ax, 'a′', x=.02, y=1.08)
        else:
            provisional(ax, y=1.2)
    ax = fig.add_subplot(outer[1])
    ladder(ax)
    letter(ax, 'b', x=.02, y=.92)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--family', default='dog')
    a = ap.parse_args()
    main(a.gain_table, a.family)
