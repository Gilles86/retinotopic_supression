"""Shared style of the 2026-11 paper figures: sizes, ROI order and colours, stamps, saving."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path('notes/figures/paper_2026-11')
DOUBLE, SINGLE = 180 / 25.4, 88 / 25.4
FG, MUTED, HP = '#1F2933', '#9AA5B1', '#D62828'
TARGET, DISTRACTOR, STIM = '#F77F00', '#0077B6', '#3D5A80'
ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
ROI_COLORS = {'V1': '#3D5A80', 'V2': '#2A9D8F', 'V3': '#8AB17D', 'hV4': '#E9C46A', 'VO': '#F4A261',
              'LO': '#E76F51', 'TO': '#B5179E', 'V3AB': '#7209B7', 'IPS': '#0077B6'}
PROVISIONAL = 'Provisional: current inputs'
# Pathways, in ROI_ORDER; panels over ROIs leave a gap between groups and name them under the axis.
ROI_GROUPS = [('Early', ('V1', 'V2', 'V3')), ('Dorsal', ('V3AB', 'IPS')), ('Occipitotemporal', ('hV4', 'VO', 'LO', 'TO'))]
GROUP_GAP = 1.0


def _roi_x():
    x, out = 0., {}
    for _, rois in ROI_GROUPS:
        for r in rois:
            out[r] = x
            x += 1
        x += GROUP_GAP
    return out


ROI_X = _roi_x()
ICON_PAD = 2.2   # x room (data units) right of the last ROI for a stimulus icon


def title_icon(ax, title, marks, color=FG, y=1.1, h=.3, fontsize=7, sub=None):
    """Panel title above the axes with the stimulus icon at its left: neither can touch data."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    w = h * bb.height / bb.width
    stimulus_icon(ax, (0, y, w, h), marks)
    ax.text(w + .02, y + h / 2 + (.06 if sub else 0), title, transform=ax.transAxes, fontsize=fontsize, color=color,
            va='center', ha='left')
    if sub:
        ax.text(w + .02, y + h / 2 - .07, sub, transform=ax.transAxes, fontsize=6, color='.35', va='center', ha='left')


def roi_icon(ax, marks, frac=(.05, .95), pad=ICON_PAD):
    """Stimulus icon in the empty room right of IPS (needs roi_axis(..., right_pad=ICON_PAD))."""
    import matplotlib.transforms as mtrans
    x0 = max(ROI_X.values()) + .75
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    return stimulus_icon(ax, (x0, frac[0], pad - .8, frac[1] - frac[0]), marks, transform=tr)

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8, 'lines.linewidth': 1.2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.pad_inches': .02,
})


def letter(ax, s, x=-.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right', color='k')


def provisional(ax, x=1., y=1.0):
    """Grey stamp on every panel that depends on gain fits."""
    ax.text(x, y, PROVISIONAL, transform=ax.transAxes, fontsize=5.5, color=MUTED, ha='right', va='bottom')


def pending(ax, what):
    """Grey placeholder of the final size."""
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='#ECEFF2', lw=0))
    ax.text(.5, .5, f'Pending: {what}', transform=ax.transAxes, ha='center', va='center', fontsize=6.5, color='#6B7785', wrap=True)


def offset_spines(ax, by=3):
    ax.spines['left'].set_position(('outward', by))
    ax.spines['bottom'].set_position(('outward', by))


def roi_axis(ax, labels=None, rotate=True, fontsize=6.5, group_fontsize=6, pad_pt=3, right_pad=0.):
    """ROI ticks at ROI_X with the pathway names under the tick labels.

    right_pad: extra x room after the last ROI (data units), e.g. for a stimulus icon that must not touch data.

    Call after the panel's layout is fixed (fixed gridspec); the names are placed below the drawn tick labels."""
    import matplotlib.transforms as mtrans
    xs = [ROI_X[r] for r in ROI_ORDER]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels or ROI_ORDER, fontsize=fontsize, rotation=90 if rotate else 0)
    ax.set_xlim(xs[0] - .6, xs[-1] + .6 + right_pad)
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    bottom = min(inv.transform((0, t.get_window_extent().y0))[1] for t in ax.get_xticklabels() if t.get_text())
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    off = mtrans.offset_copy(tr, fig=fig, y=-pad_pt, units='points')
    for name, rois in ROI_GROUPS:
        x0, x1 = ROI_X[rois[0]] - .35, ROI_X[rois[-1]] + .35
        ax.plot([x0, x1], [bottom] * 2, transform=off, color=MUTED, lw=.7, clip_on=False, solid_capstyle='butt')
        t = ax.text((x0 + x1) / 2, bottom, name, transform=mtrans.offset_copy(tr, fig=fig, y=-pad_pt - 1.5, units='points'),
                    ha='center', va='top', fontsize=group_fontsize, color='.4', clip_on=False)



def roi_positions(rois):
    import numpy as np
    return np.array([ROI_X[r] for r in rois])


def save(fig, name, legend):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', dpi=300)
    fig.savefig(OUT / f'{name}.png', dpi=200)
    (OUT / f'{name}.md').write_text(legend.strip() + '\n')
    print('Saved', OUT / f'{name}.pdf')


def stimulus_icon(ax, bounds=(.78, .7, .22, .3), marks=('target',), hp_ix=3, t_ix=1, transform=None):
    """Small outline of the display in a panel corner, marking what the panel is about.

    marks: any of 'target' (orange diamond), 'distractor' (blue dot at HP), 'hp' (red dashed ring at HP),
    'lp' (grey rings at the three LP locations), 'fields' (outlines only)."""
    import numpy as np
    from matplotlib.patches import Circle
    ins = ax.inset_axes(list(bounds), transform=transform or ax.transAxes)
    ring = 4. / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    ins.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.5))
    for i, p in enumerate(ring):
        ins.add_patch(Circle(p, .55, facecolor='w', ec=MUTED, lw=.5))
    if 'lp' in marks:
        for i, p in enumerate(ring):
            if i != hp_ix:
                ins.add_patch(Circle(p, 1.15, fill=False, ec='.5', lw=.8, ls=(0, (2, 1.2))))
    if 'hp' in marks:
        ins.add_patch(Circle(ring[hp_ix], 1.15, fill=False, ec=HP, lw=1., ls=(0, (2, 1.2))))
    if 'target' in marks:
        ins.scatter(*ring[t_ix], marker='D', s=12, color=TARGET, lw=0, zorder=4)
    if 'distractor' in marks:
        ins.scatter(*ring[hp_ix], marker='o', s=16, color=DISTRACTOR, lw=0, zorder=4)
    ins.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    ins.set_xlim(-5.2, 5.2)
    ins.set_ylim(-5.2, 5.2)
    ins.set_aspect('equal')
    ins.set_axis_off()
    ins.set_clip_on(False)
    return ins
