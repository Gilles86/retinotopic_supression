"""Shared style of the 2026-11 paper figures: sizes, ROI order and colours, stamps, saving."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path('notes/figures/paper_2026-11')
DOUBLE, SINGLE = 180 / 25.4, 88 / 25.4
FG, MUTED, HP = '#1F2933', '#9AA5B1', '#D62828'
TARGET, DISTRACTOR, STIM = '#F77F00', '#0077B6', '#3D5A80'
ROI_ORDER = ['V1', 'V2', 'V3', 'hV4', 'VO', 'LO', 'TO', 'V3AB', 'IPS']
ROI_COLORS = {'V1': '#3D5A80', 'V2': '#2A9D8F', 'V3': '#8AB17D', 'hV4': '#E9C46A', 'VO': '#F4A261',
              'LO': '#E76F51', 'TO': '#B5179E', 'V3AB': '#7209B7', 'IPS': '#0077B6'}
PROVISIONAL = 'Provisional: current inputs'

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


def save(fig, name, legend):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', dpi=300)
    fig.savefig(OUT / f'{name}.png', dpi=200)
    (OUT / f'{name}.md').write_text(legend.strip() + '\n')
    print('Saved', OUT / f'{name}.pdf')
