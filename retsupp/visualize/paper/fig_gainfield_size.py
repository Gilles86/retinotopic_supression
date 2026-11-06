"""How large the fitted gain fields are: each field at its group-median size on the display, and the gain
around the ring. Provisional until the final fits.

Usage::

    python -m retsupp.visualize.paper.fig_gainfield_size --gain-table notes/data/gain_field_parameters.tsv
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from retsupp.visualize.paper.fig45_common import joint
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, TARGET, letter, provisional, save

FWHM = 2 * np.sqrt(2 * np.log(2))
RING, APERTURE = 4., 3.17
POS = RING / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
HP_IX, T_IX = 3, 1
ROIS = ['V1', 'V3', 'hV4', 'IPS']
LEGEND = """
**Figure S. Size of the fitted gain fields.** Top, each field drawn at its group-median width as a circle of one full width at half maximum (FWHM = 2.355 σ), on the display: the sustained fields at the four ring locations (HP: red, LP: grey), and the transient fields of the target (orange) and the distractor (blue), at one example target and distractor location. Dotted: the 4° ring and the 3.17° bar aperture. Bottom, the gain along the ring (angle, HP at −45°): the sustained field summed over the four locations, and the transient field of a target at 135°, scaled to the group-median gains of m7_full (DoG pRFs, n = 30). The share of fits with the transient width at its 8° bound is given per ROI. Provisional: current inputs.
"""


def gain_on_ring(angle, centres, sigma, gains):
    xy = RING * np.c_[np.cos(angle), np.sin(angle)]
    return sum(g * np.exp(-((xy - c) ** 2).sum(1) / (2 * sigma ** 2)) for c, g in zip(centres, gains))


def main(gain_table, family='dog', name='figS_gainfield_size'):
    d = joint(gain_table, family)
    med = d.groupby('roi')[['sigma_AF', 'sigma_dyn', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn']].median()
    bound = d.groupby('roi').sigma_dyn.apply(lambda v: (v > 7.9).mean() * 100)
    fig, axes = plt.subplots(2, len(ROIS), figsize=(DOUBLE, 3.9), gridspec_kw=dict(height_ratios=[1.25, 1], hspace=.45, wspace=.3,
                                                                                    left=.08, right=.99, top=.9, bottom=.12))
    ang = np.linspace(-np.pi, np.pi, 721)
    for j, roi in enumerate(ROIS):
        m = med.loc[roi]
        ax = axes[0, j]
        for r, ls in ((RING, (0, (1, 2))), (APERTURE, (0, (1, 2)))):
            ax.add_patch(Circle((0, 0), r, fill=False, ec=MUTED, lw=.6, ls=ls))
        half = FWHM * m.sigma_AF / 2
        for i, p in enumerate(POS):
            col = HP if i == HP_IX else '.55'
            ax.add_patch(Circle(p, half, facecolor=col, alpha=.12, ec=col, lw=1.1 if i == HP_IX else .8))
        half_d = FWHM * m.sigma_dyn / 2
        ax.add_patch(Circle(POS[T_IX], half_d, fill=False, ec=TARGET, lw=1.2, ls=(0, (3, 1.5))))
        ax.add_patch(Circle(POS[HP_IX], half_d, fill=False, ec=DISTRACTOR, lw=1.2, ls=(0, (3, 1.5))))
        for p in POS:
            ax.plot(*p, 'o', ms=2.5, color=FG)
        ax.plot(0, 0, '+', color=FG, ms=4, mew=.7)
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(roi, fontsize=8, color=FG, pad=1)
        ax.text(0, -11.5, f'Sustained {FWHM * m.sigma_AF:.1f}°, transient {FWHM * m.sigma_dyn:.1f}°\n'
                          f'({bound.loc[roi]:.0f}% of transient widths at bound)', ha='center', va='top', fontsize=5.8, color=FG)
        ax2 = axes[1, j]
        gains = [m.g_HP if i == HP_IX else m.g_LP for i in range(4)]
        sus = gain_on_ring(ang, POS, m.sigma_AF, gains)
        trans = gain_on_ring(ang, [POS[T_IX]], m.sigma_dyn, [m.g_T_dyn])
        deg = np.degrees(ang)
        ax2.plot(deg, sus, color=HP, lw=1.3)
        ax2.plot(deg, trans, color=TARGET, lw=1.3, ls=(0, (3, 1.5)))
        for a in (-135, -45, 45, 135):
            ax2.axvline(a, color='#E4E7EB', lw=.6, zorder=0)
        ax2.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
        ax2.set_xticks([-135, -45, 45, 135])
        ax2.set_xticklabels(['LP', 'HP', 'LP', 'LP'], fontsize=6.5)
        ax2.set_xlim(-180, 180)
        ax2.set_ylim(-.15, .45)
        if j == 0:
            ax2.set_ylabel('Gain on the ring')
            ax2.text(.02, .97, 'Target, transient', transform=ax2.transAxes, fontsize=6.5, color=TARGET, va='top')
            ax2.text(.02, .84, 'Sustained, all four', transform=ax2.transAxes, fontsize=6.5, color=HP, va='top')
        else:
            ax2.set_yticklabels([])
        ax2.spines['left'].set_position(('outward', 3))
        ax2.spines['bottom'].set_position(('outward', 3))
    axes[1, 0].set_xlabel('Location on the ring', fontsize=7)
    letter(axes[0, 0], 'a', x=-.05, y=1.05)
    letter(axes[1, 0], 'b', x=-.3, y=1.05)
    provisional(axes[0, -1], y=1.08)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--family', default='dog')
    a = ap.parse_args()
    main(a.gain_table, a.family)
