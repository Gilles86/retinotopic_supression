"""Fig. 5b prototype: the fitted gain field over the visual field.

Group medians (n = 28, sub-01/02 excluded) of the joint gain-field fit's parameters per area, put through the model's
own field equations (``braincoder`` ``DoGAttentionFieldPRF2D._attention_modulation`` and
``DoGDynamicAttentionFieldPRF2D_v3._attention_modulation_dynamic_v3``, ``local_models._attention_modulation_target``;
signed mode, so the sign is +1):

    sustained(g)  = max(0, 1 + Σ_ℓ w_ℓ · exp(−‖g − r_ℓ‖² / 2σ_AF²)),  w = g_HP at HP, g_LP at the three LP locations
    display(g)    = sustained(g) + g_dyn · exp(−‖g − r_D‖² / 2σ_dyn²) + g_T · exp(−‖g − r_T‖² / 2σ_dyn²)

(σ_T_dyn is tied to σ_dyn in the fit; the transient terms are added to the sustained gain map, as in the forward model,
where the drive is paradigm · pRF · [sustained + transient].) The display row has the distractor at HP (gain g_HP_dyn)
and the target at the LP location opposite HP (gain g_T_dyn). HP is rotated to the top.

    python -m retsupp.visualize.paper.fig5b_gain_field_map
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, TARGET, save

AREAS = ['V1', 'V3', 'hV4', 'TO']
PARS = ['g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'sigma_AF', 'sigma_dyn']
ECC, APERTURE, EXTENT = 4.0, 3.17, 6.0
# HP rotated to the top; the other three ring locations at left, bottom (opposite) and right.
RING = {'hp': (0., ECC), 'left': (-ECC, 0.), 'opp': (0., -ECC), 'right': (ECC, 0.)}


def medians(table, exclude=(1, 2)):
    d = pd.read_csv(table, sep='\t')
    d = d[(d.family == 'dog') & (d.model == 'm7_full') & ~d.subject.isin(exclude)]
    n = d.groupby('roi').subject.nunique()
    return d.groupby('roi')[PARS].median(), n, d


def gauss(xx, yy, c, s):
    return np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * s ** 2))


def fields(p, xx, yy):
    sus = 1 + sum((p.g_HP if k == 'hp' else p.g_LP) * gauss(xx, yy, c, p.sigma_AF) for k, c in RING.items())
    sus = np.maximum(sus, 0.)
    disp = sus + p.g_HP_dyn * gauss(xx, yy, RING['hp'], p.sigma_dyn) + p.g_T_dyn * gauss(xx, yy, RING['opp'], p.sigma_dyn)
    return sus, disp


def decorate(ax, display):
    ax.add_patch(Circle((0, 0), ECC, fill=False, ec=FG, lw=.5, ls=(0, (1.5, 1.5)), alpha=.6))
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec=FG, lw=.6))
    for k, c in RING.items():
        ax.add_patch(Circle(c, .6, fill=False, ec=FG, lw=.5, alpha=.7))
    ax.add_patch(Circle(RING['hp'], 1.05, fill=False, ec=HP, lw=1., ls=(0, (2, 1.2))))
    if display:
        ax.plot(*RING['hp'], 'o', ms=3.4, color=DISTRACTOR, mec='w', mew=.4)
        ax.plot(*RING['opp'], 'D', ms=3.2, color=TARGET, mec='w', mew=.4)
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.6)
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


LEGEND = """
**Figure 5b (prototype). The fitted gain field over the visual field.** The joint gain-field fit's parametric field, drawn from the group medians of its parameters (n = 28 participants; the median of each parameter taken separately over participants, per area), not data. The display is rotated so that the high-probability (HP) distractor location is at the top (red dashed ring); the other three possible target and distractor locations are at the left, bottom (opposite HP) and right; dotted circle, the 4° ring of item locations; solid circle, the 3.17° aperture of the mapping bar; +, fixation. Colour: the multiplicative gain on the stimulus drive at each point of the visual field (1, white, no change; below 1, purple, suppression; above 1, orange, enhancement), one scale for all panels. Top row, the sustained field between search displays: a Gaussian gain field (width σ = the fitted sustained width) at each of the four locations, with the gain at HP and the common gain at the other three fitted separately. Bottom row, during a search display with the distractor at HP (blue dot) and the target at the location opposite HP (orange diamond): the sustained field plus the transient fields at the distractor and at the target (one shared transient width). {pars} The transient width is often at a bound of the fit (0.5° or 8°) in single participants, so its median is approximate. Field equations and code: `retsupp/visualize/paper/fig5b_gain_field_map.py` (the same functions as the model's forward pass).
"""


def main(table, name):
    med, n, _ = medians(table)
    g = np.linspace(-EXTENT, EXTENT, 361)
    xx, yy = np.meshgrid(g, g)
    maps = {a: fields(med.loc[a], xx, yy) for a in AREAS}
    dev = max(np.abs(m - 1).max() for pair in maps.values() for m in pair)
    lim = np.ceil(dev * 10) / 10
    norm = TwoSlopeNorm(vcenter=1., vmin=1 - lim, vmax=1 + lim)
    cmap = mpl.colormaps['PuOr_r']

    W = DOUBLE
    cw, gap, left, top_pad = 1.32, .1, .62, .3
    H = top_pad + 2 * cw + gap + .08
    fig = plt.figure(figsize=(W, H))
    rows = [('Sustained field\n(between displays)', 0), ('During a display\n(distractor at HP)', 1)]
    for r, (rlab, ri) in enumerate(rows):
        y0 = H - top_pad - (r + 1) * cw - r * gap
        for c, a in enumerate(AREAS):
            ax = fig.add_axes([(left + c * (cw + gap)) / W, y0 / H, cw / W, cw / H])
            ax.imshow(maps[a][ri], extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap=cmap, norm=norm,
                      interpolation='bilinear')
            decorate(ax, ri == 1)
            if r == 0:
                ax.set_title(f'{a} (n = {n[a]})', fontsize=7.5, pad=3)
        fig.text(.3 / W, (y0 + cw / 2) / H, rlab, rotation=90, ha='center', va='center', fontsize=7)
    cax = fig.add_axes([(left + 4 * (cw + gap) + .05) / W, (H - top_pad - 2 * cw - gap + .25) / H, .09 / W,
                        (2 * cw + gap - .5) / H])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('Gain on the stimulus drive', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(.5)

    rows_txt = []
    for a in AREAS:
        p = med.loc[a]
        sus, disp = maps[a]
        ix = np.argmin(np.hypot(xx - RING['hp'][0], yy - RING['hp'][1]))
        io = np.argmin(np.hypot(xx - RING['opp'][0], yy - RING['opp'][1]))
        rows_txt.append(
            f'{a}: gain at HP {p.g_HP:+.3f}, at the other three {p.g_LP:+.3f}, sustained width {p.sigma_AF:.2f}°; '
            f'transient gain at the distractor at HP {p.g_HP_dyn:+.3f}, at the target {p.g_T_dyn:+.3f}, width '
            f'{p.sigma_dyn:.2f}° (sustained field at the HP item {sus.ravel()[ix]:.2f}, opposite {sus.ravel()[io]:.2f}; '
            f'during the display {disp.ravel()[ix]:.2f} and {disp.ravel()[io]:.2f})')
    pars = ('Group medians: ' + '; '.join(rows_txt) + '.').replace('-', '−')
    save(fig, name, LEGEND.format(pars=pars))
    print(pars)
    print('colour limits', 1 - lim, 1 + lim)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', default='notes/data/gain_field_parameters_final2_2026-11.tsv')
    ap.add_argument('--name', default='fig5b_gain_field_map_prototype_n28')
    a = ap.parse_args()
    main(a.table, a.name)
