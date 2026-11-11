"""Fig. 5b v2: the fitted gain field over the visual field, as % change in gain, with its profile around the item ring.

Per participant (n = 28, sub-01/02 excluded), the joint gain-field fit's parameters per area are put through the model's
own field equations (the same functions as ``fig5b_gain_field_map``; signed mode):

    sustained(g)  = max(0, 1 + Σ_ℓ w_ℓ · exp(−‖g − r_ℓ‖² / 2σ_AF²)),  w = g_HP at HP, g_LP at the three LP locations
    display(g)    = sustained(g) + g_dyn · exp(−‖g − r_D‖² / 2σ_dyn²) + g_T · exp(−‖g − r_T‖² / 2σ_dyn²)

and the group field is the median over participants of these maps, pixel by pixel (not the map of the median
parameters, which is dominated by widths at a bound in single participants). Shown as % change in gain, with colour
limits per area. Below each map: the median field around the 4° item ring, by angle from HP.

    python -m retsupp.visualize.paper.fig5b_gain_field_map_v2              # sustained and display rows
    python -m retsupp.visualize.paper.fig5b_gain_field_map_v2 --sustained-only --name fig5b_gain_field_map_v2b_n28
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

from retsupp.visualize.paper.fig5b_gain_field_map import APERTURE, ECC, EXTENT, PARS, RING, fields

# Areas where the joint fit's HP − LP sustained gain is reliable at n = 28 (Wilcoxon, BH over nine areas).
AREAS = ['V3', 'hV4', 'TO']
ZERO = '#2A9D8F'   # the no-change contour (teal; magma has no neutral midpoint)
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, TARGET, offset_spines, save

NICE = np.array([2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100])


def per_subject(table, exclude=(1, 2)):
    d = pd.read_csv(table, sep='\t')
    return d[(d.family == 'dog') & (d.model == 'm7_full') & ~d.subject.isin(exclude)]


def median_maps(d, area, xx, yy):
    """Median over participants of each participant's field (pixel by pixel)."""
    sus, disp = zip(*(fields(r, xx, yy) for _, r in d[d.roi == area][PARS].iterrows()))
    return 100 * (np.median(sus, 0) - 1), 100 * (np.median(disp, 0) - 1), len(sus)


def median_param_maps(d, area, xx, yy):
    """The field of the median parameters (each parameter's median over participants)."""
    x = d[d.roi == area]
    sus, disp = fields(x[PARS].median(), xx, yy)
    return 100 * (sus - 1), 100 * (disp - 1), len(x)


def nice(x):
    return float(NICE[np.searchsorted(NICE, x)]) if x <= NICE[-1] else float(np.ceil(x / 50) * 50)


def step_for(lim):
    for s in (1, 2, 2.5, 5, 10, 20, 25):
        if lim / s <= 5:
            return s
    return lim / 4


def decorate(ax, display, labels):
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec=MUTED, lw=.5, ls=(0, (2, 2))))
    ang = np.deg2rad(np.arange(0, 360, 45))
    for k, a in enumerate(ang):
        main4 = k % 2 == 0
        ax.add_patch(Circle((ECC * np.sin(a), ECC * np.cos(a)), .75, fill=False, ec=FG, lw=.55 if main4 else .35,
                            alpha=.75 if main4 else .35))
    ax.add_patch(Circle(RING['hp'], 1.12, fill=False, ec=HP, lw=1., ls=(0, (2, 1.2))))
    if display:
        ax.plot(*RING['hp'], 'o', ms=3.2, color=DISTRACTOR, mec='w', mew=.4, zorder=5)
        ax.plot(*RING['opp'], 'D', ms=3., color=TARGET, mec='w', mew=.4, zorder=5)
    if labels:
        kw = dict(fontsize=6, color=FG, ha='center', va='center')
        ax.text(1.45, ECC + .95, 'HP', color=HP, fontsize=6.5, fontweight='bold', ha='left', va='center',
                path_effects=[pe.withStroke(linewidth=1.6, foreground='w')])
        stroke = [pe.withStroke(linewidth=1.6, foreground='w')]
        ax.text(ECC + .05, -1.25, 'next', path_effects=stroke, **kw)
        ax.text(0, -ECC - 1.3, 'opposite', path_effects=stroke, **kw)
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.6)
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal')
    ax.set_axis_off()


LEGEND = """
**Figure 5b. The fitted gain field over the visual field.** The joint gain-field fit's parametric field, not data: {method} The three areas are those where the sustained gain is reliably lower at HP than at the other three locations at n = 28 (joint fit, HP − LP: V3 −.090, q = .032; hV4 −.057, q = .028; TO −.024, q = .044; Wilcoxon signed-rank across participants, Benjamini-Hochberg across the nine areas; V1 −.071, q = .20, and the other areas are in the parameter table). Colour: % change in the gain on the stimulus drive (magma: dark, the strongest suppression; bright, no change or enhancement), on each area's own scale (colour bar under the column); the teal contour and the teal tick on the colour bar mark no change (0%), and thin grey contours mark fixed steps. The display is rotated so that the high-probability (HP) distractor location is at the top (red dashed ring); small circles, the eight item positions on the 4° ring (stronger: the four possible target and distractor locations; the two next to HP at the left and right, the one opposite HP at the bottom); dashed circle, the 3.17° aperture of the mapping bar; +, fixation. {rows} Bottom: the same field around the 4° item ring, by angle from HP (0°: HP; ±90°: next to HP; ±180°: opposite HP){profile_rows}. The sustained field narrows up the hierarchy (median width σ: V3 {s_v3:.1f}°, hV4 {s_hv4:.1f}°, TO {s_to:.1f}°; FWHM {f_v3:.1f}°, {f_hv4:.1f}° and {f_to:.1f}°) while pRFs grow, so the field's effect on a voxel's response, which integrates the field over its pRF, is broader than the field drawn here in visual space. The measured item responses of Fig. 3 are not drawn on the profiles: they are % of the mean BOLD response at pRF-selected voxels, not % change in gain. {numbers} Field equations and code: `retsupp/visualize/paper/fig5b_gain_field_map_v2.py` (the same functions as the model's forward pass).
"""
METHOD_PARAMS = ("the median over participants (n = 28) of each fitted parameter of the area (gains and widths), put through "
                 "the model's field equations.")
METHOD_MAPS = ("for each participant (n = 28) the fitted parameters of the area were put through the model's field equations, "
               "and the maps show the median over participants at each point.")
ROWS_BOTH = ("Top row, the sustained field between search displays: a Gaussian gain field at each of the four locations, "
             "with its own gain at HP and a common gain at the other three. Middle row, during a search display with the "
             "distractor at HP (blue dot) and the target opposite HP (orange diamond): the sustained field plus the transient "
             "fields at the distractor and at the target. The transient width is at a bound of the fit (0.5° or 8°) in about "
             "half of the participants per area, so the shape of the transient fields is approximate.")
ROWS_SUS = ("Top: the sustained field between search displays: a Gaussian gain field at each of the four locations, with "
            "its own gain at HP and a common gain at the other three. The transient fields during a search display are "
            "not drawn: their width is at a bound of the fit (0.5° or 8°) in about half of the participants per area, so "
            "their shape is not well determined.")


def ring_profile(maps_full, xx, yy, angles):
    th = np.deg2rad(angles)
    px, py = ECC * np.sin(th), ECC * np.cos(th)
    g = xx[0]
    ix = np.clip(np.searchsorted(g, px), 0, len(g) - 1)
    iy = np.clip(np.searchsorted(g, py), 0, len(g) - 1)
    return maps_full[iy, ix]


def main(table, name, sustained_only=False, method='params', cmap_name='magma'):
    d = per_subject(table)
    g = np.linspace(-EXTENT, EXTENT, 481)
    xx, yy = np.meshgrid(g, g)
    build = median_param_maps if method == 'params' else median_maps
    res = {a: build(d, a, xx, yy) for a in AREAS}
    angles = np.linspace(-180, 180, 721)

    W = DOUBLE
    cw, gap = 1.62, .42
    left = (W - 3 * cw - 2 * gap) / 2 + .15
    mh = cw
    nrow = 1 if sustained_only else 2
    cb_h, prof_h = .07, .8
    top_pad, between = .28, .06
    H = top_pad + nrow * mh + (nrow - 1) * between + .3 + cb_h + .4 + prof_h + (.42 if not sustained_only else .3)
    fig = plt.figure(figsize=(W, H))
    numbers = []
    for c, a in enumerate(AREAS):
        sus, disp, n = res[a]
        shown = [sus] if sustained_only else [sus, disp]
        cmap = mpl.colormaps[cmap_name[c] if isinstance(cmap_name, (list, tuple)) else cmap_name]
        allv = np.concatenate([m.ravel() for m in shown])
        lo, hi = min(np.percentile(allv, .5), 0.), max(np.percentile(allv, 99.5), 0.)
        span = hi - lo
        step = step_for(span / 1.6)
        lo_n, hi_n = np.floor(lo / step) * step, np.ceil(hi / step) * step
        if hi_n - 0 < step * .5:
            hi_n = step * .5 if hi_n <= 0 else hi_n
        norm = mpl.colors.Normalize(vmin=lo_n, vmax=max(hi_n, step * .5))
        x0 = left + c * (cw + gap)
        y = H - top_pad
        for r, m in enumerate(shown):
            y -= mh + (between if r else 0)
            ax = fig.add_axes([x0 / W, y / H, cw / W, mh / H])
            ax.imshow(m, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap=cmap, norm=norm,
                      interpolation='bilinear')
            levels = [v for v in np.arange(lo_n, hi_n + 1e-9, step) if abs(v) > 1e-9 and v > m.min() + .3 * step]
            if levels:
                ax.contour(xx, yy, m, levels=levels, colors=['w'], linewidths=.35, alpha=.45)
            if m.min() < 0 < m.max():
                ax.contour(xx, yy, m, levels=[0.], colors=[ZERO], linewidths=.8)
            decorate(ax, r == 1, labels=(c == 0 and r == 0))
            if r == 0:
                ax.set_title(f'{a}', fontsize=7.5, pad=2)
        y -= .2 + cb_h
        cax = fig.add_axes([(x0 + .15) / W, y / H, (cw - .3) / W, cb_h / H])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        ticks = list(np.arange(lo_n, norm.vmax + 1e-9, step))
        while len(ticks) > 5:
            ticks = ticks[::2] if 0 not in ticks[::2] else ticks[::2]
        if all(abs(t) > 1e-9 for t in ticks):
            ticks = sorted(ticks + [0.])
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t:g}'.replace('-', '−') for t in ticks])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.ax.axvline(0, color=ZERO, lw=1.2)
        cb.outline.set_linewidth(.4)
        if c == 0:
            cb.ax.set_title('% change in gain', fontsize=6, pad=1.5, loc='left')
        y -= .4 + prof_h
        pax = fig.add_axes([x0 / W, y / H, cw / W, prof_h / H])
        ps = ring_profile(sus, xx, yy, angles)
        pax.plot(angles, ps, color=FG, lw=1.1)
        rng = [ps]
        if not sustained_only:
            pd_ = ring_profile(disp, xx, yy, angles)
            pax.plot(angles, pd_, color=FG, lw=.9, ls=(0, (2.5, 1.5)), alpha=.75)
            rng.append(pd_)
        pax.axhline(0, color=ZERO, lw=.7, zorder=0)
        for v in (-180, -90, 0, 90, 180):
            pax.axvline(v, color=HP if v == 0 else '.8', lw=.5, zorder=0, alpha=.7 if v == 0 else 1)
        plo, phi = min(r.min() for r in rng), max(r.max() for r in rng)
        pad = .12 * max(phi - plo, 1)
        pax.set_ylim(min(plo - pad, -1), max(phi + pad, 1))
        pax.set_xlim(-180, 180)
        pax.set_xticks([-180, -90, 0, 90, 180])
        pax.set_xticklabels(['Opp.', 'Next', 'HP', 'Next', 'Opp.'], fontsize=6)
        pax.tick_params(axis='y', labelsize=6)
        if c == 0:
            pax.set_ylabel('% change in gain\naround the ring', fontsize=6.5)
        offset_spines(pax)
        at = lambda prof, deg: float(prof[np.argmin(np.abs(angles - deg))])
        line = (f'{a} (n = {n}): sustained field at HP {at(ps, 0):+.1f}%, next to HP {at(ps, 90):+.1f}%, '
                f'opposite {at(ps, 180):+.1f}%')
        if not sustained_only:
            line += f'; during the display {at(pd_, 0):+.1f}%, {at(pd_, 90):+.1f}% and {at(pd_, 180):+.1f}%'
        numbers.append(line)
    if not sustained_only:
        kax = fig.add_axes([left / W, .02 / H, 4 / W, .14 / H])
        kax.set_axis_off()
        kax.set_xlim(0, 4)
        kax.set_ylim(-1, 1)
        kax.plot([0, .3], [0, 0], color=FG, lw=1.1)
        kax.text(.36, 0, 'Sustained field', fontsize=6, va='center')
        kax.plot([1.45, 1.75], [0, 0], color=FG, lw=.9, ls=(0, (2.5, 1.5)), alpha=.75)
        kax.text(1.81, 0, 'During a display (distractor at HP, target opposite)', fontsize=6, va='center')
    sig = d[d.roi.isin(AREAS)].groupby('roi').sigma_AF.median()
    f = 2 * np.sqrt(2 * np.log(2))
    nums = ('Field around the ring: ' + '; '.join(numbers) + '.').replace('-', '−')
    legend = LEGEND.format(method=METHOD_PARAMS if method == 'params' else METHOD_MAPS,
                           rows=ROWS_SUS if sustained_only else ROWS_BOTH,
                           profile_rows='' if sustained_only else ' (solid, sustained field; dashed, during the display)',
                           numbers=nums, s_v3=sig['V3'], s_hv4=sig['hV4'], s_to=sig['TO'],
                           f_v3=f * sig['V3'], f_hv4=f * sig['hV4'], f_to=f * sig['TO'])
    save(fig, name, legend)
    print(name, nums)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', default='notes/data/gain_field_parameters_final2_2026-11.tsv')
    ap.add_argument('--name', default='fig5b_gain_field_map_v2_n28')
    ap.add_argument('--sustained-only', action='store_true')
    ap.add_argument('--method', choices=['params', 'maps'], default='params',
                    help='params: field of the median parameters (main); maps: median of the participants\' fields')
    ap.add_argument('--cmaps', nargs='+', default=['magma'], help='One colormap, or one per area')
    a = ap.parse_args()
    main(a.table, a.name, a.sustained_only, a.method, a.cmaps[0] if len(a.cmaps) == 1 else a.cmaps)
