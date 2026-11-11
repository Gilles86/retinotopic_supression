"""Fig. 5b v5: the fitted gain field, taken apart into what is learned, what the target does and what the distractor
does, then added up.

Rows (each map on its own colour scale, % change in gain on the stimulus drive):
  1  Sustained field between search displays (learned): Gaussian fields at the four possible locations, g_HP at HP and
     g_LP at the other three.
  2  Target field alone, while a display is on: g_T at the target (here opposite HP, an LP location).
  3  Distractor field alone, while a display is on: g_HP_dyn with the distractor at HP (the LP value is given in the
     panel). Drawn pale: the HP − LP distractor gain is not a reliable effect.
  4  During a search display (distractor at HP, target opposite): the sum of rows 1–3, as in the model's forward pass.

Median parameters over participants (n = 28, sub-01/02 excluded) of the joint gain-field fit per area; the transient
width is the median over participants whose width is not at a bound of the fit (0.5° or 8°).

    python -m retsupp.visualize.paper.fig5b_gain_field_map_v4
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from retsupp.visualize.paper.fig5b_gain_field_map import APERTURE, ECC, EXTENT, PARS, RING, gauss
from retsupp.visualize.paper.fig5b_gain_field_map_v2 import free_width, per_subject
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, TARGET, save

AREAS = ['V1', 'V3', 'hV4', 'TO']
ZERO = '#2A9D8F'
STROKE = [pe.withStroke(linewidth=1.8, foreground='w')]


def components(d, area, xx, yy):
    """% change maps of the sustained field, the target field, the distractor field (at HP) and their sum."""
    x = d[d.roi == area]
    p = x[PARS].median()
    w = free_width(x)
    sus = np.maximum(1 + sum((p.g_HP if k == 'hp' else p.g_LP) * gauss(xx, yy, c, p.sigma_AF)
                             for k, c in RING.items()), 0.)
    tgt = p.g_T_dyn * gauss(xx, yy, RING['opp'], w)
    dis = p.g_HP_dyn * gauss(xx, yy, RING['hp'], w)
    tot_hp = sus + tgt + dis                                                  # distractor at HP
    tot_lp = sus + tgt + p.g_LP_dyn * gauss(xx, yy, RING['left'], w)          # distractor next to HP (an LP location)
    info = dict(n=len(x), w=w, n_free=int(((x.sigma_dyn > .51) & (x.sigma_dyn < 7.9)).sum()), sigma_af=p.sigma_AF,
                g_T=p.g_T_dyn, g_D_hp=p.g_HP_dyn, g_D_lp=p.g_LP_dyn, g_HP=p.g_HP, g_LP=p.g_LP)
    return [100 * (sus - 1), 100 * tgt, 100 * dis, 100 * (tot_hp - 1), 100 * (tot_lp - 1)], info


def component_cmap(col):
    """Grey (lower) → white (no change) → the component's colour (higher), for the single-component rows."""
    return mpl.colors.LinearSegmentedColormap.from_list('c', ['#8a8f96', '#ffffff', col])


def at(m, xx, yy, xy):
    g = xx[0]
    return float(m[np.argmin(np.abs(g - xy[1])), np.argmin(np.abs(g - xy[0]))])


def step_for(span):
    for s in (1, 2, 2.5, 5, 10, 20, 25, 50):
        if span / s <= 5:
            return s
    return span / 4


def items(ax, row):
    """Ring of item positions; HP ring; target and distractor glyphs where the row has them."""
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec=MUTED, lw=.45, ls=(0, (2, 2))))
    for k, a in enumerate(np.deg2rad(np.arange(0, 360, 45))):
        main4 = k % 2 == 0
        ax.add_patch(Circle((ECC * np.sin(a), ECC * np.cos(a)), .75, fill=False, ec=FG, lw=.5 if main4 else .3,
                            alpha=.7 if main4 else .3))
    ax.add_patch(Circle(RING['hp'], 1.15, fill=False, ec=HP, lw=1., ls=(0, (2, 1.2))))
    if row in (1, 3, 4):
        ax.plot(*RING['opp'], 'D', ms=3.4, color=TARGET, mec='w', mew=.5, zorder=6)
    if row in (2, 3):
        ax.plot(*RING['hp'], 'o', ms=3.8, color=DISTRACTOR, mec='w', mew=.5, zorder=6)
    if row == 4:
        ax.plot(*RING['left'], 'o', ms=3.8, color=DISTRACTOR, mec='w', mew=.5, zorder=6)
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.6)
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal')
    ax.set_axis_off()


def pct(v):
    return f'{v:+.0f}%'.replace('-', '−') if abs(v) >= .5 else '0%'


def row_icon(fig, x, y, s, row):
    """Small glyph for the row label: the display with the component's mark."""
    ax = fig.add_axes([x, y, s[0], s[1]])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_axis_off()
    for c in RING.values():
        ax.add_patch(Circle(c, 1.1, fill=False, ec=MUTED, lw=.5))
    if row in (0, 3, 4):
        ax.add_patch(Circle(RING['hp'], 1.8, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2))))
    if row in (1, 3, 4):
        ax.plot(*RING['opp'], 'D', ms=3.6, color=TARGET, mec='w', mew=.4)
    if row in (2, 3):
        ax.plot(*RING['hp'], 'o', ms=4, color=DISTRACTOR, mec='w', mew=.4)
    if row == 4:
        ax.plot(*RING['left'], 'o', ms=4, color=DISTRACTOR, mec='w', mew=.4)
    ax.plot(0, 0, '+', color=FG, ms=2.5, mew=.5)


ROW_TEXT = [('Learned', 'Sustained field', 'lower gain at HP'),
            ('Target', 'Transient field', 'boost at the target'),
            ('Distractor', 'Transient field', 'smaller boost;\nsame at HP and LP'),
            ('Search', 'Distractor at HP', 'sum of the three'),
            ('Search', 'Distractor at LP', 'sum of the three')]
ROW_COL = [HP, TARGET, DISTRACTOR, FG, FG]
NROW = len(ROW_TEXT)

LEGEND = """
**Figure 5b. The fitted gain field, taken apart.** The joint gain-field fit's parametric field over the visual field, not data: for each area the median over participants (n = 28) of each fitted parameter, put through the model's field equations; colour, % change in the gain on the stimulus drive, each map on its own scale (colour bar under it; the two search rows share one scale per area): rows 1, 4 and 5 in magma (dark, lower gain; bright, higher; teal line and tick, no change), rows 2 and 3, one component alone, from white (no change) to the component's colour (higher gain; grey, lower); values printed at the item locations. The display is rotated so that the high-probability (HP) distractor location is at the top (red dashed ring); circles, the eight item positions on the 4° ring (stronger: the four possible target and distractor locations); dashed circle, the 3.17° aperture of the mapping bar; +, fixation. Row 1, learned, between search displays: the sustained field, a Gaussian gain field at each of the four possible locations with its own gain at HP and a common gain at the other three (median width σ: {sig}). Row 2, the target's transient field alone, with the target (orange diamond) at the location opposite HP; present only while a search display is on. Row 3, the distractor's transient field alone, with the distractor (blue dot) at HP: a smaller transient boost than the target's, the same at HP and at the other locations (median gains per area: {dis}; over areas and participants, distractor gain at HP +.063, positive in 18 of 28 participants, p = .016, at LP +.064, 20 of 28, p = .095, HP − LP +.004, p = .29; target gain +.157, 25 of 28, p = 3 × 10⁻⁷; Wilcoxon signed-rank on each participant's median over areas). Rows 4 and 5, during a search display with the target opposite HP and the distractor at HP (row 4) or at the LP location next to HP (row 5): the sum of rows 1–3, as in the model's forward pass, where the transient fields are added to the sustained gain; only the distractor's location differs between the two rows, and the line under row 5 gives the difference in gain at the distractor (HP minus LP), which is what learning the HP location changes: the sustained HP − LP difference plus the transient HP − LP difference. The transient width is at a bound of the fit (0.5° or 8°) in about half of the participants, so the target and distractor fields are drawn with the median width of the participants whose width is not at a bound ({w}; target and distractor share one width in the fit). V1 is shown for comparison: its sustained HP − LP gain is of similar size to V3's but not reliable (q = .20; V3 q = .032, hV4 .028, TO .044; Wilcoxon signed-rank across participants, Benjamini-Hochberg across nine areas). {numbers} Code: `retsupp/visualize/paper/fig5b_gain_field_map_v5.py`.
"""


def main(table, name):
    d = per_subject(table)
    g = np.linspace(-EXTENT, EXTENT, 401)
    xx, yy = np.meshgrid(g, g)
    res = {a: components(d, a, xx, yy) for a in AREAS}

    W = DOUBLE
    lab_w = 1.25                     # left column: row statements
    cw, gap = 1.28, .16
    mh, cbh, cb_gap, rgap = cw, .055, .07, .3
    top = .3
    rh = mh + cb_gap + cbh + rgap
    H = top + NROW * rh + .25
    fig = plt.figure(figsize=(W, H))
    x_left = lab_w + .05
    numbers = []
    for c, a in enumerate(AREAS):
        maps, info = res[a]
        x0 = x_left + c * (cw + gap)
        both = np.concatenate([maps[3].ravel(), maps[4].ravel()])
        for r, m in enumerate(maps):
            ytop = H - top - r * rh
            y = ytop - mh
            ax = fig.add_axes([x0 / W, y / H, cw / W, mh / H])
            src = both if r in (3, 4) else m
            lo, hi = min(src.min(), 0.), max(src.max(), 0.)
            pale = False
            if r in (1, 2):
                # one component alone: white = no change, the component's colour = higher gain, grey = lower
                lim = max(abs(lo), abs(hi), 1.)
                st = step_for(lim)
                lim = np.ceil(lim / st) * st
                lo_n, hi_n = (-lim if lo < -.5 else -lim * .25), lim
                cmap = component_cmap(TARGET if r == 1 else DISTRACTOR)
                norm = mpl.colors.TwoSlopeNorm(vcenter=0., vmin=lo_n, vmax=hi_n)
            else:
                st = step_for(max(hi - lo, 1.))
                lo_n, hi_n = np.floor(lo / st) * st, np.ceil(hi / st) * st
                if hi_n - lo_n < st:
                    hi_n = lo_n + st
                cmap = mpl.colormaps['magma']
                norm = mpl.colors.Normalize(vmin=lo_n, vmax=hi_n)
            ax.imshow(m, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap=cmap, norm=norm,
                      interpolation='bilinear', alpha=.5 if pale else 1.)
            if r in (0, 3, 4) and m.min() < 0 < m.max():
                ax.contour(xx, yy, m, levels=[0.], colors=[ZERO], linewidths=.8)
            items(ax, r)
            # values at the locations that matter for the row
            spots = {0: [('hp', 'HP'), ('opp', 'opp')], 1: [('opp', 'T')], 2: [('hp', 'D')],
                     3: [('hp', 'HP'), ('opp', 'T')], 4: [('left', 'LP'), ('opp', 'T')]}[r]
            for key, _ in spots:
                cx, cy = RING[key]
                v = at(m, xx, yy, (cx, cy))
                ax.text(cx + 1.35, cy, pct(v), fontsize=6.5, ha='left', va='center', color=FG, path_effects=STROKE,
                        zorder=7)
            if r == 0:
                ax.set_title(a + (' (not reliable)' if a == 'V1' else ''), fontsize=7.5, pad=3,
                             color='.45' if a == 'V1' else FG)
            # colour bar
            cax = fig.add_axes([(x0 + .14) / W, (y - cb_gap - cbh) / H, (cw - .28) / W, cbh / H])
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
            if pale:
                cb.solids.set_alpha(.5)
            ticks = sorted({lo_n, hi_n, 0.}) if (r in (0, 3, 4) or lo < -.5) else [0., hi_n]
            cb.set_ticks(ticks)
            cb.set_ticklabels([f'{t:g}'.replace('-', '−') for t in ticks])
            cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
            cb.ax.axvline(0, color=ZERO, lw=1.2)
            cb.outline.set_linewidth(.4)
            if c == 0:
                cb.ax.text(-.06, .5, '% change\nin gain', transform=cb.ax.transAxes, fontsize=6, ha='right',
                           va='center', color='.35', linespacing=1.)
        sus, tgt, dis, tot, tot_lp = maps
        d_hp, d_lp = at(tot, xx, yy, RING['hp']), at(tot_lp, xx, yy, RING['left'])
        info.update(d_hp=d_hp, d_lp=d_lp)
        yb = H - top - (NROW - 1) * rh - mh - cb_gap - cbh - .2
        fig.text((x0 + cw / 2) / W, yb / H, f'At the distractor, HP − LP: {pct(d_hp - d_lp)}', fontsize=6.5,
                 ha='center', va='center', color=FG)
        numbers.append(f"{a} (n = {info['n']}): sustained field {pct(at(sus, xx, yy, RING['hp']))} at HP, "
                       f"{pct(at(sus, xx, yy, RING['opp']))} at the other locations; target field "
                       f"{pct(at(tgt, xx, yy, RING['opp']))} at the target; distractor field "
                       f"{pct(at(dis, xx, yy, RING['hp']))} at HP (gain {info['g_D_hp']:+.2f}; at LP {info['g_D_lp']:+.2f}); "
                       f"during search, at the distractor {pct(d_hp)} with the distractor at HP and {pct(d_lp)} "
                       f"with it at LP (HP − LP {pct(d_hp - d_lp)}), at the target {pct(at(tot, xx, yy, RING['opp']))}")
    # row statements with glyphs
    for r, (head, sub, msg) in enumerate(ROW_TEXT):
        ytop = H - top - r * rh
        yc = ytop - mh / 2
        s = .42
        row_icon(fig, .05 / W, (yc - s / 2 + .18) / H, (s / W, s / H), r)
        fig.text(.05 / W, (yc - .17) / H, head, fontsize=7.5, weight='bold', color=ROW_COL[r], va='center')
        fig.text(.05 / W, (yc - .33) / H, sub, fontsize=6.5, color=FG, va='center')
        if msg:
            fig.text(.05 / W, (yc - .47) / H, msg, fontsize=6.5, color='.4', va='top', linespacing=1.1)
    sig = '; '.join(f"{a} {res[a][1]['sigma_af']:.1f}°" for a in AREAS)
    wtxt = '; '.join(f"{a} {res[a][1]['w']:.1f}° (from {res[a][1]['n_free']} of {res[a][1]['n']})" for a in AREAS)
    dtxt = '; '.join(f"{a} {res[a][1]['g_D_hp']:+.2f} at HP, {res[a][1]['g_D_lp']:+.2f} at LP" for a in AREAS)
    legend = LEGEND.format(sig=sig, w=wtxt, dis=dtxt.replace('-', '−'),
                           numbers=('Values: ' + '; '.join(numbers) + '.'))
    save(fig, name, legend)
    print('\n'.join(numbers))
    for a in AREAS:
        i = res[a][1]
        print(f"{a}: at the distractor HP {i['d_hp']:+.1f}%, LP {i['d_lp']:+.1f}%, HP − LP {i['d_hp'] - i['d_lp']:+.1f}%")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', default='notes/data/gain_field_parameters_final2_2026-11.tsv')
    ap.add_argument('--name', default='fig5b_gain_field_map_v5_n28')
    a = ap.parse_args()
    main(a.table, a.name)
