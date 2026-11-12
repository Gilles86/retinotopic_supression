"""Figure 5 v6: the fitted gain-field maps (new panel a) above the per-area dot plots (b-d, the earlier a, c, d).

a  Maps of the fitted parametric field over the display, V3, hV4 and TO (columns), for the sustained field, the
   target's transient field and the distractor's transient field (rows). Per area, the median over participants of each
   participant's joint-fit parameter; the sustained HP gain is drawn as median g_LP + median dS (joint), so the
   HP - LP difference on the map is the joint dS of the text; the distractor gain is averaged over its locations as in
   panel b. The transient width is fixed for display (FWHM 3.2 deg) and not interpreted.
b  Target and distractor gain per area (the earlier panel a).
c  dS estimated alone (unchanged).
d  dD and dT estimated alone (unchanged).

Also writes the supplementary figure ``figS_gain_field_map_search`` (the sum of the three fields during a search display,
distractor at HP and at the LP location next to HP, target opposite HP).

Run through ``run_excluding`` for n = 28::

    python -m retsupp.visualize.paper.run_excluding --exclude 1 2 -- retsupp.visualize.paper.fig5_gain_fields_v6 \\
        --suffix _n28

``--layout v7`` (= ``--compact``) is the compact two-row version. ``--layout v8`` puts panel a in one row of six maps over
the full width (sustained block left, transient block right, colour keys under each block, one shared key for the target
and distractor fields) and draws the sustained HP gain as median g_LP + the median dS ESTIMATED ALONE (the value panel c
plots), so the HP - LP printed on each map equals panel c; parameters go to ``..._2026-11-12_v8.tsv``.

``--layout v11`` is v10 on a reference fit given by its tables (``--gain-table``, ``--conditional-table``; e.g. the
polished fits, pRFs refitted together with the gain fields) with the sustained width of each map restricted to fits with
a reliable sustained gain (``--widths-table``, the ``_widths_thr2.tsv`` of ``field_widths_reliable``) and every sentence
that names significant areas computed from the data (joint and estimated alone, per pathway). The per-pathway pooling
is written to ``notes/data/dS_pooled_by_pathway<suffix>_2026-11-12_v11.tsv``. Driver for a whole reference fit:
``retsupp/visualize/paper/regenerate_reference.sh``.
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy import stats

import retsupp.visualize.paper.fig5_gain_fields as F
from retsupp.visualize.paper.fig45_common import conditional, joint
from retsupp.visualize.paper.fig5b_gain_field_map import APERTURE, ECC, EXTENT, RING, gauss
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, OUT, ROI_ORDER, TARGET, save

AREAS = ['V3', 'hV4', 'TO']
ZERO = '#2A9D8F'
STROKE = [pe.withStroke(linewidth=1.8, foreground='w')]
FWHM = 2 * np.sqrt(2 * np.log(2))
TR_FWHM = 3.2                     # display width of the transient fields (deg), the same for every area
TR_SIGMA = TR_FWHM / FWHM
TGT, DIS = 'opp', 'left'          # target opposite HP; distractor at the LP location next to HP (its gain is the average)


def map_parameters(gain_table, exclude=(1, 2), cond=None, widths=None):
    """Per area, the median over participants of each participant's joint-fit parameter (DoG, m7_full).

    cond: the estimated-alone table (subject, roi, dS, ...). If given, the HP gain drawn is median g_LP + the median dS
    estimated alone (what panel c plots) instead of + the median joint dS."""
    d = joint(gain_table, 'dog')
    d = d[~d.subject.isin(exclude)]
    rows = []
    for a in ROI_ORDER:
        x = d[d.roi == a]
        rows.append(dict(roi=a, n=len(x), g_HP=x.g_HP.median(), g_LP=x.g_LP.median(), dS=x.dS.median(),
                         T=x['T'].median(), D=x.D.median(), g_HP_dyn=x.g_HP_dyn.median(),
                         g_LP_dyn=x.g_LP_dyn.median(), dD=x.dD.median(), sigma_AF=x.sigma_AF.median()))
    p = pd.DataFrame(rows).set_index('roi')
    p['dS_map'] = p.dS                     # the HP - LP drawn on the map: the joint dS ...
    if cond is not None:                   # ... or dS estimated alone, the per-area median of panel c
        c = cond[~cond.subject.isin(exclude)].dropna(subset=['dS'])
        p['dS_alone'] = c.groupby('roi').dS.median().reindex(p.index)
        p['n_alone'] = c.groupby('roi').dS.size().reindex(p.index)
        p['dS_map'] = p.dS_alone
    p['g_HP_map'] = p.g_LP + p.dS_map
    if widths is not None:                 # the width among fits with a reliable sustained gain (field_widths_reliable)
        w = pd.read_csv(widths, sep='\t')
        w = w[(w.field == 'sustained') & w.subset.str.startswith('reliable')].set_index('roi')
        p['sigma_AF_all'] = p.sigma_AF
        p['n_width'] = w['n'].reindex(p.index)
        p['sigma_AF'] = (w['median'].reindex(p.index) / FWHM).fillna(p.sigma_AF)
    p['fwhm_AF'] = FWHM * p.sigma_AF
    return p


def q_by_area(df, term):
    """BH q per area (Wilcoxon against 0) of one term, as panel c and the parameter table."""
    ps = [stats.wilcoxon(df[df.roi == r][term].dropna()).pvalue for r in ROI_ORDER]
    return dict(zip(ROI_ORDER, stats.false_discovery_control(ps)))


def and_list(xs):
    xs = list(xs)
    return 'no area' if not xs else xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


def pathway_pooling(cond, term='dS'):
    """Per participant the median of the term over the areas of each pathway (and over all nine), median over
    participants (%, bootstrapped 95% CI), participants < 0, Wilcoxon; BH across the three pathways."""
    from retsupp.visualize.paper.fig45_common import median_ci
    from retsupp.visualize.paper.paper_style import ROI_GROUPS
    rows = []
    for name, rois in list(ROI_GROUPS) + [('All areas', tuple(ROI_ORDER))]:
        v = cond[cond.roi.isin(rois)].groupby('subject')[term].median().dropna() * 100
        m, lo, hi = median_ci(v.to_numpy())
        rows.append(dict(group=name, n=len(v), median_pct=m, ci_lo=lo, ci_hi=hi, k_neg=int((v < 0).sum()),
                         p=stats.wilcoxon(v).pvalue))
    g = pd.DataFrame(rows)
    k = g.group != 'All areas'
    g.loc[k, 'q_BH3'] = stats.false_discovery_control(g.loc[k, 'p'])
    return g


def fields(p, xx, yy):
    """Fractional change in gain: sustained, target field, distractor field (each alone)."""
    sus = np.maximum(1 + sum((p.g_HP_map if k == 'hp' else p.g_LP) * gauss(xx, yy, c, p.sigma_AF)
                             for k, c in RING.items()), 0.) - 1
    tgt = p['T'] * gauss(xx, yy, RING[TGT], TR_SIGMA)
    dis = p.D * gauss(xx, yy, RING[DIS], TR_SIGMA)
    return sus, tgt, dis


def search_fields(p, xx, yy):
    """During a search display, target opposite HP: sustained + target + distractor field, distractor at HP (gain median
    g_LP_dyn + median dD, the joint transient HP - LP) or at the LP location next to HP (median g_LP_dyn)."""
    sus, tgt, _ = fields(p, xx, yy)
    at_hp = sus + tgt + (p.g_LP_dyn + p.dD) * gauss(xx, yy, RING['hp'], TR_SIGMA)
    at_lp = sus + tgt + p.g_LP_dyn * gauss(xx, yy, RING['left'], TR_SIGMA)
    return at_hp, at_lp


def value_at(m, xx, yy, xy):
    g = xx[0]
    return float(m[np.argmin(np.abs(g - xy[1])), np.argmin(np.abs(g - xy[0]))])


def fmt(v):
    return f'{v:+.2f}'.replace('-', '−') if abs(v) >= .005 else '0.00'


def draw_display(ax, marks=()):
    """Display outline: the 3.17 deg bar aperture, the eight item positions (the four possible target and distractor
    locations stronger), HP ring at the top, fixation; marks: (kind, location key)."""
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec=MUTED, lw=.5))
    ax.add_patch(Circle((0, 0), ECC, fill=False, ec=FG, lw=.35, ls=(0, (1.5, 1.5)), alpha=.45))
    for k, a in enumerate(np.deg2rad(np.arange(0, 360, 45))):
        main4 = k % 2 == 0
        ax.add_patch(Circle((ECC * np.sin(a), ECC * np.cos(a)), .75, fill=False, ec=FG, lw=.5 if main4 else .3,
                            alpha=.7 if main4 else .3))
    ax.add_patch(Circle(RING['hp'], 1.15, fill=False, ec=HP, lw=1., ls=(0, (2, 1.2))))
    for kind, key in marks:
        if kind == 'target':
            ax.plot(*RING[key], 'D', ms=3.4, color=TARGET, mec='w', mew=.5, zorder=6)
        else:
            ax.plot(*RING[key], 'o', ms=3.8, color=DISTRACTOR, mec='w', mew=.5, zorder=6)
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.6)
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal')
    ax.set_axis_off()


def component_cmap(col):
    return mpl.colors.LinearSegmentedColormap.from_list('c', ['#8a8f96', '#ffffff', col])


def row_icon(fig, rect, row):
    ax = fig.add_axes(rect)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_axis_off()
    for c in RING.values():
        ax.add_patch(Circle(c, 1.1, fill=False, ec=MUTED, lw=.5))
    if row == 0:
        ax.add_patch(Circle(RING['hp'], 1.8, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2))))
    if row == 1:
        ax.plot(*RING[TGT], 'D', ms=3.6, color=TARGET, mec='w', mew=.4)
    if row == 2:
        ax.plot(*RING[DIS], 'o', ms=4, color=DISTRACTOR, mec='w', mew=.4)
    ax.plot(0, 0, '+', color=FG, ms=2.5, mew=.5)


ROWS_V6 = ("Middle row, the target's transient field alone (target, orange diamond, at an LP location, here opposite "
           "HP); value, the target gain of panel b. Bottom row, the distractor's transient field alone (distractor, blue "
           "dot, drawn at an LP location next to HP); value, the distractor gain averaged over its locations, as in "
           "panel b (the transient HP − LP difference is in panel d).")
ROWS_V7 = ("Bottom row, the transient fields of the target (orange, around the orange diamond, at an LP location, here "
           "opposite HP) and of the distractor (blue, around the blue dot, drawn at an LP location next to HP) in one "
           "map; values, the target gain of panel b and the distractor gain averaged over its locations, as in panel b "
           "(the transient HP − LP difference is in panel d); both on one scale, so their colours compare directly.")
ROWS_V8 = ("Right three maps, the transient fields of the target (orange, around the orange diamond, at an LP location, "
           "here opposite HP) and of the distractor (blue, around the blue dot, drawn at an LP location next to HP) in "
           "one map; values, the target gain of panel b and the distractor gain averaged over its locations, as in "
           "panel b (the transient HP − LP difference is in panel d); both on one scale, so their colours compare "
           "directly.")
ROWS = [('Sustained field', 'lower at the HP location', HP),
        ('Target field', 'transient gain at the target', TARGET),
        ('Distractor field', 'smaller transient gain', DISTRACTOR)]


def panel_maps(fig, p, x_lab, x0, y_top, cw, gap, rgap, per_area_sustained):
    """Panel a: 3 x 3 maps; one colour bar per row on the right (shared across areas), or per-area ranges for the
    sustained row. Returns the bottom y (inches) and the colour-scale description."""
    W, H = fig.get_size_inches()
    g = np.linspace(-EXTENT, EXTENT, 401)
    xx, yy = np.meshgrid(g, g)
    maps = {a: fields(p.loc[a], xx, yy) for a in AREAS}
    scales = []
    y = y_top
    for r, (head, sub, col) in enumerate(ROWS):
        y -= cw
        vals = np.concatenate([maps[a][r].ravel() for a in AREAS])
        if r == 0:
            cmap = mpl.colormaps['magma']
            lo = np.floor(vals.min() / .05) * .05
            norm_shared = mpl.colors.Normalize(vmin=lo, vmax=0.)
        else:
            cmap = component_cmap(col)
            hi = np.ceil(vals.max() / .1) * .1
            norm_shared = mpl.colors.TwoSlopeNorm(vcenter=0., vmin=-hi * .25, vmax=hi)
        for c, a in enumerate(AREAS):
            m = maps[a][r]
            xa = x0 + c * (cw + gap)
            ax = fig.add_axes([xa / W, y / H, cw / W, cw / H])
            norm = norm_shared
            if r == 0 and per_area_sustained:
                lo_a = np.floor(m.min() / .01) * .01
                hi_a = max(np.ceil(m.max() / .01) * .01, 0.)
                norm = mpl.colors.Normalize(vmin=lo_a, vmax=hi_a if hi_a > lo_a else lo_a + .01)
                ax.text(.5, -.135, f'range {fmt(lo_a)} to {fmt(hi_a)}', transform=ax.transAxes, fontsize=6,
                        ha='center', va='top', color='.45')
            ax.imshow(m, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap=cmap, norm=norm,
                      interpolation='bilinear')
            if r == 0 and m.min() < 0 < m.max():
                ax.contour(xx, yy, m, levels=[0.], colors=[ZERO], linewidths=.8)
            marks = [(), [('target', TGT)], [('distractor', DIS)]][r]
            draw_display(ax, marks)
            # one number per panel, in the units of the text
            if r == 0:
                cx, cy = RING['hp']
                txt, v = 'HP − LP ', p.loc[a, 'dS']
            elif r == 1:
                cx, cy = RING[TGT]
                txt, v = '', p.loc[a, 'T']
            else:
                cx, cy = RING[DIS]
                txt, v = '', p.loc[a, 'D']
            if r == 0:     # under the map, off the data
                ax.text(.5, -.04, txt + fmt(v), transform=ax.transAxes, fontsize=6.5, ha='center', va='top',
                        color=FG)
            elif r == 1:
                ax.text(cx + 1.3, cy, fmt(v), fontsize=6.5, ha='left', va='center', color=FG, path_effects=STROKE,
                        zorder=7)
            else:
                ax.text(cx + .75, cy - .95, fmt(v), fontsize=6.5, ha='left', va='top', color=FG, path_effects=STROKE,
                        zorder=7)
            if r == 0:
                ax.set_title(a, fontsize=7.5, pad=3, color=FG)
        # colour bar for the row (shared across areas)
        xc = x0 + 3 * cw + 2 * gap + .12
        cax = fig.add_axes([xc / W, (y + .12) / H, .07 / W, (cw - .24) / H])
        sm = mpl.cm.ScalarMappable(norm=norm_shared if not (r == 0 and per_area_sustained) else
                                   mpl.colors.Normalize(0, 1), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
        if r == 0 and per_area_sustained:
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Lower', 'Higher'])
            scales.append('sustained row: each area on its own scale (range under each map)')
        else:
            v0, v1 = norm_shared.vmin, norm_shared.vmax
            ticks = [v0, 0.] if r == 0 else [0., v1]     # transient rows: grey below 0 only marks 'lower'
            cb.set_ticks(ticks)
            cb.set_ticklabels([fmt(t) if t else '0' for t in ticks])
            cb.ax.axhline(0, color=ZERO, lw=1.2)
            scales.append(f'{head.lower()}: one scale for the three areas, {fmt(v0)} to {fmt(v1)}')
        if r == 0:
            cb.ax.set_title('Gain', fontsize=6, pad=3, color='.35')
        # row statement with a glyph
        yc = y + cw / 2
        s = .4
        row_icon(fig, [x_lab / W, (yc + .05) / H, s / W, s / H], r)
        fig.text(x_lab / W, (yc - .1) / H, head, fontsize=7, weight='bold', color=col, va='center')
        fig.text(x_lab / W, (yc - .26) / H, sub, fontsize=6.5, color='.4', va='center')
        y -= rgap + (.24 if r == 0 else 0)
    return y, scales


def transient_rgb(tgt, dis, vmax):
    """Target (orange) and distractor (blue) fields in one map, both on the same scale 0..vmax (white = no change)."""
    def rgb(c):
        return np.array(mpl.colors.to_rgb(c))
    at = np.clip(tgt / vmax, 0, 1)[..., None]
    ad = np.clip(dis / vmax, 0, 1)[..., None]
    img = np.ones(tgt.shape + (3,))
    img = img * (1 - at) + rgb(TARGET) * at
    img = img * (1 - ad) + rgb(DISTRACTOR) * ad
    return img


def panel_maps_v7(fig, p, x0, y_top, cw, gap, rgap, x_key):
    """Panel a, compact: row 1 the sustained field (each area on its own scale, range under the map); row 2 the target
    and distractor transient fields in one map (one shared scale 0..vmax for both components and all areas)."""
    W, H = fig.get_size_inches()
    g = np.linspace(-EXTENT, EXTENT, 361)
    xx, yy = np.meshgrid(g, g)
    maps = {a: fields(p.loc[a], xx, yy) for a in AREAS}
    vmax = np.ceil(max(max(maps[a][1].max(), maps[a][2].max()) for a in AREAS) / .1) * .1
    ranges = []
    y1 = y_top - cw
    y2 = y1 - rgap - cw
    for c, a in enumerate(AREAS):
        sus, tgt, dis = maps[a]
        xa = x0 + c * (cw + gap)
        ax = fig.add_axes([xa / W, y1 / H, cw / W, cw / H])
        lo_a = np.floor(sus.min() / .01) * .01
        hi_a = max(np.ceil(sus.max() / .01) * .01, 0.)
        hi_a = hi_a if hi_a > lo_a else lo_a + .01
        ax.imshow(sus, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap='magma',
                  norm=mpl.colors.Normalize(lo_a, hi_a), interpolation='bilinear')
        if sus.min() < 0 < sus.max():
            ax.contour(xx, yy, sus, levels=[0.], colors=[ZERO], linewidths=.8)
        draw_display(ax)
        ax.set_title(a, fontsize=7.5, pad=2, color=FG)
        ax.text(.5, -.03, f'HP − LP {fmt(p.loc[a, "dS"])}', transform=ax.transAxes, fontsize=6.5, ha='center',
                va='top', color=FG)
        ax.text(.5, -.135, f'range {fmt(lo_a)} to {fmt(hi_a)}', transform=ax.transAxes, fontsize=6, ha='center',
                va='top', color='.45')
        ranges.append(f'{a} {fmt(lo_a)} to {fmt(hi_a)}')
        ax = fig.add_axes([xa / W, y2 / H, cw / W, cw / H])
        ax.imshow(transient_rgb(tgt, dis, vmax), extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower',
                  interpolation='bilinear')
        draw_display(ax, [('target', TGT), ('distractor', DIS)])
        tx, ty = RING[TGT]
        ax.text(tx + 1.0, ty + .2, fmt(p.loc[a, 'T']), fontsize=6.5, ha='left', va='center', color=FG,
                path_effects=STROKE, zorder=7)
        dx, dy = RING[DIS]
        ax.text(dx + .75, dy - .95, fmt(p.loc[a, 'D']), fontsize=6.5, ha='left', va='top', color=FG,
                path_effects=STROKE, zorder=7)
    # row labels: a small title over each row, left-aligned with the first map
    for yy_, head, sub, col in ((y1 + cw + .2, 'Sustained field', 'lower at the HP location', HP),
                                (y2 + cw + .1, 'Transient fields', 'target and distractor, while a display is on', FG)):
        t = fig.text(x0 / W, yy_ / H, head, fontsize=7, weight='bold', color=col, va='center', ha='left')
        fig.canvas.draw()
        xe = t.get_window_extent().x1 / fig.dpi
        fig.text((xe + .08) / W, yy_ / H, sub, fontsize=6.5, color='.4', va='center', ha='left')
    # keys at the right
    kx = x_key
    cax = fig.add_axes([kx / W, (y1 + .15) / H, .07 / W, (cw - .3) / H])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='magma'), cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Lower', 'Higher'])
    cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
    fig.text((kx - .02) / W, (y1 + cw - .02) / H, 'Gain (scale\nper area)', fontsize=6, color='.35', va='bottom',
             ha='left', linespacing=1.)
    for k, (col, lab) in enumerate(((TARGET, 'Target'), (DISTRACTOR, 'Distractor'))):
        cm = mpl.colors.LinearSegmentedColormap.from_list('c', ['#ffffff', col])
        cax = fig.add_axes([(kx + k * .42) / W, (y2 + .15) / H, .07 / W, (cw - .3) / H])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=cm), cax=cax)
        cb.set_ticks([0, vmax])
        cb.set_ticklabels(['0', fmt(vmax)] if k == 1 else ['', ''])   # one scale for both: labels on the right bar
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
        fig.text((kx + k * .42 + .035) / W, (y2 + cw - .1) / H, lab, fontsize=6, color=col, va='bottom', ha='center')
    return y2 - .05, ranges, vmax


def fmt3(v):
    return f'{v:+.3f}'.replace('-', '−')


def panel_maps_v8(fig, p, x0, y_top, cw, gap, bgap, under):
    """Panel a in one row over the full width: three sustained maps (each area on its own scale, a colour bar under each
    map), a gap, three maps with the target and distractor transient fields (one shared scale, one key under the block:
    an orange and a blue bar on the same ticks). Returns the bottom y (inches), the sustained ranges and vmax."""
    W, H = fig.get_size_inches()
    g = np.linspace(-EXTENT, EXTENT, 361)
    xx, yy = np.meshgrid(g, g)
    maps = {a: fields(p.loc[a], xx, yy) for a in AREAS}
    vmax = np.ceil(max(max(maps[a][1].max(), maps[a][2].max()) for a in AREAS) / .1) * .1
    ranges = []
    y = y_top - cw
    xs = x0
    xt = x0 + 3 * cw + 2 * gap + bgap
    bar_h = .055
    for c, a in enumerate(AREAS):
        sus, tgt, dis = maps[a]
        # sustained
        xa = xs + c * (cw + gap)
        ax = fig.add_axes([xa / W, y / H, cw / W, cw / H])
        lo_a = np.floor(sus.min() / .01) * .01
        hi_a = max(np.ceil(sus.max() / .01) * .01, 0.)
        hi_a = hi_a if hi_a > lo_a else lo_a + .01
        norm = mpl.colors.Normalize(lo_a, hi_a)
        ax.imshow(sus, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap='magma', norm=norm,
                  interpolation='bilinear')
        if sus.min() < 0 < sus.max():
            ax.contour(xx, yy, sus, levels=[0.], colors=[ZERO], linewidths=.8)
        draw_display(ax)
        ax.set_title(a, fontsize=7.5, pad=2, color=FG)
        ax.text(.5, -.03, f'HP − LP {fmt3(p.loc[a, "dS_map"])}', transform=ax.transAxes, fontsize=6.5, ha='center',
                va='top', color=FG)
        bw = .56 * cw
        cax = fig.add_axes([(xa + (cw - bw) / 2) / W, (y - .24 - bar_h) / H, bw / W, bar_h / H])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='magma'), cax=cax, orientation='horizontal')
        cb.set_ticks([lo_a, hi_a])
        cb.set_ticklabels([fmt(lo_a), '0' if hi_a == 0 else fmt(hi_a)])
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
        if c == 0:
            cb.ax.text(-.07, .5, 'Gain', transform=cb.ax.transAxes, fontsize=6, ha='right', va='center', color='.35')
        ranges.append(f'{a} {fmt(lo_a)} to {fmt(hi_a)}')
        # transient: target and distractor in one map
        xa = xt + c * (cw + gap)
        ax = fig.add_axes([xa / W, y / H, cw / W, cw / H])
        ax.imshow(transient_rgb(tgt, dis, vmax), extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower',
                  interpolation='bilinear')
        draw_display(ax, [('target', TGT), ('distractor', DIS)])
        ax.set_title(a, fontsize=7.5, pad=2, color=FG)
        tx, ty = RING[TGT]
        ax.text(tx + 1.0, ty + .2, fmt(p.loc[a, 'T']), fontsize=6.5, ha='left', va='center', color=FG,
                path_effects=STROKE, zorder=7)
        dx, dy = RING[DIS]
        ax.text(dx + .75, dy - .95, fmt(p.loc[a, 'D']), fontsize=6.5, ha='left', va='top', color=FG,
                path_effects=STROKE, zorder=7)
    # block headings: bold head, grey statement (target and distractor in their colours)
    yh = y + cw + .29
    for xb, head, col, parts in (
            (xs, 'Sustained field', HP, [('lower at the HP location', '.4')]),
            (xt, 'Transient fields', FG, [('target', TARGET), (' and ', '.4'), ('distractor', DISTRACTOR),
                                          (', while a display is on', '.4')])):
        t = fig.text(xb / W, yh / H, head, fontsize=7, weight='bold', color=col, va='center', ha='left')
        fig.canvas.draw()
        xe = t.get_window_extent().x1 / fig.dpi + .08
        for txt, cc in parts:
            t = fig.text(xe / W, yh / H, txt, fontsize=6.5, color=cc, va='center', ha='left')
            fig.canvas.draw()
            xe = t.get_window_extent().x1 / fig.dpi
    # one key for the transient block: target (orange) over distractor (blue), touching, on the same ticks
    bw = 1.5
    xk = xt + (3 * cw + 2 * gap - bw) / 2
    kh = .09                                  # each bar high enough for its 6 pt label
    yk = y - .08 - kh                         # top bar; the second bar directly below it
    for k, (col, lab) in enumerate(((TARGET, 'Target'), (DISTRACTOR, 'Distractor'))):
        cm = mpl.colors.LinearSegmentedColormap.from_list('c', ['#ffffff', col])
        cax = fig.add_axes([xk / W, (yk - k * kh) / H, bw / W, kh / H])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=cm), cax=cax,
                          orientation='horizontal')
        ticks = [0, vmax / 2, vmax]
        cb.set_ticks(ticks if k == 1 else [])
        if k == 1:
            cb.set_ticklabels(['0', fmt(vmax / 2), fmt(vmax)])
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
        cb.ax.text(-.03, .5, lab, transform=cb.ax.transAxes, fontsize=6, ha='right', va='center', color=col)
    fig.text((xk + bw + .07) / W, yk / H, 'Gain (one scale)', fontsize=6, color='.35', ha='left', va='center')
    return y - under, ranges, vmax


def pct(v, dec=None):
    """Gain (fraction) as a signed percentage: one decimal below 10%, else none."""
    x = 100 * v
    dec = (1 if abs(x) < 10 else 0) if dec is None else dec
    return (f'{x:+.{dec}f}%' if round(abs(x), dec) else f'0%').replace('-', '−')


HP_CMAP = mpl.colors.LinearSegmentedColormap.from_list('hp', ['#ffffff', '#F4A6A6', HP, '#7A1010'])


def panel_maps_v9(fig, p, x0, y_top, cw, gap, bgap, under, bars=True):
    """As v8, in percent, with the sustained block as HP − LP DIFFERENCE maps: only the HP-specific part of the
    sustained field (dS estimated alone × the field's Gaussian at HP), white → HP red on one scale shared by the three
    areas whose deepest value is the deepest area's dS (no headroom); a thin red contour at the field's FWHM around HP;
    the HP − LP value in bold next to HP; a small inset per map with the sustained gain at LP (grey) and at HP (red)."""
    W, H = fig.get_size_inches()
    g = np.linspace(-EXTENT, EXTENT, 361)
    xx, yy = np.meshgrid(g, g)
    maps = {a: fields(p.loc[a], xx, yy) for a in AREAS}
    diff = {a: 100 * p.loc[a, 'dS_map'] * gauss(xx, yy, RING['hp'], p.loc[a, 'sigma_AF']) for a in AREAS}
    r = np.hypot(xx, yy)                                  # soft circular edge: 1 inside 4.9°, cosine to 0 at 6°
    fade = np.where(r < 4.9, 1., np.where(r > EXTENT, 0., .5 * (1 + np.cos(np.pi * (r - 4.9) / (EXTENT - 4.9)))))
    dmin = min(diff[a].min() for a in AREAS)            # the deepest area's HP − LP (%), the end of the scale
    norm = mpl.colors.Normalize(vmin=dmin, vmax=0.)
    vmax = np.ceil(max(max(maps[a][1].max(), maps[a][2].max()) for a in AREAS) / .1) * .1
    y = y_top - cw
    xs = x0
    xt = x0 + 3 * cw + 2 * gap + bgap
    for c, a in enumerate(AREAS):
        _, tgt, dis = maps[a]
        xa = xs + c * (cw + gap)
        ax = fig.add_axes([xa / W, y / H, cw / W, cw / H])
        # reversed so that 0 is white and the deepest value is dark red; faded to white towards a circular edge so
        # the field does not end in a hard cut at the square border
        rgba = HP_CMAP(mpl.colors.Normalize(0, -dmin)(-diff[a]))
        rgba[..., :3] = rgba[..., :3] * fade[..., None] + (1 - fade[..., None])
        ax.imshow(rgba, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', interpolation='bilinear')
        half = diff[a].min() / 2                         # FWHM contour of the HP-specific field
        ax.contour(xx, yy, diff[a], levels=[half], colors=[HP], linewidths=.6)
        draw_display(ax)
        ax.set_title(a, fontsize=7.5, pad=2, color=FG)
        hx, hy = RING['hp']
        ax.text(hx + 1.35, hy + .15, pct(p.loc[a, 'dS_map']), fontsize=7, weight='bold', color=HP, ha='left',
                va='center', path_effects=STROKE, zorder=7)
        if bars:      # the full sustained gain at LP and at HP (%), which the difference map leaves out
            ax.text(.5, -.005, f'Gain at LP {pct(p.loc[a, "g_LP"])}\nat HP {pct(p.loc[a, "g_HP_map"])}',
                    transform=ax.transAxes, fontsize=6, ha='center', va='top', color='.4', linespacing=1.1)
        # transient: target and distractor in one map, in percent
        xa = xt + c * (cw + gap)
        ax = fig.add_axes([xa / W, y / H, cw / W, cw / H])
        img = transient_rgb(tgt, dis, vmax)
        img = img * fade[..., None] + (1 - fade[..., None])
        ax.imshow(img, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', interpolation='bilinear')
        draw_display(ax, [('target', TGT), ('distractor', DIS)])
        ax.set_title(a, fontsize=7.5, pad=2, color=FG)
        tx, ty = RING[TGT]
        ax.text(tx + 1.0, ty + .2, pct(p.loc[a, 'T'], 0), fontsize=6.5, ha='left', va='center', color=FG,
                path_effects=STROKE, zorder=7)
        dx, dy = RING[DIS]
        ax.text(dx + .75, dy - .95, pct(p.loc[a, 'D'], 0), fontsize=6.5, ha='left', va='top', color=FG,
                path_effects=STROKE, zorder=7)
    # block headings
    yh = y + cw + .29
    for xb, head, col, parts in (
            (xs, 'Sustained field', HP, [('at HP minus at the other locations', '.4')]),
            (xt, 'Transient fields', FG, [('target', TARGET), (' and ', '.4'), ('distractor', DISTRACTOR),
                                          (', while a display is on', '.4')])):
        t = fig.text(xb / W, yh / H, head, fontsize=7, weight='bold', color=col, va='center', ha='left')
        fig.canvas.draw()
        xe = t.get_window_extent().x1 / fig.dpi + .08
        for txt, cc in parts:
            t = fig.text(xe / W, yh / H, txt, fontsize=6.5, color=cc, va='center', ha='left')
            fig.canvas.draw()
            xe = t.get_window_extent().x1 / fig.dpi
    # key under the sustained block: one bar, 0 .. deepest HP − LP
    bw, kh = 1.5, .07
    xk = xs + (3 * cw + 2 * gap - bw) / 2
    yk = y - .245 - kh
    cax = fig.add_axes([xk / W, yk / H, bw / W, kh / H])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(dmin, 0), cmap=HP_CMAP.reversed()), cax=cax,
                      orientation='horizontal')
    cb.set_ticks([dmin, dmin / 2, 0])
    cb.set_ticklabels([f'{dmin:+.1f}%'.replace('-', '−'), f'{dmin / 2:+.1f}%'.replace('-', '−'), '0'])
    cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
    fig.text((xk + bw + .07) / W, (yk + kh / 2) / H, 'HP − LP gain', fontsize=6, color='.35', ha='left',
             va='center')
    # key under the transient block (as v8, in %)
    xk = xt + (3 * cw + 2 * gap - bw) / 2
    kh = .09
    yk = y - .08 - kh
    for k, (col, lab) in enumerate(((TARGET, 'Target'), (DISTRACTOR, 'Distractor'))):
        cm = mpl.colors.LinearSegmentedColormap.from_list('c', ['#ffffff', col])
        cax = fig.add_axes([xk / W, (yk - k * kh) / H, bw / W, kh / H])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=cm), cax=cax,
                          orientation='horizontal')
        cb.set_ticks([0, vmax / 2, vmax] if k == 1 else [])
        if k == 1:
            cb.set_ticklabels(['0', f'+{100 * vmax / 2:.0f}%', f'+{100 * vmax:.0f}%'])
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
        cb.ax.text(-.03, .5, lab, transform=cb.ax.transAxes, fontsize=6, ha='right', va='center', color=col)
    fig.text((xk + bw + .07) / W, yk / H, 'Gain (one scale)', fontsize=6, color='.35', ha='left', va='center')
    return y - under, dmin, vmax


def main(gain_table, conditional_table, slope_table, sigma_table, suffix, per_area_sustained=None, compact=False,
         layout=None, widths_table=None, fit_label=None):
    # the same inputs and options as fig5_gain_fields_n28 (final fits, b-d unchanged in content)
    F.FINAL, F.SUFFIX, F.WIDTHS_SUPP = True, suffix, True
    F.GAIN_TABLE = gain_table
    js = {f: joint(gain_table, f) for f in F.FAM if f in pd.read_csv(gain_table, sep='\t', usecols=['family']).family.unique()}
    cond = conditional(conditional_table)
    t = F.summarise(js, cond)
    dt = cond[['subject', 'roi', 'dT']].dropna()
    d = js['dog']
    n = d.subject.nunique()
    layout = layout or ('v7' if compact else 'v6')
    compact = layout in ('v7', 'v8', 'v9', 'v10', 'v11')
    v11 = layout == 'v11'      # v11: v10 on a reference fit, reliable-gain widths, significance sentences from the data
    v10 = layout in ('v10', 'v11')      # v10: v9 with panel c led by the all-areas pooled estimate
    v9 = layout in ('v9', 'v10', 'v11')
    v8 = layout in ('v8', 'v9', 'v10', 'v11')        # v9: the v8 layout and parameters, in percent, sustained block as HP − LP maps
    p = map_parameters(gain_table, cond=cond if v8 else None, widths=widths_table if v11 else None)

    from retsupp.visualize.paper.fig7_behaviour import icons, row
    from retsupp.visualize.paper.paper_style import stimulus_icon

    if per_area_sustained is None:
        # does one shared scale hide the HP dip in V3 or hV4? (HP - LP smaller than 20% of the shared range)
        g = np.linspace(-EXTENT, EXTENT, 201)
        xx, yy = np.meshgrid(g, g)
        rng = -min(fields(p.loc[a], xx, yy)[0].min() for a in AREAS)
        per_area_sustained = any(abs(p.loc[a, 'dS']) < .2 * rng for a in ('V3', 'hV4'))

    W = DOUBLE
    cw, gap, rgap = 1.2, .26, .18
    x_lab, x0 = .05, 1.6
    H = .32 + 3 * cw + 2 * rgap + .24 + .25 + 1.95 + 2.35
    if compact:
        cw, gap, rgap = 1.05, .22, .5
        x0 = .3
        bh = .82                         # panel b height
        H = .34 + .15 + 2 * cw + rgap + .05 + (.3 + bh + .44) + (.62 + .85 + .55) + .2
    if v8:
        cw, gap, bgap, under = 1.0, .1, .42, (.5 if v9 else .44)
        x0 = (W - 6 * cw - 4 * gap - bgap) / 2
        H = .58 + cw + under + (.3 + bh + .44) + (.62 + .85 + .55) + (.14 if v9 else .2)
    print('figure height (in):', round(H, 2))
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    Y = lambda dd: H - dd

    # a: the maps
    fig.text(.02 / W, Y(.14) / H, 'a', fontsize=8, fontweight='bold', va='center', ha='left')
    fig.text(.2 / W, Y(.14) / H, 'The fitted gain fields over the display (group medians)', fontsize=7.5, color=FG,
             va='center', ha='left')
    if v9:
        y_bottom, dmin, vmax = panel_maps_v9(fig, p, x0, Y(.58), cw, gap, bgap, under)
        scales = [f'sustained maps: HP − LP, one scale for the three areas, from 0 (white) to the deepest area\'s '
                  f'HP − LP, {dmin:+.1f}% (red; no headroom, so that area saturates)'.replace('-', '−'),
                  f'transient maps: target (orange) and distractor (blue) on one scale, 0 to +{100 * vmax:.0f}%, for both '
                  'and all three areas (the two bars of the key share their ticks)']
        ranges = []
    elif v8:
        y_bottom, ranges, vmax = panel_maps_v8(fig, p, x0, Y(.58), cw, gap, bgap, under)
        scales = [f'sustained maps: each area on its own scale, colour bar under each map ({"; ".join(ranges)})',
                  f'transient maps: target (orange) and distractor (blue) on one scale, 0 to {fmt(vmax)}, for both '
                  'and all three areas (the two bars of the key share their ticks)']
    elif compact:
        y_bottom, ranges, vmax = panel_maps_v7(fig, p, x0, Y(.34 + .15), cw, gap, rgap, x_key=x0 + 3 * cw + 2 * gap + .3)
        scales = [f'sustained row: each area on its own scale ({"; ".join(ranges)})',
                  f'transient row: target (orange) and distractor (blue) on one scale, 0 to {fmt(vmax)}, for both '
                  'and all three areas']
    else:
        y_bottom, scales = panel_maps(fig, p, x_lab, x0, Y(.34), cw, gap, rgap, per_area_sustained)

    # b: target and distractor gain per area (the earlier a)
    yb = y_bottom - .05
    d_rois = t[(t.model == 'DoG') & (t.term == 'D') & (t.q < .05)].roi.tolist()
    d_where = ' in ' + ', '.join(d_rois) if d_rois else ' nowhere reliably'
    t_rois = t[(t.model == 'DoG') & (t.term == 'T') & (t.q < .05)].roi.tolist()
    t_where = (' in every area' if len(t_rois) == 9 else ' beyond V1' if set(t_rois) == set(ROI_ORDER) - {'V1'}
               else f' in {len(t_rois)} of 9 areas')
    row(fig, yb, 'b', [('Gain at the ', FG), ('target', TARGET), (t_where, FG),
                       (', at the ', FG), ('distractor', DISTRACTOR), (d_where, FG)], [], x_let=.02, x_title=.2)
    LX, RX, PW = .55, 4.05, 2.95
    bh_ = bh if compact else 1.18
    gb = .42 if compact else .48
    ax = at(LX, yb - gb - bh_, 6.45, bh_)
    sm = lambda term: t[(t.model == 'DoG') & (t.term == term)].set_index('roi').loc[ROI_ORDER]
    if v9:      # percent: the same data and statistics, × 100
        k = 100.
        d100 = d.assign(T=d['T'] * k, D=d.D * k)
        sm100 = lambda term: sm(term).assign(median=lambda x: x['median'] * k, ci_low=lambda x: x.ci_low * k,
                                             ci_high=lambda x: x.ci_high * k)
        F.dot_panel(ax, [(d100, 'T', TARGET, sm100('T')), (d100, 'D', DISTRACTOR, sm100('D'))], (-40, 120),
                    'Gain change (%)')
        ax.set_yticks([0, 50, 100])
    else:
        F.dot_panel(ax, [(d, 'T', TARGET, sm('T')), (d, 'D', DISTRACTOR, sm('D'))], (-.4, 1.2), 'Gain')
        ax.set_yticks([0, .5, 1])

    # c, d: as before
    y2 = H - (yb - gb - bh_ - .55)
    ch = .85 if compact else 1.0
    hp_lp = [lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('hp',))),
             lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('lp',)))]
    row(fig, Y(y2), 'c', [('Sustained gain: less at ', FG), ('HP', HP)], [(hp_lp, HP, 'Whole block, at HP − at LP')],
        x_let=.02, x_title=.2, key_y=Y(y2 + .27))
    ax = at(LX, Y(y2 + .62 + ch), PW, ch)
    if v9:
        F.dot_panel(ax, [(cond.assign(dS=cond.dS * 100), 'dS', HP, None)], (-12, 6), 'HP − LP (%)')
        ax.set_yticks([-10, -5, 0, 5])
        if v10:
            pooled_c = pooled_all_areas(ax, cond, (-12, 6))
    else:
        F.dot_panel(ax, [(cond, 'dS', HP, None)], (-.12, .06), 'Gain, HP − LP')
        ax.set_yticks([-.1, 0, .05])
    title_d = [('Transient gain at HP: slightly less for ', FG), ('distractor', DISTRACTOR), (' and ', FG),
               ('target', TARGET)]
    if stats.wilcoxon(cond.groupby('subject').dD.median().dropna()).pvalue >= .05:
        title_d = [('Transient gain at HP: slightly less, ', FG), ('target', TARGET), (' and ', FG),
                   ('distractor', DISTRACTOR), (' alike', FG)]
    keys_d = [(icons('location', 'distractor'), DISTRACTOR, 'Each trial'), (icons('location', 'target'), TARGET, '')]
    row(fig, Y(y2), 'd', title_d, keys_d, x_let=RX - .5, x_title=RX - .32, key_y=Y(y2 + .27))
    fig.text((RX + PW) / W, Y(y2 + .27) / H, 'As Fig. 1c, d', fontsize=6, color='.45', ha='right', va='center')
    ax = at(RX, Y(y2 + .62 + ch), PW, ch)
    if v9:
        F.dot_panel(ax, [(cond.assign(dD=cond.dD * 100), 'dD', DISTRACTOR, None),
                         (dt.assign(dT=dt.dT * 100), 'dT', TARGET, None)], (-35, 30), 'HP − LP (%)')
        ax.set_yticks([-20, 0, 20])
    else:
        F.dot_panel(ax, [(cond, 'dD', DISTRACTOR, None), (dt, 'dT', TARGET, None)], (-.35, .3), 'Gain, HP − LP')
        ax.set_yticks([-.2, 0, .2])

    # legend: the earlier legend for b-d, re-lettered, plus a
    fw, fw_roi, fw_bound = F.sustained_fwhm(js, 'dog')
    fmt_ = dict(spacing=4 * np.sqrt(2), fwhm=fw, fw_bound=100 * fw_bound,
                fw_roi='; '.join(f'{r} {v:.1f}°' for r, v in fw_roi.items()), **F.fwhm_text(d),
                **F.legend_findings(cond, dt, js))
    L = F.LEGEND
    i0 = L.index(' b, The sustained gain field over the search display')
    i1 = L.index('{width_note}{overlap_note}') + len('{width_note}{overlap_note}')
    old = (L[:i0] + L[i1:]).format(**fmt_)
    old = old.replace(f'(n = 30)', f'(n = {n})').replace('a, c, d: dots are subjects', 'b–d: dots are subjects')
    old = old.replace(' a, Gain at the target (orange)', ' b, Gain at the target (orange)')
    old = old.replace(' Provisional: current inputs.', '')
    head, rest = old.strip().split('** ', 1)
    head = ('**Figure 5. The gain fields: gain at the target and the distractor, and less sustained gain at the '
            'high-probability (HP) location.**')
    ds_txt = (lambda a: f"HP − LP = dS estimated alone {fmt3(p.loc[a, 'dS_map'])}, the median of panel c; joint-fit dS "
              f"{fmt3(p.loc[a, 'dS'])}") if v8 else (lambda a: f"HP − LP = joint dS {fmt(p.loc[a, 'dS'])}")
    f3 = (lambda v: f'{v:+.3f}'.replace('-', '−')) if v8 else fmt
    vals = '; '.join(
        f"{a}: sustained gain at LP {f3(p.loc[a, 'g_LP'])}, at HP {f3(p.loc[a, 'g_HP_map'])} "
        f"({ds_txt(a)}; median g_HP {fmt(p.loc[a, 'g_HP'])}), width σ "
        f"{p.loc[a, 'sigma_AF']:.2f}° (FWHM {p.loc[a, 'fwhm_AF']:.1f}°); target gain {fmt(p.loc[a, 'T'])}; "
        f"distractor gain {fmt(p.loc[a, 'D'])}" for a in AREAS)
    alone_sig = []
    if v8:      # areas where dS estimated alone is reliable, the test of panel c (Wilcoxon, BH across the nine ROIs)
        pv = [stats.wilcoxon(cond[cond.roi == r].dS.dropna()).pvalue for r in ROI_ORDER]
        alone_sig = [r for r, q in zip(ROI_ORDER, stats.false_discovery_control(pv)) if q < .05 and r in AREAS]
    neg = [r for r in ROI_ORDER if p.loc[r, 'g_LP'] < 0 and p.loc[r, 'g_HP_map'] < 0]
    a_txt = (
        "a, The fitted parametric gain field over the search display, not data: for each area, the median over "
        f"participants (n = {n}) of each participant's joint-fit parameter (all terms fitted together), put through the "
        "model's field equations; value, the fractional change in gain on the stimulus drive (−0.10: 10% less). The "
        "display is rotated so that the HP location is at the top (red dashed ring); small circles, the eight item "
        "positions on the 4° ring (dotted; stronger: the four possible target and distractor locations); grey circle, "
        "the 3.17° aperture of the mapping bar; +, fixation. "
        + ("Left three maps, the sustained field, present throughout the block: a Gaussian field at each of the four "
           "possible locations, with the median joint-fit gain at the low-probability (LP) locations and, at HP, that "
           "gain plus the area's median HP − LP difference estimated alone (dS, the value panel c plots), so that the "
           "HP − LP difference printed under each map equals panel c's estimate for that area (the joint-fit dS is "
           "larger and is given below); width, the area's median fitted width. " if v8 else
           "Top row, the sustained field, present throughout the block: "
           "a Gaussian field at each of the four possible locations, with the median gain at the low-probability (LP) "
           "locations and, at HP, that gain plus the median HP − LP difference of the joint fit, so that the HP − LP "
           "difference printed on each map is the joint dS of panel c's legend and the text (panel c shows dS estimated "
           "alone, which is smaller); width, the area's median fitted width. ")
        + "The sustained gain is negative at all four "
        f"locations in most areas ({len(neg)} of 9: {', '.join(neg)}). "
        + (ROWS_V8 if v8 else ROWS_V7 if compact else ROWS_V6) + " "
        "The transient fields exist only while a search display is on; their fitted width is often at a bound of the "
        f"fit (0.5° or 8°), so it is set for display to the same FWHM of {TR_FWHM:.1f}° in every area and is not "
        "interpreted (field widths per subject: Supplementary Fig. S, field widths). Colour scales: "
        + '; '.join(scales)
        + (' (magma for the sustained field, teal line and tick at no change; the transient rows from white, no '
           'change, to the component colour)' if not compact else
           ' (magma for the sustained field; the transient fields from white, no change, to orange, target, and '
           'blue, distractor)' if v8 else
           ' (magma for the sustained field, teal line at no change; the transient fields from white, no change, to '
           'orange, target, and blue, distractor)')
        + '. Values used: ' + vals + '. V3, hV4 and TO are the areas where the sustained gain is reliably lower at HP '
        'than at LP in the joint fit (q < .05; V1, V2 and the other areas are in the parameter table)'
        + ('; estimated alone, as drawn here, the difference is reliable in ' + ' and '.join(alone_sig)
           + ' (panel c)' if v8 else '') + '. The sum of the '
        'three fields during a search display is in Supplementary Fig. S (gain field during search).')
    general, rest = rest.split(' b–d: ', 1)          # 'Gain-field model fits ... (+0.30: 30% more).' leads
    legend = head + ' ' + general.strip() + ' ' + a_txt + ' b–d: ' + rest
    legend = legend.replace('-0.', '−0.')
    if v9:
        legend = percent_legend(legend, p, n, dmin, data_driven=(js['dog'], cond) if v11 else None)
    if v10:
        pw = None
        if v11:
            pw = pathway_pooling(cond)
            pw.to_csv(f'notes/data/dS_pooled_by_pathway{suffix}_2026-11-12_v11.tsv', sep='\t', index=False)
        legend = pooled_legend(legend, pooled_c, pw)
    if v11:
        lab = fit_label or ('Gain-field model fits with DoG pRFs, the pRFs refitted together with the gain fields')
        legend = legend.replace('Gain-field model fits with DoG pRFs (n = ', f'{lab} (n = ', 1)
        old = '(median joint-fit LP gain; at HP, that plus the HP − LP difference drawn)'
        assert old in legend
        legend = legend.replace(old, old[:-1] + '; with the pRFs refitted, the level shared by all four locations '
                                'trades off against the item weight and the pRF amplitude, so read its sign, not its '
                                'size)')
    save(fig, f'fig5_gain_fields_{layout}' + suffix, legend)
    print('scales:', scales, 'per-area sustained:', per_area_sustained)
    print(p.loc[AREAS].round(4).to_string())
    if not compact:
        search_figure(p, n, suffix)
    if layout == 'v9':
        search_figure_v2(p, n, suffix)
    if v11:
        search_figure_v2(p, n, suffix, name='figS_gain_field_map_search_v3', dd=(js['dog'], cond))
    (p if v8 else p.drop(columns='dS_map')).to_csv(
        f'notes/data/fig5_gain_field_map_parameters{suffix}_2026-11-12{"_" + layout if v8 else ""}.tsv', sep='\t',
             float_format='%.5f')
    return p


POOL_X = -1.85      # x position of the all-areas estimate in panel c (left of V1, past a divider)


def pooled_all_areas(ax, cond, ylim):
    """Panel c (v10): the all-areas pooled estimate first and larger. Per participant the median dS over the nine areas
    (as sensitivity_excl_subjects.pooled and key_numbers), then the median over participants with its bootstrapped 95%
    CI, subject dots, Wilcoxon; the nine areas stay as they are, to the right of a thin divider."""
    from retsupp.visualize.paper.fig7_behaviour import XP, XMAX
    from retsupp.visualize.paper.fig45_common import median_ci
    lo, hi = ylim
    s = cond.groupby('subject').dS.median().dropna() * 100
    m, a, b = median_ci(s.to_numpy())
    p = stats.wilcoxon(s).pvalue
    k = int((s < 0).sum())
    v = s.to_numpy()
    xx = POOL_X + np.random.default_rng(99).uniform(-.22, .22, len(v))
    ok = (v >= lo) & (v <= hi)
    ax.scatter(xx[ok], v[ok], s=7, color=HP, alpha=.35, lw=0, zorder=1)
    ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=8, marker='v', color=HP, alpha=.5, lw=0, clip_on=False)
    ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=8, marker='^', color=HP, alpha=.5, lw=0, clip_on=False)
    ax.errorbar(POOL_X, m, yerr=[[m - a], [b - m]], fmt='o', ms=7, color=HP, mec='w', mew=.8, lw=2.0, capsize=0,
                zorder=5)
    ax.text(POOL_X, hi + .02 * (hi - lo), F.stars(p), ha='center', va='bottom', fontsize=8.5, color=HP, clip_on=False)
    ax.text(POOL_X + .38, m, f'{k}/{len(v)}', fontsize=6, color=HP, ha='left', va='center', zorder=6,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
    ax.axvline(-.95, color='.85', lw=.6, zorder=0)
    ax.hlines(0, POOL_X - .6, -.6, color='.75', lw=.6, ls='--', zorder=0)
    ax.set_xlim(POOL_X - .6, XMAX + .6)
    ticks = [POOL_X] + list(ax.get_xticks())
    labels = ['All\nareas'] + [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6)
    ax.get_xticklabels()[0].set_fontweight('bold')
    ax.get_xticklabels()[0].set_color(HP)
    return dict(median=m, lo=a, hi=b, k=k, n=len(v), p=p)


def pooled_legend(legend, pc, pathways=None):
    """v10 legend: panel c's sentence led by the all-areas pooled estimate, plus the groups (dS_pooled_by_pathway).
    pathways (v11): the per-pathway table computed from the data; the clause naming where it is reliable follows it."""
    g = (pd.read_csv('notes/data/dS_pooled_by_pathway_n28_2026-11-12.tsv', sep='\t') if pathways is None
         else pathways).set_index('group')
    clause = 'Present in early and occipitotemporal areas, not reliable in dorsal areas'
    if pathways is not None:
        names = {'Early': 'early', 'Dorsal': 'dorsal', 'Occipitotemporal': 'occipitotemporal'}
        yes = [names[k] for k in names if g.loc[k, 'q_BH3'] < .05]
        no = [names[k] for k in names if g.loc[k, 'q_BH3'] >= .05]
        clause = (f'Present in {and_list(yes)} areas' + (f', not reliable in {and_list(no)} areas' if no else '')
                  if yes else 'Not reliable in any single pathway')
    def grp(name, lab):
        r = g.loc[name]
        return f"{lab} {r.median_pct:+.1f}% (q = {r.q_BH3:.3f})".replace('-', '−').replace('= 0.', '= .')
    mant, ex = f"{pc['p']:.1e}".split('e-')
    pexp = f"{mant} × 10" + ''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in str(int(ex))).join(['⁻', ''])
    txt = (f"c, leftmost and larger: all areas pooled (per participant the median dS over the nine areas; median over "
           f"participants {pc['median']:+.1f}% [{pc['lo']:+.1f}, {pc['hi']:+.1f}], negative in {pc['k']} of {pc['n']} "
           f"participants, Wilcoxon p = {pexp}); the claim rests on this pooled effect, consistent across "
           f"participants, rather than on any single area. {clause} (the same pooling per pathway, BH across the three: "
           f"{grp('Early', 'early')}, {grp('Dorsal', 'dorsal')}, {grp('Occipitotemporal', 'occipitotemporal')}). "
           f"Right of the divider, the nine areas as before: ").replace('+-', '−').replace('-', '−')
    i = legend.find(' c, ')
    assert i >= 0, 'panel c sentence not found'
    return legend[:i + 1] + txt + legend[i + 4:]


def to_pct(m):
    """A signed gain in the legend text (fraction) as % (one decimal below 10%)."""
    sgn, num = m.group(1), m.group(2)
    x = float(num) * 100
    dec = 1 if x < 10 else 0
    return f'{sgn}{x:.{dec}f}%'


def percent_legend(legend, p, n, dmin, data_driven=None):
    """v9 legend: panel a rewritten (HP − LP difference maps, percent); every signed gain value in b–d (the
    per-ROI and pooled estimates and their CIs) converted to percent; differences are percentage points of gain."""
    import re
    head_end = legend.index(' a, The fitted parametric gain field')
    a_end = legend.index(' b–d: ')
    head = legend[:head_end]
    head = head.replace('a gain is the fractional change in drive at the centre of a field (+0.30: 30% more)',
                        'gains are in % (the change in drive at the centre of a field: +30%, 30% more); differences '
                        'between gains (HP − LP) are in percentage points of gain, written as %')
    wn = (lambda a: f", median of the {int(p.loc[a, 'n_width'])} fits with a reliable sustained gain; all fits "
          f"{FWHM * p.loc[a, 'sigma_AF_all']:.1f}°") if 'n_width' in p else (lambda a: '')
    vals = '; '.join(
        f"{a}: sustained gain at LP {pct(p.loc[a, 'g_LP'])}, at HP {pct(p.loc[a, 'g_HP_map'])} (HP − LP = dS "
        f"estimated alone {pct(p.loc[a, 'dS_map'])}, the median of panel c; joint-fit dS {pct(p.loc[a, 'dS'])}), "
        f"width σ {p.loc[a, 'sigma_AF']:.2f}° (FWHM {p.loc[a, 'fwhm_AF']:.1f}°{wn(a)}); target gain {pct(p.loc[a, 'T'], 0)}; "
        f"distractor gain {pct(p.loc[a, 'D'], 0)}" for a in AREAS)
    width_txt = "Gaussian at HP (the area's median fitted width)"
    sig_txt = ("V3, hV4 and TO are the areas where the sustained gain is reliably lower at HP than at LP in the joint fit "
               "(q < .05); estimated alone, as drawn, the difference is reliable in V3 and TO (panel c).")
    if data_driven is not None:
        jd, cond = data_driven
        qj, qa = q_by_area(jd, 'dS'), q_by_area(cond, 'dS')
        sj = [r for r in ROI_ORDER if qj[r] < .05]
        sa = [r for r in ROI_ORDER if qa[r] < .05]
        near = [f'{r} q = {qa[r]:.3f}'.replace('= 0.', '= .') for r in AREAS if .05 <= qa[r] < .1]
        sig_txt = (f"The sustained gain is reliably lower at HP than at LP (q < .05) in {and_list(sj)} in the joint fit "
                   f"and, estimated alone as drawn, in {and_list(sa)} (panel c)"
                   + (f"; {', '.join(near)} estimated alone" if near else '') + '.')
        if 'n_width' in p:
            width_txt = ("Gaussian at HP (width: the area's median full width at half maximum among fits whose sustained "
                         "gain is reliably different from zero, |gain| > 2 SE; Supplementary Fig., field widths)")
    neg = [r for r in ROI_ORDER if p.loc[r, 'g_LP'] < 0 and p.loc[r, 'g_HP_map'] < 0]
    a_txt = (
        " a, The fitted parametric gain field over the search display, not data: for each area, the median over "
        f"participants (n = {n}) of each participant's fitted parameters, put through the model's field equations. The "
        "display is rotated so that the HP location is at the top (red dashed ring); small circles, the eight item "
        "positions on the 4° ring (dotted; stronger: the four possible target and distractor locations); grey circle, "
        "the 3.17° aperture of the mapping bar; +, fixation. Left three maps, the sustained field at HP minus at the "
        "low-probability (LP) locations: only the HP-specific part of the sustained field, the area's median HP − LP "
        "difference estimated alone (dS, the value panel c plots, printed in bold next to HP) times the field's "
        + width_txt + "; thin red line, the half-maximum (FWHM) contour of that field. "
        "One scale for the three areas, white (no difference) to dark red at the deepest area's HP − LP "
        f"({dmin:+.1f}%; no headroom, so that area saturates). ".replace("-", "−") + "The sustained field itself is a suppression at all four "
        "possible locations in most areas "
        f"({len(neg)} of 9: {', '.join(neg)}); the grey lines under each map give the sustained gain at LP and at HP "
        "(median joint-fit LP gain; at HP, that plus the HP − LP difference drawn). Right three maps, the transient "
        "fields of the target (orange, around the orange diamond, at an LP location, here opposite HP) and of the "
        "distractor (blue, around the blue dot, drawn at an LP location next to HP) in one map; values, the target gain "
        "of panel b and the distractor gain averaged over its locations, as in panel b (the transient HP − LP "
        "difference is in panel d); both on one scale, 0 to the largest value, so their colours compare directly. The "
        "transient fields exist only while a search display is on; their fitted width is often at a bound of the fit "
        f"(0.5° or 8°), so it is set for display to the same FWHM of {TR_FWHM:.1f}° in every area and is not "
        "interpreted (field widths per participant: Supplementary Fig., field widths). Values used: " + vals + ". "
        + sig_txt + " The sum of the three "
        "fields during a search display is in Supplementary Fig. (gain field during search).")
    rest = legend[a_end:]
    # signed decimals (gains, differences of gains, their CIs) -> %; q and p values are unsigned and stay
    rest = re.sub(r'([+−-])(\d+\.\d+)(?![\d°])', lambda m: to_pct(m) if m.group(1) != '-' else to_pct(m), rest)
    rest = rest.replace('dS = g_HP − g_LP, the sustained gain', 'dS = g_HP − g_LP (in percentage points), the '
                        'sustained gain')
    return (head + a_txt + rest).replace('−-', '−')


SEARCH_LEGEND = """
**Supplementary Figure S. The fitted gain field during a search display: distractor at the HP location vs at an LP location.** The fitted parametric field, not data, as in Fig. 5a: for each area (V3, hV4, TO), the median over participants (n = {n}) of each participant's joint-fit parameter, put through the model's field equations; value, the fractional change in gain on the stimulus drive. Each map is the sum of the sustained field and the transient target and distractor fields of Fig. 5a, as in the model's forward pass, during a search display with the target (orange diamond) opposite the HP location and the distractor (blue dot) at HP (top row) or at the LP location next to HP (bottom row); only the distractor's location differs between the two rows. Sustained field as in Fig. 5a (median LP gain; HP gain = median LP gain + median joint dS). Distractor gain: at an LP location the median g_LP_dyn, at HP that plus the median joint transient HP − LP difference (dD), the same rule as for the sustained field; target gain, the median g_T_dyn. Transient width set for display to FWHM {fwhm:.1f}° in every area, not interpreted. The joint transient HP − LP difference is not reliable in any area (Fig. 5d legend: smallest q = 0.29), so the transient part of the HP − LP difference drawn here is not an effect. The line under each column gives the gain at the distractor with the distractor at HP minus with the distractor at LP: the sustained HP − LP difference plus the transient HP − LP difference. Colour: magma, one scale per area shared by the two rows (colour bar under each column); teal line and tick, no change. HP at the top (red dashed ring); small circles, the eight item positions on the 4° ring; grey circle, the 3.17° bar aperture. Values: {values}.
"""


def search_figure(p, n, suffix):
    W = DOUBLE
    cw, gap = 1.55, .35
    x_lab = .05
    x0 = 1.45
    H = .32 + 2 * cw + .12 + .62
    fig = plt.figure(figsize=(W, H))
    g = np.linspace(-EXTENT, EXTENT, 401)
    xx, yy = np.meshgrid(g, g)
    values = []
    for c, a in enumerate(AREAS):
        at_hp, at_lp = search_fields(p.loc[a], xx, yy)
        allv = np.concatenate([at_hp.ravel(), at_lp.ravel()])
        lo, hi = np.floor(allv.min() / .1) * .1, np.ceil(allv.max() / .1) * .1
        norm = mpl.colors.Normalize(vmin=min(lo, 0.), vmax=max(hi, .1))
        xa = x0 + c * (cw + gap)
        d_hp, d_lp = value_at(at_hp, xx, yy, RING['hp']), value_at(at_lp, xx, yy, RING['left'])
        for r, (m, key) in enumerate(((at_hp, 'hp'), (at_lp, 'left'))):
            y = H - .32 - (r + 1) * cw - r * .12
            ax = fig.add_axes([xa / W, y / H, cw / W, cw / H])
            ax.imshow(m, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap='magma', norm=norm,
                      interpolation='bilinear')
            if m.min() < 0 < m.max():
                ax.contour(xx, yy, m, levels=[0.], colors=[ZERO], linewidths=.8)
            draw_display(ax, [('target', 'opp'), ('distractor', key)])
            cx, cy = RING[key]
            v = value_at(m, xx, yy, (cx, cy))
            ax.text(cx + (1.3 if key == 'hp' else 0), cy - (0 if key == 'hp' else 1.45), fmt(v), fontsize=6.5,
                    ha='left' if key == 'hp' else 'center', va='center', color=FG, path_effects=STROKE, zorder=7)
            if r == 0:
                ax.set_title(a, fontsize=7.5, pad=3)
        cax = fig.add_axes([(xa + .15) / W, (H - .32 - 2 * cw - .12 - .14) / H, (cw - .3) / W, .055 / H])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='magma'), cax=cax, orientation='horizontal')
        ticks = sorted({round(norm.vmin, 2), 0., round(norm.vmax, 2)})
        cb.set_ticks(ticks)
        cb.set_ticklabels([fmt(t) if t else '0' for t in ticks])
        cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
        cb.ax.axvline(0, color=ZERO, lw=1.2)
        cb.outline.set_linewidth(.4)
        if c == 0:
            cb.ax.text(-.06, .5, 'Gain', transform=cb.ax.transAxes, fontsize=6, ha='right', va='center', color='.35')
        fig.text((xa + cw / 2) / W, (H - .32 - 2 * cw - .12 - .45) / H, f'At the distractor, HP − LP: {fmt(d_hp - d_lp)}', fontsize=6.5,
                 ha='center', va='center', color=FG)
        values.append(f'{a}: at the distractor {fmt(d_hp)} with the distractor at HP, {fmt(d_lp)} at LP (HP − LP '
                      f'{fmt(d_hp - d_lp)}; sustained HP − LP {fmt(p.loc[a, "dS"])}, transient HP − LP {fmt(p.loc[a, "dD"])}); '
                      f'at the target {fmt(value_at(at_hp, xx, yy, RING["opp"]))}')
    for r, (head, sub) in enumerate((('Distractor at HP', 'target opposite HP'),
                                     ('Distractor at LP', 'next to HP; target opposite HP'))):
        yc = H - .32 - r * (cw + .12) - cw / 2
        s = .4
        ax = fig.add_axes([x_lab / W, (yc + .05) / H, s / W, s / H])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.set_axis_off()
        for cc in RING.values():
            ax.add_patch(Circle(cc, 1.1, fill=False, ec=MUTED, lw=.5))
        ax.add_patch(Circle(RING['hp'], 1.8, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2))))
        ax.plot(*RING['opp'], 'D', ms=3.6, color=TARGET, mec='w', mew=.4)
        ax.plot(*RING['hp' if r == 0 else 'left'], 'o', ms=4, color=DISTRACTOR, mec='w', mew=.4)
        ax.plot(0, 0, '+', color=FG, ms=2.5, mew=.5)
        fig.text(x_lab / W, (yc - .1) / H, head, fontsize=7, weight='bold', color=DISTRACTOR, va='center')
        fig.text(x_lab / W, (yc - .26) / H, sub, fontsize=6.5, color='.4', va='center')
    save(fig, 'figS_gain_field_map_search' + suffix,
         SEARCH_LEGEND.format(n=n, fwhm=TR_FWHM, values='; '.join(values)).replace('-0.', '−0.'))
    print('\n'.join(values))


SEARCH_LEGEND_V2 = """
**Supplementary Figure. The fitted gain field during a search display: distractor at the HP location vs at an LP location.** The fitted parametric field, not data, as in Fig. 5a: for each area (V3, hV4, TO), the median over participants (n = {n}) of each participant's fitted parameters, put through the model's field equations; value, the change in gain on the stimulus drive in % (differences between gains in percentage points, written as %). Each map is the sum of the sustained field and the transient target and distractor fields, as in the model's forward pass, during a search display with the target (orange diamond) opposite the HP location and the distractor (blue dot) at HP (top row) or at the LP location next to HP (bottom row); only the distractor's location differs between the two rows. Sustained field: median joint-fit gain at the LP locations; at HP, that plus the area's median HP − LP difference estimated alone (dS), exactly as in Fig. 5a. Distractor gain: at an LP location the median joint-fit LP distractor gain; at HP, that plus the median joint-fit transient HP − LP difference (dD); target gain, the median joint-fit target gain. Transient width set for display to FWHM {fwhm:.1f}° in every area, not interpreted. The transient HP − LP difference is not reliable in any area, jointly fitted or estimated alone (Fig. 5d legend), so the transient part of the HP − LP difference drawn here is not an effect. The line under each column gives the gain at the distractor with the distractor at HP minus with the distractor at LP: the sustained plus the transient HP − LP difference. Colour: magma, one scale per area shared by the two rows (colour bar under each column); teal line and tick, no change. HP at the top (red dashed ring); small circles, the eight item positions on the 4° ring; grey circle, the 3.17° bar aperture. Values: {values}.
"""


def search_figure_v2(p, n, suffix, name='figS_gain_field_map_search_v2', dd=None):
    """dd (v11): (joint fit, estimated-alone table); the sentence on the transient HP − LP difference is computed."""
    """As search_figure, in percent, with the sustained HP gain = median g_LP + median dS estimated alone (Fig. 5 v9)."""
    W = DOUBLE
    cw, gap = 1.55, .35
    x_lab = .05
    x0 = 1.45
    H = .32 + 2 * cw + .12 + .62
    fig = plt.figure(figsize=(W, H))
    g = np.linspace(-EXTENT, EXTENT, 401)
    xx, yy = np.meshgrid(g, g)
    values = []
    for c, a in enumerate(AREAS):
        at_hp, at_lp = (100 * m for m in search_fields(p.loc[a], xx, yy))
        allv = np.concatenate([at_hp.ravel(), at_lp.ravel()])
        lo, hi = np.floor(allv.min() / 10) * 10, np.ceil(allv.max() / 10) * 10
        norm = mpl.colors.Normalize(vmin=min(lo, 0.), vmax=max(hi, 10.))
        xa = x0 + c * (cw + gap)
        d_hp, d_lp = value_at(at_hp, xx, yy, RING['hp']), value_at(at_lp, xx, yy, RING['left'])
        for r, (m, key) in enumerate(((at_hp, 'hp'), (at_lp, 'left'))):
            y = H - .32 - (r + 1) * cw - r * .12
            ax = fig.add_axes([xa / W, y / H, cw / W, cw / H])
            ax.imshow(m, extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), origin='lower', cmap='magma', norm=norm,
                      interpolation='bilinear')
            if m.min() < 0 < m.max():
                ax.contour(xx, yy, m, levels=[0.], colors=[ZERO], linewidths=.8)
            draw_display(ax, [('target', 'opp'), ('distractor', key)])
            cx, cy = RING[key]
            v = value_at(m, xx, yy, (cx, cy))
            ax.text(cx + (1.3 if key == 'hp' else 0), cy - (0 if key == 'hp' else 1.45), pct(v / 100, 0), fontsize=6.5,
                    ha='left' if key == 'hp' else 'center', va='center', color=FG, path_effects=STROKE, zorder=7)
            if r == 0:
                ax.set_title(a, fontsize=7.5, pad=3)
        cax = fig.add_axes([(xa + .15) / W, (H - .32 - 2 * cw - .12 - .14) / H, (cw - .3) / W, .055 / H])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='magma'), cax=cax, orientation='horizontal')
        ticks = sorted({round(norm.vmin), 0., round(norm.vmax)})
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t:+.0f}%'.replace('-', '−') if t else '0' for t in ticks])
        cb.ax.tick_params(labelsize=6, length=1.8, pad=1)
        cb.ax.axvline(0, color=ZERO, lw=1.2)
        cb.outline.set_linewidth(.4)
        if c == 0:
            cb.ax.text(-.06, .5, 'Gain (%)', transform=cb.ax.transAxes, fontsize=6, ha='right', va='center', color='.35')
        fig.text((xa + cw / 2) / W, (H - .32 - 2 * cw - .12 - .45) / H, f'At the distractor, HP − LP: {pct((d_hp - d_lp) / 100)}', fontsize=6.5,
                 ha='center', va='center', color=FG)
        values.append(f'{a}: at the distractor {pct(d_hp / 100)} with the distractor at HP, {pct(d_lp / 100)} at LP '
                      f'(HP − LP {pct((d_hp - d_lp) / 100)}; sustained HP − LP, dS estimated alone, '
                      f'{pct(p.loc[a, "dS_map"])}; transient HP − LP, joint dD, {pct(p.loc[a, "dD"])}); '
                      f'at the target {pct(value_at(at_hp, xx, yy, RING["opp"]) / 100, 0)}')
    for r, (head, sub) in enumerate((('Distractor at HP', 'target opposite HP'),
                                     ('Distractor at LP', 'next to HP; target opposite HP'))):
        yc = H - .32 - r * (cw + .12) - cw / 2
        s = .4
        ax = fig.add_axes([x_lab / W, (yc + .05) / H, s / W, s / H])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.set_axis_off()
        for cc in RING.values():
            ax.add_patch(Circle(cc, 1.1, fill=False, ec=MUTED, lw=.5))
        ax.add_patch(Circle(RING['hp'], 1.8, fill=False, ec=HP, lw=.9, ls=(0, (2, 1.2))))
        ax.plot(*RING['opp'], 'D', ms=3.6, color=TARGET, mec='w', mew=.4)
        ax.plot(*RING['hp' if r == 0 else 'left'], 'o', ms=4, color=DISTRACTOR, mec='w', mew=.4)
        ax.plot(0, 0, '+', color=FG, ms=2.5, mew=.5)
        fig.text(x_lab / W, (yc - .1) / H, head, fontsize=7, weight='bold', color=DISTRACTOR, va='center')
        fig.text(x_lab / W, (yc - .26) / H, sub, fontsize=6.5, color='.4', va='center')
    leg = SEARCH_LEGEND_V2.format(n=n, fwhm=TR_FWHM, values='; '.join(values))
    if dd is not None:
        qj, qa = q_by_area(dd[0], 'dD'), q_by_area(dd[1], 'dD')
        sj = [f'{r} (q = {qj[r]:.3f})' for r in ROI_ORDER if qj[r] < .05]
        sa = [f'{r} (q = {qa[r]:.3f})' for r in ROI_ORDER if qa[r] < .05]
        txt = ('The transient HP − LP difference (dD) is reliable in ' + (and_list(sj) if sj else 'no area')
               + ' fitted jointly and in ' + (and_list(sa) if sa else 'no area') + ' estimated alone (Fig. 5d legend)'
               + ('' if any(r in AREAS for r in [x.split()[0] for x in sj]) else
                  '; in the three areas drawn it is not reliable, so the transient part of the HP − LP difference '
                  'drawn here is not an effect') + '.')
        old = ('The transient HP − LP difference is not reliable in any area, jointly fitted or estimated alone '
               '(Fig. 5d legend), so the transient part of the HP − LP difference drawn here is not an effect.')
        assert old in leg
        leg = leg.replace(old, txt.replace('= 0.', '= .')).replace(
            "exactly as in Fig. 5a.", "exactly as in Fig. 5a, with the width of Fig. 5a.")
    save(fig, name + suffix, leg.replace('-', '−'))
    print('\n'.join(values))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters_final2_2026-11.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_final2_2026-11.tsv')
    ap.add_argument('--slope-table', default='notes/data/hierarchy_regression_final2_2026-11.tsv')
    ap.add_argument('--sigma-table', default='notes/data/fig1_prf_sigma.tsv')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--sustained-scale', choices=['auto', 'shared', 'per-area'], default='auto')
    ap.add_argument('--compact', action='store_true', help='v7: panel a in two rows (sustained; target + distractor '
                    'transient fields in one map), smaller panels; writes fig5_gain_fields_v7')
    ap.add_argument('--layout', choices=['v6', 'v7', 'v8', 'v9', 'v10', 'v11'], default=None, help='v8: panel a as one row of six maps '
                    'over the full width, keys under the blocks, sustained HP − LP = dS estimated alone (panel c); '
                    'writes fig5_gain_fields_v8 and the parameter table with suffix _v8')
    ap.add_argument('--widths-table', default=None,
                    help='v11: field_widths_reliable ..._widths_thr2.tsv; the sustained width of each map is the median '
                         'among fits with a reliable sustained gain')
    ap.add_argument('--fit-label', default=None, help='v11: how the legend names the fits (first sentence)')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.slope_table, a.sigma_table, a.suffix,
         None if a.sustained_scale == 'auto' else a.sustained_scale == 'per-area', a.compact, a.layout,
         a.widths_table, a.fit_label)
