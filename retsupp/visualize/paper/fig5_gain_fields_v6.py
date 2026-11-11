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


def map_parameters(gain_table, exclude=(1, 2)):
    """Per area, the median over participants of each participant's joint-fit parameter (DoG, m7_full)."""
    d = joint(gain_table, 'dog')
    d = d[~d.subject.isin(exclude)]
    rows = []
    for a in ROI_ORDER:
        x = d[d.roi == a]
        rows.append(dict(roi=a, n=len(x), g_HP=x.g_HP.median(), g_LP=x.g_LP.median(), dS=x.dS.median(),
                         T=x['T'].median(), D=x.D.median(), g_HP_dyn=x.g_HP_dyn.median(),
                         g_LP_dyn=x.g_LP_dyn.median(), dD=x.dD.median(), sigma_AF=x.sigma_AF.median()))
    p = pd.DataFrame(rows).set_index('roi')
    p['g_HP_map'] = p.g_LP + p.dS          # the HP gain drawn: HP - LP on the map = the joint dS
    p['fwhm_AF'] = FWHM * p.sigma_AF
    return p


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


def main(gain_table, conditional_table, slope_table, sigma_table, suffix, per_area_sustained=None):
    # the same inputs and options as fig5_gain_fields_n28 (final fits, b-d unchanged in content)
    F.FINAL, F.SUFFIX, F.WIDTHS_SUPP = True, suffix, True
    F.GAIN_TABLE = gain_table
    js = {f: joint(gain_table, f) for f in F.FAM if f in pd.read_csv(gain_table, sep='\t', usecols=['family']).family.unique()}
    cond = conditional(conditional_table)
    t = F.summarise(js, cond)
    dt = cond[['subject', 'roi', 'dT']].dropna()
    d = js['dog']
    n = d.subject.nunique()
    p = map_parameters(gain_table)

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
    print('figure height (in):', round(H, 2))
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    Y = lambda dd: H - dd

    # a: the maps
    fig.text(.02 / W, Y(.14) / H, 'a', fontsize=8, fontweight='bold', va='center', ha='left')
    fig.text(.2 / W, Y(.14) / H, 'The fitted gain fields over the display (group medians)', fontsize=7.5, color=FG,
             va='center', ha='left')
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
    ax = at(LX, yb - 1.66, 6.45, 1.18)
    sm = lambda term: t[(t.model == 'DoG') & (t.term == term)].set_index('roi').loc[ROI_ORDER]
    F.dot_panel(ax, [(d, 'T', TARGET, sm('T')), (d, 'D', DISTRACTOR, sm('D'))], (-.4, 1.2), 'Gain')
    ax.set_yticks([0, .5, 1])

    # c, d: as before
    y2 = H - (yb - 1.66 - .55)
    hp_lp = [lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('hp',))),
             lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('lp',)))]
    row(fig, Y(y2), 'c', [('Sustained gain: less at ', FG), ('HP', HP)], [(hp_lp, HP, 'Whole block, at HP − at LP')],
        x_let=.02, x_title=.2, key_y=Y(y2 + .27))
    ax = at(LX, Y(y2 + .62 + 1.0), PW, 1.0)
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
    ax = at(RX, Y(y2 + .62 + 1.0), PW, 1.0)
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
    vals = '; '.join(
        f"{a}: sustained gain at LP {fmt(p.loc[a, 'g_LP'])}, at HP {fmt(p.loc[a, 'g_HP_map'])} "
        f"(HP − LP = joint dS {fmt(p.loc[a, 'dS'])}; median g_HP {fmt(p.loc[a, 'g_HP'])}), width σ "
        f"{p.loc[a, 'sigma_AF']:.2f}° (FWHM {p.loc[a, 'fwhm_AF']:.1f}°); target gain {fmt(p.loc[a, 'T'])}; "
        f"distractor gain {fmt(p.loc[a, 'D'])}" for a in AREAS)
    neg = [r for r in ROI_ORDER if p.loc[r, 'g_LP'] < 0 and p.loc[r, 'g_HP_map'] < 0]
    a_txt = (
        "a, The fitted parametric gain field over the search display, not data: for each area, the median over "
        f"participants (n = {n}) of each participant's joint-fit parameter (all terms fitted together), put through the "
        "model's field equations; value, the fractional change in gain on the stimulus drive (−0.10: 10% less). The "
        "display is rotated so that the HP location is at the top (red dashed ring); small circles, the eight item "
        "positions on the 4° ring (dotted; stronger: the four possible target and distractor locations); grey circle, "
        "the 3.17° aperture of the mapping bar; +, fixation. Top row, the sustained field, present throughout the block: "
        "a Gaussian field at each of the four possible locations, with the median gain at the low-probability (LP) "
        "locations and, at HP, that gain plus the median HP − LP difference of the joint fit, so that the HP − LP "
        "difference printed on each map is the joint dS of panel c's legend and the text (panel c shows dS estimated "
        "alone, which is smaller); width, the area's median fitted width. The sustained gain is negative at all four "
        f"locations in most areas ({len(neg)} of 9: {', '.join(neg)}). Middle row, the target's transient field alone "
        "(target, orange diamond, at an LP location, here opposite HP); value, the target gain of panel b. Bottom row, "
        "the distractor's transient field alone (distractor, blue dot, drawn at an LP location next to HP); value, the "
        "distractor gain averaged over its locations, as in panel b (the transient HP − LP difference is in panel d). "
        "The transient fields exist only while a search display is on; their fitted width is often at a bound of the "
        f"fit (0.5° or 8°), so it is set for display to the same FWHM of {TR_FWHM:.1f}° in every area and is not "
        "interpreted (field widths per subject: Supplementary Fig. S, field widths). Colour scales: "
        + '; '.join(scales) + ' (magma for the sustained field, teal line and tick at no change; the transient rows '
        'from white, no change, to the component colour). Values used: ' + vals + '. V3, hV4 and TO are the areas '
        'where the sustained gain is reliably lower at HP than at LP in the joint fit (q < .05; V1, V2 and the other '
        'areas are in the parameter table). The sum of the three fields during a search display is in Supplementary '
        'Fig. S (gain field during search).')
    general, rest = rest.split(' b–d: ', 1)          # 'Gain-field model fits ... (+0.30: 30% more).' leads
    legend = head + ' ' + general.strip() + ' ' + a_txt + ' b–d: ' + rest
    legend = legend.replace('-0.', '−0.')
    save(fig, 'fig5_gain_fields_v6' + suffix, legend)
    print('scales:', scales, 'per-area sustained:', per_area_sustained)
    print(p.loc[AREAS].round(4).to_string())
    search_figure(p, n, suffix)
    p.to_csv(f'notes/data/fig5_gain_field_map_parameters{suffix}_2026-11-12.tsv', sep='\t', float_format='%.5f')
    return p


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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters_final2_2026-11.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_final2_2026-11.tsv')
    ap.add_argument('--slope-table', default='notes/data/hierarchy_regression_final2_2026-11.tsv')
    ap.add_argument('--sigma-table', default='notes/data/fig1_prf_sigma.tsv')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--sustained-scale', choices=['auto', 'shared', 'per-area'], default='auto')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.slope_table, a.sigma_table, a.suffix,
         None if a.sustained_scale == 'auto' else a.sustained_scale == 'per-area')
