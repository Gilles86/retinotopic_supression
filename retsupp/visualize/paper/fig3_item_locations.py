"""Figure 3: the measured responses at the item locations, without a model.

Voxels whose pRF centre lies within 2° of an item location, deconvolved (nideconv) around search onset
(``deconvolve_location_responses.py``; read by ``fig45_common.item_timecourses``).

Row 1, what the item is: the target and the distractor evoke more response than a neutral item (a raw, b, c differences
per pathway). Row 2, where the item is relative to the run's HP location (HP, next to HP, opposite;
``deconvolve_location_responses.py --relative``): d each location type minus the mean of the four, per pathway; e HP − next
and next − opposite per item type, with the location type × item ANOVA and planned contrasts (``rel_stats``). The HP vs LP
ANOVA of ``fig_hp_heatmap`` (earlier deconvolution) stays in the legend.

The model's gain fields are Figure 5 (``fig5_gain_fields.py``).

Usage::

    python -m retsupp.visualize.paper.fig3_item_locations                     # Fourier basis (main)
    python -m retsupp.visualize.paper.fig3_item_locations --stats-only \\
        --deconv-folder notes/data/event_deconv_dct_2026-11                   # DCT basis: statistics only
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig2_prf_fits import PATHWAY, PATHWAY_COL
from retsupp.visualize.paper.fig45_common import item_timecourses
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, TARGET, letter, offset_spines, save

NEUTRAL = '#7B8794'
LP_GREY = '#7B8794'
PRF_TEAL = '#2A9D8F'
PEAK = (3.2, 6.4)   # s after search onset: the window tested
SHORT = {'Early': 'Early', 'Dorsal': 'Dorsal', 'Occipitotemporal': 'Occ.-temporal', 'All areas': 'All areas'}
ITEMS = [('target', 'Target', TARGET), ('distractor', 'Distractor', DISTRACTOR), ('neutral', 'Neutral item', NEUTRAL)]
# Where the item lies relative to the run's HP location; colours and order as Fig. 6d, g.
LOC_COLORS = {'hp': HP, 'next': '#52606D', 'opposite': '#9AA5B1'}
LOC_TYPES = [('hp', 'HP'), ('next', 'Next to HP'), ('opposite', 'Opposite')]
REL_FOLDER = 'notes/data/event_deconv_relative_2026-11'
REL_DCT_FOLDER = 'notes/data/event_deconv_relative_dct_2026-11'
ROW2_TITLE = 'Where the item is: in early areas, least response at HP, less next to it than opposite it, for every item type'

LEGEND = """
**Figure 3. At the item locations, without a model: more response to the target and the distractor, and, in early areas, least response at the high-probability (HP) location, less next to it than opposite it, for every item.** All panels: deconvolved response (nideconv, Fourier basis) around search onset of voxels whose pRF centre lies within 2° of an item location (inset in a: the display with one such pRF on an item, and the response it gives), ROIs pooled weighted by voxel count, each subject's pre-onset mean (−4 to 0 s) subtracted; mean over subjects ± SEM (n = 30). Pathways: early, V1–V3; dorsal, V3AB and IPS; occipitotemporal, hV4, VO, LO and TO. Grey band: the window tested, {w0:.1f}–{w1:.1f} s. a, Response at a pRF's item location when the target, the distractor or a neutral item is there, all at low-probability (LP) locations, all nine ROIs pooled. b, Target minus neutral item per pathway (both averaged over HP and LP locations). c, Distractor minus neutral item per pathway (both at LP locations). Stars in b, c: the mean over the window against 0, Wilcoxon signed-rank test, Benjamini-Hochberg over the three pathways per panel; {diff_tests}. d, e, Where the item lies relative to the run's HP location (HP rotates between runs, so each location is labelled per run): HP (red), next to HP (the two neighbouring LP locations, left and right seen from fixation facing HP, pooled; dark grey) or opposite HP (light grey), colours and order as Fig. 6d, g; deconvolved with the item type × location type as twelve event types. d, Each location type minus the mean of the four locations, item types pooled, per pathway. e, Each location type relative to the mean of the four locations over the window, item types pooled, in % of each subject's mean response over the four locations and three item types in that pathway; mean ± within-subject SEM (dorsal pathway in b–e: n = {n_dor}, two subjects lack dorsal voxels at some item location). Brackets: planned contrasts (HP vs next to HP, next to HP vs opposite, HP vs opposite) on the mean over item types, Wilcoxon signed-rank test, uncorrected, drawn where p < .05 (* p < .05). The item types do not differ in their location effect (location type × item type below), so they are pooled. Per item type (target, distractor, neutral), HP − next and next − opposite: {items}. Per pathway, repeated-measures ANOVA location type (HP, next, opposite) × item type (target, distractor, neutral), with the planned contrasts (mean ± SEM over subjects): {rel}. HP against all three LP locations (the earlier HP vs LP deconvolution, two event types per item; ANOVA location (HP, LP) × item type): main effect of location {hplp}; location × item {inter}. DCT basis (robustness), b, c and HP vs LP: {dct} Location types, DCT basis: {rel_dct}.
"""


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def fmt_p(p):
    return 'p < .001' if p < .001 else f'p = {p:.3f}'.replace('0.', '.')


def item_pair(p, term):
    """Per subject and time: the item's response and the neutral item's, matched as in the difference."""
    w = p.pivot_table(index=['subject', 'grp', 't'], columns='cls', values='r')
    if term == 'T':   # same mix of HP and LP locations for the target and the neutral item
        return (w.target_hp + w.target_lp) / 2, (w.neutral_hp + w.neutral_lp) / 2
    return w.distractor_lp, w.neutral_lp   # the distractor at LP locations, against the neutral item at LP locations


def diff_tests(pw, term):
    """Item minus neutral item per pathway: per-subject time courses, window means, Wilcoxon p, BH q."""
    item, neutral = item_pair(pw, term)
    x = (item - neutral).rename('r').reset_index()
    out = {}
    for grp in PATHWAY:
        g = x[x.grp == grp].pivot(index='subject', columns='t', values='r')
        v = g.loc[:, (g.columns >= PEAK[0]) & (g.columns <= PEAK[1])].mean(1)
        out[grp] = dict(tc=g, mean=v.mean(), p=stats.wilcoxon(v).pvalue, n=len(v))
    for grp, q in zip(PATHWAY, stats.false_discovery_control([out[g]['p'] for g in PATHWAY])):
        out[grp]['q'] = q
    return out


def hp_lp(pw):
    """Per pathway: time courses at an item location when it is HP vs LP (item types pooled), and the window test."""
    w = pw.pivot_table(index=['grp', 'subject', 't'], columns='cls', values='r')
    x = pd.DataFrame({'hp': w[['target_hp', 'distractor_hp', 'neutral_hp']].mean(1),
                      'lp': w[['target_lp', 'distractor_lp', 'neutral_lp']].mean(1)})
    res = {}
    for g in PATHWAY:
        hp, lp = x.loc[g].hp.unstack('t'), x.loc[g].lp.unstack('t')
        inw = (hp.columns >= PEAK[0]) & (hp.columns <= PEAK[1])
        v = (hp - lp).loc[:, inw].mean(1).dropna()
        res[g] = dict(hp=hp, lp=lp, diff=v.mean(), rel=100 * v.mean() / lp.loc[v.index, inw].mean(1).mean(),
                      p=stats.wilcoxon(v).pvalue, n=len(v))
    return res


def all_stats(folder):
    from retsupp.visualize.paper.fig_hp_heatmap import hp_tests
    pw = item_timecourses(folder, groups={r: g for g, rs in PATHWAY.items() for r in rs})
    dt, dd = diff_tests(pw, 'T'), diff_tests(pw, 'D')
    hl = hp_lp(pw)
    diff, res = hp_tests(folder)
    return pw, dt, dd, hl, diff, res


def rel_stats(folder):
    """Location type relative to the run's HP location (``deconvolve_location_responses.py --relative``).

    Per subject and group (pathways and all areas): the window means per item × location (HP, next left, next right,
    opposite) in % of the subject's mean over all twelve; next = mean of left and right. Returns (tcs, cells, res):
    tcs, per subject, group and time, each location type minus the mean of the four locations (items pooled, % signal);
    cells, one row per subject, group, item and location type; res, one row per group with the repeated-measures
    ANOVA location type (HP, next, opposite) × item, the planned contrasts (Wilcoxon on the mean over items), left vs
    right, and the HP vs all LP main effect (ANOVA location (HP, LP) × item, LP = mean of the three LP locations)."""
    import pingouin as pg
    groups = {r: g for g, rs in PATHWAY.items() for r in rs}
    p = pd.concat([item_timecourses(folder, groups), item_timecourses(folder, None).assign(grp='All areas')])
    w = p.pivot_table(index=['subject', 'grp', 't'], columns='cls', values='r')
    loc = {c: w[[f'{it}_{c}' for it, _, _ in ITEMS]].mean(1) for c in ('hp', 'nextleft', 'nextright', 'opposite')}
    m4 = sum(loc.values()) / 4
    tcs = pd.DataFrame({'hp': loc['hp'] - m4, 'next': (loc['nextleft'] + loc['nextright']) / 2 - m4,
                        'opposite': loc['opposite'] - m4}).reset_index()

    pk = p[p.t.between(*PEAK)].groupby(['subject', 'grp', 'cls']).r.mean().unstack().dropna()
    pk = pk.div(pk.mean(1), axis=0) * 100
    cells = pk.stack().rename('r').reset_index()
    cells[['item', 'loc']] = cells.cls.str.split('_', expand=True)
    cells['loc3'] = cells['loc'].replace({'nextleft': 'next', 'nextright': 'next'})
    fa = lambda a, src: (float(a.loc[a.Source == src, 'F'].iloc[0]), float(a.loc[a.Source == src, 'p-unc'].iloc[0]),
                         int(a.loc[a.Source == src, 'ddof1'].iloc[0]), int(a.loc[a.Source == src, 'ddof2'].iloc[0]))
    res = []
    for g in list(PATHWAY) + ['All areas']:
        d = cells[cells.grp == g]
        d3 = d.groupby(['subject', 'item', 'loc3']).r.mean().reset_index()
        a3 = pg.rm_anova(data=d3, dv='r', within=['loc3', 'item'], subject='subject')
        d2 = d.assign(l2=np.where(d['loc'] == 'hp', 'hp', 'lp')).groupby(['subject', 'item', 'l2']).r.mean().reset_index()
        a2 = pg.rm_anova(data=d2, dv='r', within=['l2', 'item'], subject='subject')
        x = d.pivot_table(index='subject', columns='loc', values='r')   # mean over items
        nxt = (x.nextleft + x.nextright) / 2
        row = dict(grp=g, n=len(x))
        for k, src in (('loc', 'loc3'), ('int', 'loc3 * item')):
            row[f'{k}_F'], row[f'{k}_p'], row[f'{k}_df1'], row[f'{k}_df2'] = fa(a3, src)
        row['lp_F'], row['lp_p'], row['lp_df1'], row['lp_df2'] = fa(a2, 'l2')
        for k, v in (('hp_next', x.hp - nxt), ('next_opp', nxt - x.opposite), ('hp_opp', x.hp - x.opposite),
                     ('left_right', x.nextleft - x.nextright),
                     ('hp_lp', x.hp - (x.nextleft + x.nextright + x.opposite) / 3)):
            row[k], row[f'{k}_se'], row[f'{k}_p'] = v.mean(), v.sem(), stats.wilcoxon(v).pvalue
        res.append(row)
    return tcs, cells, pd.DataFrame(res).set_index('grp')


def rel_report(folder, res):
    print(f'--- location types, {folder}')
    cols = ['n', 'loc_F', 'loc_p', 'int_F', 'int_p', 'lp_F', 'lp_p', 'hp_next', 'hp_next_p', 'next_opp', 'next_opp_p',
            'hp_opp', 'hp_opp_p', 'left_right', 'left_right_p']
    print(res[cols].round(3).to_string())


def item_text(cells):
    """Per group and item type: HP − next and next − opposite (% of the mean), Wilcoxon p, for the legend."""
    out = []
    for g in list(PATHWAY) + ['All areas']:
        w = cells[cells.grp == g].pivot_table(index=['subject', 'item'], columns='loc3', values='r').unstack('item')
        parts = []
        for it, lab, _ in ITEMS:
            a, b = w['hp'][it] - w['next'][it], w['next'][it] - w['opposite'][it]
            parts.append(f"{lab.lower().replace(' item', '')} {a.mean():+.1f}%, {fmt_p(stats.wilcoxon(a).pvalue)} and "
                         f"{b.mean():+.1f}%, {fmt_p(stats.wilcoxon(b).pvalue)}")
        out.append(f"{g.lower()}: " + ', '.join(parts))
    return '; '.join(out)


def rel_text(res, groups=None):
    """Per group: the ANOVA and the planned contrasts, for the legend."""
    out = []
    for g, r in res.loc[groups or list(res.index)].iterrows():
        out.append(f"{g.lower()}: location type F({int(r.loc_df1)}, {int(r.loc_df2)}) = {r.loc_F:.2f}, {fmt_p(r.loc_p)}, "
                   f"location type × item F({int(r.int_df1)}, {int(r.int_df2)}) = {r.int_F:.2f}, {fmt_p(r.int_p)}; "
                   f"HP − next {r.hp_next:+.1f} ± {r.hp_next_se:.1f}%, {fmt_p(r.hp_next_p)}; "
                   f"next − opposite {r.next_opp:+.1f} ± {r.next_opp_se:.1f}%, {fmt_p(r.next_opp_p)}; "
                   f"HP − opposite {r.hp_opp:+.1f}%, {fmt_p(r.hp_opp_p)}; "
                   f"next left − next right {r.left_right:+.1f}%, {fmt_p(r.left_right_p)}")
    return '; '.join(out)


def report(folder, dt, dd, hl, res):
    print(f'--- {folder}')
    for lab, d in (('Target - neutral', dt), ('Distractor - neutral', dd)):
        for g, r in d.items():
            print(f"{lab}, {g}: {r['mean']:+.3f} % signal, p = {r['p']:.2g}, q = {r['q']:.2g}, n = {r['n']}")
    for g, r in hl.items():
        print(f"HP - LP, {g}: {r['diff']:+.3f} % signal ({r['rel']:+.1f}% of LP), p = {r['p']:.3f}, n = {r['n']}")
    print(res.round(4).to_string())


def short_stats(dt, dd, hl, res):
    """One line of the key statistics, for the legend (used for the DCT basis)."""
    s = [f"target − neutral " + ', '.join(f"{g.lower()} {fmt_p(dt[g]['p'])}" for g in PATHWAY),
         f"distractor − neutral " + ', '.join(f"{g.lower()} {fmt_p(dd[g]['p'])}" for g in PATHWAY),
         'HP − LP ' + ', '.join(f"{g.lower()} {hl[g]['rel']:+.0f}%, {fmt_p(hl[g]['p'])}" for g in PATHWAY),
         'main effect of location ' + ', '.join(f"{g.lower()} {fmt_p(r.main_p)}" for g, r in res.iterrows()),
         'location × item ' + ', '.join(f"{g.lower()} {fmt_p(r.int_p)}" for g, r in res.iterrows())]
    return '; '.join(s) + '.'


# ---------------------------------------------------------------- panels

def icon(fig, x0, y0, w):
    """The display (eight items, 4° ring) with one pRF on an item (pRF centre within 2° of it), and the response it gives."""
    from matplotlib.patches import Circle, FancyArrowPatch
    W, H = fig.get_size_inches()
    xr, yr = (-5.3, 12.6), (-5.3, 5.3)
    h = w * (yr[1] - yr[0]) / (xr[1] - xr[0])
    ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    ax.add_patch(Circle((0, 0), 3.17, fill=False, ec=MUTED, lw=.4))
    a = np.deg2rad(45)
    item = 4 * np.array([np.cos(a), np.sin(a)])
    for k in range(8):
        b = np.deg2rad(k * 45)
        ax.add_patch(Circle((4 * np.cos(b), 4 * np.sin(b)), .6, facecolor='w', ec=FG, lw=.5, zorder=3))
    ax.add_patch(Circle(item, 2., fill=False, ec='.45', lw=.5, ls=(0, (1.5, 1.2)), zorder=2))
    prf = item + np.array([-.6, -.5])
    ax.add_patch(Circle(prf, 1.25, facecolor=PRF_TEAL, ec='none', alpha=.45, zorder=4))
    ax.plot(*prf, 'o', ms=1.6, color=PRF_TEAL, zorder=5)
    ax.plot(0, 0, '+', color=FG, ms=3, mew=.5)
    # The response: a small time course right of the display.
    t = np.linspace(0, 1, 60)
    r = (t / .3) ** 2 * np.exp(-(t - .3) / .09 * (t > .3)) * (t <= .3) + np.exp(-((t - .3) / .2) ** 2) * (t > .3)
    r = r - .25 * np.exp(-((t - .62) / .14) ** 2)
    xs, ys = 7.2 + 5.2 * t, -.6 + 3.8 * r
    ax.plot([7.2, 12.4], [-.6, -.6], color=MUTED, lw=.5)
    ax.plot([7.2, 7.2], [-1.4, 3.6], color=MUTED, lw=.5)
    ax.plot(xs, ys, color=PRF_TEAL, lw=1.1)
    ax.add_patch(FancyArrowPatch(prf + np.array([1.1, .5]), (7.7, 2.6), arrowstyle='-|>', mutation_scale=5, color=FG, lw=.6,
                                 connectionstyle='arc3,rad=-.3', shrinkA=1, shrinkB=2, zorder=6))
    ax.text(9.8, -1.5, 'Time', fontsize=5.5, color='.4', ha='center', va='top')
    ax.text(2.8, -4.6, 'pRF within 2°\nof an item', fontsize=5.5, color=PRF_TEAL, ha='center', va='top', linespacing=1.)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return ax


def tc(ax, m, se, col, lw=1.3, alpha=1.):
    ax.fill_between(m.index, m - se, m + se, color=col, alpha=.18 * alpha, lw=0)
    ax.plot(m.index, m.values, color=col, lw=lw, alpha=alpha)


def time_axis(ax, xlabel=True):
    ax.axvspan(*PEAK, color='.93', lw=0, zorder=0)
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.set_xlim(-4, 20)
    ax.set_xticks([0, 10, 20])
    if xlabel:
        ax.set_xlabel('Time from search onset (s)')
    offset_spines(ax)


def panel_raw(ax, pa):
    """All nine ROIs pooled: target, distractor and neutral item, each at LP locations."""
    w = pa.pivot_table(index=['subject', 't'], columns='cls', values='r')
    peaks = {}
    for cls, lab, col in ITEMS[::-1]:
        q = w[f'{cls}_lp'].unstack('t')
        tc(ax, q.mean(), q.sem(), col)
        peaks[lab] = q.mean().max()
    for k, (cls, lab, col) in enumerate(ITEMS):
        ax.text(1., .97 - k * .12, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
    time_axis(ax)
    ax.set_ylim(-.3, 1.1)
    ax.set_yticks([0, .5, 1])
    ax.set_ylabel('Response (% signal)')


def panel_diff(ax, d, ylabel=True):
    for k, grp in enumerate(PATHWAY):
        g = d[grp]['tc']
        tc(ax, g.mean(), g.sem(), PATHWAY_COL[grp], lw=1.4)
        ax.text(.9, .99 - k * .12, SHORT[grp], transform=ax.transAxes, color=PATHWAY_COL[grp], fontsize=6.5, ha='right', va='top')
        ax.text(.91, 1.005 - k * .12, stars(d[grp]['q']), transform=ax.transAxes, color=PATHWAY_COL[grp], fontsize=8.5, ha='left',
                va='top')
    time_axis(ax)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_ylim(-.1, .38)
    ax.set_yticks([-.1, 0, .1, .2, .3])
    if ylabel:
        ax.set_ylabel('Extra response (% signal)')
    else:
        ax.set_yticklabels([])


def loc_icon(ax, bounds, shade, transform=None):
    """The four item locations with HP rotated to the top and ringed in red; the location type(s) in ``shade`` filled
    in their colour (as Fig. 6d)."""
    from matplotlib.patches import Circle
    ins = ax.inset_axes(list(bounds), transform=transform or ax.transAxes)
    which = {90: 'hp', 180: 'next', 0: 'next', 270: 'opposite'}
    ins.add_patch(Circle((0, 0), 3.17, fill=False, ec='.6', lw=.5))
    for a, c in which.items():
        q = 4. * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ins.add_patch(Circle(q, 1.05, facecolor=LOC_COLORS[c] if c in shade else 'w',
                             ec=HP if c == 'hp' else '.55', lw=.9 if c == 'hp' else .5, zorder=3))
    ins.plot(0, 0, '+', color=FG, ms=2.5, mew=.5)
    ins.set_xlim(-5.3, 5.3)
    ins.set_ylim(-5.3, 5.3)
    ins.set_aspect('equal')
    ins.set_axis_off()
    return ins


def panel_loc_tc(ax, t, name, ylim=(-.125, .1)):
    """Each location type minus the mean of the four locations, item types pooled."""
    t = t[t.grp == name]
    for c, lab in LOC_TYPES[::-1]:
        x = t.pivot(index='subject', columns='t', values=c)
        tc(ax, x.mean(), x.sem(), LOC_COLORS[c], lw=1.4 if c == 'hp' else 1.1)
    time_axis(ax, xlabel=False)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_ylim(*ylim)
    ax.set_yticks([-.1, -.05, 0, .05, .1])
    ax.set_yticklabels(['−.1', '', '0', '', '.1'])
    ax.set_title(SHORT[name], fontsize=6.5, color=FG, pad=2)


def panel_loc_types(ax, cells, res):
    """e: per pathway, each location type relative to the mean of the four locations (% of the mean response), item
    types pooled; mean ± within-subject SEM; brackets where a planned contrast has p < .05 (Fourier)."""
    import matplotlib.transforms as mtrans
    order = list(PATHWAY) + ['All areas']
    off = {'hp': -.26, 'next': 0., 'opposite': .26}
    pairs = [('hp_next', 'hp', 'next'), ('next_opp', 'next', 'opposite'), ('hp_opp', 'hp', 'opposite')]
    tops, lows = [], []
    for i, g in enumerate(order):
        x0 = i + (.4 if g == 'All areas' else 0)
        x = cells[cells.grp == g].pivot_table(index='subject', columns='loc', values='r')   # mean over item types
        m4 = x[['hp', 'nextleft', 'nextright', 'opposite']].mean(1)
        v = pd.DataFrame({'hp': x.hp - m4, 'next': (x.nextleft + x.nextright) / 2 - m4, 'opposite': x.opposite - m4})
        k = v.shape[1]
        norm = v.sub(v.mean(1), axis=0)
        se = norm.std(ddof=1) / np.sqrt(len(v)) * np.sqrt(k / (k - 1))
        for c, _ in LOC_TYPES:
            ax.errorbar(x0 + off[c], v[c].mean(), yerr=se[c], fmt='o', ms=4.5, color=LOC_COLORS[c], mec='w', mew=.6,
                        lw=1.2, capsize=0, zorder=3)
        top = (v.mean() + se).max()
        lows.append((v.mean() - se).min())
        level = 0
        for key, a_, b_ in pairs:
            if res.loc[g, f'{key}_p'] < .05:
                y = top + 1.2 + level * 2.6
                xa, xb = x0 + off[a_], x0 + off[b_]
                ax.plot([xa, xa, xb, xb], [y - .6, y, y, y - .6], color=FG, lw=.6)
                ax.text((xa + xb) / 2, y - .35, stars(res.loc[g, f'{key}_p']), ha='center', va='bottom', fontsize=7,
                        color=FG)
                level += 1
        tops.append(top + 1.2 + level * 2.6)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.axvline(len(order) - 1.5 + .2, color='.85', lw=.6, zorder=0)
    ax.set_xticks([i + (.4 if g == 'All areas' else 0) for i, g in enumerate(order)])
    ax.set_xticklabels([SHORT[g] for g in order], fontsize=6.5)
    ax.set_xlim(-.5, len(order) - .1)
    lo, hi = np.floor(min(lows)) - 1, max(tops) + .5
    ax.set_ylim(lo, hi)
    ticks = [t for t in range(-30, 31, 10) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t}'.replace('-', '−') for t in ticks])
    ax.spines['left'].set_bounds(lo, max(ticks))
    ax.set_ylabel('Response vs mean of\nthe four locations (%)')
    offset_spines(ax)
    # Key above the panel: the display icons for the three location types (HP ringed red at the top, the type filled).
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    h = .27
    w_ = h * bb.height / bb.width
    for j, (c, lab) in enumerate(LOC_TYPES):
        x = .06 + j * .33
        loc_icon(ax, (x, 1.03, w_, h), {c})
        ax.text(x + w_ + .025, 1.03 + h / 2, lab, transform=ax.transAxes, fontsize=6.5,
                color=LOC_COLORS[c] if c != 'opposite' else '.5', va='center', ha='left')
    return max(tops)


def header(fig, y, text, x0=.012, x1=.99):
    fig.text(x0, y, text, fontsize=7.5, fontweight='bold', color=FG, va='bottom', ha='left')
    fig.add_artist(plt.Line2D([x0, x1], [y - .006] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


def main(folder, name='fig3_item_locations', stats_only=False, dct_folder='notes/data/event_deconv_dct_2026-11',
         rel_folder=REL_FOLDER, rel_dct_folder=REL_DCT_FOLDER):
    pw, dt, dd, hl, diff, res = all_stats(folder)
    report(folder, dt, dd, hl, res)
    if stats_only:
        return dt, dd, hl, res
    _, dt2, dd2, hl2, _, res2 = all_stats(dct_folder)
    report(dct_folder, dt2, dd2, hl2, res2)
    pa = item_timecourses(folder, groups=None)
    tcs, cells, rres = rel_stats(rel_folder)
    rel_report(rel_folder, rres)
    _, _, rres2 = rel_stats(rel_dct_folder)
    rel_report(rel_dct_folder, rres2)

    W, H = DOUBLE, 4.55
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    header(fig, 4.3 / H, 'What the item is: more response to the target and the distractor than to a neutral item')
    icon(fig, .06, 3.05, 1.18)
    ax = at(1.72, 2.62, 1.5, 1.28)
    panel_raw(ax, pa)
    ax.text(0, 1.06, 'Response at an item location', transform=ax.transAxes, fontsize=7, color=FG, va='bottom')
    fig.text(.012, (2.62 + 1.28 * 1.06) / H, 'a', fontsize=8, fontweight='bold', va='bottom', ha='left')
    ax = at(3.75, 2.62, 1.5, 1.28)
    panel_diff(ax, dt)
    ax.text(0, 1.06, 'Target − neutral item', transform=ax.transAxes, fontsize=7, color=TARGET, va='bottom')
    letter(ax, 'b', x=-.3, y=1.06)
    ax = at(5.45, 2.62, 1.5, 1.28)
    panel_diff(ax, dd, ylabel=False)
    ax.text(0, 1.06, 'Distractor − neutral item', transform=ax.transAxes, fontsize=7, color=DISTRACTOR, va='bottom')
    letter(ax, 'c', x=-.08, y=1.06)

    header(fig, 1.98 / H, ROW2_TITLE)
    for k, g in enumerate(PATHWAY):
        ax = at(.66 + k * .98, .5, .9, .98)
        panel_loc_tc(ax, tcs, g)
        if k == 0:
            ax.set_ylabel('Δ vs mean of the four\nlocations (% signal)')
            letter(ax, 'd', x=-.5, y=1.27)
            # Key for d, as glyphs, in the row above the panels.
            for j, (c, lab) in enumerate(LOC_TYPES):
                x = .1 + j * .95
                ax.plot([x, x + .14], [1.27, 1.27], transform=ax.transAxes, color=LOC_COLORS[c], lw=1.4, clip_on=False)
                ax.text(x + .18, 1.27, lab, transform=ax.transAxes, color=LOC_COLORS[c] if c != 'opposite' else '.5',
                        fontsize=6.5, va='center')
        else:
            ax.set_yticklabels([])
        if k == 1:
            ax.set_xlabel('Time from search onset (s)')
    ax = at(4.42, .5, 2.55, .98)
    print('e: highest bracket', panel_loc_types(ax, cells, rres))
    letter(ax, 'e', x=-.2, y=1.27)

    fmt_d = lambda d, lab: f"{lab}: " + ', '.join(f"{g.lower()} {d[g]['mean']:+.2f}% signal, {fmt_p(d[g]['p'])}"
                                                   for g in PATHWAY)
    an = '; '.join(f"{g.lower()} {r.main:+.1f}%, F(1, {int(r.n) - 1}) = {r.main_F:.2f}, {fmt_p(r.main_p)}"
                   for g, r in res.iterrows())
    inter = '; '.join(f"{g.lower()} F(2, {2 * (int(r.n) - 1)}) = {r.int_F:.2f}, {fmt_p(r.int_p)}" for g, r in res.iterrows())
    legend = LEGEND.format(w0=PEAK[0], w1=PEAK[1], diff_tests=fmt_d(dt, 'b') + '; ' + fmt_d(dd, 'c'),
                           n_dor=int(rres.loc['Dorsal', 'n']), rel=rel_text(rres), rel_dct=rel_text(rres2), items=item_text(cells),
                           hplp=an, inter=inter, dct=short_stats(dt2, dd2, hl2, res2))
    legend = legend.replace('-0.', '−0.').replace('(-', '(−').replace(' -', ' −')
    save(fig, name, legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deconv-folder', default='notes/data/event_deconv_2026-11',
                    help='deconvolved item responses; notes/data/event_deconv_dct_2026-11 for the DCT basis')
    ap.add_argument('--dct-folder', default='notes/data/event_deconv_dct_2026-11',
                    help='second basis, reported in the legend')
    ap.add_argument('--stats-only', action='store_true', help='print the statistics, draw nothing')
    ap.add_argument('--name', default='fig3_item_locations')
    ap.add_argument('--rel-folder', default=REL_FOLDER, help='responses by location type (Fourier), for d and e')
    ap.add_argument('--rel-dct-folder', default=REL_DCT_FOLDER, help='responses by location type (DCT), legend')
    a = ap.parse_args()
    main(a.deconv_folder, a.name, a.stats_only, a.dct_folder, a.rel_folder, a.rel_dct_folder)
