"""Figure 4: the measured responses at the item locations, without a model.

Voxels whose pRF centre lies within 2° of an item location, deconvolved (nideconv) around search onset
(``deconvolve_location_responses.py``; read by ``fig45_common.item_timecourses``).

Row 1, what the item is: the target and the distractor evoke more response than a neutral item (a raw, b, c differences
per pathway). Row 2, where the item is: items at the HP location evoke less response, alike for every item type
(d HP vs LP time courses per pathway, e HP − LP per item type with the location × item ANOVA of ``fig_hp_heatmap``).

The model's gain fields are Figure 5 (``fig5_gain_fields.py``).

Usage::

    python -m retsupp.visualize.paper.fig4_item_locations                     # Fourier basis (main)
    python -m retsupp.visualize.paper.fig4_item_locations --stats-only \\
        --deconv-folder notes/data/event_deconv_dct_2026-11                   # DCT basis: statistics only
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig3_prf_fits import PATHWAY, PATHWAY_COL
from retsupp.visualize.paper.fig45_common import item_timecourses
from retsupp.visualize.paper.paper_style import DISTRACTOR, DOUBLE, FG, HP, MUTED, TARGET, letter, offset_spines, save

NEUTRAL = '#7B8794'
LP_GREY = '#7B8794'
PRF_TEAL = '#2A9D8F'
PEAK = (3.2, 6.4)   # s after search onset: the window tested
SHORT = {'Early': 'Early', 'Dorsal': 'Dorsal', 'Occipitotemporal': 'Occ.-temporal', 'All areas': 'All areas'}
ITEMS = [('target', 'Target', TARGET), ('distractor', 'Distractor', DISTRACTOR), ('neutral', 'Neutral item', NEUTRAL)]

LEGEND = """
**Figure 4. At the item locations, without a model: more response to the target and the distractor, and less response at the high-probability (HP) location for every item.** All panels: deconvolved response (nideconv, Fourier basis) around search onset of voxels whose pRF centre lies within 2° of an item location (inset in a: the display with one such pRF on an item, and the response it gives), ROIs pooled weighted by voxel count, each subject's pre-onset mean (−4 to 0 s) subtracted; mean over subjects ± SEM (n = 30). Pathways: early, V1–V3; dorsal, V3AB and IPS; occipitotemporal, hV4, VO, LO and TO. Grey band: the window tested, {w0:.1f}–{w1:.1f} s. a, Response at a pRF's item location when the target, the distractor or a neutral item is there, all at low-probability (LP) locations, all nine ROIs pooled. b, Target minus neutral item per pathway (both averaged over HP and LP locations). c, Distractor minus neutral item per pathway (both at LP locations). Stars in b, c: the mean over the window against 0, Wilcoxon signed-rank test, Benjamini-Hochberg over the three pathways per panel; {diff_tests}. d, Response at an item location when it is the HP location (red) vs an LP location (grey), target, distractor and neutral item pooled, per pathway; the percentage is the HP − LP difference over the window relative to the LP response; {hp_tests}. e, HP − LP per item type over the window, in % of each subject's mean LP response over the three item types in that pathway; points and bars, mean ± SEM over subjects (dorsal pathway in b–e: n = {n_dor}, two subjects lack dorsal voxels at some item location). Red rules: the main effect of location (mean over item types); stars, repeated-measures ANOVA location (HP, LP) × item type (target, distractor, neutral) per pathway, uncorrected; {anova}. The location × item interaction is not significant in any pathway ({inter}); the planned contrast of the distractor against the mean of the target and the neutral item (the measured counterpart of the model's distractor-specific term) is closest to significance in the occipitotemporal pathway ({dvo}). DCT basis (robustness): the HP − LP differences keep their sign and size, but the main effect of location is not significant in any pathway; {dct}
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
        ax.text(1., .99 - k * .12, f"{SHORT[grp]} {stars(d[grp]['q'])}", transform=ax.transAxes, color=PATHWAY_COL[grp],
                fontsize=6.5, ha='right', va='top')
    time_axis(ax)
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.set_ylim(-.1, .38)
    ax.set_yticks([-.1, 0, .1, .2, .3])
    if ylabel:
        ax.set_ylabel('Extra response (% signal)')
    else:
        ax.set_yticklabels([])


def panel_hp_tc(ax, r, name, key=False, ylim=(-.25, 1.2)):
    for k, (c, col, lab) in enumerate((('lp', LP_GREY, 'LP location'), ('hp', HP, 'HP location'))):
        tc(ax, r[c].mean(), r[c].sem(), col)
        if key:
            ax.text(1.02, .97 - k * .13, lab, transform=ax.transAxes, color=col, fontsize=6, ha='right', va='top')
    top = r['lp'].mean().max()
    st = stars(r['p'])
    if st:
        ax.text(np.mean(PEAK), top + .06, st, ha='center', va='bottom', fontsize=7, color=HP)
    ax.text(PEAK[1] + .7, top - .1, f"{r['rel']:.0f}%".replace('-', '−'), ha='left', va='center', fontsize=6,
            color=HP if st else '.45')
    time_axis(ax, xlabel=False)
    ax.set_ylim(*ylim)
    ax.set_yticks([0, .5, 1])
    ax.set_title(SHORT[name], fontsize=6.5, color=FG, pad=2)


def panel_items(ax, diff, res):
    """HP − LP per item type and pathway (mean ± SEM); red rule, the mean over items (main effect of location)."""
    order = list(PATHWAY) + ['All areas']
    off = {'target': -.25, 'distractor': 0., 'neutral': .25}
    for i, g in enumerate(order):
        x = i + (.35 if g == 'All areas' else 0)
        w = diff[diff.grp == g].pivot(index='subject', columns='item', values='diff')
        for it, _, col in ITEMS:
            v = w[it]
            ax.errorbar(x + off[it], v.mean(), yerr=v.sem(), fmt='o', ms=4, color=col, mec='w', mew=.6, lw=1.1, capsize=0, zorder=3)
        m = w.mean(1).mean()
        ax.plot([x - .4, x + .4], [m, m], color=HP, lw=1.6, solid_capstyle='butt', zorder=4)
        ax.text(x, 12.5, stars(res.loc[g, 'main_p']) or 'n.s.', ha='center', va='bottom',
                fontsize=7 if stars(res.loc[g, 'main_p']) else 6, color=HP if res.loc[g, 'main_p'] < .05 else '.5')
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.axvline(len(order) - 1.5 + .175, color='.85', lw=.6, zorder=0)
    ax.set_xticks([i + (.35 if g == 'All areas' else 0) for i, g in enumerate(order)])
    ax.set_xticklabels([SHORT[g] for g in order], fontsize=6.5)
    ax.set_xlim(-.55, len(order) - .1)
    ax.set_ylim(-31, 12)
    ax.set_yticks([-20, -10, 0, 10])
    ax.set_ylabel('HP − LP (% of LP)')
    offset_spines(ax)
    # Key: item colours, and the red rule for the main effect, drawn as glyphs.
    tf = ax.transAxes
    for k, (it, lab, col) in enumerate(ITEMS):
        ax.plot(.03 + k * .25, .07, 'o', ms=3.5, color=col, mec='w', mew=.5, transform=tf, clip_on=False)
        ax.text(.055 + k * .25, .07, lab.replace(' item', ''), transform=tf, fontsize=6, color=col, va='center')
    ax.plot([.79, .85], [.07, .07], color=HP, lw=1.6, transform=tf, solid_capstyle='butt')
    ax.text(.87, .07, 'All items', transform=tf, fontsize=6, color=HP, va='center')


def header(fig, y, text, x0=.012, x1=.99):
    fig.text(x0, y, text, fontsize=7.5, fontweight='bold', color=FG, va='bottom', ha='left')
    fig.add_artist(plt.Line2D([x0, x1], [y - .006] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


def main(folder, name='fig4_item_locations', stats_only=False, dct_folder='notes/data/event_deconv_dct_2026-11'):
    pw, dt, dd, hl, diff, res = all_stats(folder)
    report(folder, dt, dd, hl, res)
    if stats_only:
        return dt, dd, hl, res
    _, dt2, dd2, hl2, _, res2 = all_stats(dct_folder)
    report(dct_folder, dt2, dd2, hl2, res2)
    pa = item_timecourses(folder, groups=None)

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

    header(fig, 1.98 / H, 'Where the item is: less response at the HP location in early areas, alike for every item type')
    for k, g in enumerate(PATHWAY):
        ax = at(.5 + k * 1.08, .5, .98, 1.08)
        panel_hp_tc(ax, hl[g], g, key=k == 0)
        if k == 0:
            ax.set_ylabel('Response (% signal)')
            letter(ax, 'd', x=-.4, y=1.12)
        else:
            ax.set_yticklabels([])
        if k == 1:
            ax.set_xlabel('Time from search onset (s)')
    ax = at(4.25, .5, 2.7, 1.08)
    panel_items(ax, diff, res)
    letter(ax, 'e', x=-.17, y=1.12)

    fmt_d = lambda d, lab: f"{lab}: " + ', '.join(f"{g.lower()} {d[g]['mean']:+.2f}% signal, {fmt_p(d[g]['p'])}"
                                                   for g in PATHWAY)
    hp_txt = ', '.join(f"{g.lower()} {hl[g]['diff']:+.3f}% signal ({hl[g]['rel']:.0f}%), {fmt_p(hl[g]['p'])}"
                       for g in PATHWAY)
    an = '; '.join(f"{g.lower()} F(1, {int(r.n) - 1}) = {r.main_F:.2f}, {fmt_p(r.main_p)}" for g, r in res.iterrows())
    inter = '; '.join(f"{g.lower()} F(2, {2 * (int(r.n) - 1)}) = {r.int_F:.2f}, {fmt_p(r.int_p)}" for g, r in res.iterrows())
    dvo = f"{res.loc['Occipitotemporal', 'd_vs_others']:+.1f}%, Wilcoxon {fmt_p(res.loc['Occipitotemporal', 'd_vs_others_p'])}"
    legend = LEGEND.format(w0=PEAK[0], w1=PEAK[1], diff_tests=fmt_d(dt, 'b') + '; ' + fmt_d(dd, 'c'), hp_tests=hp_txt,
                           n_dor=int(res.loc['Dorsal', 'n']), anova=an, inter=inter, dvo=dvo,
                           dct=short_stats(dt2, dd2, hl2, res2)).replace('-0.', '−0.').replace('(-', '(−').replace(' -', ' −')
    save(fig, name, legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deconv-folder', default='notes/data/event_deconv_2026-11',
                    help='deconvolved item responses; notes/data/event_deconv_dct_2026-11 for the DCT basis')
    ap.add_argument('--dct-folder', default='notes/data/event_deconv_dct_2026-11',
                    help='second basis, reported in the legend')
    ap.add_argument('--stats-only', action='store_true', help='print the statistics, draw nothing')
    ap.add_argument('--name', default='fig4_item_locations')
    a = ap.parse_args()
    main(a.deconv_folder, a.name, a.stats_only, a.dct_folder)
