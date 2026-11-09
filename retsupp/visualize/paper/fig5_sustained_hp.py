"""Figure 5: less sustained gain at the HP location. Provisional until the final fits.

Usage::

    python -m retsupp.visualize.paper.fig5_sustained_hp
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import generalization_row, conditional, joint, median_ci
from retsupp.visualize.paper.paper_style import PROVISIONAL, SINGLE, DISTRACTOR, DOUBLE, FG, HP, MUTED, ROI_COLORS, ROI_ORDER, letter, ROI_X, title_icon, offset_spines, pending, provisional, roi_axis, roi_icon, save

LEGEND = """
**Figure 5. Less sustained gain at the location where the distractor is likely.** a, Sketch of the result, drawn to scale: the sustained gain field over the search display (red shading), one Gaussian per item (white discs, 4° eccentricity, neighbouring items {spacing:.1f}° apart; grey circle, the 3.17° aperture of the bar stimulus), with less gain at the high-probability (HP) than at the low-probability (LP) locations; the HP − LP difference is exaggerated for display. The dashed circle is the full width at half maximum (FWHM = 2.355 σ) of the fitted sustained field, the median over all subjects and ROIs ({fwhm:.1f}°; joint m7 fit, DoG pRFs); per-ROI medians: {fw_roi}. {fw_bound:.0f}% of the fits sit at a bound of σ (0.5° or 8°) and constrain the width poorly. The field reaches well beyond the item and overlaps those of its neighbours, so the lower gain at HP is not confined to the item location. b, dS = g_HP − g_LP, the sustained gain at the HP location minus that at the LP locations, per ROI, each term estimated alone on m4 (conditional estimate, DoG pRFs): dots are subjects (n = 30; triangles, beyond the axis), points and bars the median and bootstrapped 95% CI. Stars: Wilcoxon signed-rank test against 0, Benjamini-Hochberg across ROIs; * q < .05, ** q < .01, *** q < .001. Squares under the data (New subjects): leave-one-subject-out test per ROI, the term added with the size estimated from the other 29 subjects (+δ) vs the opposite size (−δ); filled, a majority of subjects predicted better with +δ (one-sided Wilcoxon signed-rank test, Benjamini-Hochberg q < .05 across ROIs); open, not significant. c, The same in the measured responses, without a model, per pathway (early, V1–V3; dorsal, V3AB and IPS; occipitotemporal, hV4, VO, LO and TO): deconvolved response (nideconv) of voxels whose pRF centre lies within 2° of an item location, when that location is HP (red) vs LP (grey), all items (target, distractor, neutral) pooled and ROIs pooled within a pathway (weighted by voxel count); mean ± SEM over subjects. The HP − LP difference averaged over the shaded window (3.2–6.4 s), Wilcoxon signed-rank test against 0, the percentage beside the curves giving it relative to the LP response: early, {c_ear}; dorsal, {c_dor}; occipitotemporal, {c_occ}. Only the early areas show a significantly lower response at HP (*); in the dorsal and occipitotemporal areas the difference has the same sign but is not significant. d, The transient distractor gain, dD = g_HP_dyn − g_LP_dyn: the gain the distractor draws when it appears at the HP location minus when it appears at an LP location, estimated alone as in b. It is negative in V3AB, IPS, hV4 and VO. When dD is fitted together with dS and the other terms (joint m7 fit, supplementary figure), no ROI survives correction: dD stays at −7 to −10% in V3AB, IPS, hV4 and VO (Benjamini-Hochberg q ≈ .09–.15) and is positive in V1–V3, while dS is −4 to −5% in V1–V3 (q ≥ .07) and close to 0 elsewhere; over all nine ROIs (median per subject) dD is not distinguishable from 0 (57% of subjects below 0, p = .44). Provisional: current inputs.
"""
POOLED_LEGEND = """
**Figure S. dS and dD per subject, median over the nine ROIs.** a, dS as the conditional estimate (each term alone on m4) and from the joint m7 fit (DoG pRFs); the share of negative subjects and the Wilcoxon p are given above. b, The same for dD: negative when estimated alone, not distinguishable from 0 in the joint fit. Points and bars: median and bootstrapped 95% CI; n = 30. Provisional: current inputs.
"""
POWER_LEGEND = """
**Figure S. Within-subject cross-validation cannot test dS.** When a known dS is put into the data (x-axis), the share of subjects whose held-out prediction improves (y-axis) passes 50% only near −.05, about three times the estimate (dashed). Provisional: current inputs.
"""


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def panel_roi(ax, cond, term='dS', color=HP, ylim=(-.12, .06), ylabel='dS: HP − LP sustained gain', annot=('Less gain at HP', (2, -.016), (3.2, -.095)), band_label=True):
    lo, hi = ylim
    ps, med = [], []
    for i, roi in enumerate(ROI_ORDER):
        v = cond[cond.roi == roi][term].dropna().to_numpy()
        rng = np.random.default_rng(i)
        xx = ROI_X[roi] + rng.uniform(-.17, .17, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=7, color=color, alpha=.35, lw=0, zorder=1)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=10, marker='^', color=color, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=10, marker='v', color=color, alpha=.6, lw=0, clip_on=False)
        m, a, b = median_ci(v)
        ax.errorbar(ROI_X[roi], m, yerr=[[m - a], [b - m]], fmt='o', ms=6, color=color, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        ps.append(stats.wilcoxon(v).pvalue)
    for i, q in enumerate(stats.false_discovery_control(ps)):
        ax.text(ROI_X[ROI_ORDER[i]], hi + .03 * (hi - lo), stars(q), ha='center', va='bottom', fontsize=6.5, color=color, clip_on=False)
    ax.hlines(0, -.6, ROI_X[ROI_ORDER[-1]] + .6, color='.6', lw=.8, ls='--', zorder=0)
    ax.annotate(annot[0], xy=annot[1], xytext=annot[2], fontsize=7, color=color, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1., mutation_scale=7, shrinkA=2, shrinkB=4))
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    generalization_row(ax, term, color, lo, hi, label=band_label)
    roi_axis(ax)


def panel_pooled(ax, cond, js, term='dS', color=HP, ylim=(-.12, .06)):
    sets = [('Conditional', cond.groupby('subject')[term].median()), ('Joint (m7)', js['dog'].groupby('subject')[term].median())]
    lo, hi = ylim
    for i, (lab, v) in enumerate(sets):
        v = v.dropna().to_numpy()
        rng = np.random.default_rng(10 + i)
        xx = i + rng.uniform(-.15, .15, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=10, color=color, alpha=.35, lw=0)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=12, marker='v', color=color, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=12, marker='^', color=color, alpha=.6, lw=0, clip_on=False)
        m, a, b = median_ci(v)
        ax.errorbar(i, m, yerr=[[m - a], [b - m]], fmt='o', ms=6, color=color, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        p = stats.wilcoxon(v).pvalue
        ptxt = 'p < .001' if p < .001 else f'p = {p:.3f}'.replace('0.', '.')
        ax.text(i, hi + .03 * (hi - lo), f'{(v < 0).mean() * 100:.0f}% < 0\n{ptxt}', ha='center', va='bottom', fontsize=6, color=FG,
                clip_on=False)
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s[0] for s in sets], fontsize=6.5)
    ax.set_xlim(-.5, len(sets) - .5)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(f'{term}, median over ROIs')
    offset_spines(ax)


def panel_power(ax, table, estimate=-.016):
    d = pd.read_csv(table, sep='\t')
    x = -d.planted_dS.to_numpy()
    ax.plot(x, d.pct_improved, 'o-', color=FG, ms=4, lw=1.3)
    ax.axhline(50, color='.6', lw=.8, ls='--', zorder=0)
    ax.text(.2, 52, 'Chance', fontsize=6, color='.45', ha='right', va='bottom')
    ax.axvline(-estimate, color=HP, lw=1, ls='--')
    ax.text(-estimate * 1.1, 97, 'Our\nestimate', fontsize=6.5, color=HP, va='top', ha='left')
    ax.annotate('Break-even\nnear .05', xy=(.05, 50), xytext=(.075, 20), fontsize=6.5, color=FG, ha='left',
                arrowprops=dict(arrowstyle='-|>', color=FG, lw=.9, mutation_scale=6, shrinkA=2, shrinkB=3))
    ax.set_xscale('symlog', linthresh=.01)
    ax.set_xticks([0, .01, .03, .1, .2])
    ax.set_xticklabels(['0', '.01', '.03', '.1', '.2'])
    ax.set_xlabel('Size of a planted dS (−)')
    ax.set_ylabel('Subjects improved (%)')
    ax.set_ylim(-3, 103)
    offset_spines(ax)


FWHM = 2 * np.sqrt(2 * np.log(2))
PATHWAY = {'Early': ('V1', 'V2', 'V3'), 'Dorsal': ('V3AB', 'IPS'), 'Occipitotemporal': ('hV4', 'VO', 'LO', 'TO')}
LP_GREY = '#7B8794'


def sustained_fwhm(js):
    """FWHM (°) of the sustained field, m7_full DoG: median over all subject x ROI fits, and per-ROI medians."""
    d = js['dog']
    per_roi = FWHM * d.groupby('roi').sigma_AF.median().reindex(ROI_ORDER)
    at_bound = ((d.sigma_AF < .51) | (d.sigma_AF > 7.9)).mean()
    return FWHM * d.sigma_AF.median(), per_roi, at_bound


def panel_sketch(ax, fwhm, lo_hi, g_lp=1., g_hp=.2, hp_ix=3, ecc=4., aperture=3.17, item_r=.75):
    """The sustained gain field over the display, drawn to scale with the fitted FWHM; weaker at HP (exaggerated)."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    sigma = fwhm / FWHM
    ring = ecc / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    L = 7.
    xy = np.linspace(-L, L, 400)
    X, Y = np.meshgrid(xy, xy)
    G = sum((g_hp if i == hp_ix else g_lp) * np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * sigma ** 2))
            for i, p in enumerate(ring))
    cmap = LinearSegmentedColormap.from_list('gain', ['#FFFFFF', HP])
    im = ax.imshow(G, extent=(-L, L, -L, L), origin='lower', cmap=cmap, vmin=0, vmax=G.max() * 1.05, alpha=.8, zorder=0)
    edge = Circle((0, 0), L - .05, fill=False, ec='none', lw=0)
    ax.add_patch(edge)
    im.set_clip_path(edge)
    ax.add_patch(Circle((0, 0), aperture, fill=False, ec=MUTED, lw=.6, zorder=1))
    for p in ring:
        ax.add_patch(Circle(p, item_r, facecolor='w', ec=MUTED, lw=.6, zorder=2))
    ax.plot(0, 0, '+', color=FG, ms=4, mew=.6)
    hp = ring[hp_ix]
    # The fitted half-maximum extent of one item's field, around the HP item.
    ax.add_patch(Circle(hp, fwhm / 2, fill=False, ec=HP, lw=1., ls=(0, (2.5, 1.5)), zorder=3))
    ax.text(hp[0] - .4, hp[1] - fwhm / 2 - .3, f'Fitted FWHM {fwhm:.1f}°', color=HP, fontsize=6, ha='center', va='top')
    ax.text(hp[0], hp[1], 'HP', color=HP, fontsize=5.5, ha='center', va='center', fontweight='bold', zorder=4)
    for i, p in enumerate(ring):
        if i != hp_ix:
            ax.text(p[0], p[1], 'LP', color='.4', fontsize=5.5, ha='center', va='center', zorder=4)
    # Distance between neighbouring items, along the top edge.
    a, b = ring[1], ring[0]
    y = a[1] + 1.25
    ax.annotate('', xy=(b[0] - .05, y), xytext=(a[0] + .05, y),
                arrowprops=dict(arrowstyle='<|-|>', color=FG, lw=.7, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(0, y + .3, f'{np.linalg.norm(a - b):.1f}°', color=FG, fontsize=6, ha='center', va='bottom')
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect('equal')
    ax.set_axis_off()


def hp_lp_pathways(window=(3.2, 6.4)):
    """Per subject and pathway: time courses at an item location when it is HP vs LP (all item types pooled)."""
    from retsupp.visualize.paper.fig45_common import item_timecourses
    grp = {r: g for g, rois in PATHWAY.items() for r in rois}
    p = item_timecourses(groups=grp)
    w = p.pivot_table(index=['grp', 'subject', 't'], columns='cls', values='r')
    x = pd.DataFrame({'hp': w[['target_hp', 'distractor_hp', 'neutral_hp']].mean(1),
                      'lp': w[['target_lp', 'distractor_lp', 'neutral_lp']].mean(1)})
    res = {}
    for g in PATHWAY:
        xg = x.loc[g]
        hp, lp = xg.hp.unstack('t'), xg.lp.unstack('t')
        inw = (hp.columns >= window[0]) & (hp.columns <= window[1])
        v = (hp - lp).loc[:, inw].mean(1).dropna()
        lpw = lp.loc[v.index, inw].mean(1)
        res[g] = dict(hp=hp, lp=lp, diff=v.mean(), rel=100 * v.mean() / lpw.mean(), p=stats.wilcoxon(v).pvalue, n=len(v))
    return res


def panel_hp_tc(ax, r, name, window=(3.2, 6.4), key=False, ylim=(-.25, 1.2)):
    """Measured response at an item location when it is the HP vs an LP location, one pathway."""
    ax.axvspan(*window, color='.93', lw=0, zorder=0)
    for k, (c, col, lab) in enumerate((('lp', LP_GREY, 'LP location'), ('hp', HP, 'HP location'))):
        g = r[c]
        m, se = g.mean(), g.sem()
        ax.fill_between(m.index, m - se, m + se, color=col, alpha=.18, lw=0)
        ax.plot(m.index, m.values, color=col, lw=1.3)
        if key:
            ax.text(.98, .97 - k * .13, lab, transform=ax.transAxes, color=col, fontsize=6, ha='right', va='top')
    top = r['lp'].mean().max()
    st = stars(r['p'])
    if st:
        ax.text(np.mean(window), top + .05, st, ha='center', va='bottom', fontsize=7, color=HP)
    ax.text(window[1] + .6, top - .07, f"{r['rel']:.0f}%".replace('-', '−'), ha='left', va='center', fontsize=6,
            color=HP if st else '.45')
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.set_xlim(-4, 20)
    ax.set_xticks([0, 10, 20])
    ax.set_ylim(*ylim)
    ax.set_yticks([0, .5, 1])
    ax.set_title(name if name != 'Occipitotemporal' else 'Occ.-temporal', fontsize=6.5, color=FG, pad=2)
    offset_spines(ax)


def main(gain_table, conditional_table, power_table, name='fig5_sustained_hp'):
    cond = conditional(conditional_table)
    js = {f: joint(gain_table, f) for f in ('dog', 'dn')}
    fw, fw_roi, fw_bound = sustained_fwhm(js)
    tcs = hp_lp_pathways()
    W, H = DOUBLE, 5.1
    fig = plt.figure(figsize=(W, H))
    # Row 1: a the sustained field to scale, b dS per ROI (the result).
    # Row 2: c the measured HP vs LP response per pathway, d dD (secondary).
    ax = fig.add_axes([.02 / W, 3.0 / H, 1.5 / W, 1.5 / H])
    panel_sketch(ax, fw, (fw_roi.min(), fw_roi.max()))
    ax.text(.08, 1.1, 'Sustained gain field', transform=ax.transAxes, fontsize=7, color=FG, va='center')
    ax.text(.42, -.06, 'Less gain where the\ndistractor is likely', transform=ax.transAxes,
            fontsize=6.5, color=HP, ha='center', va='top')
    letter(ax, 'a', x=.03, y=1.07)
    ax = fig.add_axes([2.3 / W, 3.2 / H, 4.7 / W, 1.3 / H])
    panel_roi(ax, cond, ylabel='Sustained gain, HP − LP', annot=('Less gain at HP', (ROI_X['V3'], -.016), (3.0, -.1)))
    title_icon(ax, 'Less sustained gain at HP', ('hp', 'lp'), HP, h=.27, y=1.1)
    letter(ax, 'b', x=-.16, y=1.19)
    for k, g in enumerate(PATHWAY):
        ax = fig.add_axes([(.5 + k * 1.12) / W, .62 / H, 1.0 / W, .95 / H])
        panel_hp_tc(ax, tcs[g], g, key=k == 0)
        if k == 0:
            ax.set_ylabel('Response (% signal)')
            ax.text(-.36, 1.36, 'Items at HP evoke less response in early areas', transform=ax.transAxes, fontsize=7,
                    color=HP, va='center')
            letter(ax, 'c', x=-.36, y=1.44)
        else:
            ax.set_yticklabels([])
        if k == 1:
            ax.set_xlabel('Time from search onset (s)')
    ax = fig.add_axes([4.3 / W, .75 / H, 2.7 / W, 1.15 / H])
    panel_roi(ax, cond, 'dD', DISTRACTOR, (-.3, .2), 'Transient gain, HP − LP',
              ('Less at HP', (ROI_X['V3AB'] - .1, -.045), (1.2, -.26)), band_label=False)
    title_icon(ax, 'Transient distractor gain, HP − LP', ('distractor', 'lp'), DISTRACTOR, h=.3, y=1.08,
               sub='Estimated alone; not significant per ROI in the joint fit')
    letter(ax, 'd', x=-.2, y=1.22)
    fig.text(.005, .005, PROVISIONAL, fontsize=5.5, color=MUTED, ha='left', va='bottom')
    fmt = lambda r: (f"{r['diff']:+.3f}% signal ({r['rel']:.0f}% of the LP response), p = {r['p']:.3f}, n = {r['n']}".replace('0.', '.')
                     .replace('-', '−'))
    save(fig, name, LEGEND.format(spacing=4 * np.sqrt(2), fwhm=fw, fw_lo=fw_roi.min(), fw_hi=fw_roi.max(), fw_bound=100 * fw_bound,
                                  fw_roi='; '.join(f'{r} {v:.1f}°' for r, v in fw_roi.items()),
                                  **{f'c_{g[:3].lower()}': fmt(tcs[g]) for g in PATHWAY}))
    print('FWHM median', fw, 'per ROI', fw_roi.round(2).to_dict(), 'at bound', fw_bound)
    for g in PATHWAY:
        print('c', g, {k: v for k, v in tcs[g].items() if k not in ('hp', 'lp')})
    fig = plt.figure(figsize=(SINGLE, 2.3))
    gs = fig.add_gridspec(1, 2, wspace=.6, left=.17, right=.98, top=.8, bottom=.15)
    ax = fig.add_subplot(gs[0])
    panel_pooled(ax, cond, js)
    letter(ax, 'a', x=-.35, y=1.12)
    ax = fig.add_subplot(gs[1])
    panel_pooled(ax, cond, js, 'dD', DISTRACTOR, (-.3, .2))
    letter(ax, 'b', x=-.35, y=1.12)
    save(fig, 'figS_pooled_dS_dD', POOLED_LEGEND)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2), gridspec_kw=dict(left=.17, right=.97, top=.92, bottom=.2))
    panel_power(ax, power_table)
    save(fig, 'figS_cv_power', POWER_LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--power-table', default='notes/data/cv_power_simulation_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.power_table)
