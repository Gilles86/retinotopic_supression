"""Figure 5: the gain fields, the model's account of the responses at the item locations (Fig. 3). Provisional until the final fits.

Top row: a, target and distractor gain per ROI; b, leave-one-subject-out test of each term per ROI (ROIs as rows); c, the
sustained field to scale. Bottom row: d, the sustained HP − LP gain (dS); e, distractor (dD) and target (dT) HP − LP gain
side by side, the gain analogue of Fig. 1c, d.
The gain against pRF size (the hierarchy) is the supplementary figure ``figS_gain_vs_prf_size``.

Also writes the parameter table ``table_parameters_2026-11.{md,tsv}`` (all terms, DoG and DN, medians, CIs, tests,
Bayes factors, field sizes) and the supplementary figures ``figS_field_size``, ``figS_pooled_dS_dD`` and ``figS_cv_power``.

Usage::

    python -m retsupp.visualize.paper.fig5_gain_fields
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, generalization_row, joint, jzs_bf10, median_ci
from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, ICON_PAD, MUTED, OUT, PROVISIONAL, ROI_ORDER, ROI_X,
                                                 SINGLE, TARGET, letter, offset_spines, provisional, roi_axis, roi_icon,
                                                 roi_positions, save)

FWHM = 2 * np.sqrt(2 * np.log(2))
TERMS = [('T', 'Target gain', TARGET), ('D', 'Distractor gain', DISTRACTOR), ('dS', 'Sustained gain, HP − LP', HP),
         ('dD', 'Distractor gain, HP − LP', '#5E3C99')]
FAM = {'dog': 'DoG', 'dn': 'DN'}
LEGEND = """
**Figure 5. The gain fields: gain at the target and the distractor, and less sustained gain at the high-probability (HP) location.** Gain-field model fits with DoG pRFs (n = 30); a gain is the fractional change in drive at the centre of a field (+0.30: 30% more). a, d, e: dots are subjects (triangles, beyond the axis), points and bars the median and bootstrapped 95% CI; stars, Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs, * q < .05, ** q < .01, *** q < .001 (exact p and q per ROI in the parameter table). a, Gain at the target (orange) and at the distractor (blue, averaged over its locations) while the search display is on, all terms fitted jointly. The gain against pRF size is in a supplementary figure. b, Do the terms predict new subjects? Leave-one-subject-out test per ROI and term: the term added with the size estimated from the other 29 subjects (+δ) vs the opposite size (−δ); rows are ROIs, columns terms; numbers, the percentage of subjects predicted better with +δ (cell colour: red above 50%, blue below); bold: q < .05 (non-significant cells muted), one-sided Wilcoxon signed-rank test, Benjamini-Hochberg across the nine ROIs per term (q per cell below). The distractor gain is tested on top of the target term. c, The sustained gain field over the search display, drawn to scale (red shading): one Gaussian per item (white discs, 4° eccentricity, neighbouring items {spacing:.1f}° apart; grey circle, the 3.17° aperture of the bar stimulus), with less gain at the HP than at the LP locations (the HP − LP difference exaggerated for display). Dashed circle: the full width at half maximum (FWHM = 2.355 σ) of the fitted sustained field, the median over all subjects and ROIs ({fwhm:.1f}°; per-ROI medians: {fw_roi}; {fw_bound:.0f}% of the fits at a bound of σ, 0.5° or 8°). The field reaches beyond the item and overlaps those of its neighbours. d, dS = g_HP − g_LP, the sustained gain at the HP location minus that at the low-probability (LP) locations, each term estimated alone on the item model with sustained gain (conditional estimate). e, The gain analogue of Fig. 1c, d: dD = g_HP_dyn − g_LP_dyn, the gain the distractor draws at the HP location minus at an LP location (blue), next to dT, the target gain at the HP location minus at an LP location (orange), each estimated alone as in d; the display icons are those of Fig. 1 (HP ringed in red; the item at HP minus at an LP location). Less gain for a distractor at HP in V3AB, IPS, hV4 and VO; the target gain shows no reliable HP − LP difference after correction (smallest q = .07, V3; a positive trend in IPS, q = .10), and none when fitted jointly with all other terms either (q ≥ .36). Stars per series, one row each. When dS and dD are fitted together with all other terms (supplementary figure and parameter table), no ROI survives correction: dD stays at −7 to −10% in V3AB, IPS, hV4 and VO (q ≈ .09–.15), dS is −4 to −5% in V1–V3 (q ≥ .07). The measured responses at the item locations (Fig. 3) show less response at HP for every item type, significant in the early areas, without a distractor-specific difference. DN pRFs, Bayes factors and all other terms are in the parameter table. Provisional: current inputs.
"""
LEGEND_HIERARCHY = """
**Figure S. Gain at the target and the distractor against pRF size.** Median target and distractor gain per ROI (as in Fig. 5a) against the ROI's pRF size (median over subjects of the voxel-weighted mean pRF σ at 1–2.5° eccentricity, log scale). Lines: slope per doubling of pRF size, the median of the per-subject OLS slopes over the nine ROIs, drawn through the centre of the points; target {t_slope}; distractor {d_slope} (Wilcoxon signed-rank test of the per-subject slopes, Benjamini-Hochberg over the eight parameters tested). The slope may partly reflect a trade-off in the fits: within subjects, the target gain correlates with the fitted weight of the items relative to the bar (ρ = .25, p = .011) and negatively with the size of the gain field (ρ = −.22, p = .049), and the model-free enhancement at the item locations (Fig. 3) rises much less up the hierarchy. Provisional: current inputs.
"""
LEGEND_SIZE = """
**Figure S. Size of the fitted gain fields.** Full width at half maximum (2.355 σ) of the sustained and transient fields per ROI (m7_full, DoG pRFs), median over subjects and bootstrapped 95% CI; dotted line, the 4° radius of the ring. Many fits sit at a bound of σ (0.5° or 8°; up to 73% for the transient field in IPS, see the parameter table): where a gain is small, its width is poorly constrained. Provisional: current inputs.
"""


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def summarise(js, cond):
    rows = []
    for fam, d in js.items():
        for term, _, _ in TERMS:
            ps = []
            for roi in ROI_ORDER:
                v = d[d.roi == roi][term].dropna().to_numpy()
                m, lo, hi = median_ci(v)
                p = stats.wilcoxon(v).pvalue
                ps.append(p)
                c = cond[cond.roi == roi][term].dropna() if (fam == 'dog' and term in cond.columns) else pd.Series(dtype=float)
                rows.append(dict(model=FAM[fam], term=term, roi=roi, n=len(v), median=m, ci_low=lo, ci_high=hi,
                                 pct_negative=(v < 0).mean() * 100, p=p, bf10=jzs_bf10(v),
                                 conditional_median=c.median() if len(c) else np.nan,
                                 conditional_p=stats.wilcoxon(c).pvalue if len(c) > 5 else np.nan))
            for r, qq in zip(rows[-len(ROI_ORDER):], stats.false_discovery_control(ps)):
                r['q'] = qq
    return pd.DataFrame(rows)


def write_table(t, js):
    t.to_csv(OUT / 'table_parameters_2026-11.tsv', sep='\t', index=False)
    fmt = lambda r: f"{r['median']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
    lines = ['**Table S. Gain-field parameters per ROI (m7_full, n = 30).** Median [bootstrapped 95% CI], Wilcoxon p, BH q '
             'across 9 ROIs, JZS Bayes factor BF10 (r = .707), and for DoG the conditional median. Gains are the fractional '
             'change in drive at the centre of a field (+0.30: 30% more). Provisional: current inputs.', '']
    for term, lab, _ in TERMS:
        lines += [f'**{lab}**', '', '| ROI | DoG median [CI] | p | q | BF10 | DoG conditional | DN median [CI] | p | q | BF10 |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for roi in ROI_ORDER:
            a = t[(t.model == 'DoG') & (t.term == term) & (t.roi == roi)].iloc[0]
            b = t[(t.model == 'DN') & (t.term == term) & (t.roi == roi)].iloc[0]
            lines.append(f"| {roi} | {fmt(a)} | {a.p:.3g} | {a.q:.3g} | {a.bf10:.2g} | {a.conditional_median:+.3f} | "
                         f"{fmt(b)} | {b.p:.3g} | {b.q:.3g} | {b.bf10:.2g} |")
        lines.append('')
    lines += ['**Field size (FWHM = 2.355 σ, degrees), median [95% CI], % of fits at a bound**', '',
              '| ROI | DoG sustained | at bound | DoG transient | at bound | DN sustained | at bound | DN transient | at bound |',
              '|---|---|---|---|---|---|---|---|---|']
    for roi in ROI_ORDER:
        cells = []
        for fam in ('dog', 'dn'):
            for par in ('sigma_AF', 'sigma_dyn'):
                v = js[fam][js[fam].roi == roi][par].dropna()
                m, lo, hi = (FWHM * x for x in median_ci(v))
                cells += [f'{m:.1f} [{lo:.1f}, {hi:.1f}]', f'{((v < .51) | (v > 7.9)).mean() * 100:.0f}%']
        lines.append(f'| {roi} | ' + ' | '.join(cells) + ' |')
    (OUT / 'table_parameters_2026-11.md').write_text('\n'.join(lines) + '\n')


def panel_gain(ax, d, t, term, col, ylim, note, note_xy):
    x = roi_positions(ROI_ORDER)
    lo, hi = ylim
    for i, roi in enumerate(ROI_ORDER):
        v = d[d.roi == roi][term].dropna().to_numpy()
        rng = np.random.default_rng(i)
        xx = ROI_X[roi] + rng.uniform(-.17, .17, len(v))
        ok = (v >= lo) & (v <= hi)
        ax.scatter(xx[ok], v[ok], s=7, color=col, alpha=.35, lw=0, zorder=1)
        ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=10, marker='^', color=col, alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=10, marker='v', color=col, alpha=.6, lw=0, clip_on=False)
    r = t[(t.model == 'DoG') & (t.term == term)].set_index('roi').loc[ROI_ORDER]
    ax.errorbar(x, r['median'], yerr=[r['median'] - r.ci_low, r.ci_high - r['median']], fmt='o', ms=6, color=col, mec='w', mew=.8,
                lw=1.6, capsize=0, zorder=4)
    for xi, (_, row) in zip(x, r.iterrows()):
        ax.text(xi, hi + .03 * (hi - lo), stars(row.q), ha='center', va='bottom', fontsize=8.5, color=col, clip_on=False)
    ax.hlines(0, x[0] - .6, x[-1] + .6, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    offset_spines(ax)
    generalization_row(ax, {'D': 'DonT'}.get(term, term), col, lo, hi)   # the distractor on top of the target term


def panel_size(ax, d):
    x = roi_positions(ROI_ORDER)
    for j, (par, lab, col) in enumerate((('sigma_AF', 'Sustained field', HP), ('sigma_dyn', 'Transient field', DISTRACTOR))):
        m = FWHM * np.array([median_ci(d[d.roi == r][par]) for r in ROI_ORDER])
        ax.errorbar(x + (j - .5) * .28, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='o', ms=5.5, color=col, mec='w',
                    mew=.8, lw=1.5, capsize=0)
    ax.axhline(4, color='.6', lw=.7, ls=':', zorder=0)
    ax.text(x[0] - .55, 3.6, 'Ring radius', fontsize=6, color='.45', ha='left', va='top')
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 4, 8, 12, 16, 20])
    ax.set_ylabel('Field size, FWHM (°)')
    offset_spines(ax)
    roi_axis(ax)


def size_figure(d):
    fig = plt.figure(figsize=(SINGLE, 2.4))
    ax = fig.add_axes([.17, .3, .8, .6])
    panel_size(ax, d)
    ax.set_title('Size of the fitted fields', fontsize=7, loc='left', color=FG, pad=6)
    for j, (lab, col) in enumerate((('Sustained field', HP), ('Transient field', DISTRACTOR))):
        ax.text(.98, .95 - j * .09, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
    provisional(ax, y=1.08)
    save(fig, 'figS_field_size', LEGEND_SIZE)



POOLED_LEGEND = """
**Figure S. dS and dD per subject, median over the nine ROIs.** a, dS as the conditional estimate (each term alone on m4) and from the joint m7 fit (DoG pRFs); the share of negative subjects and the Wilcoxon p are given above. b, The same for dD: negative when estimated alone, not distinguishable from 0 in the joint fit. Points and bars: median and bootstrapped 95% CI; n = 30. Provisional: current inputs.
"""
POWER_LEGEND = """
**Figure S. Within-subject cross-validation cannot test dS.** When a known dS is put into the data (x-axis), the share of subjects whose held-out prediction improves (y-axis) passes 50% only near −.05, about three times the estimate (dashed). Provisional: current inputs.
"""



def panel_roi(ax, cond, term='dS', color=HP, ylim=(-.12, .06), ylabel='dS: HP − LP sustained gain', annot=('Less gain at HP', (2, -.016), (3.2, -.095)), band_label=True, band=True):
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
        ax.text(ROI_X[ROI_ORDER[i]], hi + .03 * (hi - lo), stars(q), ha='center', va='bottom', fontsize=8.5, color=color, clip_on=False)
    ax.hlines(0, -.6, ROI_X[ROI_ORDER[-1]] + .6, color='.6', lw=.8, ls='--', zorder=0)
    ax.annotate(annot[0], xy=annot[1], xytext=annot[2], fontsize=7, color=color, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1., mutation_scale=7, shrinkA=2, shrinkB=4))
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    if band:
        generalization_row(ax, term, color, lo, hi, label=band_label)


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




def bare_roi_axis(ax, right_pad=ICON_PAD):
    """ROI positions as in roi_axis(..., right_pad=right_pad), without labels: for the upper panel of a stacked pair."""
    xs = [ROI_X[r] for r in ROI_ORDER]
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.set_xlim(xs[0] - .6, xs[-1] + .6 + right_pad)


def panel_hierarchy(ax, t, size, slopes):
    """Median target and distractor gain per ROI against pRF size; line: median per-subject slope per doubling."""
    lx = np.log2(size.loc[ROI_ORDER].to_numpy())
    for k, (term, col, lab, name) in enumerate((('T', TARGET, 'Target gain', 'Target'),
                                                 ('D', DISTRACTOR, 'Distractor gain', 'Distractor'))):
        r = t[(t.model == 'DoG') & (t.term == term)].set_index('roi').loc[ROI_ORDER]
        ax.plot(lx, r['median'], 'o', ms=4.5, color=col, mec='w', mew=.6, zorder=3)
        s = slopes.loc[lab]
        x0, y0 = lx.mean(), r['median'].mean()
        xx = np.array([lx.min() - .1, lx.max() + .1])
        ax.plot(xx, y0 + s.median_slope * (xx - x0), color=col, lw=1., ls=(0, (3, 1.5)), zorder=2)
        st = '***' if s.q < .001 else '**' if s.q < .01 else '*' if s.q < .05 else 'n.s.'
        ax.text(.03, .99 - k * .1, f'{name}: slope {s.median_slope:+.2f} {st}'.replace('-', '−'), transform=ax.transAxes, color=col,
                fontsize=6.5, va='top')
        if term == 'T':
            lab_pos = {'V1': (0, -.05), 'V2': (0, .045), 'V3': (0, .045), 'hV4': (0, .045), 'LO': (.17, -.01),
                       'V3AB': (0, .045)}
            for roi, xi, yi in zip(ROI_ORDER, lx, r['median']):
                if roi in lab_pos:
                    dx, dy = lab_pos[roi]
                    ax.text(xi + dx, yi + dy, roi, fontsize=5.5, color='.4', ha='center', va='center')
            xs = lx[[ROI_ORDER.index(q) for q in ('VO', 'TO', 'IPS')]]
            ax.text(xs.max() + .02, .03, 'VO, TO, IPS', fontsize=5.5,
                    color='.4', ha='right', va='center')
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ticks = [.6, 1., 1.6]
    ax.set_xticks(np.log2(ticks))
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.set_xlim(np.log2(.5), np.log2(1.8))
    ax.set_ylim(-.12, .58)
    ax.set_yticks([0, .2, .4])
    ax.set_xlabel('pRF size, σ (°)')
    ax.set_ylabel('Median gain')
    offset_spines(ax)


def panel_td(ax, d, t, ylim=(-.4, 1.2), off=.2):
    """Target and distractor gain side by side per ROI; stars in two rows above the data, one per term."""
    lo, hi = ylim
    for k, (term, col) in enumerate((('T', TARGET), ('D', DISTRACTOR))):
        dx = (k - .5) * 2 * off
        for i, roi in enumerate(ROI_ORDER):
            v = d[d.roi == roi][term].dropna().to_numpy()
            rng = np.random.default_rng(10 * k + i)
            xx = ROI_X[roi] + dx + rng.uniform(-.09, .09, len(v))
            ok = (v >= lo) & (v <= hi)
            ax.scatter(xx[ok], v[ok], s=6, color=col, alpha=.35, lw=0, zorder=1)
            ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=9, marker='^', color=col, alpha=.6, lw=0, clip_on=False)
            ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=9, marker='v', color=col, alpha=.6, lw=0, clip_on=False)
        r = t[(t.model == 'DoG') & (t.term == term)].set_index('roi').loc[ROI_ORDER]
        x = roi_positions(ROI_ORDER) + dx
        ax.errorbar(x, r['median'], yerr=[r['median'] - r.ci_low, r.ci_high - r['median']], fmt='o', ms=5, color=col, mec='w',
                    mew=.7, lw=1.5, capsize=0, zorder=4)
        for xi, (_, row) in zip(roi_positions(ROI_ORDER), r.iterrows()):
            ax.text(xi, hi + (.02 + .1 * k) * (hi - lo), stars(row.q), ha='center', va='bottom', fontsize=8.5, color=col,
                    clip_on=False)
    ax.hlines(0, ROI_X[ROI_ORDER[0]] - .6, ROI_X[ROI_ORDER[-1]] + .6, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_yticks([0, .5, 1])
    ax.set_ylabel('Gain')
    offset_spines(ax)


def panel_location(ax, cond, dt, ylim=(-.35, .3), off=.2):
    """Distractor (dD) and target (dT) gain, HP − LP, side by side per ROI, each estimated alone; stars per series above its
    own points (Wilcoxon, BH across ROIs)."""
    lo, hi = ylim
    for k, (df, term, col) in enumerate(((cond, 'dD', DISTRACTOR), (dt, 'dT', TARGET))):
        dx = (k - .5) * 2 * off
        ps = []
        for i, roi in enumerate(ROI_ORDER):
            v = df[df.roi == roi][term].dropna().to_numpy()
            rng = np.random.default_rng(10 * k + i)
            xx = ROI_X[roi] + dx + rng.uniform(-.09, .09, len(v))
            ok = (v >= lo) & (v <= hi)
            ax.scatter(xx[ok], v[ok], s=6, color=col, alpha=.35, lw=0, zorder=1)
            ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=9, marker='^', color=col, alpha=.6, lw=0, clip_on=False)
            ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=9, marker='v', color=col, alpha=.6, lw=0, clip_on=False)
            m, a, b = median_ci(v)
            ax.errorbar(ROI_X[roi] + dx, m, yerr=[[m - a], [b - m]], fmt='o', ms=5, color=col, mec='w', mew=.7, lw=1.5,
                        capsize=0, zorder=4)
            ps.append(stats.wilcoxon(v).pvalue)
        for roi, q in zip(ROI_ORDER, stats.false_discovery_control(ps)):
            ax.text(ROI_X[roi] + dx, hi + .03 * (hi - lo), stars(q), ha='center', va='bottom', fontsize=8.5, color=col,
                    clip_on=False)
    ax.hlines(0, ROI_X[ROI_ORDER[0]] - .6, ROI_X[ROI_ORDER[-1]] + .6, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_yticks([-.2, 0, .2])
    ax.set_ylabel('Gain\nHP − LP')
    offset_spines(ax)


LOSO_TERMS = [('T', 'Target', TARGET), ('DonT', 'Distractor', DISTRACTOR), ('dS', 'Sustained, HP − LP', HP),
              ('dD', 'Distractor, HP − LP', DISTRACTOR)]


def panel_loso(ax):
    """Leave-one-subject-out per ROI and term: % of subjects predicted better with the others' size of the term."""
    import matplotlib as mpl
    from retsupp.visualize.paper.fig45_common import generalization
    cmap = mpl.colormaps['RdBu_r']
    for i, (term, lab, col) in enumerate(LOSO_TERMS):
        g = generalization(term)
        for _, r in g.iterrows():
            x = ROI_X[r.roi]
            ax.add_patch(plt.Rectangle((x - .47, i - .45), .94, .9, color=cmap(.5 + .8 * (r.pct - 50) / 100), lw=0))
            sig = r.pct > 50 and r.q < .05
            strong = abs(r.pct - 50) > 30
            tc = ('w' if strong else FG) if sig else ('#EEF1F4' if strong else '.45')
            ax.text(x, i, f'{r.pct:.0f}', ha='center', va='center', fontsize=6.5, color=tc,
                    fontweight='bold' if sig else 'normal')
        ax.text(1.01, 1 - (i + .5) / len(LOSO_TERMS), lab, transform=ax.transAxes, ha='left', va='center', fontsize=6.5,
                color=col)
    ax.set_ylim(len(LOSO_TERMS) - .5, -.5)
    ax.set_yticks([])
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='x', length=0)


def dot_panel(ax, series, ylim, ylabel):
    """Per ROI, one or two series: subject dots (small), median and bootstrapped 95% CI on top, stars per series (BH across
    ROIs; one row of stars per series). series: (table, column, colour, summary with median/ci_low/ci_high/q or None)."""
    from retsupp.visualize.paper.fig7_behaviour import XP, XMAX, roi_axis7
    lo, hi = ylim
    off = [0.] if len(series) == 1 else [-.2, .2]
    jit = .15 if len(series) == 1 else .08
    for k, (df, col, color, sm) in enumerate(series):
        if sm is None:
            rows = []
            for roi in ROI_ORDER:
                v = df[df.roi == roi][col].dropna().to_numpy()
                m, a, b = median_ci(v)
                rows.append(dict(roi=roi, median=m, ci_low=a, ci_high=b, p=stats.wilcoxon(v).pvalue))
            sm = pd.DataFrame(rows).set_index('roi')
            sm['q'] = stats.false_discovery_control(sm.p)
        for i, roi in enumerate(ROI_ORDER):
            v = df[df.roi == roi][col].dropna().to_numpy()
            x = XP[roi] + off[k]
            xx = x + np.random.default_rng(10 * k + i).uniform(-jit, jit, len(v))
            ok = (v >= lo) & (v <= hi)
            ax.scatter(xx[ok], v[ok], s=4, color=color, alpha=.3, lw=0, zorder=1)
            ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=6, marker='^', color=color, alpha=.5, lw=0, clip_on=False)
            ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=6, marker='v', color=color, alpha=.5, lw=0, clip_on=False)
            r = sm.loc[roi]
            ax.errorbar(x, r['median'], yerr=[[r['median'] - r.ci_low], [r.ci_high - r['median']]], fmt='o', ms=4.5, color=color,
                        mec='w', mew=.6, lw=1.3, capsize=0, zorder=4)
            ax.text(XP[roi], hi + (.02 + .12 * k) * (hi - lo), stars(r.q), ha='center', va='bottom', fontsize=8.5, color=color,
                    clip_on=False)
    ax.hlines(0, -.6, XMAX + .6, color='.75', lw=.6, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)
    roi_axis7(ax, fontsize=6)


def panel_loso_t(ax):
    """Panel e transposed: ROIs as rows (pathway gaps as in the other panels), the four terms as columns. Fill: % of subjects
    predicted better with the others' size of the term (red above 50%, blue below); bold, q < .05."""
    import matplotlib as mpl
    from retsupp.visualize.paper.fig45_common import generalization
    from retsupp.visualize.paper.fig7_behaviour import XP, XMAX
    cmap = mpl.colormaps['RdBu_r']
    heads = [('Target', TARGET), ('Distractor', DISTRACTOR), ('Sustained', HP), ('Distractor', DISTRACTOR)]
    for j, (term, _, _) in enumerate(LOSO_TERMS):
        g = generalization(term)
        for _, r in g.iterrows():
            y = XP[r.roi]
            sig = r.pct > 50 and r.q < .05
            ax.add_patch(plt.Rectangle((j - .47, y - .45), .94, .9, color=cmap(.5 + .8 * (r.pct - 50) / 100), lw=0,
                                       alpha=1. if sig else .4))
            tc = ('w' if abs(r.pct - 50) > 30 else FG) if sig else MUTED
            ax.text(j, y, f'{r.pct:.0f}', ha='center', va='center', fontsize=6, color=tc, fontweight='bold' if sig else 'normal')
        ax.text(j, -.65, heads[j][0], ha='center', va='bottom', fontsize=6, color=heads[j][1])
    ax.plot([1.6, 3.4], [-1.55] * 2, color=MUTED, lw=.6, clip_on=False)
    ax.text(2.5, -1.62, 'HP − LP', ha='center', va='bottom', fontsize=6, color='.4')
    ax.set_xlim(-.5, len(LOSO_TERMS) - .5)
    ax.set_ylim(XMAX + .5, -.5)
    ax.set_yticks([XP[r] for r in ROI_ORDER])
    ax.set_yticklabels(ROI_ORDER, fontsize=6)
    ax.set_xticks([])
    for sp in ('left', 'bottom', 'top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=2)


def loso_table():
    """Panel e as a table: % of subjects predicted better with +δ, and the BH q, per term and ROI."""
    from retsupp.visualize.paper.fig45_common import generalization
    lines = ['', '**Panel b, per cell.** % of subjects predicted better with +δ (q, one-sided Wilcoxon, BH across ROIs).', '',
             '| Term | ' + ' | '.join(ROI_ORDER) + ' |', '|---|' + '---|' * len(ROI_ORDER)]
    for term, lab, _ in LOSO_TERMS:
        g = generalization(term).set_index('roi').loc[ROI_ORDER]
        lines.append(f'| {lab} | ' + ' | '.join(f'{r.pct:.0f} ({r.q:.2g})' for _, r in g.iterrows()) + ' |')
    return '\n'.join(lines) + '\n'


def hierarchy_figure(t, size, sl):
    fig = plt.figure(figsize=(SINGLE, 2.5))
    ax = fig.add_axes([.17, .2, .78, .72])
    panel_hierarchy(ax, t, size, sl)
    fmt_sl = lambda r: (f"{r.median_slope:+.2f} per doubling (95% CI [{r.ci_low:+.2f}, {r.ci_high:+.2f}], "
                        f"{r.pct_positive:.0f}% of subjects positive, p = {r.p:.2g}, q = {r.q:.2g})")
    save(fig, 'figS_gain_vs_prf_size', LEGEND_HIERARCHY.format(t_slope=fmt_sl(sl.loc['Target gain']),
                                                              d_slope=fmt_sl(sl.loc['Distractor gain'])))


def header(fig, y, text, x0=.012, x1=.99):
    fig.text(x0, y, text, fontsize=7.5, fontweight='bold', color=FG, va='bottom', ha='left')
    fig.add_artist(plt.Line2D([x0, x1], [y - .006] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


def main(gain_table, conditional_table, power_table, slope_table, sigma_table, name='fig5_gain_fields',
         dt_table='notes/data/conditional_dT_dog_2026-11.tsv'):
    from retsupp.visualize.paper.hierarchy_regression import roi_size
    js = {f: joint(gain_table, f) for f in FAM}
    cond = conditional(conditional_table)
    t = summarise(js, cond)
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(t, js)
    d = js['dog']
    fw, fw_roi, fw_bound = sustained_fwhm(js)
    sl = pd.read_csv(slope_table, sep='\t')
    sl = sl[sl.rois == 'all nine'].set_index('parameter')
    size = roi_size(sigma_table)

    from retsupp.visualize.paper.fig7_behaviour import icons, row
    W, H = DOUBLE, 4.62
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    Y = lambda dd: H - dd           # inches from the top
    LX, RX, PW = .55, 4.05, 2.95    # left and right columns of per-ROI panels

    # Row 1: a (target and distractor gain) | b (new subjects) and c (the sustained field to scale).
    row(fig, Y(.16), 'a', [('Gain at the ', FG), ('target', TARGET), (' beyond V1, at the ', FG), ('distractor', DISTRACTOR),
                            (' in V3AB, IPS, hV4, VO', FG)], [], x_let=.02, x_title=.2)
    ax = at(LX, Y(1.82), PW, 1.18)
    sm = lambda term: t[(t.model == 'DoG') & (t.term == term)].set_index('roi').loc[ROI_ORDER]
    dot_panel(ax, [(d, 'T', TARGET, sm('T')), (d, 'D', DISTRACTOR, sm('D'))], (-.4, 1.2), 'Gain')
    ax.set_yticks([0, .5, 1])
    ax = at(RX + .12, Y(2.12), 1.74, 1.45)
    panel_loso_t(ax)
    fig.text((RX - .13) / W, Y(.16) / H, 'b', fontsize=8, fontweight='bold', va='center', ha='left')
    fig.text((RX + .05) / W, Y(.16) / H, 'Predicts new subjects (%)', fontsize=7.5, color=FG, va='center', ha='left')
    ax = at(5.98, Y(1.58), .92, .92)
    panel_sketch(ax, fw, (fw_roi.min(), fw_roi.max()))
    fig.text(6.0 / W, Y(.16) / H, 'c', fontsize=8, fontweight='bold', va='center', ha='left')
    fig.text(6.16 / W, Y(.16) / H, 'Sustained field', fontsize=7, color=FG, va='center', ha='left')
    fig.text(6.57 / W, Y(.36) / H, 'To scale', fontsize=6, color='.45', va='center', ha='center')

    # Row 2: the gain at the HP location, d (sustained) | e (transient, distractor vs target; Fig. 1c, d); each with its
    # icon key under the title, so the two axes and titles line up.
    y2 = 2.38
    from retsupp.visualize.paper.paper_style import stimulus_icon
    hp_lp = [lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('hp',))),
             lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('lp',)))]
    row(fig, Y(y2), 'd', [('Sustained gain: less at ', FG), ('HP', HP)], [(hp_lp, HP, 'Whole block, at HP − at LP')],
        x_let=.02, x_title=.2, key_y=Y(y2 + .27))
    ax = at(LX, Y(y2 + .62 + 1.0), PW, 1.0)
    dot_panel(ax, [(cond, 'dS', HP, None)], (-.12, .06), 'Gain, HP − LP')
    ax.set_yticks([-.1, 0, .05])
    row(fig, Y(y2), 'e', [('Less transient gain at HP for the ', FG), ('distractor', DISTRACTOR), (', not the ', FG), ('target', TARGET)],
        [(icons('location', 'distractor'), DISTRACTOR, 'Each trial'), (icons('location', 'target'), TARGET, '')],
        x_let=RX - .5, x_title=RX - .32, key_y=Y(y2 + .27))
    fig.text((RX + PW) / W, Y(y2 + .27) / H, 'As Fig. 1c, d', fontsize=6, color='.45', ha='right', va='center')
    ax = at(RX, Y(y2 + .62 + 1.0), PW, 1.0)
    dot_panel(ax, [(cond, 'dD', DISTRACTOR, None), (pd.read_csv(dt_table, sep='\t'), 'dT', TARGET, None)], (-.35, .3),
              'Gain, HP − LP')
    ax.set_yticks([-.2, 0, .2])
    fig.text(.995, .005, PROVISIONAL, fontsize=6, color=MUTED, ha='right', va='bottom')
    save(fig, name, LEGEND.format(spacing=4 * np.sqrt(2), fwhm=fw, fw_bound=100 * fw_bound,
                                  fw_roi='; '.join(f'{r} {v:.1f}°' for r, v in fw_roi.items())) + loso_table())
    hierarchy_figure(t, size, sl)
    print('FWHM median', fw, 'per ROI', fw_roi.round(2).to_dict(), 'at bound', fw_bound)
    for term in ('T', 'D', 'dS', 'dD'):
        r = t[(t.model == 'DoG') & (t.term == term)]
        print(term, 'joint DoG', ', '.join(f'{a.roi} {a["median"]:+.3f} (q={a.q:.2g})' for _, a in r.iterrows()))
    for term in ('dS', 'dD'):
        ps = [stats.wilcoxon(cond[cond.roi == roi][term].dropna()).pvalue for roi in ROI_ORDER]
        meds = [cond[cond.roi == roi][term].median() for roi in ROI_ORDER]
        print(term, 'conditional', ', '.join(f'{r} {m:+.3f} (q={q:.2g})' for r, m, q in
                                             zip(ROI_ORDER, meds, stats.false_discovery_control(ps))))

    size_figure(d)
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--power-table', default='notes/data/cv_power_simulation_2026-11.tsv')
    ap.add_argument('--slope-table', default='notes/data/hierarchy_regression_2026-11.tsv',
                    help='per-subject slopes on log pRF size, from hierarchy_regression.py')
    ap.add_argument('--sigma-table', default='notes/data/fig1_prf_sigma.tsv')
    ap.add_argument('--name', default='fig5_gain_fields')
    ap.add_argument('--dt-table', default='notes/data/conditional_dT_dog_2026-11.tsv',
                    help='target gain HP − LP, estimated alone')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.power_table, a.slope_table, a.sigma_table, a.name, a.dt_table)
