"""Figure 5: the gain fields, the model's account of the responses at the item locations (Fig. 4). Provisional until the final fits.

Row 1: gain at the target and the distractor, per ROI and against pRF size (the hierarchy).
Row 2: gain at the HP location: the sustained field to scale, the sustained HP − LP gain (dS) and the
distractor-specific HP − LP gain (dD) per ROI.

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
**Figure 5. The gain fields: gain at the target and the distractor grows up the visual hierarchy; at the high-probability (HP) location there is less sustained gain.** Gain-field model fits with DoG pRFs (n = 30); a gain is the fractional change in drive at the centre of a field (+0.30: 30% more). Per-ROI panels (b, c, e, f): dots are subjects (triangles, beyond the axis), points and bars the median and bootstrapped 95% CI; stars, Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs, * q < .05, ** q < .01, *** q < .001. Squares under the data (New subjects): leave-one-subject-out test per ROI, the term added with the size estimated from the other 29 subjects (+δ) vs the opposite size (−δ); filled, a majority of subjects predicted better with +δ (one-sided Wilcoxon signed-rank test, Benjamini-Hochberg q < .05 across ROIs); open, not significant. The display icon right of each panel marks the item or location the gain applies to. a, Median target and distractor gain per ROI (as in b, c) against the ROI's pRF size (median over subjects of the voxel-weighted mean pRF σ at 1–2.5° eccentricity, log scale). Lines: slope per doubling of pRF size, the median of the per-subject OLS slopes over the nine ROIs, drawn through the centre of the points; target {t_slope}; distractor {d_slope} (Wilcoxon signed-rank test of the per-subject slopes, Benjamini-Hochberg over the eight parameters tested). b, Gain at the target location while the search display is on (all terms fitted jointly). c, Gain at the distractor, averaged over its locations (joint fit); for the New-subjects test, the distractor gain added on top of the target term. d, The sustained gain field over the search display, drawn to scale (red shading): one Gaussian per item (white discs, 4° eccentricity, neighbouring items {spacing:.1f}° apart; grey circle, the 3.17° aperture of the bar stimulus), with less gain at the HP than at the low-probability (LP) locations (the HP − LP difference exaggerated for display). Dashed circle: the full width at half maximum (FWHM = 2.355 σ) of the fitted sustained field, the median over all subjects and ROIs ({fwhm:.1f}°; per-ROI medians: {fw_roi}; {fw_bound:.0f}% of the fits at a bound of σ, 0.5° or 8°). The field reaches beyond the item and overlaps those of its neighbours. e, dS = g_HP − g_LP, the sustained gain at the HP location minus that at the LP locations, each term estimated alone on the item model with sustained gain (conditional estimate). f, dD = g_HP_dyn − g_LP_dyn, the gain the distractor draws at the HP location minus at an LP location, estimated alone as in e. When dS and dD are fitted together with all other terms (supplementary figure and parameter table), no ROI survives correction: dD stays at −7 to −10% in V3AB, IPS, hV4 and VO (q ≈ .09–.15), dS is −4 to −5% in V1–V3 (q ≥ .07). The measured responses at the item locations (Fig. 4) show less response at HP for every item type, significant in the early areas, without a distractor-specific difference. DN pRFs, Bayes factors and all other terms are in the parameter table. Provisional: current inputs.
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
        ax.text(xi, hi + .03 * (hi - lo), stars(row.q), ha='center', va='bottom', fontsize=6.5, color=col, clip_on=False)
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




def bare_roi_axis(ax):
    """ROI positions as in roi_axis(..., right_pad=ICON_PAD), without labels: for the upper panel of a stacked pair."""
    xs = [ROI_X[r] for r in ROI_ORDER]
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.set_xlim(xs[0] - .6, xs[-1] + .6 + ICON_PAD)


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


def header(fig, y, text, x0=.012, x1=.99):
    fig.text(x0, y, text, fontsize=7.5, fontweight='bold', color=FG, va='bottom', ha='left')
    fig.add_artist(plt.Line2D([x0, x1], [y - .006] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


def main(gain_table, conditional_table, power_table, slope_table, sigma_table, name='fig5_gain_fields'):
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

    W, H = DOUBLE, 6.8
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    RX, RW = 2.62, 4.4      # right column: the stacked per-ROI panels share their x positions

    header(fig, 6.58 / H, 'Gain at the target and the distractor: larger up the visual hierarchy')
    ax = at(.5, 4.15, 1.4, 1.75)
    panel_hierarchy(ax, t, size, sl)
    letter(ax, 'a', x=-.24, y=1.02)
    spec = (('T', TARGET, 'Target gain', ('target',), 5.3), ('D', DISTRACTOR, 'Distractor gain', ('distractor',), 4.05))
    for k, (term, col, lab, marks, y0) in enumerate(spec):
        ax = at(RX, y0, RW, 1.02)
        panel_gain(ax, d, t, term, col, (-.4, 1.2), None, None)
        ax.set_ylabel(lab)
        if k == 0:
            bare_roi_axis(ax)
        else:
            roi_axis(ax, right_pad=ICON_PAD)
        roi_icon(ax, marks, frac=(.35, .95))
        letter(ax, 'bc'[k], x=-.13, y=1.0)

    header(fig, 3.2 / H, 'Gain at the HP location: less sustained gain there, and less distractor gain in dorsal and ventral areas '
                          '(each term estimated alone)')
    ax = at(.05, .85, 1.95, 1.95)
    panel_sketch(ax, fw, (fw_roi.min(), fw_roi.max()))
    ax.text(.5, 1.0, 'Sustained gain field, to scale', transform=ax.transAxes, fontsize=7, color=FG, ha='center', va='bottom')
    ax.text(.5, -.02, 'Less gain where the\ndistractor is likely', transform=ax.transAxes, fontsize=6.5, color=HP, ha='center',
            va='top')
    letter(ax, 'd', x=.04, y=1.0)
    ax = at(RX, 2.0, RW, .9)
    panel_roi(ax, cond, ylabel='Sustained gain,\nHP − LP', annot=('Less gain at HP', (ROI_X['V3'] + .15, -.016), (4.6, -.1)))
    bare_roi_axis(ax)
    roi_icon(ax, ('hp', 'lp'), frac=(.3, .95))
    letter(ax, 'e', x=-.13, y=1.0)
    ax = at(RX, .66, RW, .88)
    panel_roi(ax, cond, 'dD', DISTRACTOR, (-.3, .2), 'Distractor gain,\nHP − LP',
              ('Less at HP', (ROI_X['V3AB'] - .15, -.045), (1.3, -.25)), band_label=True)
    roi_axis(ax, right_pad=ICON_PAD)
    roi_icon(ax, ('distractor', 'lp'), frac=(.3, .95))
    letter(ax, 'f', x=-.13, y=1.0)

    fig.text(.005, .005, PROVISIONAL, fontsize=5.5, color=MUTED, ha='left', va='bottom')
    fmt_sl = lambda r: (f"{r.median_slope:+.2f} per doubling (95% CI [{r.ci_low:+.2f}, {r.ci_high:+.2f}], "
                        f"{r.pct_positive:.0f}% of subjects positive, p = {r.p:.2g}, q = {r.q:.2g})")
    save(fig, name, LEGEND.format(spacing=4 * np.sqrt(2), fwhm=fw, fw_bound=100 * fw_bound,
                                  fw_roi='; '.join(f'{r} {v:.1f}°' for r, v in fw_roi.items()),
                                  t_slope=fmt_sl(sl.loc['Target gain']), d_slope=fmt_sl(sl.loc['Distractor gain'])))
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
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.power_table, a.slope_table, a.sigma_table, a.name)
