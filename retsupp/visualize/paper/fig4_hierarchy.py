"""Figure 4: the target and the distractor draw gain, more so higher in the hierarchy. Provisional until the final fits.

Also writes ``table_parameters_2026-11.{md,tsv}`` (all terms, DoG and DN, medians, CIs, tests, Bayes factors, field sizes).

Usage::

    python -m retsupp.visualize.paper.fig4_hierarchy --gain-table notes/data/gain_field_parameters.tsv
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, generalization_row, joint, jzs_bf10, median_ci
from retsupp.visualize.paper.paper_style import (SINGLE, DISTRACTOR, DOUBLE, FG, HP, MUTED, OUT, ROI_COLORS, ROI_ORDER, TARGET, letter,
                                                 offset_spines, provisional, save, stimulus_icon, ICON_PAD, ROI_X, roi_axis,
                                                 roi_icon, roi_positions, title_icon)

FWHM = 2 * np.sqrt(2 * np.log(2))
TERMS = [('T', 'Target gain', TARGET), ('D', 'Distractor gain', DISTRACTOR), ('dS', 'Sustained gain, HP − LP', HP),
         ('dD', 'Distractor gain, HP − LP', '#5E3C99')]
FAM = {'dog': 'DoG', 'dn': 'DN'}
LEGEND = """
**Figure 4. The target and the distractor draw gain, increasingly along the visual hierarchy.** a, Gain at the target location while the search display is on, per ROI (m7_full, DoG pRFs): dots are subjects (n = 30; triangles, beyond the axis), points and bars are the median and bootstrapped 95% CI. Stars: Wilcoxon signed-rank test against 0, Benjamini-Hochberg across ROIs; * q < .05, ** q < .01, *** q < .001. b, The same for the distractor, averaged over its locations. c, Per subject, the slope of the gain on log₂ pRF size over the nine ROIs (OLS; pRF size: group median at 1-2.5° eccentricity, from the pRF fit), so the change in gain per doubling of pRF size; median and 95% CI beside the dots; Wilcoxon against 0, Benjamini-Hochberg over the eight parameters tested (hierarchy_regression_2026-11.tsv). DN pRFs, Bayes factors and all other terms are in the parameter table. Squares under the data (New subjects): leave-one-subject-out test per ROI, the term added with the size estimated from the other 29 subjects (+δ) vs the opposite size (−δ); filled, a majority of subjects predicted better with +δ (Wilcoxon signed-rank, Benjamini-Hochberg q < .05 across ROIs); open, not significant; cross, significantly worse; for the distractor (b), its gain added on top of the target term (m3b vs m1). Provisional: current inputs.
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


def panel_slopes(ax, sl, reg):
    """Per-subject slope of target and distractor gain on log2 pRF size: one dot per subject."""
    rows = [('Target gain', 'Target', TARGET), ('Distractor gain', 'Distractor', DISTRACTOR)]
    for i, (par, lab, col) in enumerate(rows):
        v = sl[(sl.parameter == par) & (sl.rois == 'all nine')].slope.to_numpy()
        rng = np.random.default_rng(i)
        ax.scatter(i + rng.uniform(-.17, .17, len(v)), v, s=7, color=col, alpha=.35, lw=0, zorder=1)
        m, lo, hi = median_ci(v)
        ax.errorbar(i + .32, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=6, color=col, mec='w', mew=.8, lw=1.6, capsize=0, zorder=4)
        q = reg[(reg.parameter == par) & (reg.rois == 'all nine')].q.iloc[0]
        ax.text(i + .1, .7, stars(q), ha='center', va='bottom', fontsize=6.5, color=col)
    ax.hlines(0, -.5, 1.6, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_xticks([0.1, 1.1])
    ax.set_xticklabels([r[1] for r in rows])
    ax.set_xlim(-.5, 1.6)
    ax.set_ylim(-.4, .7)
    ax.set_yticks([-.4, -.2, 0, .2, .4, .6])
    ax.set_ylabel('Change in gain per\ndoubling of pRF size')
    offset_spines(ax)


def slope_note(reg, parameter, unit=''):
    r = reg[(reg.parameter == parameter) & (reg.rois == 'all nine')].iloc[0]
    return f'{r.median_slope:+.2f}{unit} per pRF-size doubling{stars(r.q)}'.replace('+0.', '+.').replace('-0.', '−.').replace('-', '−')


def size_figure(d):
    fig = plt.figure(figsize=(SINGLE, 2.4))
    ax = fig.add_axes([.17, .3, .8, .6])
    panel_size(ax, d)
    ax.set_title('Size of the fitted fields', fontsize=7, loc='left', color=FG, pad=6)
    for j, (lab, col) in enumerate((('Sustained field', HP), ('Transient field', DISTRACTOR))):
        ax.text(.98, .95 - j * .09, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='top')
    provisional(ax, y=1.08)
    save(fig, 'figS_field_size', LEGEND_SIZE)


def main(gain_table, conditional_table, name='fig4_hierarchy', regression_table='notes/data/hierarchy_regression_2026-11.tsv'):
    reg = pd.read_csv(regression_table, sep='\t')
    js = {f: joint(gain_table, f) for f in FAM}
    t = summarise(js, conditional(conditional_table))
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(t, js)
    d = js['dog']
    fig = plt.figure(figsize=(DOUBLE, 2.75))
    gs = fig.add_gridspec(1, 3, wspace=.45, left=.11, right=.99, top=.74, bottom=.22)
    ax = fig.add_subplot(gs[0])
    panel_gain(ax, d, t, 'T', TARGET, (-.4, 1.2), None, None)
    roi_axis(ax)
    title_icon(ax, 'Target: more gain up the hierarchy', ('target',), TARGET, h=.24)
    ax.set_ylabel('Gain at the field centre')
    letter(ax, 'a', x=-.2, y=1.26)
    provisional(ax, x=3.3, y=1.36)
    ax = fig.add_subplot(gs[1])
    panel_gain(ax, d, t, 'D', DISTRACTOR, (-.4, 1.2), None, None)
    roi_axis(ax)
    title_icon(ax, 'Distractor: less gain than the target', ('distractor',), DISTRACTOR, h=.24)
    letter(ax, 'b', x=-.12, y=1.26)
    ax = fig.add_subplot(gs[2])
    panel_slopes(ax, pd.read_csv(regression_table.replace('regression', 'slopes'), sep='\t'), reg)
    ax.text(0, 1.22, 'Slope over ROIs, per subject', transform=ax.transAxes, fontsize=7, color=FG, va='center', ha='left')
    letter(ax, 'c', x=-.3, y=1.26)
    save(fig, name, LEGEND)
    size_figure(d)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table)
