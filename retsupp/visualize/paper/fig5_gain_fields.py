"""Figure 5: the gain fields, the model's account of the responses at the item locations (Fig. 3). Provisional until the final fits.

Top row: a, target and distractor gain per ROI; b, the sustained field to scale. Bottom row: c, the sustained HP − LP gain
(dS); d, distractor (dD) and target (dT) HP − LP gain side by side, the gain analogue of Fig. 1c, d. The leave-one-subject-out
test per ROI is the supplementary figure ``figS_loso_per_roi``.
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
                                                 roi_positions, save as _save)

FWHM = 2 * np.sqrt(2 * np.log(2))
GAIN_TABLE = None
SUFFIX = ''      # --suffix: every output gets it (new names; the earlier files stay)
FINAL = False    # --final: final fits; findings in the legends computed from the data, no provisional stamps
WIDTHS_SUPP = False   # --widths-supplement: b is the to-scale sketch only; the FWHM plot goes to figS_field_widths
OUTPUTS = None        # --outputs: only these figures (main, loso, hierarchy, size, pooled); the table is always written
PERCENT = False       # --percent: the pooled and hierarchy supplements in % (gains × 100), labels without model codes
FIT_LABEL = None      # --fit-label: how the parameter table and supplements name the fits


def wanted(k):
    return OUTPUTS is None or k in OUTPUTS


def save(fig, name, legend):
    if FINAL:
        legend = legend.replace(' Provisional: current inputs.', '').replace('Provisional: current inputs.', '')
    _save(fig, name + SUFFIX, legend)
TERMS = [('T', 'Target gain', TARGET), ('D', 'Distractor gain', DISTRACTOR), ('dS', 'Sustained gain, HP − LP', HP),
         ('dD', 'Distractor gain, HP − LP', DISTRACTOR)]
FAM = {'dog': 'DoG', 'dn': 'DN', 'gauss': 'Gaussian'}
LEGEND = """
**Figure 5. The gain fields: gain at the target and the distractor, and less sustained gain at the high-probability (HP) location.** Gain-field model fits with DoG pRFs (n = 30); a gain is the fractional change in drive at the centre of a field (+0.30: 30% more). a, c, d: dots are subjects (triangles, beyond the axis), points and bars the median and bootstrapped 95% CI; stars, Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs, * q < .05, ** q < .01, *** q < .001 (exact p and q per ROI in the parameter table). a, Gain at the target (orange) and at the distractor (green, averaged over its locations) while the search display is on, all terms fitted jointly. The gain against pRF size is in a supplementary figure. b, The sustained gain field over the search display, drawn to scale (purple shading): one Gaussian per item (white discs, 4° eccentricity, neighbouring items {spacing:.1f}° apart; grey circle, the 3.17° aperture of the bar stimulus), with less gain at the HP than at the LP locations (the HP − LP difference exaggerated for display). Purple dashed circle around the HP item: the full width at half maximum (FWHM = 2.355 σ) of the fitted sustained field, the median over all subjects and ROIs ({fwhm:.1f}°); thin grey circles, the median per ROI. Below: FWHM per subject and ROI of the sustained field (purple) and of the transient field shared by the target and distractor gain (grey), side by side (log axis; dots, subjects; points and bars, median and bootstrapped 95% CI; open triangles at the axis edge, fits at a bound of σ, 0.5° or 8°, i.e. FWHM 1.2° or 18.8°); dotted line, the spacing of neighbouring items. Sustained: {fw_bound:.0f}% of all fits at a bound (per ROI {fw_bound_roi}); median FWHM [interquartile range over subjects] {fw_iqr}. Transient: often at a bound (per ROI {tr_bound_roi}; IPS mostly at the upper bound), so its width is poorly constrained and trades off with the transient gain amplitude; median FWHM [IQR] {tr_iqr}. The sustained width is the well-determined one.{width_note}{overlap_note} c, dS = g_HP − g_LP, the sustained gain at the HP location minus that at the low-probability (LP) locations, each term estimated alone on the item model with sustained gain (conditional estimate). d, The gain analogue of Fig. 1c, d: dD = g_HP_dyn − g_LP_dyn, the gain the distractor draws at the HP location minus at an LP location (green), next to dT, the target gain at the HP location minus at an LP location (orange), each estimated alone as in c; the display icons are those of Fig. 1 (HP ringed in purple; the item at HP minus at an LP location). {d_findings} {loso_ref} DN pRFs, Bayes factors and all other terms are in the parameter table. Provisional: current inputs.
"""
LEGEND_LOSO = """
**Figure S. Do the gain terms predict new subjects?** Leave-one-subject-out test per ROI and term: the term added with the size estimated from the other 29 subjects (+δ) vs the opposite size (−δ); numbers, the percentage of subjects predicted better with +δ (cell colour: charcoal above 50%, gold below); bold: q < .05 (non-significant cells muted), one-sided Wilcoxon signed-rank test, Benjamini-Hochberg across the nine ROIs per term (q per cell below). The distractor gain is tested on top of the target term, whose size and field width are the subject's own fit ("with target"); added to the pRF model alone it predicts new subjects no better (52–67% per ROI, TO 8%). These tests use the earlier gain fits (canonical HRF, before the selection on cross-run reliability) and will be redone on the final fits. Provisional.
"""
LEGEND_DN = """
**Figure S. The gain fields with divisive-normalization (DN) pRFs (as Fig. 5, which uses DoG pRFs).** The same gain-field model (m7_full) fitted on DN pRFs (earlier fits, canonical HRF; the final DN fits with per-ROI HRFs are pending); n = 30. Dots are subjects (triangles, beyond the axis), points and bars the median and bootstrapped 95% CI; stars, Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the nine ROIs, * q < .05, ** q < .01, *** q < .001. a, Gain at the target and at the distractor (averaged over its locations), all terms fitted jointly; {dn_a}. b, Field sizes as in Fig. 5b: sustained FWHM, median {fwhm:.1f}° ({fw_bound:.0f}% of the fits at a σ bound; per ROI {fw_bound_roi}; median [IQR] {fw_iqr}); transient FWHM median [IQR] {tr_iqr}, at a bound in {tr_bound_roi}. c, d: each term estimated alone, as in Fig. 5 (for DN, each subject's minimum of the exact ΔSSE curve of the term added alone; ROIs with at least 20 voxels, so n per ROI: {dn_n}); there is no DN target split by HP yet, so d has no target series. c, dS = g_HP − g_LP, estimated alone: {dn_ds_alone}; pooled over ROIs, per-subject median −0.014, p = .001. Joint DN fit for comparison: {dn_ds}. d, dD = g_HP_dyn − g_LP_dyn, estimated alone: {dn_dd_alone}. Joint DN fit: {dn_dd}. Provisional.
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
    t.to_csv(OUT / f'table_parameters_2026-11{SUFFIX}.tsv', sep='\t', index=False)
    k = 100. if PERCENT else 1.
    if PERCENT:
        fmt = lambda r: (f"{k * r['median']:+.1f}% [{k * r.ci_low:+.1f}, {k * r.ci_high:+.1f}]").replace('-', '−')
    else:
        fmt = lambda r: f"{r['median']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
    n_sub = t['n'].max() if 'n' in t else 30
    lines = [f'**Table S. Gain-field parameters per ROI (gain-field model with all terms fitted jointly, n = {n_sub}'
             + (f'; {FIT_LABEL}' if FIT_LABEL else '') + ').** '
             'Median [bootstrapped 95% CI], Wilcoxon p, BH q '
             'across 9 ROIs, JZS Bayes factor BF10 (r = .707), and for DoG the median of the estimate with each term added '
             'alone. ' + ('Gains in % (the change in drive at the centre of a field: +30%, 30% more); HP − LP '
                          'differences in percentage points. DN columns: the DN fits with the pRFs held at the fit '
                          'without gain fields (not refitted). The estimated-alone column ("DoG conditional") of '
                          'the target and distractor gain adds the term to base models that were not refitted; report '
                          'those two from the joint fit.' if PERCENT else
                          'Gains are the fractional change in drive at the centre of a field (+0.30: 30% more).'), '']
    for term, lab, _ in TERMS:
        lines += [f'**{lab}**', '', '| ROI | DoG median [CI] | p | q | BF10 | DoG conditional | DN median [CI] | p | q | BF10 |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for roi in ROI_ORDER:
            a = t[(t.model == 'DoG') & (t.term == term) & (t.roi == roi)].iloc[0]
            b = t[(t.model == 'DN') & (t.term == term) & (t.roi == roi)].iloc[0]
            lines.append(f"| {roi} | {fmt(a)} | {a.p:.3g} | {a.q:.3g} | {a.bf10:.2g} | "
                         + (f"{k * a.conditional_median:+.1f}%".replace('-', '−') if PERCENT else f"{a.conditional_median:+.3f}")
                         + f" | "
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
    text = '\n'.join(lines) + '\n'
    if FINAL:
        text = text.replace(' Provisional: current inputs.', '')
    (OUT / f'table_parameters_2026-11{SUFFIX}.md').write_text(text)


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
    if not FINAL:
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
    k = 100. if PERCENT else 1.
    labs = ('Estimated\nalone', 'Joint fit') if PERCENT else ('Conditional', 'Joint (m7)')
    sets = [(labs[0], cond.groupby('subject')[term].median() * k), (labs[1], js['dog'].groupby('subject')[term].median() * k)]
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
        if PERCENT:     # exact p
            ptxt = f'p = {p:.2g}'.replace('0.', '.').replace('e-0', ' × 10^-').replace('e-', ' × 10^-')
            if '10^-' in ptxt:
                m_, e_ = ptxt.split(' × 10^-')
                ptxt = f'{m_} × 10' + ''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in e_).join(['⁻', ''])
        ax.text(i, hi + .03 * (hi - lo), f'{(v < 0).mean() * 100:.0f}% < 0\n{ptxt}', ha='center', va='bottom', fontsize=6, color=FG,
                clip_on=False)
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s[0] for s in sets], fontsize=6.5)
    ax.set_xlim(-.5, len(sets) - .5)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(f'{term}, median over areas (%)' if PERCENT else f'{term}, median over ROIs')
    offset_spines(ax)


def pooled_legend_pct(cond, js):
    """figS_pooled_dS_dD legend in % with the numbers computed (estimated alone vs joint fit)."""
    def one(v):
        v = v.dropna() * 100
        m, a, b = median_ci(v.to_numpy())
        p = stats.wilcoxon(v).pvalue
        return (f'{m:+.1f}% [{a:+.1f}, {b:+.1f}], {int((v < 0).sum())} of {len(v)} participants < 0, '
                f'Wilcoxon p = {p:.2g}').replace('-', '−')
    jd = js['dog']
    txt = {f'{term}_{w}': one(df.groupby('subject')[term].median())
           for term in ('dS', 'dD') for w, df in (('alone', cond), ('joint', jd))}
    lab = (FIT_LABEL[0].upper() + FIT_LABEL[1:]) if FIT_LABEL else 'Gain-field model fits with DoG pRFs'
    return (f"\n**Figure S. The HP − LP gain differences per participant, median over the nine areas.** {lab}; gains in % (differences in percentage points). Left in each panel, each term "
            f"estimated alone (added alone to the model with item weight, target, distractor and sustained gain); right, "
            f"the joint fit with all terms together. Dots, participants; points and bars, the median and bootstrapped "
            f"95% CI; above, the share of participants below 0 and the Wilcoxon p. a, dS, the sustained gain at HP "
            f"minus at the LP locations: estimated alone {txt['dS_alone']}; joint fit {txt['dS_joint']}. b, dD, the "
            f"distractor gain at HP minus at LP: estimated alone {txt['dD_alone']}; joint fit {txt['dD_joint']}. "
            f"n = {cond.subject.nunique()}.\n")


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



def sustained_fwhm(js, family='dog'):
    """FWHM (°) of the sustained field, m7_full: median over all subject x ROI fits, and per-ROI medians."""
    d = js[family]
    per_roi = FWHM * d.groupby('roi').sigma_AF.median().reindex(ROI_ORDER)
    at_bound = ((d.sigma_AF < .51) | (d.sigma_AF > 7.9)).mean()
    return FWHM * d.sigma_AF.median(), per_roi, at_bound


def panel_sketch(ax, fwhm, lo_hi, g_lp=1., g_hp=.2, hp_ix=3, ecc=4., aperture=3.17, item_r=.75, roi_fwhm=None, label=True):
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
    if roi_fwhm is not None:       # per-ROI medians, thin grey rings
        for f in roi_fwhm:
            ax.add_patch(Circle(hp, f / 2, fill=False, ec='.35', lw=.4, alpha=.7, zorder=3))
    ax.add_patch(Circle(hp, fwhm / 2, fill=False, ec=HP, lw=1., ls=(0, (2.5, 1.5)), zorder=3))
    if label:
        ax.text(hp[0] - .4, hp[1] - fwhm / 2 - .3, f'Fitted FWHM {fwhm:.1f}°', color=HP, fontsize=6, ha='center', va='top')
    ax.text(hp[0], hp[1], 'HP', color=HP, fontsize=6, ha='center', va='center', fontweight='bold', zorder=4)
    for i, p in enumerate(ring):
        if i != hp_ix:
            ax.text(p[0], p[1], 'LP', color='.4', fontsize=6, ha='center', va='center', zorder=4)
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




def legend_findings(cond, dt, js):
    """The findings of panels c, d in words with their numbers (final fits): per-ROI BH results and the pooled tests."""
    def per_roi(df, term):
        ps, meds = [], []
        for r in ROI_ORDER:
            v = df[df.roi == r][term].dropna()
            ps.append(stats.wilcoxon(v).pvalue)
            meds.append(v.median())
        q = stats.false_discovery_control(ps)
        sig = [f'{r} {m:+.3f} (q = {qq:.2g})' for r, m, qq in zip(ROI_ORDER, meds, q) if qq < .05]
        return sig, min(zip(q, ROI_ORDER))

    def pooled(df, term):
        v = df.groupby('subject')[term].median().dropna()
        m, a, b = median_ci(v)
        p = stats.wilcoxon(v).pvalue
        return f'per-subject median over ROIs {m:+.3f} [{a:+.3f}, {b:+.3f}], {(v < 0).mean() * 100:.0f}% of subjects < 0, p = {p:.2g}'

    both = cond[['subject', 'roi', 'dD']].merge(dt, on=['subject', 'roi'])
    both['dDmT'] = both.dD - both.dT
    out = []
    for df, term, lab in ((cond, 'dS', 'dS'), (cond, 'dD', 'The distractor term (dD)'), (dt, 'dT', 'The target term (dT)'),
                          (both, 'dDmT', 'Their difference (dD − dT)')):
        sig, (qmin, rmin) = per_roi(df, term)
        where = ('reliable in ' + ', '.join(sig)) if sig else f'no ROI q < .05 (smallest q = {qmin:.2g}, {rmin})'
        out.append(f'{lab}: {where}; {pooled(df, term)}')
    jd = js['dog']
    joint_txt = []
    for term in ('dS', 'dD'):
        sig, (qmin, rmin) = per_roi(jd, term)
        joint_txt.append(f'{term} ' + (('q < .05 in ' + ', '.join(sig)) if sig else f'no ROI q < .05 (smallest {qmin:.2g}, {rmin})')
                         + f'; {pooled(jd, term)}')
    d_findings = ('Estimated alone (c, d): ' + '. '.join(out) + '. Fitted jointly with all other terms: '
                  + '; '.join(joint_txt) + '. ' + 'The measured responses at the item locations (Fig. 3) are lower at '
                  'HP for every item type, without an item-specific difference.') if FINAL else ''
    d_findings = d_findings.replace('-0.', '−0.').replace('+0.', '+0.')
    loso_ref = ('How much each term adds to out-of-subject prediction (leave-one-subject-out) is in the supplementary '
                'table of model terms.') if FINAL else ''
    return dict(d_findings=d_findings, loso_ref=loso_ref,
                overlap_note=(' At half maximum the sustained field (median FWHM above) stays within the spacing of '
                              'neighbouring items in most ROIs (early areas ≈ 5°, others ≈ 3°).') if FINAL else
                ' The field reaches beyond the item and overlaps those of its neighbours.',
                width_note='' if FINAL else ' Widths of all fits; filtering on fit improvement pending on the final fits.',
                ds_text=out[0])


def dn_numbers(dn, dog, cond):
    """Per-ROI medians (with BH q) of the joint DN terms next to the joint DoG ones, for the DN legend."""
    def fmt(d, term):
        ps = [stats.wilcoxon(d[d.roi == r][term].dropna()).pvalue for r in ROI_ORDER]
        q = stats.false_discovery_control(ps)
        return ', '.join(f'{r} {d[d.roi == r][term].median():+.3f}{" (q = %.3g)" % qq if qq < .05 else ""}'
                         for r, qq in zip(ROI_ORDER, q))
    return dict(dn_a=f'target {fmt(dn, "T")}; distractor {fmt(dn, "D")} (DoG joint: target {fmt(dog, "T")}; distractor '
                     f'{fmt(dog, "D")})',
                dn_ds=f'{fmt(dn, "dS")} (DoG joint: {fmt(dog, "dS")})',
                dn_dd=f'{fmt(dn, "dD")} (DoG joint: {fmt(dog, "dD")})',
                dn_ds_alone=fmt(cond, 'dS'), dn_dd_alone=fmt(cond, 'dD'),
                dn_n=', '.join(f'{r} {cond[cond.roi == r].dS.notna().sum()}' for r in ROI_ORDER))


def fwhm_text(d):
    """Per-ROI median FWHM with the subjects' IQR, and the share of fits at a σ bound, for the legend (sustained and transient)."""
    out = {}
    for par, kx in (('sigma_AF', 'fw'), ('sigma_dyn', 'tr')):
        iqr, bnd = [], []
        for roi in ROI_ORDER:
            sg = d[d.roi == roi][par].dropna()
            q1, m, q3 = (FWHM * sg).quantile([.25, .5, .75])
            iqr.append(f'{roi} {m:.1f}° [{q1:.1f}–{q3:.1f}]')
            bnd.append(f'{roi} {100 * ((sg < .51) | (sg > 7.9)).mean():.0f}%')
        out[f'{kx}_iqr'], out[f'{kx}_bound_roi'] = '; '.join(iqr), ', '.join(bnd)
    return out


TRANSIENT = '#6B7785'   # the transient field (shared by the target and distractor gain): neutral grey


def fwhm_strip(ax, d, pars=(('sigma_AF', HP),), keep=None, weight=None, bound_counts=False, fs=6):
    """FWHM of the fitted fields per subject and ROI (log axis): faint dots, median and bootstrapped 95% CI; fits at a σ
    bound (0.5° or 8°) as open triangles at the axis edge; dotted line, the spacing of neighbouring items. pars: one or two
    (column, colour), side by side per ROI.

    keep: optional {parameter: boolean column} — only fits where the field earns its place (e.g. a likelihood-ratio
    criterion on the fit improvement from adding it) are drawn and summarised. weight: optional {parameter: column} — the
    summary becomes a weighted median (e.g. by gain); the CI stays the unweighted bootstrap of the kept fits."""
    from retsupp.visualize.paper.fig7_behaviour import XP, XMAX, roi_axis7
    lo, hi = 1.1, 21.
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    off = [0.] if len(pars) == 1 else [-.2, .2]
    jit = .18 if len(pars) == 1 else .09
    for k, (par, col) in enumerate(pars):
        for i, roi in enumerate(ROI_ORDER):
            dr = d[d.roi == roi]
            if keep and par in keep:
                dr = dr[dr[keep[par]].astype(bool)]
            dr = dr.dropna(subset=[par])
            sg = dr[par].to_numpy()
            f = FWHM * sg
            bound = (sg < .51) | (sg > 7.9)
            x = XP[roi] + off[k]
            xx = x + np.random.default_rng(10 * k + i).uniform(-jit, jit, len(f))
            ax.scatter(xx[~bound], f[~bound], s=3, color=col, alpha=.3, lw=0, zorder=1)
            low, high = bound & (sg < .51), bound & (sg > 7.9)
            if bound_counts:      # fits at a σ bound as a count per area, above (upper bound) and below (lower bound)
                for sel, yv, va in ((high, hi * 1.1, 'bottom'), (low, lo / 1.16, 'top')):
                    if sel.any():
                        ax.text(x, yv, str(int(sel.sum())), color=col, fontsize=fs, ha='center', va=va, clip_on=False)
            else:
                for sel, yv, mk in ((low, lo, 'v'), (high, hi, '^')):
                    if sel.any():           # an empty scatter with open markers still draws one at the figure origin
                        ax.scatter(xx[sel], np.full(sel.sum(), yv), s=8, marker=mk, facecolor='none', edgecolor=col, lw=.5,
                                   alpha=.7, clip_on=False, zorder=2)
            if not len(f):
                continue
            m, a, b = median_ci(f)
            if weight and par in weight:
                w = dr[weight[par]].to_numpy()
                o = np.argsort(f)
                m = f[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), .5)]
            ax.errorbar(x, m, yerr=[[max(m - a, 0)], [max(b - m, 0)]], fmt='o', ms=3.8, color=col, mec='w', mew=.5, lw=1.1, capsize=0, zorder=4)
    sp = 4 * np.sqrt(2)
    ax.hlines(sp, -.6, XMAX + .6, color='.55', lw=.6, ls=':', zorder=0)
    ax.set_yticks([2, 5, 10, 20])
    ax.set_yticklabels(['2', '5', '10', '20'], fontsize=fs)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel('FWHM (°)', fontsize=fs + .5)
    offset_spines(ax)
    if bound_counts:      # room under the axis for the lower-bound counts
        ax.tick_params(axis='x', pad=12)
    roi_axis7(ax, fontsize=fs)


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
        r = t[(t.model == 'DoG') & (t.term == term)].set_index('roi').loc[ROI_ORDER].copy()
        kk = 100. if PERCENT else 1.
        r['median'] = r['median'] * kk
        ax.plot(lx, r['median'], 'o', ms=4.5, color=col, mec='w', mew=.6, zorder=3)
        s = slopes.loc[lab].copy()
        s['median_slope'] = s.median_slope * kk
        x0, y0 = lx.mean(), r['median'].mean()
        xx = np.array([lx.min() - .1, lx.max() + .1])
        ax.plot(xx, y0 + s.median_slope * (xx - x0), color=col, lw=1., ls=(0, (3, 1.5)), zorder=2)
        st = '***' if s.q < .001 else '**' if s.q < .01 else '*' if s.q < .05 else 'n.s.'
        sl_txt = f'{s.median_slope:+.0f}% per doubling' if PERCENT else f'{s.median_slope:+.2f}'
        ax.text(.03, .99 - k * .1, f'{name}: slope {sl_txt} {st}'.replace('-', '−'), transform=ax.transAxes, color=col,
                fontsize=6.5, va='top')
        if term == 'T':
            lab_pos = {'V1': (0, -.05), 'V2': (0, .045), 'V3': (0, .045), 'hV4': (0, .045), 'LO': (.17, -.01),
                       'V3AB': (0, .045)}
            for roi, xi, yi in zip(ROI_ORDER, lx, r['median']):
                if roi in lab_pos:
                    dx, dy = lab_pos[roi]
                    ax.text(xi + dx, yi + dy * kk, roi, fontsize=6 if PERCENT else 5.5, color='.4', ha='center',
                            va='center')
            xs = lx[[ROI_ORDER.index(q) for q in ('VO', 'TO', 'IPS')]]
            ax.text(xs.max() + .02, -6 if PERCENT else .03, 'VO, TO, IPS', fontsize=6 if PERCENT else 5.5,
                    color='.4', ha='right', va='center')
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ticks = [.6, 1., 1.6]
    ax.set_xticks(np.log2(ticks))
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.set_xlim(np.log2(.5), np.log2(1.8))
    if PERCENT:
        ax.set_ylim(-12, 82)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_ylabel('Median gain (%)')
    else:
        ax.set_ylim(-.12, .58)
        ax.set_yticks([0, .2, .4])
        ax.set_ylabel('Median gain')
    ax.set_xlabel('pRF size, σ (°)')
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
    cmap = mpl.colors.LinearSegmentedColormap.from_list('loso', ['#8C6D1F', '#FFFFFF', '#3E4C59'])   # % better: gold below 50%, charcoal above (no condition hue)
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
    predicted better with the others' size of the term (charcoal above 50%, gold below); bold, q < .05."""
    import matplotlib as mpl
    from retsupp.visualize.paper.fig45_common import generalization
    from retsupp.visualize.paper.fig7_behaviour import XP, XMAX
    cmap = mpl.colors.LinearSegmentedColormap.from_list('loso', ['#8C6D1F', '#FFFFFF', '#3E4C59'])   # % better: gold below 50%, charcoal above (no condition hue)
    heads = [('Target', TARGET), ('Distractor\n(with target)', DISTRACTOR), ('Sustained', HP), ('Distractor', DISTRACTOR)]
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
    lines = ['', '**Per cell.** % of subjects predicted better with +δ (q, one-sided Wilcoxon, BH across ROIs).', '',
             '| Term | ' + ' | '.join(ROI_ORDER) + ' |', '|---|' + '---|' * len(ROI_ORDER)]
    for term, lab, _ in LOSO_TERMS:
        g = generalization(term).set_index('roi').loc[ROI_ORDER]
        lines.append(f'| {lab} | ' + ' | '.join(f'{r.pct:.0f} ({r.q:.2g})' for _, r in g.iterrows()) + ' |')
    return '\n'.join(lines) + '\n'


def loso_figure():
    """Supplementary figure: the leave-one-subject-out table per ROI and term (formerly Fig. 5b)."""
    W, H = SINGLE, 2.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.62 / W, .08 / H, (W - .75) / W, 1.5 / H])
    panel_loso_t(ax)
    fig.text(.05 / W, (H - .12) / H, 'Predicts new subjects (% of subjects)', fontsize=7.5, color=FG, va='center', ha='left')
    leg = LEGEND_LOSO
    if FINAL:
        leg = leg.replace(' These tests use the earlier gain fits (canonical HRF, before the selection on cross-run reliability) and will be redone on the final fits. Provisional.',
                          ' Final DoG fits (one HRF per subject × ROI, voxels reliable across runs); signal voxels, those whose held-out response beats the model-free null.')
        leg = leg.replace(' added to the pRF model alone it predicts new subjects no better (52–67% per ROI, TO 8%).', '.')
    save(fig, 'figS_loso_per_roi', leg + loso_table())


def hierarchy_figure(t, size, sl):
    fig = plt.figure(figsize=(SINGLE, 2.5))
    ax = fig.add_axes([.17, .2, .78, .72])
    panel_hierarchy(ax, t, size, sl)
    fmt_sl = lambda r: (f"{r.median_slope:+.2f} per doubling (95% CI [{r.ci_low:+.2f}, {r.ci_high:+.2f}], "
                        f"{r.pct_positive:.0f}% of subjects positive, p = {r.p:.2g}, q = {r.q:.2g})")
    if PERCENT:
        fmt_sl = lambda r: (f"{100 * r.median_slope:+.0f}% per doubling (95% CI [{100 * r.ci_low:+.0f}%, "
                            f"{100 * r.ci_high:+.0f}%], {r.pct_positive:.0f}% of participants positive, p = {r.p:.2g}, "
                            f"q = {r.q:.2g})").replace('-', '−')
    leg = LEGEND_HIERARCHY.format(t_slope=fmt_sl(sl.loc['Target gain']), d_slope=fmt_sl(sl.loc['Distractor gain']))
    if FINAL and GAIN_TABLE:
        from scipy.stats import spearmanr
        g = pd.read_csv(GAIN_TABLE, sep='\t')
        g = g[(g.family == 'dog') & (g.model == 'm7_full') & g.roi.isin(ROI_ORDER)]
        rho = {c: g.groupby('subject')[['g_T_dyn', c]].apply(lambda x: spearmanr(x.g_T_dyn, x[c]).statistic).dropna()
               for c in ('item_scale', 'sigma_dyn')}
        txt = lambda c: f'median ρ = {rho[c].median():+.2f}, p = {stats.wilcoxon(rho[c]).pvalue:.2g}'.replace('-', '−')
        leg = leg.replace('within subjects, the target gain correlates with the fitted weight of the items relative to the bar (ρ = .25, p = .011) and negatively with the size of the gain field (ρ = −.22, p = .049)',
                          f'within subjects (Spearman ρ over the nine ROIs, Wilcoxon across subjects), the target gain does not track the fitted weight of the items relative to the bar ({txt("item_scale")}) but falls with the width of the transient field ({txt("sigma_dyn")})')
    if PERCENT:
        leg = (leg.replace('(as in Fig. 5a)', '(as in Fig. 5b; gains in %)')
               .replace('Median target and distractor gain per ROI', 'Median target and distractor gain per area')
               .replace("the ROI's pRF size", "the area's pRF size").replace('nine ROIs', 'nine areas'))
        if FIT_LABEL:
            leg = leg.replace(' Median target and distractor gain per area', f' {FIT_LABEL[0].upper() + FIT_LABEL[1:]}. Median target and distractor gain per area', 1)
    save(fig, 'figS_gain_vs_prf_size', leg)


def header(fig, y, text, x0=.012, x1=.99):
    fig.text(x0, y, text, fontsize=7.5, fontweight='bold', color=FG, va='bottom', ha='left')
    fig.add_artist(plt.Line2D([x0, x1], [y - .006] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


WIDTHS_LEGEND = """
**Figure S. Width of the gain fields.** Full width at half maximum (FWHM = 2.355 σ) per subject and area of the sustained field (purple) and of the transient field shared by the target and the distractor gain (grey), from the joint gain-field fits with DoG pRFs (m7, final fits; n = 30; log axis). Dots are subjects; points and bars, the median and its bootstrapped 95% CI over all fits, including those at a bound. Numbers above and below a series: fits at the upper (σ = 8°, FWHM 18.8°) or lower (σ = 0.5°, FWHM 1.2°) bound of σ, not drawn as dots. Dotted line: the spacing of neighbouring items ({spacing:.1f}°). Sustained: {fw_bound:.0f}% of all fits at a bound (per area {fw_bound_roi}); median FWHM [interquartile range over subjects] {fw_iqr}. Transient: often at a bound (per area {tr_bound_roi}; IPS mostly at the upper bound), so its width is poorly constrained and trades off with the transient gain amplitude; median FWHM [IQR] {tr_iqr}. The sustained width is the well-determined one.{overlap_note}
"""


def widths_figure(d, fmt):
    """figS_field_widths: the FWHM plot of Fig. 5b on its own, fits at a bound as counts per area."""
    W, H = DOUBLE, 2.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.75 / W, .66 / H, (W - 1.0) / W, 1.08 / H])
    fwhm_strip(ax, d, (('sigma_AF', HP), ('sigma_dyn', TRANSIENT)), bound_counts=True, fs=7)
    # Key, drawn with the panel's own marks.
    y = (H - .12) / H
    x = .75 / W
    for col, lab in ((HP, 'Sustained'), (TRANSIENT, 'Transient (target, distractor)')):
        fig.text(x, y, '●', color=col, fontsize=7, va='center', ha='left')
        t = fig.text(x + .12 / W, y, lab, color=col, fontsize=7, va='center', ha='left')
        fig.canvas.draw()
        x = t.get_window_extent().x1 / fig.dpi / W + .25 / W
    k = fig.add_axes([x, y - .02 / H, .22 / W, .04 / H])
    k.plot([0, 1], [0, 0], color='.55', lw=.6, ls=':')
    k.set_axis_off()
    t = fig.text(x + .27 / W, y, f'Item spacing, {4 * np.sqrt(2):.1f}°', color='.45', fontsize=7, va='center', ha='left')
    fig.canvas.draw()
    x = t.get_window_extent().x1 / fig.dpi / W + .25 / W
    fig.text(x, y, 'Numbers: fits at the σ bound (above, upper; below, lower)', color='.45', fontsize=7, va='center',
             ha='left')
    save(fig, 'figS_field_widths', WIDTHS_LEGEND.format(**fmt))


def main(gain_table, conditional_table, power_table, slope_table, sigma_table, name='fig5_gain_fields',
         dt_table='notes/data/conditional_dT_dog_2026-11.tsv', family='dog',
         dn_conditional_table='notes/data/conditional_exact_dog_dn_2026-11.tsv'):
    from retsupp.visualize.paper.hierarchy_regression import roi_size
    global GAIN_TABLE
    GAIN_TABLE = gain_table
    fams = pd.read_csv(gain_table, sep='\t', usecols=['family']).family.unique()
    js = {f: joint(gain_table, f) for f in FAM if f in fams}
    cond = conditional(conditional_table)
    t = summarise(js, cond)
    # dT estimated alone: its own table, or (dt_table == 'cond') the dT column of the conditional table
    dt = cond[['subject', 'roi', 'dT']].dropna() if dt_table == 'cond' else pd.read_csv(dt_table, sep='\t')
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(t, js)
    d = js[family]
    fw, fw_roi, fw_bound = sustained_fwhm(js, family)
    dn = family != 'dog'   # the robustness versions (DN, Gaussian pRFs)
    if dn:
        # Terms estimated alone: each subject's minimum of the exact ΔSSE curve ('own'), ROIs with ≥ 20 voxels.
        c = pd.read_csv(dn_conditional_table, sep='\t')
        c = c[(c.family == family) & (c.nvox >= 20) & c.roi.isin(ROI_ORDER)]
        cond = c.pivot_table(index=['subject', 'roi'], columns='mech', values='own').reset_index()
        dt = cond[['subject', 'roi', 'dT']].dropna() if 'dT' in cond else None
    sl = pd.read_csv(slope_table, sep='\t')
    sl = sl[sl.rois == 'all nine'].set_index('parameter')
    size = roi_size(sigma_table)

    from retsupp.visualize.paper.fig7_behaviour import icons, row
    W, H = DOUBLE, 4.84
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    Y = lambda dd: H - dd           # inches from the top
    LX, RX, PW = .55, 4.05, 2.95    # left and right columns of per-ROI panels

    # Row 1: a (target and distractor gain, wide) | b (the sustained field to scale).
    d_rois = t[(t.model == FAM[family]) & (t.term == 'D') & (t.q < .05)].roi.tolist()
    d_where = ' in ' + ', '.join(d_rois) if d_rois else ' nowhere reliably'
    t_rois = t[(t.model == FAM[family]) & (t.term == 'T') & (t.q < .05)].roi.tolist()
    t_where = (' in every area' if len(t_rois) == 9 else ' beyond V1' if set(t_rois) == set(ROI_ORDER) - {'V1'}
               else f' in {len(t_rois)} of 9 areas')
    row(fig, Y(.16), 'a', [('Gain at the ', FG), ('target', TARGET), (t_where, FG),
                            (', at the ', FG), ('distractor', DISTRACTOR), (d_where, FG)], [],
        x_let=.02, x_title=.2)
    ax = at(LX, Y(1.82), 3.7, 1.18)
    sm = lambda term: t[(t.model == FAM[family]) & (t.term == term)].set_index('roi').loc[ROI_ORDER]
    dot_panel(ax, [(d, 'T', TARGET, sm('T')), (d, 'D', DISTRACTOR, sm('D'))], (-.4, 1.6) if dn else (-.4, 1.2), 'Gain')
    ax.set_yticks([0, .5, 1, 1.5] if dn else [0, .5, 1])
    BX = 4.62                       # left edge of b
    if WIDTHS_SUPP:
        # b: the sustained field to scale, larger; the FWHM plot is figS_field_widths.
        ax = at(BX - .05, Y(2.05), 1.75, 1.75)
        panel_sketch(ax, fw, (fw_roi.min(), fw_roi.max()), roi_fwhm=fw_roi.values, label=False)
        fig.text((BX - .25) / W, Y(.16) / H, 'b', fontsize=8, fontweight='bold', va='center', ha='left')
        fig.text((BX - .07) / W, Y(.16) / H, 'The sustained field, to scale', fontsize=7.5, color=FG, va='center', ha='left')
        from matplotlib.patches import Circle as _C
        kx, kt, kb = BX + 1.72, Y(.75), Y(1.75)
        key = fig.add_axes([kx / W, kb / H, (7.07 - kx) / W, (kt - kb) / H])
        key.set_xlim(0, 7.07 - kx)
        key.set_ylim(kb - kt, 0)
        key.set_axis_off()
        key.add_patch(_C((.07, -.1), .05, fill=False, ec=HP, lw=1., ls=(0, (2.5, 1.5))))
        key.text(.16, -.1, f'FWHM,\nmedian {fw:.1f}°', fontsize=6.5, color=HP, va='center')
        key.add_patch(_C((.07, -.45), .05, fill=False, ec='.35', lw=.4))
        key.text(.16, -.45, 'Median\nper area', fontsize=6.5, color='.35', va='center')
    else:
        ax = at(BX + .05, Y(1.22), .94, .94)
        panel_sketch(ax, fw, (fw_roi.min(), fw_roi.max()), roi_fwhm=fw_roi.values, label=False)
        fig.text((BX - .25) / W, Y(.16) / H, 'b', fontsize=8, fontweight='bold', va='center', ha='left')
        fig.text((BX - .07) / W, Y(.16) / H, 'Size of the gain fields', fontsize=7.5, color=FG, va='center', ha='left')
        kx, kt, kb = BX + 1.12, Y(.36), Y(1.3)      # key column: x, top, bottom (inches; 1 data unit = 1 inch)
        key = fig.add_axes([kx / W, kb / H, (7.07 - kx) / W, (kt - kb) / H])
        key.set_xlim(0, 7.07 - kx)
        key.set_ylim(kb - kt, 0)
        key.set_axis_off()
        from matplotlib.patches import Circle as _C
        ys = -np.arange(6) * .155 - .07
        key.add_patch(_C((.07, ys[0]), .05, fill=False, ec=HP, lw=1., ls=(0, (2.5, 1.5))))
        key.text(.17, ys[0], f'FWHM, median {fw:.1f}°', fontsize=6, color=HP, va='center')
        key.add_patch(_C((.07, ys[1]), .05, fill=False, ec='.35', lw=.4))
        key.text(.17, ys[1], 'Median per ROI', fontsize=6, color='.35', va='center')
        key.text(.0, ys[2], 'Sustained field, to scale', fontsize=6, color='.45', va='center')
        for yy, col, lab in ((ys[3], HP, 'Sustained'), (ys[4], TRANSIENT, 'Transient (target, distractor)')):
            key.errorbar(.07, yy, yerr=.045, fmt='o', ms=3.5, color=col, mec='w', mew=.5, lw=1.1, capsize=0)
            key.text(.17, yy, lab, fontsize=6, color=col, va='center')
        key.plot([.01, .13], [ys[5]] * 2, color='.55', lw=.6, ls=':')
        key.text(.17, ys[5], f'Item spacing, {4 * np.sqrt(2):.1f}°', fontsize=6, color='.45', va='center')
        ax = at(BX + .1, Y(2.07), 7.0 - BX - .1, .62)
        fwhm_strip(ax, d, (('sigma_AF', HP), ('sigma_dyn', TRANSIENT)))

    # Row 2: the gain at the HP location, c (sustained) | d (transient, distractor vs target; Fig. 1c, d); each with its
    # icon key under the title, so the two axes and titles line up.
    y2 = 2.6
    from retsupp.visualize.paper.paper_style import stimulus_icon
    hp_lp = [lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('hp',))),
             lambda a: (a.set_axis_off(), stimulus_icon(a, (0, 0, 1, 1), ('lp',)))]
    title_c = [('Sustained gain: less at ', FG), ('HP', HP)]
    row(fig, Y(y2), 'c', title_c, [(hp_lp, HP, 'Whole block, at HP − at LP')], x_let=.02, x_title=.2, key_y=Y(y2 + .27))
    ax = at(LX, Y(y2 + .62 + 1.0), PW, 1.0)
    dot_panel(ax, [(cond, 'dS', HP, None)], (-.14, .08) if dn else (-.12, .06), 'Gain, HP − LP')
    ax.set_yticks([-.1, 0, .05])
    title_d = ([('Transient gain at HP: less for the ', FG), ('distractor', DISTRACTOR)] if not FINAL else
               [('Transient gain, HP − LP: ', FG), ('distractor', DISTRACTOR), (' and ', FG), ('target', TARGET)] if dn else
               [('Transient gain at HP: slightly less for ', FG), ('distractor', DISTRACTOR), (' and ', FG),
                ('target', TARGET)])
    if FINAL and not dn and 'dD' in cond:
        pd_ = stats.wilcoxon(cond.groupby('subject').dD.median().dropna()).pvalue
        if pd_ >= .05:   # the distractor term alone is not reliable: say what holds (no difference from the target)
            title_d = [('Transient gain at HP: slightly less, ', FG), ('target', TARGET), (' and ', FG),
                       ('distractor', DISTRACTOR), (' alike', FG)]
    keys_d = [(icons('location', 'distractor'), DISTRACTOR, 'Each trial')] + \
        ([] if (dn and dt is None) else [(icons('location', 'target'), TARGET, '')])
    row(fig, Y(y2), 'd', title_d, keys_d, x_let=RX - .5, x_title=RX - .32, key_y=Y(y2 + .27))
    fig.text((RX + PW) / W, Y(y2 + .27) / H, 'As Fig. 1c, d', fontsize=6, color='.45', ha='right', va='center')
    ax = at(RX, Y(y2 + .62 + 1.0), PW, 1.0)
    series_d = [(cond, 'dD', DISTRACTOR, None)] + ([] if dt is None else [(dt, 'dT', TARGET, None)])
    dot_panel(ax, series_d, (-.3, .25) if dn else (-.35, .3), 'Gain, HP − LP')
    ax.set_yticks([-.2, 0, .2])
    fig.text(.995, .005, ('' if FINAL else 'DN pRFs, earlier fits (canonical HRF)') if dn else ('' if FINAL else PROVISIONAL),
             fontsize=6, color=MUTED, ha='right',
             va='bottom')
    if dn:
        leg = LEGEND_DN
        if FINAL:
            lab = {'dn': 'divisive-normalization (DN) pRFs', 'gauss': 'Gaussian pRFs'}[family]
            short = FAM[family]
            leg = (leg.replace('divisive-normalization (DN) pRFs', lab)
                   .replace('fitted on DN pRFs (earlier fits, canonical HRF; the final DN fits with per-ROI HRFs are pending)',
                            f'fitted on {short} pRFs with their own HRF per subject × ROI, fitted jointly with the pRFs (final fits, voxels reliable across runs)')
                   .replace('(for DN, each', '(each').replace('; there is no DN target split by HP yet, so d has no target series', '; orange in d, the target term dT, estimated alone')
                   .replace('pooled over ROIs, per-subject median −0.014, p = .001', '{pooled_ds}')
                   .replace('Joint DN fit', f'Joint {short} fit').replace(' Provisional.', ''))
            v = cond.groupby('subject').dS.median().dropna()
            pooled_ds = (f'pooled over ROIs, per-subject median {v.median():+.3f}, {(v < 0).mean() * 100:.0f}% of subjects < 0, '
                         f'p = {stats.wilcoxon(v).pvalue:.2g}').replace('-', '−')
            if dt is not None:
                w = dt.groupby('subject').dT.median().dropna()
                pooled_ds += ''
            leg = leg.replace('{pooled_ds}', pooled_ds)
        save(fig, name, leg.format(fwhm=fw, fw_bound=100 * fw_bound, **fwhm_text(d), **dn_numbers(d, js['dog'], cond)))
        return
    fmt = dict(spacing=4 * np.sqrt(2), fwhm=fw, fw_bound=100 * fw_bound,
               fw_roi='; '.join(f'{r} {v:.1f}°' for r, v in fw_roi.items()), **fwhm_text(d), **legend_findings(cond, dt, js))
    if WIDTHS_SUPP:
        i0 = LEGEND.index(' Below: FWHM per subject and ROI')
        i1 = LEGEND.index('{width_note}{overlap_note}') + len('{width_note}{overlap_note}')
        leg = (LEGEND[:i0] + ' The width of the sustained and the transient field per subject and area is in Supplementary '
               'Fig. S (field widths).' + LEGEND[i1:]).replace('thin grey circles, the median per ROI',
                                                                'thin grey circles, the median per area')
        save(fig, name, leg.format(**fmt))
        widths_figure(d, fmt)
        return
    if wanted('main'):
        save(fig, name, LEGEND.format(**fmt))
    if wanted('loso'):
        loso_figure()
    if wanted('hierarchy'):
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

    if wanted('size'):
        size_figure(d)
    fig = plt.figure(figsize=(SINGLE, 2.3))
    gs = fig.add_gridspec(1, 2, wspace=.6, left=.17, right=.98, top=.8, bottom=.15)
    ax = fig.add_subplot(gs[0])
    k = 100. if PERCENT else 1.
    panel_pooled(ax, cond, js, ylim=(-.12 * k, .06 * k))
    letter(ax, 'a', x=-.35, y=1.12)
    ax = fig.add_subplot(gs[1])
    panel_pooled(ax, cond, js, 'dD', DISTRACTOR, (-.3 * k, .2 * k))
    letter(ax, 'b', x=-.35, y=1.12)
    if wanted('pooled'):
        save(fig, 'figS_pooled_dS_dD', pooled_legend_pct(cond, js) if PERCENT else POOLED_LEGEND)
    if FINAL:      # the power simulation ran on the earlier fits; not redrawn
        return
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
    ap.add_argument('--name', default=None, help='default fig5_gain_fields, or figS_gain_fields_dn for --family dn')
    ap.add_argument('--family', default='dog', choices=['dog', 'dn', 'gauss'])
    ap.add_argument('--robust-alone-table', default='notes/data/conditional_exact_dog_dn_2026-11.tsv',
                    help='exact own estimates (family, subject, roi, mech, own, nvox) for --family dn/gauss')
    ap.add_argument('--dt-table', default='notes/data/conditional_dT_dog_2026-11.tsv',
                    help='target gain HP − LP, estimated alone')
    ap.add_argument('--suffix', default='', help='appended to every output name')
    ap.add_argument('--final', action='store_true', help='final fits: findings computed in the legends, no provisional stamps')
    ap.add_argument('--loso-table', default=None, help='per-ROI LOSO table for figS_loso_per_roi (default fig45_common.LOSO_TABLE)')
    ap.add_argument('--widths-supplement', action='store_true',
                    help='b: the sustained field to scale only; the FWHM plot as figS_field_widths (fits at a bound counted)')
    ap.add_argument('--outputs', nargs='+', default=None, choices=['main', 'loso', 'hierarchy', 'size', 'pooled'],
                    help='only these figures (default all); the parameter table is always written')
    ap.add_argument('--percent', action='store_true', help='pooled and hierarchy supplements in %%, no model codes')
    ap.add_argument('--fit-label', default=None, help='how the table and supplements name the fits')
    a = ap.parse_args()
    SUFFIX, FINAL, WIDTHS_SUPP = a.suffix, a.final, a.widths_supplement
    OUTPUTS, PERCENT, FIT_LABEL = a.outputs, a.percent, a.fit_label
    if a.loso_table:
        import retsupp.visualize.paper.fig45_common as _fc
        _fc.LOSO_TABLE = a.loso_table
    main(a.gain_table, a.conditional_table, a.power_table, a.slope_table, a.sigma_table,
         a.name or ('fig5_gain_fields' if a.family == 'dog' else f'figS_gain_fields_{a.family}'), a.dt_table, a.family,
         a.robust_alone_table)
