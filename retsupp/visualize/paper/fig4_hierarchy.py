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
**Figure 4. The target and, less so, the distractor draw gain, increasingly along the visual hierarchy.** Rows: target (a-c) and distractor (d-f); columns: the gain-field model (a, d), the measured responses (b, e) and their difference from a neutral item (c, f). a, Gain at the target location while the search display is on, per ROI (m7_full, DoG pRFs): dots are subjects (n = 30; triangles, beyond the axis), points and bars are the median and bootstrapped 95% CI. Stars: Wilcoxon signed-rank test against 0, Benjamini-Hochberg across ROIs; * q < .05, ** q < .01, *** q < .001. The target gain rises with pRF size: per subject, the OLS slope of the gain on log₂ pRF size over the nine ROIs (pRF size: group median at 1-2.5° eccentricity) has a median of +0.24 per doubling of pRF size (95% CI [+0.15, +0.27], 90% of subjects positive; Wilcoxon p < .001, Benjamini-Hochberg over the eight parameters tested, q < .001). Squares under the data (New subjects): leave-one-subject-out test per ROI, the term added with the size estimated from the other 29 subjects (+δ) vs the opposite size (−δ); filled, a majority of subjects predicted better with +δ (one-sided Wilcoxon signed-rank test, Benjamini-Hochberg q < .05 across ROIs); open, not significant. b, Deconvolved response at a pRF's item location (ROIs pooled weighted by voxel count; mean over subjects ± SEM; each subject's pre-onset mean, −4 to 0 s, subtracted) when the target is there and when a neutral item is there, both averaged over HP and LP locations; dark lines, higher areas (V3AB, IPS, hV4, VO, LO, TO); pale lines, early areas (V1-V3). c, Target minus neutral item per pathway, mean ± SEM over subjects. Stars: the mean over 3.2-6.4 s (grey band) against 0, Wilcoxon signed-rank test, Benjamini-Hochberg over the three pathways (all q < 10⁻⁸). d, As a, for the distractor gain averaged over its locations; for the New-subjects test, the distractor gain added on top of the target term (m3b vs m1). Its slope on log₂ pRF size is +0.06 per doubling (95% CI [+0.03, +0.11], 77% of subjects positive; p = .005, q = .010). e, f, As b, c, for the distractor at LP locations against a neutral item at LP locations (all q < .001). The measured distractor response exceeds the neutral item's in the occipitotemporal pathway as a whole, including LO and TO, where the model's distractor gain is not significant (d). DN pRFs, Bayes factors and all other terms are in the parameter table. Provisional: current inputs.
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



PEAK = (3.2, 6.4)   # s after search onset: the window tested in the difference time courses


def pstars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


def item_pair(p, term):
    """Per subject and time: the item's response and the neutral item's, matched as in the difference (c, f)."""
    w = p.pivot_table(index=['subject', 'grp', 't'], columns='cls', values='r')
    if term == 'T':   # same mix of HP and LP locations for the target and the neutral item
        item, neutral = (w.target_hp + w.target_lp) / 2, (w.neutral_hp + w.neutral_lp) / 2
    else:             # the distractor at LP locations only, against the neutral item at LP locations
        item, neutral = w.distractor_lp, w.neutral_lp
    return item, neutral


def long(s):
    return s.rename('r').reset_index()


def panel_raw(ax, raw, term, col, lab):
    """Measured response at the item's location vs a neutral item's, in early and in higher areas."""
    from retsupp.visualize.paper.fig45_common import tc_line
    from retsupp.visualize.paper.fig_event_responses_2x3 import NEUTRAL
    item, neutral = item_pair(raw, term)
    x = pd.concat([long(item).assign(cls='item'), long(neutral).assign(cls='neutral')])
    peaks = {}
    for grp, alpha, lw in (('Early areas', .5, 1.), ('Higher areas', 1., 1.4)):
        g = x[x.grp == grp]
        for cls, c in (('neutral', NEUTRAL), ('item', col)):
            q = g[g.cls == cls].pivot(index='subject', columns='t', values='r')
            m, se = q.mean(), q.sem()
            ax.fill_between(m.index, m - se, m + se, color=c, alpha=.15 * alpha, lw=0)
            ax.plot(m.index, m.values, color=c, lw=lw, alpha=alpha)
            if cls == 'item':
                peaks[grp] = (m.idxmax(), m.max())
    for grp, (t, v) in peaks.items():
        ax.text(t + 1.8, v, grp, fontsize=6, color='.35' if grp.startswith('Early') else FG, va='bottom', ha='left')
    ax.text(1., .41, lab, transform=ax.transAxes, color=col, fontsize=6.5, ha='right', va='center')
    ax.text(1., .31, 'Neutral item', transform=ax.transAxes, color=NEUTRAL, fontsize=6.5, ha='right', va='center')
    ax.axhline(0, color='.75', lw=.6, zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.set_ylim(-.3, 1.4)
    ax.set_yticks([0, .5, 1])
    ax.set_xticks([0, 10, 20])
    ax.set_ylabel('Response (% signal)')
    ax.set_xlabel('Time from search onset (s)')
    offset_spines(ax)


def panel_diff(ax, p, term, window_label=False):
    """Item minus neutral item per pathway, mean ± SEM; stars: the peak window (grey) against 0."""
    from retsupp.visualize.paper.fig3_prf_fits import PATHWAY, PATHWAY_COL
    item, neutral = item_pair(p, term)
    x = long(item - neutral)
    ps, out = [], {}
    for grp in PATHWAY:
        g = x[x.grp == grp].pivot(index='subject', columns='t', values='r')
        m, se = g.mean(), g.sem()
        ax.fill_between(m.index, m - se, m + se, color=PATHWAY_COL[grp], alpha=.18, lw=0)
        ax.plot(m.index, m.values, color=PATHWAY_COL[grp], lw=1.4)
        v = g.loc[:, (g.columns >= PEAK[0]) & (g.columns <= PEAK[1])].mean(1)
        ps.append(stats.wilcoxon(v).pvalue)
        out[grp] = v.mean()
    qs = stats.false_discovery_control(ps)
    for k, (grp, q) in enumerate(zip(PATHWAY, qs)):
        print(f'{term} - neutral, {grp}: mean {out[grp]:+.3f}, p = {ps[k]:.2g}, q = {q:.2g}')
        lab = f'{grp} {stars(q) or "n.s."}'
        ax.text(1.02, .99 - k * .11, lab, transform=ax.transAxes, color=PATHWAY_COL[grp], fontsize=6.5, ha='right', va='top')
    ax.axvspan(*PEAK, color='.93', lw=0, zorder=0)
    if window_label:
        ax.text(sum(PEAK) / 2, -.105, 'Tested', fontsize=5.5, color='.45', ha='center', va='bottom')
    ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.set_xticks([0, 10, 20])
    ax.set_ylim(-.12, .4)
    ax.set_yticks([-.1, 0, .1, .2, .3])
    ax.set_xlabel('Time from search onset (s)')
    ax.set_ylabel('Extra response (% signal)')
    offset_spines(ax)
    return ps, qs


def main(gain_table, conditional_table, name='fig4_hierarchy', deconv_folder='notes/data/event_deconv_2026-11'):
    from retsupp.visualize.paper.fig3_prf_fits import PATHWAY
    from retsupp.visualize.paper.fig45_common import item_timecourses
    js = {f: joint(gain_table, f) for f in FAM}
    t = summarise(js, conditional(conditional_table))
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(t, js)
    d = js['dog']
    raw = item_timecourses(deconv_folder, groups={r: ('Early areas' if r in PATHWAY['Early'] else 'Higher areas') for r in ROI_ORDER})
    pw = item_timecourses(deconv_folder, groups={r: g for g, rs in PATHWAY.items() for r in rs})

    # Rows tell one story each (target, distractor); columns: model gain | measured response | difference.
    W, H = DOUBLE, 5.0
    cols = [(.62, 2.2), (3.42, 1.55), (5.5, 1.5)]   # (left, width) in inches
    rows = [(3.05, 1.35), (.67, 1.35)]                # (bottom, height) in inches
    ax_at = lambda r, c: fig.add_axes([cols[c][0] / W, rows[r][0] / H, cols[c][1] / W, rows[r][1] / H])
    fig = plt.figure(figsize=(W, H))
    spec = (('T', TARGET, 'Target', 'Target: more gain up the hierarchy', 'Target minus neutral item', ('target',)),
            ('D', DISTRACTOR, 'Distractor', 'Distractor: less gain than the target', 'Distractor minus neutral item',
             ('distractor',)))
    first = []
    for r, (term, col, lab, title, dtitle, icon) in enumerate(spec):
        ax = ax_at(r, 0)
        panel_gain(ax, d, t, term, col, (-.4, 1.2), None, None)
        roi_axis(ax)
        ax.set_ylabel('Gain at the field centre')
        title_icon(ax, title, icon, col, h=.26, y=1.1)
        letter(ax, 'ad'[r], x=-.2, y=1.19)
        ax = ax_at(r, 1)
        panel_raw(ax, raw, term, col, lab)
        ax.text(0, 1.23, f'{lab} vs neutral item', transform=ax.transAxes, fontsize=7, color=FG, va='center')
        letter(ax, 'be'[r], x=-.3, y=1.19)
        ax = ax_at(r, 2)
        panel_diff(ax, pw, term, window_label=r == 0)
        ax.text(0, 1.23, dtitle, transform=ax.transAxes, fontsize=7, color=col, va='center')
        letter(ax, 'cf'[r], x=-.3, y=1.19)
        if r == 0:
            provisional(ax, x=1.05, y=1.36)
    save(fig, name, LEGEND)
    size_figure(d)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--deconv-folder', default='notes/data/event_deconv_2026-11',
                    help='deconvolved item responses; notes/data/event_deconv_dct_2026-11 for the DCT basis')
    ap.add_argument('--name', default='fig4_hierarchy')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.name, a.deconv_folder)
