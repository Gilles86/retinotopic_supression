"""Figure 4: gain terms across the visual hierarchy, and the full parameter table. Provisional until the final fits.

Writes the figure and ``table_parameters_2026-11.{md,tsv}`` (medians, CIs, tests and Bayes factors).

Usage::

    python -m retsupp.visualize.paper.fig4_hierarchy --gain-table notes/data/gain_field_parameters.tsv
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, joint, jzs_bf10, median_ci
from retsupp.visualize.paper.paper_style import (DISTRACTOR, DOUBLE, FG, HP, MUTED, OUT, ROI_ORDER, TARGET, letter, offset_spines,
                                                 pending, provisional, save)

FWHM = 2 * np.sqrt(2 * np.log(2))
TERMS = [('T', 'Target gain', TARGET), ('D', 'Singleton gain', DISTRACTOR), ('dS', 'Sustained gain, HP − LP', HP),
         ('dD', 'Singleton gain, HP − LP', '#5E3C99')]
FAM = {'dog': 'DoG', 'dn': 'DN'}
LEGEND = """
**Figure 4. Gain terms across the visual hierarchy.** a-d, Median across subjects (n = 30) per ROI of the target gain, the singleton gain (averaged over locations), and the HP − LP differences of the sustained and the singleton gain, from m7_full under DoG (filled) and DN (open); bars are bootstrapped 95% confidence intervals. Stars: Wilcoxon signed-rank test against 0, Benjamini-Hochberg across the 9 ROIs per term and model; * q < .05, ** q < .01, *** q < .001. e, Size of the sustained and transient fields (DoG) as full width at half maximum (2.355 σ), median with bootstrapped 95% CI, and the share of fits at a bound (σ = .5° or 8°) under each ROI; dotted line, the 4° radius of the ring. f, Term × ROI hierarchical model, pending. Bayes factors and all values are in the parameter table. All values provisional: current inputs.
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
            q = stats.false_discovery_control(ps)
            for r, qq in zip(rows[-len(ROI_ORDER):], q):
                r['q'] = qq
    return pd.DataFrame(rows)


def write_table(t):
    t.to_csv(OUT / 'table_parameters_2026-11.tsv', sep='\t', index=False)
    fmt = lambda r: f"{r['median']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
    lines = ['**Table S. Gain-field parameters per ROI (m7_full, n = 30).** Median [bootstrapped 95% CI], Wilcoxon p, BH q '
             'across 9 ROIs, JZS Bayes factor BF10 (r = .707), and for DoG the conditional median. Provisional: current inputs.', '']
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
                v = JS[fam][JS[fam].roi == roi][par].dropna()
                m, lo, hi = (FWHM * x for x in median_ci(v))
                cells += [f'{m:.1f} [{lo:.1f}, {hi:.1f}]', f'{((v < .51) | (v > 7.9)).mean() * 100:.0f}%']
        lines.append(f'| {roi} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Gain values are the fractional change in the drive at the centre of a field: +0.30 means 30% more drive '
              'there during the event, −0.05 means 5% less throughout the block.', '']
    (OUT / 'table_parameters_2026-11.md').write_text('\n'.join(lines))


def panel_term(ax, t, term, lab, col):
    x = np.arange(len(ROI_ORDER))
    for j, name in enumerate(('DoG', 'DN')):
        d = t[(t.model == name) & (t.term == term)].set_index('roi').loc[ROI_ORDER]
        xx = x + (j - .5) * .3
        ax.errorbar(xx, d['median'], yerr=[d['median'] - d.ci_low, d.ci_high - d['median']], fmt='o', ms=5, mfc=col if j == 0 else 'w',
                    mec=col, mew=1.2, color=col, lw=1.3, capsize=0)
        for xi, (_, r) in zip(xx, d.iterrows()):
            s = stars(r.q)
            if s:
                ax.text(xi, r.ci_high, s, ha='center', va='bottom', fontsize=7, color=col)
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER, fontsize=6.5)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title(lab, fontsize=7.5, color=col, pad=4)
    offset_spines(ax)


def panel_widths(ax, js):
    x = np.arange(len(ROI_ORDER))
    d = js['dog']
    for j, (par, lab, col) in enumerate((('sigma_AF', 'Sustained', HP), ('sigma_dyn', 'Transient', DISTRACTOR))):
        m = FWHM * np.array([median_ci(d[d.roi == r][par]) for r in ROI_ORDER])
        ax.errorbar(x + (j - .5) * .3, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='o', ms=5, color=col, lw=1.3, capsize=0)
        at_bound = [((d[d.roi == r][par] < .51) | (d[d.roi == r][par] > 7.9)).mean() * 100 for r in ROI_ORDER]
        for xi, b in zip(x, at_bound):
            ax.text(xi + (j - .5) * .3, -.8 - j * 2.1, f'{b:.0f}', fontsize=5.5, color=col, ha='center', va='top')
        ax.text(.02, .97 - j * .1, lab, transform=ax.transAxes, color=col, fontsize=7, va='top')
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_ORDER, fontsize=6.5)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylim(-5, 19.5)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.axhline(4, color='.7', lw=.6, ls=':', zorder=0)
    ax.text(len(ROI_ORDER) - .5, 4, 'Ring radius', fontsize=5.5, color='.5', ha='right', va='bottom')
    ax.set_ylabel('Field FWHM (°)')
    ax.text(-.02, .05, '% at\nbound', transform=ax.transAxes, fontsize=5.5, color=MUTED, ha='right')
    ax.set_title('Field widths, DoG', fontsize=7.5, color=FG, pad=10)
    offset_spines(ax)


def main(gain_table, conditional_table, name='fig4_hierarchy'):
    global JS
    js = {f: joint(gain_table, f) for f in FAM}
    JS = js
    t = summarise(js, conditional(conditional_table))
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(t)
    fig = plt.figure(figsize=(DOUBLE, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=.75, wspace=.35, left=.075, right=.99, top=.9, bottom=.1)
    for k, (term, lab, col) in enumerate(TERMS):
        ax = fig.add_subplot(gs[k // 2, k % 2] if k < 2 else gs[1, k - 2])
        panel_term(ax, t, term, lab, col)
        if k in (0, 2):
            ax.set_ylabel('Gain (m7)')
        letter(ax, 'abcd'[k], x=-.12, y=1.06)
        provisional(ax, y=1.2)
    ax = fig.add_subplot(gs[0, 2])
    panel_widths(ax, js)
    letter(ax, 'e', x=-.15, y=1.06)
    ax = fig.add_subplot(gs[1, 2])
    pending(ax, 'term × ROI hierarchical model')
    letter(ax, 'f', x=-.02, y=1.06)
    fig.text(.075, .965, '●  DoG     ○  DN', fontsize=7.5, color=FG)
    save(fig, name, LEGEND)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table)
