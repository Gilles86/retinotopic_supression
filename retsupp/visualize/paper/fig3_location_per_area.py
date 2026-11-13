"""Fig. 3d, e per area: the location-type effect (HP, next to HP, opposite HP; items pooled) in each of the nine areas.

Same measure as Fig. 3e (``fig3_item_locations.rel_stats``) and ``richter_followup.loc_values``: per subject and area,
the window mean (3.2–6.4 s after search onset) of the relative deconvolution per item × location type, in % of the
subject's mean over the twelve cells; subject × area pairs whose mean response is not positive are left out; the
pilots sub-01, sub-02 are left out (n ≤ 28). Items are pooled (mean over target, distractor, neutral); next = mean of
the two neighbouring locations. Per area: one-way repeated-measures ANOVA on location type (pingouin ``rm_anova``,
uncorrected F; Greenhouse-Geisser p also reported), and the three contrasts (HP − opposite, next − opposite, HP − next)
with mean ± SEM, bootstrapped 95% CI of the mean (10 000 resamples), Wilcoxon signed-rank p, and Benjamini-Hochberg
q across the nine areas per contrast.

    python -m retsupp.visualize.paper.fig3_location_per_area
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper import richter_followup as rf
from retsupp.visualize.paper.paper_style import ROI_ORDER

CONTRASTS = [('hp_opp', 'HP − opposite'), ('next_opp', 'Next − opposite'), ('hp_next', 'HP − next')]


def per_area_values(folder):
    loc = rf.loc_values(folder, {r: r for r in ROI_ORDER})
    return pd.DataFrame({'hp': loc['hp'], 'next': (loc['nextleft'] + loc['nextright']) / 2,
                         'opposite': loc['opposite']}).reset_index()


def stats_table(v):
    import pingouin as pg
    rows = []
    for r in ROI_ORDER:
        x = v[v.grp == r].set_index('subject')[['hp', 'next', 'opposite']]
        long = x.reset_index().melt(id_vars='subject', var_name='loc', value_name='r')
        a = pg.rm_anova(data=long, dv='r', within='loc', subject='subject', correction=True)
        row = dict(area=r, n=len(x), F=float(a['F'].iloc[0]), df1=int(a['ddof1'].iloc[0]),
                   df2=int(a['ddof2'].iloc[0]), p_anova=float(a['p-unc'].iloc[0]),
                   p_anova_gg=float(a['p-GG-corr'].iloc[0]) if 'p-GG-corr' in a else np.nan)
        for k, _ in CONTRASTS:
            d = {'hp_opp': x.hp - x.opposite, 'next_opp': x.next - x.opposite, 'hp_next': x.hp - x.next}[k]
            lo, hi = rf.boot_ci(d, np.mean)
            row.update({f'{k}_mean': d.mean(), f'{k}_sem': d.sem(), f'{k}_ci_lo': lo, f'{k}_ci_hi': hi,
                        f'{k}_k_neg': int((d < 0).sum()), f'{k}_p': stats.wilcoxon(d).pvalue})
        rows.append(row)
    t = pd.DataFrame(rows)
    for k, _ in CONTRASTS:
        t[f'{k}_q'] = stats.false_discovery_control(t[f'{k}_p'])
    t['q_anova'] = stats.false_discovery_control(t['p_anova'])
    return t


def fmt_p(p):
    return f'{p:.2g}' if p < .001 else f'{p:.3f}'.lstrip('0')


def markdown(t):
    m = lambda x: f'{x:+.1f}'.replace('-', '−')   # noqa: E731
    lines = ['## Per area: location type (Fig. 3d, e), n = 28', '',
             'Items pooled; % of each participant\'s mean response over the twelve item × location cells in that area '
             '(window 3.2–6.4 s); participants whose mean response in an area is not positive are left out (n per area). '
             'ANOVA: one-way repeated-measures, location type (HP, next to HP, opposite HP), uncorrected F, BH q across '
             'the nine areas. Contrasts: mean ± SEM [bootstrapped 95% CI of the mean, 10 000 resamples], participants '
             'below 0, Wilcoxon signed-rank p, Benjamini-Hochberg q across the nine areas per contrast. Exploratory per '
             'area; the pathway-level tests are in Fig. 3. Data: `notes/data/fig3_location_per_area_n28_2026-11-12.tsv`; '
             'code: `retsupp/visualize/paper/fig3_location_per_area.py`.', '',
             '| Area | n | Location type F (p, q) | HP − opposite (%) | p / q | Next − opposite (%) | p / q | HP − next (%) | p / q |',
             '|---|---|---|---|---|---|---|---|---|']
    for _, r in t.iterrows():
        cells = [f"F({r.df1}, {r.df2}) = {r.F:.2f} ({fmt_p(r.p_anova)}, {fmt_p(r.q_anova)})"]
        for k, _ in CONTRASTS:
            cells.append(f"{m(r[f'{k}_mean'])} ± {r[f'{k}_sem']:.1f} [{m(r[f'{k}_ci_lo'])}, {m(r[f'{k}_ci_hi'])}], "
                         f"{r[f'{k}_k_neg']}/{r.n} < 0")
            cells.append(f"{fmt_p(r[f'{k}_p'])} / {fmt_p(r[f'{k}_q'])}")
        lines.append(f'| {r.area} | {r.n} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    for k, lab in CONTRASTS:
        sig = [r.area for _, r in t.iterrows() if r[f'{k}_q'] < .05]
        lines.append(f"- {lab}, q < .05: {', '.join(sig) if sig else 'none'}.")
    sig = [r.area for _, r in t.iterrows() if r.q_anova < .05]
    lines.append(f"- Location type ANOVA, q < .05: {', '.join(sig) if sig else 'none'}.")
    return '\n'.join(lines)


def main(folder, out_tsv, out_vals):
    v = per_area_values(folder)
    v.to_csv(out_vals, sep='\t', index=False, float_format='%.5g')
    t = stats_table(v)
    t.to_csv(out_tsv, sep='\t', index=False, float_format='%.5g')
    md = markdown(t)
    print(md)
    return t, v, md


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rel-folder', default='notes/data/event_deconv_relative_final2_2026-11')
    ap.add_argument('--out', default='notes/data/fig3_location_per_area_n28_2026-11-12.tsv')
    ap.add_argument('--out-values', default='notes/data/fig3_location_per_area_values_n28_2026-11-12.tsv')
    a = ap.parse_args()
    main(a.rel_folder, a.out, a.out_values)


def figure(v, t, name='figS_item_locations_per_area_n28'):
    """Supplementary figure in the style of Fig. 3e, one small panel per area (own y-scale: single-area percentages
    differ widely in spread), each location type relative to the mean of the four locations, mean ± within-subject
    SEM; brackets for contrasts with q < .05 (BH across the nine areas)."""
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.fig2_prf_fits import PATHWAY
    from retsupp.visualize.paper.fig3_item_locations import LOC_COLORS, LOC_TYPES, loc_icon
    from retsupp.visualize.paper.paper_style import LOC_EDGE, LOC_TEXT
    from retsupp.visualize.paper.paper_style import DOUBLE, FG, MUTED, offset_spines, save
    W, H = DOUBLE, 2.05
    fig = plt.figure(figsize=(W, H))
    left, right, gap, pgap = .62, .05, .34, .5
    pw = (W - left - right - 8 * gap - 2 * (pgap - gap)) / 9
    off = {'hp': -.28, 'next': 0., 'opposite': .28}
    pairs = [('hp_next', 'hp', 'next'), ('next_opp', 'next', 'opposite'), ('hp_opp', 'hp', 'opposite')]
    x = left
    axes = []
    by_area = t.set_index('area')
    for g, rois in PATHWAY.items():
        for k, r in enumerate([r for r in ROI_ORDER if r in rois]):
            ax = fig.add_axes([x / W, .42 / H, pw / W, 1.05 / H])
            axes.append(ax)
            d = v[v.grp == r].set_index('subject')
            m4 = (d.hp + 2 * d.next + d.opposite) / 4
            rel = pd.DataFrame({c: d[c] - m4 for c in ('hp', 'next', 'opposite')})
            norm = rel.sub(rel.mean(1), axis=0)
            se = norm.std(ddof=1) / np.sqrt(len(rel)) * np.sqrt(3 / 2)
            for c, _ in LOC_TYPES:
                ax.errorbar(off[c], rel[c].mean(), yerr=se[c], fmt='o', ms=3.6, color=LOC_COLORS[c], mec=LOC_EDGE[c], mew=.5,
                            lw=1., capsize=0, zorder=3)
            top = (rel.mean() + se).max()
            low = (rel.mean() - se).min()
            span = max(top - low, 4)
            level = 0
            for key, a_, b_ in pairs:
                q = by_area.loc[r, f'{key}_q']
                if q < .05:
                    y = top + .12 * span + level * .2 * span
                    ax.plot([off[a_], off[a_], off[b_], off[b_]], [y - .05 * span, y, y, y - .05 * span],
                            color=FG, lw=.6)
                    ax.text((off[a_] + off[b_]) / 2, y, '***' if q < .001 else '**' if q < .01 else '*',
                            ha='center', va='bottom', fontsize=7, color=FG)
                    level += 1
            hi = top + .12 * span + level * .2 * span + .08 * span
            lo = low - .1 * span
            ax.set_ylim(lo, hi)
            ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
            ax.set_xlim(-.55, .55)
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=6)
            ax.set_title(f'{r}\nn = {len(rel)}', fontsize=6.5, color=FG, pad=2, linespacing=1.1)
            ax.text(0, -.06, f"F = {by_area.loc[r, 'F']:.1f}\nq = {fmt_p(by_area.loc[r, 'q_anova'])}".replace('-', '−'),
                    transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6, color='.4')
            offset_spines(ax)
            ax.spines['bottom'].set_visible(False)
            x += pw + gap
        x += pgap - gap
    axes[0].set_ylabel('Response vs mean of\nthe four locations (%)', fontsize=6.5)
    # pathway labels under the panels
    x = left
    for g, rois in PATHWAY.items():
        n = len(rois)
        w = n * pw + (n - 1) * gap
        fig.add_artist(plt.Line2D([(x + .03) / W, (x + w - .03) / W], [.13 / H] * 2, color=MUTED, lw=.6))
        fig.text((x + w / 2) / W, .1 / H, g, ha='center', va='top', fontsize=6.5, color='.45')
        x += w + pgap
    # key: location types as display icons
    keyax = fig.add_axes([left / W, (H - .2) / H, 4.5 / W, .18 / H])
    keyax.set_axis_off()
    keyax.set_xlim(0, 4.5)
    keyax.set_ylim(0, 1)
    for j, (c, lab) in enumerate(LOC_TYPES):
        xx = j * 1.25
        loc_icon(keyax, (xx / 4.5, 0, .18 / 4.5 * (1.0), 1), {c})
        keyax.text(xx + .24, .5, lab, fontsize=6.5, color=LOC_TEXT[c], va='center')
    legend = LEGEND.format(table=markdown(t).split('\n', 2)[2])
    save(fig, name, legend)
    return fig


LEGEND = """**Figure S. Item responses by location relative to HP, per area (n = 28).** As Fig. 3e, for each of the nine areas separately: each location type (HP, purple; next to HP, the two neighbouring locations pooled, mid purple; opposite HP, light purple) relative to the mean of the four item locations over the window 3.2–6.4 s after search onset, item types pooled, in % of each participant's mean response over the twelve item × location cells in that area; mean ± within-subject SEM. Each panel has its own y-scale: in a single area the mean response is small for some participants, so the percentages spread widely (TO, IPS, VO); medians are in the table below. n per panel: participants with voxels at every item location and a positive mean response in that area (the pilots sub-01 and sub-02 are left out of the paper). Under each panel: repeated-measures ANOVA on location type (one-way, items pooled), F and Benjamini-Hochberg q across the nine areas. Brackets: contrasts (HP vs next to HP, next to HP vs opposite, HP vs opposite) with q < .05, Wilcoxon signed-rank test, Benjamini-Hochberg across the nine areas per contrast; * q < .05, ** q < .01. Exploratory per area; the inference is at the pathway level (Fig. 3e).

{table}
"""
