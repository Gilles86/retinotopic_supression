"""Robustness of the Fig. 3d, e location effects (HP, next to HP, opposite HP; items pooled), n = 28.

The Fig. 3 measure divides each participant's location responses by their mean response in the area (in %). Where
that mean is small (TO, VO, IPS) the values blow up. Variants computed here, from the same relative-location
deconvolutions (``notes/data/event_deconv_relative_final2_2026-11``) and the same window (3.2–6.4 s after search onset):

* ``rel``      — the Fig. 3 measure (% of the participant's mean over the twelve cells; pairs with a mean ≤ 0 left out).
* ``abs``      — absolute: window mean in % signal change, no division, no exclusion.
* ``abs_v5``, ``abs_v10`` — absolute, keeping a participant × area only if each of its four item locations has at
  least 5 / 10 selected voxels.

Per area (nine) and per pathway / all areas. Pathways and all areas are pooled three ways: voxel-weighted time
courses (as in Fig. 3), the per-participant median over areas, and the per-participant mean over areas. Statistics per
contrast (HP − opposite, next − opposite, HP − next): median and bootstrapped 95% CI (10 000), mean ± SEM, k of n
below 0, Wilcoxon p; Page's L for the ordered alternative HP < next < opposite; Benjamini-Hochberg q across the nine
areas (and across the three pathways). Leave-one-participant-out influence: the largest Wilcoxon / Page p over the
n drops.

    python -m retsupp.visualize.paper.fig3_robustness
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.paper_style import ROI_ORDER

PEAK = (3.2, 6.4)
ITEMS = ('target', 'distractor', 'neutral')
LOCS = ('hp', 'nextleft', 'nextright', 'opposite')
PATHWAYS = {'Early': ['V1', 'V2', 'V3'], 'Dorsal': ['V3AB', 'IPS'], 'Occipitotemporal': ['hV4', 'LO', 'TO', 'VO']}
MIN_SUBJECT = 3
CONTRASTS = (('hp_opp', 'HP − opposite'), ('next_opp', 'Next − opposite'), ('hp_next', 'HP − next'))
RNG = np.random.default_rng(20261112)


def load(folder):
    d = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))])
    d = d[d.roi.isin(ROI_ORDER) & (d.subject >= MIN_SUBJECT)].copy()
    d['t'] = d.t.round(2)
    return d


def window_cells(d, rois_to_grp, floor=0):
    """Per subject × group × cell (item_location): baseline-corrected window mean, voxel-weighted over ROIs and the
    four physical locations, as fig45_common.item_timecourses. floor: drop subject × ROI pairs where any physical
    location has fewer voxels."""
    d = d.copy()
    if floor:
        nmin = d.groupby(['subject', 'roi'])['n_voxels'].min()
        keep = nmin[nmin >= floor].index
        d = d.set_index(['subject', 'roi']).loc[lambda x: x.index.isin(keep)].reset_index()
    d['grp'] = d.roi.map(rois_to_grp)
    d = d.dropna(subset=['grp'])
    d['w'] = d.response * d.n_voxels
    g = d.groupby(['subject', 'grp', 'cls', 't'])
    p = (g.w.sum() / g.n_voxels.sum()).rename('r').reset_index()
    base = p[p.t < 0].groupby(['subject', 'grp', 'cls']).r.mean()
    p['r'] -= p.set_index(['subject', 'grp', 'cls']).index.map(base)
    return p[p.t.between(*PEAK)].groupby(['subject', 'grp', 'cls']).r.mean().unstack().dropna()


def locations(pk, relative):
    if relative:
        pk = pk[pk.mean(1) > 0]
        pk = pk.div(pk.mean(1), axis=0) * 100
    loc = {c: pk[[f'{it}_{c}' for it in ITEMS]].mean(1) for c in LOCS}
    return pd.DataFrame({'hp': loc['hp'], 'next': (loc['nextleft'] + loc['nextright']) / 2,
                         'opposite': loc['opposite']}).reset_index()


def values(d, variant, units):
    """Per subject × unit the three location responses, for units 'area' (nine ROIs) or a pathway pooling."""
    relative = variant == 'rel'
    floor = {'abs_v5': 5, 'abs_v10': 10}.get(variant, 0)
    if units == 'area':
        return locations(window_cells(d, {r: r for r in ROI_ORDER}, floor), relative)
    if units == 'voxel_weighted':
        g = {r: p for p, rs in PATHWAYS.items() for r in rs}
        v = locations(window_cells(d, g, floor), relative)
        allv = locations(window_cells(d, {r: 'All areas' for r in ROI_ORDER}, floor), relative)
        return pd.concat([v, allv])
    per_area = values(d, variant, 'area')
    agg = 'median' if units == 'median_over_areas' else 'mean'
    out = []
    for p, rs in list(PATHWAYS.items()) + [('All areas', ROI_ORDER)]:
        x = per_area[per_area.grp.isin(rs)].groupby('subject')[['hp', 'next', 'opposite']].agg(agg)
        out.append(x.reset_index().assign(grp=p))
    return pd.concat(out)


def boot(x, f):
    x = np.asarray(x)
    b = f(x[RNG.integers(0, len(x), (10000, len(x)))], axis=1)
    return np.percentile(b, [2.5, 97.5])


def page_p(X):
    return stats.page_trend_test(X, predicted_ranks=[1, 2, 3]).pvalue


def test_unit(x, loo=False):
    X = x[['hp', 'next', 'opposite']].to_numpy()
    row = dict(n=len(X))
    if len(X) < 6:          # too few participants to test
        return row
    diffs = {'hp_opp': X[:, 0] - X[:, 2], 'next_opp': X[:, 1] - X[:, 2], 'hp_next': X[:, 0] - X[:, 1]}
    for k, d in diffs.items():
        lo, hi = boot(d, np.median)
        row.update({f'{k}_median': np.median(d), f'{k}_ci_lo': lo, f'{k}_ci_hi': hi, f'{k}_mean': d.mean(),
                    f'{k}_sem': d.std(ddof=1) / np.sqrt(len(d)), f'{k}_k_neg': int((d < 0).sum()),
                    f'{k}_p': stats.wilcoxon(d).pvalue})
    row['page_p'] = page_p(X)
    row['full_order_k'] = int(((X[:, 0] < X[:, 1]) & (X[:, 1] < X[:, 2])).sum())
    if loo:
        idx = np.arange(len(X))
        row['loo_max_p_hp_opp'] = max(stats.wilcoxon(np.delete(diffs['hp_opp'], i)).pvalue for i in idx)
        row['loo_max_page_p'] = max(page_p(np.delete(X, i, 0)) for i in idx)
    return row


def run(d):
    rows = []
    for variant in ('rel', 'abs', 'abs_v5', 'abs_v10'):
        for units in ('area', 'voxel_weighted', 'median_over_areas', 'mean_over_areas'):
            v = values(d, variant, units)
            for g, x in v.groupby('grp'):
                loo = g in ('V1', 'V2', 'TO', 'Early', 'All areas')
                rows.append(dict(variant=variant, pooling=units, unit=g, **test_unit(x, loo)))
    t = pd.DataFrame(rows)
    for (variant, pooling), m in t.groupby(['variant', 'pooling']).groups.items():
        sub = t.loc[m]
        fam = (sub.unit.isin(ROI_ORDER) if pooling == 'area' else sub.unit.isin(PATHWAYS)) & sub.page_p.notna()
        for c in [f'{k}_p' for k, _ in CONTRASTS] + ['page_p']:
            q = pd.Series(np.nan, index=sub.index)
            q[fam] = stats.false_discovery_control(sub.loc[fam, c])
            t.loc[sub.index, c.replace('_p', '_q')] = q
    order = {u: i for i, u in enumerate(ROI_ORDER + list(PATHWAYS) + ['All areas'])}
    return t.sort_values(['variant', 'pooling', 'unit'], key=lambda s: s.map(order) if s.name == 'unit' else s)


def main(folder, out, out_values):
    d = load(folder)
    t = run(d)
    t.to_csv(out, sep='\t', index=False, float_format='%.5g')
    vals = pd.concat([values(d, v, 'area').assign(variant=v) for v in ('rel', 'abs')])
    vals.to_csv(out_values, sep='\t', index=False, float_format='%.5g')
    cnt = d.groupby(['subject', 'roi'])['n_voxels'].min().unstack()
    print('voxels at the sparsest item location, median [min, max] per area:')
    print(cnt.describe().T[['50%', 'min', 'max']].round(0).astype(int).to_string())
    for f in (5, 10):
        print(f'n per area with every location >= {f} voxels:', (cnt >= f).sum().to_dict())
    return t, vals, cnt


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', default='notes/data/event_deconv_relative_final2_2026-11')
    ap.add_argument('--out', default='notes/data/fig3_robustness_n28_2026-11-12.tsv')
    ap.add_argument('--out-values', default='notes/data/fig3_robustness_values_n28_2026-11-12.tsv')
    a = ap.parse_args()
    main(a.folder, a.out, a.out_values)


LEGEND_ABS = """**Figure S. Item responses by location relative to HP, per area, in absolute units (n = 28).** As Fig. 3e, for each of the nine areas: the response at each location type (HP, red; next to HP, the two neighbouring locations pooled, dark grey; opposite HP, light grey), window 3.2–6.4 s after search onset, item types pooled, in % signal change and expressed relative to each participant's mean over the four item locations in that area (no division by the participant's mean response, so no participant is left out for a small or negative mean). Dots: participants (triangles: beyond the axis); diamonds and bars: median and bootstrapped 95% CI (10 000 resamples). Stars: Page's L test for the ordered alternative HP < next to HP < opposite, Benjamini-Hochberg q across the nine areas: {stars}. HP − opposite (Wilcoxon signed-rank, BH across the nine areas): {hp_opp}. n per area: participants with selected voxels at every item location ({n}); in IPS, LO, TO, VO and hV4 a participant's item locations hold a median of 1–2 voxels each (V1 8, V2 12, V3 7), so single-area values there rest on few voxels. Exploratory per area; the inference is at the pathway level (Fig. 3e). Data: `notes/data/fig3_robustness_n28_2026-11-12.tsv`; code: `retsupp/visualize/paper/fig3_robustness.py`.
"""


def figure_abs(vals, t, name='figS_item_locations_per_area_abs_n28', clip=.45):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.fig3_item_locations import LOC_COLORS, LOC_TYPES, loc_icon
    from retsupp.visualize.paper.paper_style import DOUBLE, FG, MUTED, ROI_GROUPS, offset_spines, save
    v = vals[vals.variant == 'abs']
    ta = t[(t.variant == 'abs') & (t.pooling == 'area')].set_index('unit')
    W, H = DOUBLE, 2.35
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.62 / W, .52 / H, (W - .72) / W, 1.45 / H])
    off = {'hp': -.27, 'next': 0., 'opposite': .27}
    xs, x = {}, 0.
    for g, rois in ROI_GROUPS:
        for r in rois:
            xs[r] = x
            x += 1.
        x += .55
    rng = np.random.default_rng(3)
    for r in ROI_ORDER:
        d = v[v.grp == r]
        m4 = (d.hp + 2 * d.next + d.opposite) / 4
        for c, _ in LOC_TYPES:
            y = (d[c] - m4).to_numpy()
            xx = xs[r] + off[c] + rng.uniform(-.06, .06, len(y))
            inside = np.abs(y) <= clip
            ax.scatter(xx[inside], y[inside], s=4, color=LOC_COLORS[c], alpha=.45, lw=0, zorder=2)
            for s in (1, -1):
                o = s * y > clip
                ax.scatter(xx[o], np.full(o.sum(), s * clip), s=7, marker='^' if s > 0 else 'v',
                           color=LOC_COLORS[c], alpha=.6, lw=0, zorder=2)
            lo, hi = boot(y, np.median)
            ax.plot([xs[r] + off[c]] * 2, [lo, hi], color=LOC_COLORS[c], lw=1.2, zorder=3, solid_capstyle='butt')
            ax.plot(xs[r] + off[c], np.median(y), 'D', ms=3.6, color=LOC_COLORS[c], mec='w', mew=.5, zorder=4)
        q = ta.loc[r, 'page_q']
        st = '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''
        if st:
            ax.text(xs[r], clip * 1.02, st, ha='center', va='bottom', fontsize=7, color=FG)
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_ylim(-clip * 1.06, clip * 1.18)
    ax.set_xlim(-.6, x - .55 + .05)
    ax.set_yticks([-.4, -.2, 0, .2, .4])
    ax.set_xticks([xs[r] for r in ROI_ORDER])
    ax.set_xticklabels([f'{r}\nn = {int(ta.loc[r, "n"])}' for r in ROI_ORDER], fontsize=6.5, linespacing=1.1)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Response vs mean of the four\nlocations (% signal change)', fontsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5)
    offset_spines(ax)
    ax.spines['bottom'].set_visible(False)
    for g, rois in ROI_GROUPS:
        x0, x1 = xs[rois[0]] - .4, xs[rois[-1]] + .4
        ax.plot([x0, x1], [-.2, -.2], transform=ax.get_xaxis_transform(), color=MUTED, lw=.6, clip_on=False)
        ax.text((x0 + x1) / 2, -.23, g, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.5,
                color='.45')
    keyax = fig.add_axes([.62 / W, (H - .22) / H, 4.5 / W, .18 / H])
    keyax.set_axis_off()
    keyax.set_xlim(0, 4.5)
    keyax.set_ylim(0, 1)
    for j, (c, lab) in enumerate(LOC_TYPES):
        xx = j * 1.25
        loc_icon(keyax, (xx / 4.5, 0, .18 / 4.5, 1), {c})
        keyax.text(xx + .24, .5, lab, fontsize=6.5, color=LOC_COLORS[c] if c != 'opposite' else '.5', va='center')
    fp = lambda p: (f'{p:.2g}' if p < .001 else f'{p:.3f}'.lstrip('0'))   # noqa: E731
    stars = '; '.join(f'{r} q = {fp(ta.loc[r, "page_q"])}' for r in ROI_ORDER)
    hp_opp = '; '.join(f'{r} {ta.loc[r, "hp_opp_median"]:+.3f}% [{ta.loc[r, "hp_opp_ci_lo"]:+.3f}, '
                       f'{ta.loc[r, "hp_opp_ci_hi"]:+.3f}], {int(ta.loc[r, "hp_opp_k_neg"])}/{int(ta.loc[r, "n"])} < 0, '
                       f'q = {fp(ta.loc[r, "hp_opp_q"])}' for r in ROI_ORDER).replace('-', '−')
    n = ', '.join(f'{r} {int(ta.loc[r, "n"])}' for r in ROI_ORDER)
    save(fig, name, LEGEND_ABS.format(stars=stars, hp_opp=hp_opp, n=n))
    return fig
