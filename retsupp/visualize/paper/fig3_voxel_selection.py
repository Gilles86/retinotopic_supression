"""Fig. 3e under looser voxel selections: HP, next to HP and opposite per pathway, current vs r > 0.2 vs r > 0.1.

Same deconvolutions as Fig. 3e (``deconvolve_location_responses.py --relative``, Fourier) on location series exported
with ``export_location_timeseries.py --root <ladder root>`` (voxels and pRF centres of that root's m7_full). Per
selection:

* the Fig. 3e statistics (``fig3_item_locations.rel_stats``: % of each subject's mean response, repeated-measures ANOVA
  location type × item type, planned contrasts on the mean over item types with Wilcoxon signed-rank tests);
* the same contrasts in absolute units (% signal; ``fig3_robustness.window_cells``), mean ± SEM, Wilcoxon p, which
  compare precision across selections without the per-subject division;
* voxels per item location per area (per subject × area the median over the four locations; median over subjects).

Run with the pilots left out::

    python -m retsupp.visualize.paper.run_excluding --exclude 1 2 -- retsupp.visualize.paper.fig3_voxel_selection
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig2_prf_fits import PATHWAY

ARMS = {
    'current': ('Current (Fig. 3)', 'notes/data/event_deconv_relative_final2_2026-11'),
    'refit': ('Current, refitted pRFs', 'notes/data/event_deconv_relative_polished_2026-11-12'),
    'r0.2': ('r > 0.2', 'notes/data/event_deconv_relative_relnull_r0.2_2026-11-12'),
    'r0.1': ('r > 0.1', 'notes/data/event_deconv_relative_relnull_r0.1_2026-11-12'),
}
ROIS = [r for rs in PATHWAY.values() for r in rs]
CONTRASTS = (('hp_opp', 'HP − opposite'), ('hp_next', 'HP − next'), ('next_opp', 'Next − opposite'))


def absolute(folder):
    """Per subject × group: HP, next, opposite window means in % signal (items pooled), as fig3_robustness."""
    from retsupp.visualize.paper.fig3_robustness import load, locations, window_cells
    d = load(folder)
    grp = {r: g for g, rs in PATHWAY.items() for r in rs}
    v = pd.concat([locations(window_cells(d, grp), relative=False),
                   locations(window_cells(d, {r: 'All areas' for r in ROIS}), relative=False)])
    return v


def abs_stats(v):
    rows = []
    for g, x in v.groupby('grp'):
        row = dict(grp=g, n_abs=len(x))
        for k, a, b in (('hp_opp', 'hp', 'opposite'), ('hp_next', 'hp', 'next'), ('next_opp', 'next', 'opposite')):
            dlt = x[a] - x[b]
            row[f'{k}_abs'], row[f'{k}_abs_se'], row[f'{k}_abs_p'] = dlt.mean(), dlt.sem(), stats.wilcoxon(dlt).pvalue
        row['mean_resp_abs'] = x[['hp', 'next', 'opposite']].mean(1).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index('grp')


def voxel_counts(folder):
    d = pd.concat([pd.read_csv(f, sep='\t', usecols=['subject', 'roi', 'loc', 'n_voxels'])
                   for f in sorted(Path(folder).glob('sub-*_2026-11.tsv'))]).drop_duplicates()
    d = d[d.roi.isin(ROIS) & (d.subject >= 3)]
    per = d.groupby(['subject', 'roi']).n_voxels.agg(['median', 'min', 'count'])
    # a missing location counts as 0 voxels at that location
    per['min'] = np.where(per['count'] < 4, 0, per['min'])
    return per.reset_index()


def main(arms, out_prefix):
    from retsupp.visualize.paper.fig3_item_locations import rel_stats
    res, vox, cells_all, vals_all = [], [], [], []
    for key in arms:
        lab, folder = ARMS[key]
        if not Path(folder).exists() or not any(Path(folder).glob('sub-*_2026-11.tsv')):
            print('missing', folder)
            continue
        tcs, cells, r = rel_stats(folder)
        a = abs_stats(v := absolute(folder))
        res.append(r.join(a).assign(selection=key, label=lab).reset_index())
        vox.append(voxel_counts(folder).assign(selection=key))
        cells_all.append(cells.assign(selection=key))
        vals_all.append(v.assign(selection=key))
    res = pd.concat(res)
    vox = pd.concat(vox)
    res.to_csv(f'{out_prefix}_stats.tsv', sep='\t', index=False, float_format='%.5g')
    vox.to_csv(f'{out_prefix}_voxels.tsv', sep='\t', index=False)
    pd.concat(cells_all).to_csv(f'{out_prefix}_cells.tsv', sep='\t', index=False, float_format='%.5g')
    pd.concat(vals_all).to_csv(f'{out_prefix}_abs_values.tsv', sep='\t', index=False, float_format='%.5g')

    pd.set_option('display.width', 250)
    cols = ['selection', 'grp', 'n', 'loc_F', 'loc_p', 'int_p', 'hp_opp', 'hp_opp_se', 'hp_opp_p', 'hp_next', 'hp_next_se',
            'hp_next_p', 'next_opp', 'next_opp_se', 'next_opp_p']
    print('--- relative (% of mean response, Fig. 3e)')
    print(res[cols].sort_values(['grp', 'selection']).round(3).to_string(index=False))
    cols = ['selection', 'grp', 'n_abs', 'mean_resp_abs', 'hp_opp_abs', 'hp_opp_abs_se', 'hp_opp_abs_p', 'hp_next_abs',
            'hp_next_abs_se', 'hp_next_abs_p', 'next_opp_abs', 'next_opp_abs_se', 'next_opp_abs_p']
    print('--- absolute (% signal)')
    print(res[cols].sort_values(['grp', 'selection']).round(4).to_string(index=False))
    print('--- voxels per item location (median over subjects of the per-subject median over the four locations; '
          'min location in brackets)')
    t = vox.groupby(['roi', 'selection'])[['median', 'min']].median().unstack('selection')
    print(t.reindex(ROIS).to_string())
    return res, vox



# ---------------------------------------------------------------- supplementary figure

SEL_STYLE = {'current': ('Current', '#1F2933'), 'r0.2': ('r > 0.2', '#3D5A80'), 'r0.1': ('r > 0.1', '#98C1D9')}
GROUPS = ['Early', 'Dorsal', 'Occipitotemporal', 'All areas']
GROUP_LAB = {'Early': 'Early', 'Dorsal': 'Dorsal', 'Occipitotemporal': 'Occ.-temporal', 'All areas': 'All areas'}


def figure(out_prefix, blur_root, name='figS_item_locations_voxel_selection_n28'):
    import matplotlib.pyplot as plt
    from retsupp.visualize.paper.item_location_blur import voxel_table
    from retsupp.visualize.paper.paper_style import DOUBLE, FG, HP, letter, offset_spines, save
    from retsupp.visualize.paper.fig3_item_locations import LOC_COLORS
    res = pd.read_csv(f'{out_prefix}_stats.tsv', sep='\t').set_index(['selection', 'grp'])
    vox = pd.read_csv(f'{out_prefix}_voxels.tsv', sep='\t')
    blur = voxel_table(blur_root, range(3, 31), ROIS, .75, centre_only=True)
    b = blur.groupby(['subject', 'roi'])[['a_self', 'a_next', 'a_opp']].mean()
    b = pd.DataFrame({'next': b.a_next / b.a_self, 'opposite': b.a_opp / b.a_self}).groupby('roi').median()
    sels = list(SEL_STYLE)

    W, H = DOUBLE, 3.9
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    top, bot, hh = 2.35, .42, 1.2
    xr = np.arange(len(ROIS))

    # a: voxels per item location
    ax = at(.7, top, 3.6, hh)
    for k, s in enumerate(sels):
        lab, col = SEL_STYLE[s]
        m = vox[vox.selection == s].groupby('roi')['median'].median().reindex(ROIS)
        ax.plot(xr + (k - 1) * .12, m.values, 'o-', color=col, ms=3.5, lw=1, mec='w', mew=.4)
        ax.text(1.0, .97 - k * .12, lab, transform=ax.transAxes, color=col if s != 'r0.1' else '#5B8FB9',
                fontsize=6.5, ha='right', va='top')
    ax.set_yscale('log')
    ax.set_yticks([3, 10, 30, 60])
    ax.set_yticklabels(['3', '10', '30', '60'])
    ax.set_ylim(2, 90)
    ax.set_xticks(xr)
    ax.set_xticklabels(ROIS, fontsize=6.5)
    ax.set_xlim(-.5, len(ROIS) - .5)
    ax.set_ylabel('Voxels per item\nlocation (median)')
    offset_spines(ax)
    ax.text(0, 1.06, 'Looser selections add voxels at every item location', transform=ax.transAxes, fontsize=7,
            color=FG, va='bottom')
    letter(ax, 'a', x=-.17, y=1.06)

    # b: leak from one item location to the next and the opposite one (pRF centres, current selection)
    ax = at(5.0, top, 1.95, hh)
    for k, c in enumerate(('next', 'opposite')):
        ax.plot(xr + (k - .5) * .2, 100 * b[c].reindex(ROIS).values, 'o', color=LOC_COLORS[c], ms=3.5, mec='w', mew=.4)
    ax.text(.03, .97, 'To next (5.7°)', transform=ax.transAxes, color=LOC_COLORS['next'], fontsize=6.5, va='top')
    ax.text(.03, .85, 'To opposite (8°)', transform=ax.transAxes, color='.5', fontsize=6.5, va='top')
    ax.set_xticks(xr)
    ax.set_xticklabels(ROIS, fontsize=6, rotation=90)
    ax.set_xlim(-.5, len(ROIS) - .5)
    ax.set_ylim(0, 45)
    ax.set_yticks([0, 20, 40])
    ax.set_ylabel('Leak (% of own\nlocation)')
    offset_spines(ax)
    ax.text(0, 1.06, 'pRF blur between item locations', transform=ax.transAxes, fontsize=7, color=FG, va='bottom')
    letter(ax, 'b', x=-.3, y=1.06)

    # c, d: HP − opposite and HP − next per pathway, % signal, three selections
    for j, (key, title, lt) in enumerate((('hp_opp', 'HP − opposite', 'c'), ('hp_next', 'HP − next to HP', 'd'))):
        ax = at(.7 + j * 3.4, bot, 2.9, hh)
        for gi, g in enumerate(GROUPS):
            x0 = gi + (.3 if g == 'All areas' else 0)
            for k, s in enumerate(sels):
                r = res.loc[(s, g)]
                m, se, p = r[f'{key}_abs'], r[f'{key}_abs_se'], r[f'{key}_abs_p']
                x = x0 + (k - 1) * .22
                col = SEL_STYLE[s][1]
                ax.errorbar(x, m, yerr=se, fmt='o', ms=4, color=col, mec='w', mew=.5, lw=1.1, capsize=0)
                if p < .05:
                    ax.text(x, m - se - .006, '*', ha='center', va='top', fontsize=7, color=col)
        ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
        ax.axvline(len(GROUPS) - 1.5 + .15, color='.85', lw=.6, zorder=0)
        ax.set_xticks([i + (.3 if g == 'All areas' else 0) for i, g in enumerate(GROUPS)])
        ax.set_xticklabels([GROUP_LAB[g] for g in GROUPS], fontsize=6.5)
        ax.set_xlim(-.5, len(GROUPS) - .2)
        ax.set_ylim(-.26, .06)
        ax.set_yticks([-.2, -.1, 0])
        ax.set_yticklabels(['−0.2', '−0.1', '0'])
        if j == 0:
            ax.set_ylabel('Difference (% signal)')
            for k, s in enumerate(sels):
                lab, col = SEL_STYLE[s]
                ax.text(.02 + k * .2, .06, lab, transform=ax.transAxes, color=col if s != 'r0.1' else '#5B8FB9',
                        fontsize=6.5, va='bottom')
        offset_spines(ax)
        ax.text(0, 1.06, title, transform=ax.transAxes, fontsize=7, color=HP, va='bottom')
        letter(ax, lt, x=-.2 if j == 0 else -.1, y=1.06)

    fmt = lambda s, g, k: (f"{res.loc[(s, g), f'{k}_abs']:+.3f} ± {res.loc[(s, g), f'{k}_abs_se']:.3f}% signal, "
                           f"p = {res.loc[(s, g), f'{k}_abs_p']:.3f}").replace('p = 0.', 'p = .')
    parts = []
    for g in GROUPS:
        parts.append(f"{g.lower()}: " + '; '.join(f"{SEL_STYLE[s][0]} {fmt(s, g, 'hp_opp')}" for s in sels))
    legend = f"""
**Supplementary Figure. The location effect of Fig. 3e under looser voxel selections: more voxels, smaller standard errors, but a smaller effect, so no gain in precision; pRF blur cannot account for the weak effect outside early areas.** Same pipeline as Fig. 3d, e (voxels whose pRF centre lies within 2° of an item location; their BOLD minus the bar-only prediction, deconvolved with nideconv, Fourier basis, item type × location type relative to the run's HP location; window 3.2–6.4 s after search onset; areas pooled weighted by voxel count; n = 28). Current: the gain-field fits' voxels (p_signal > 0.5, top 500 by R², leave-one-run-out reliability > 0) as in Fig. 3. r > 0.2, r > 0.1: every voxel of the area with model-free cross-run reliability above 0.2 or 0.1 and p ≤ .01 against a circular-shift null (BOLD SD ≤ 20%), with pRFs refitted together with the gain fields (m7_full); bar-only prediction from those pRFs with the gains at 0. a, Voxels per item location (per subject the median over the four locations, median over subjects). b, The response at one item location that a pRF centred on another location picks up, as a share of what it picks up at its own location (pRF centres, current selection, integrated over a 1.5° item; median over subjects); for a modulation confined to the HP item, HP − opposite keeps 1 minus the leak to the opposite location and HP − next 1 minus the leak to the next location (pooled per pathway: early ≥ 99%, dorsal 99% and 92%, occipitotemporal 97% and 90%). c, d, HP minus opposite and HP minus next to HP (mean over the item types), in % signal, mean ± SEM over subjects; * p < .05, Wilcoxon signed-rank test, uncorrected. HP − opposite: {'; '.join(parts)}. Missing fits: r > 0.2, sub-06 IPS, sub-25 VO, sub-26 VO; r > 0.1, sub-17 V3 (left out of those pools).
"""
    save(fig, name, legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--out-prefix', default='notes/data/fig3_voxel_selection_n28_2026-11-12')
    ap.add_argument('--figure', action='store_true', help='draw the supplementary figure from the written tables')
    ap.add_argument('--blur-root', default='/data/ds-retsupp/derivatives/'
                    'af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished')
    a = ap.parse_args()
    if a.figure:
        figure(a.out_prefix, a.blur_root)
    else:
        main(a.arms, a.out_prefix)
