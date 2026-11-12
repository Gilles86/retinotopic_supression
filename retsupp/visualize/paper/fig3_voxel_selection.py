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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--out-prefix', default='notes/data/fig3_voxel_selection_n28_2026-11-12')
    a = ap.parse_args()
    main(a.arms, a.out_prefix)
