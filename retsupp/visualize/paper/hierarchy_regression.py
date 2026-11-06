"""Phase D3: do the gain-field parameters rise or fall along the visual hierarchy?

The hierarchy axis is each ROI's pRF size: the median over subjects of the voxel-weighted mean pRF σ at
1-2.5° eccentricity (``fig1_prf_sigma.tsv``; m4 pRFs, no gain fits involved), on a log scale. One axis for
all subjects, so subjects without enough voxels in an ROI for the size estimate still count.

Per subject, the parameter is regressed on log pRF size over the nine ROIs (OLS slope, per doubling of
pRF size); the slopes are tested against 0 across subjects (Wilcoxon signed-rank; n = 30). The same is
done within the two pathways that leave early cortex: V1-V3 then dorsal (V3AB, IPS), and V1-V3 then
occipitotemporal (hV4, VO, LO, TO).

Output: ``notes/data/hierarchy_regression_2026-11.tsv``.

Usage::

    python -m retsupp.visualize.paper.hierarchy_regression
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import conditional, joint, median_ci
from retsupp.visualize.paper.paper_style import ROI_GROUPS, ROI_ORDER

FWHM = 2 * np.sqrt(2 * np.log(2))
# (column, source, label)
PARAMS = [('T', 'joint', 'Target gain'), ('D', 'joint', 'Distractor gain'),
          ('sus_fwhm', 'joint', 'Sustained field FWHM (°)'), ('dyn_fwhm', 'joint', 'Transient field FWHM (°)'),
          ('dS', 'conditional', 'dS (conditional)'), ('dD', 'conditional', 'dD (conditional)'),
          ('dS', 'joint', 'dS (joint m7)'), ('dD', 'joint', 'dD (joint m7)')]


def roi_size(table, ecc=(1., 2.5)):
    d = pd.read_csv(table, sep='\t')
    d = d[d.ecc.between(*ecc) & d.roi.isin(ROI_ORDER)]
    per_sub = d.groupby(['roi', 'subject']).apply(lambda g: np.average(g.sd, weights=g.n), include_groups=False)
    return per_sub.groupby('roi').median()


def slopes(d, col, x):
    out = []
    for s, g in d.groupby('subject'):
        g = g.dropna(subset=[col])
        if len(g) < 3:
            continue
        xx = np.log2(x.loc[g.roi].to_numpy())
        out.append(np.polyfit(xx, g[col].to_numpy(), 1)[0])
    return np.array(out)


def main(gain_table, conditional_table, sigma_table, out):
    size = roi_size(sigma_table)
    print('pRF size per ROI (°):', size.round(2).to_dict())
    js = joint(gain_table, 'dog')
    js['sus_fwhm'] = FWHM * js.sigma_AF
    js['dyn_fwhm'] = FWHM * js.sigma_dyn
    src = {'joint': js, 'conditional': conditional(conditional_table)}
    early = dict(ROI_GROUPS)['Early']
    sets = {'all nine': ROI_ORDER,
            'early + dorsal': list(early) + list(dict(ROI_GROUPS)['Dorsal']),
            'early + occipitotemporal': list(early) + list(dict(ROI_GROUPS)['Occipitotemporal'])}
    rows = []
    for col, s, lab in PARAMS:
        for name, rois in sets.items():
            d = src[s][src[s].roi.isin(rois)]
            b = slopes(d, col, size)
            m, lo, hi = median_ci(b)
            rows.append(dict(parameter=lab, rois=name, n=len(b), median_slope=m, ci_low=lo, ci_high=hi,
                             pct_positive=(b > 0).mean() * 100, p=stats.wilcoxon(b).pvalue))
    t = pd.DataFrame(rows)
    t['q'] = np.nan
    for name in sets:
        k = t.rois == name
        t.loc[k, 'q'] = stats.false_discovery_control(t.loc[k, 'p'])
    t.to_csv(out, sep='\t', index=False)
    with pd.option_context('display.width', 200):
        print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gain-table', default='notes/data/gain_field_parameters.tsv')
    ap.add_argument('--conditional-table', default='notes/data/conditional_estimates_dog_2026-11.tsv')
    ap.add_argument('--sigma-table', default='notes/data/fig1_prf_sigma.tsv')
    ap.add_argument('--out', default='notes/data/hierarchy_regression_2026-11.tsv')
    a = ap.parse_args()
    main(a.gain_table, a.conditional_table, a.sigma_table, a.out)
