"""Summary: does the DoG centre/surround tilt κ predict the neighbour pattern (Fig. 3e, Fig. 6)?

Reads the deconvolved noise-free predictions of ``predict_location_timeseries_kappa.py`` (one folder per variant: m7,
dogk, dogk0) and the data's deconvolution, and computes per subject and group (early V1–V3, all areas) the Fig. 3e
location contrasts (HP − next, next − opposite, HP − opposite; % of the subject's mean response, ``rel_stats``). The
κ-induced change is dogk − dogk0 (same parameters, κ := 0). Also the predicted bar drive per quadrant relative to HP
(V1–V3; per ROI the run mean per quadrant in % of the four-quadrant mean, then averaged over V1–V3, as Fig. 6e).

    python -m retsupp.visualize.paper.kappa_neighbour_summary --pred notes/data/kappa_neighbour_2026-11 \\
        --data notes/data/event_deconv_relative_final2_2026-11 --out notes/data/kappa_neighbour_2026-11.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig3_item_locations import rel_stats

VARIANTS = ('m7', 'dogk', 'dogk0')
GROUPS = ('Early', 'All areas')


def contrasts(folder):
    _tcs, cells, _res = rel_stats(folder)
    x = cells.pivot_table(index=['subject', 'grp'], columns='loc', values='r')   # mean over items
    nxt = (x.nextleft + x.nextright) / 2
    return pd.DataFrame({'hp_next': x.hp - nxt, 'next_opp': nxt - x.opposite, 'hp_opp': x.hp - x.opposite}).reset_index()


def bar_contrasts(folder):
    b = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(Path(folder).glob('sub-*_barquadrant.tsv'))])
    q = b.groupby(['subject', 'variant', 'roi', 'rel']).drive.mean().unstack()
    q = q.div(q.mean(1), axis=0) * 100
    q = q.groupby(['subject', 'variant']).mean()
    nxt = (q.nextleft + q.nextright) / 2
    return pd.DataFrame({'hp_next': q.hp - nxt, 'next_opp': nxt - q.opposite, 'hp_opp': q.hp - q.opposite}).reset_index()


def describe(v):
    v = v.dropna()
    p = stats.wilcoxon(v).pvalue if len(v) > 5 and (v != 0).any() else np.nan
    return dict(n=len(v), mean=v.mean(), median=v.median(), sem=v.sem(), pct_pos=(v > 0).mean() * 100, p=p)


def main(pred, data, out):
    rows = []
    c = {v: contrasts(Path(pred) / f'event_deconv_{v}') for v in VARIANTS}
    c['data'] = contrasts(data)
    c['kappa_effect'] = (c['dogk'].set_index(['subject', 'grp']) - c['dogk0'].set_index(['subject', 'grp'])).reset_index()
    for src, d in c.items():
        for g in GROUPS:
            for k in ('hp_next', 'next_opp', 'hp_opp'):
                rows.append(dict(readout='item_response_pct', source=src, group=g, contrast=k,
                                 **describe(d[d.grp == g][k])))
    b = bar_contrasts(Path(pred) / 'bar')
    bw = {v: b[b.variant == v].set_index('subject') for v in VARIANTS}
    bw['kappa_effect'] = bw['dogk'].drop(columns='variant') - bw['dogk0'].drop(columns='variant')
    for src, d in bw.items():
        for k in ('hp_next', 'next_opp', 'hp_opp'):
            rows.append(dict(readout='bar_drive_pct_V1-V3', source=src, group='Early', contrast=k, **describe(d[k])))
    res = pd.DataFrame(rows)
    res.to_csv(out, sep='\t', index=False, float_format='%.5g')
    with pd.option_context('display.width', 200):
        print(res.round(4).to_string(index=False))
    kf = pd.concat([pd.read_csv(f, sep='\t') for f in sorted((Path(pred) / 'bar').glob('sub-*_kappa_fit.tsv'))])
    kf.to_csv(Path(out).with_name(Path(out).stem + '_fits.tsv'), sep='\t', index=False, float_format='%.5g')
    early = kf[kf.roi.isin(['V1', 'V2', 'V3'])]
    print('\nκ, early (median over subject x ROI):', early.kappa.median(), ' all:', kf.kappa.median())
    print('pRF centre sd, early median:', early.sd_median.median(), ' surround size (x sd):', early.srf_size_median.median(),
          ' surround amplitude:', early.srf_amplitude_median.median(), ' sigma_AF early:', early.sigma_AF.median())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', default='notes/data/kappa_neighbour_2026-11')
    ap.add_argument('--data', default='notes/data/event_deconv_relative_final2_2026-11')
    ap.add_argument('--out', default='notes/data/kappa_neighbour_2026-11.tsv')
    a = ap.parse_args()
    main(a.pred, a.data, a.out)
