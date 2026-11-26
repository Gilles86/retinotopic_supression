"""How well the three pRF models (Gaussian, DoG, DN) fit the same voxels, per area.

Per participant x area, on the voxels the three families' fits share (same voxel_id), the in-sample R² of the full-data
fit (all runs; ~3,000 time points per voxel, so the 2 (DoG) or 4 (DN) extra parameters per voxel add about 0.001 R²
by chance). Model: the pRF-only fit (m0_prf, no gain fields) and the joint zero-sum gain-field fit (m5_TDS_dS_zs).
Per cell: the median over voxels of R²(family) − R²(Gaussian), and that gain over what the extra parameters add by
chance in-sample (k (1 − R²) / n_t, k = 2 for DoG, 4 for DN; DoG and DN contain the Gaussian, so in-sample they never fit
worse and a share of voxels 'better fitted' means nothing); across participants: median, bootstrapped 95% CI,
participants above (0 or 1), Wilcoxon. Pathways: per participant the mean of its areas.

Input: the *_cv-r2.tsv files rsynced to <root>/<family>/<model>/sub-XX/.

    PYTHONPATH=. ~/mambaforge/envs/bauer/bin/python -m retsupp.visualize.paper.prf_family_fit_quality \\
        --root /data/ds-retsupp/derivatives/prf_family_r2_2026-11-26 --out notes/data/prf_family_fit_quality_2026-11-26
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.visualize.paper.fig45_common import median_ci
from retsupp.visualize.paper.gain_meta_zs import PW_OF
from retsupp.visualize.paper.paper_style import ROI_ORDER


def load(root, fam, model):
    fs = sorted(Path(root, fam, model).glob('sub-*/*_cv-r2.tsv'))
    d = pd.concat([pd.read_csv(f, sep='\t', usecols=['subject', 'roi', 'voxel_id', 'train_r2']) for f in fs])
    return d[~d.subject.isin([1, 2]) & d.roi.isin(ROI_ORDER)].rename(columns={'train_r2': fam})


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells, summ = [], []
    for model in ('m0_prf', 'm5_zs'):
        d = load(a.root, 'gauss', model)
        for fam in ('dog', 'dn'):
            d = d.merge(load(a.root, fam, model), on=['subject', 'roi', 'voxel_id'])
        d['n_t'] = 258 * np.where(d.subject.isin([20, 24]), 11, 12)   # sub-20 and sub-24 have 11 runs
        for fam in ('dog', 'dn'):
            d[f'{fam}_minus_gauss'] = d[fam] - d.gauss
        g = d.groupby(['subject', 'roi'])
        c = pd.DataFrame({'n_vox': g.size(), 'r2_gauss': g.gauss.median(), 'r2_dog': g.dog.median(),
                          'r2_dn': g.dn.median(),
                          'dog_minus_gauss': g.dog_minus_gauss.median(), 'dn_minus_gauss': g.dn_minus_gauss.median(),
                          'n_t': g.n_t.first()}).reset_index()
        # what the extra parameters add by chance in-sample (nested least squares): k (1 − R²) / n_t
        c['chance_dog'] = 2 * (1 - c.r2_gauss) / c.n_t
        c['chance_dn'] = 4 * (1 - c.r2_gauss) / c.n_t
        c['dog_over_chance'] = c.dog_minus_gauss / c.chance_dog
        c['dn_over_chance'] = c.dn_minus_gauss / c.chance_dn
        c['model'] = model
        c['pathway'] = c.roi.map(PW_OF)
        cells.append(c)
        # per participant: area values, then pathway = mean of its areas
        units = [(r, c[c.roi == r]) for r in ROI_ORDER]
        for pw in ('early', 'dorsal', 'occipitotemporal'):
            units.append((pw, c[c.pathway == pw].groupby('subject').mean(numeric_only=True).reset_index()))
        units.append(('all', c.groupby('subject').mean(numeric_only=True).reset_index()))
        for unit, x in units:
            for col in ('r2_gauss', 'r2_dog', 'r2_dn', 'dog_minus_gauss', 'dn_minus_gauss', 'chance_dog', 'chance_dn',
                        'dog_over_chance', 'dn_over_chance'):
                v = x[col].dropna().to_numpy()
                m, lo, hi = median_ci(v)
                ref = 1. if 'over' in col else 0.
                p = stats.wilcoxon(v - ref).pvalue if ('minus' in col or 'over' in col) else np.nan
                summ.append(dict(model=model, unit=unit, measure=col, n=len(v), median=m, ci_lo=lo, ci_hi=hi,
                                 k_above=int((v > ref).sum()), p=p))
    pd.concat(cells).to_csv(out / 'cells.tsv', sep='\t', index=False)
    s = pd.DataFrame(summ)
    s.to_csv(out / 'summary.tsv', sep='\t', index=False)
    for model in ('m0_prf', 'm5_zs'):
        print(model)
        print(s[s.model == model].pivot_table(index='unit', columns='measure', values='median')
              .loc[['all', 'early', 'dorsal', 'occipitotemporal'] + ROI_ORDER].round(4).to_string())
        print(s[(s.model == model) & s.measure.str.contains('over')][['unit', 'measure', 'median', 'ci_lo', 'ci_hi',
                                                                      'k_above', 'n', 'p']].round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default='notes/data/prf_family_fit_quality_2026-11-26')
    main(ap.parse_args())
