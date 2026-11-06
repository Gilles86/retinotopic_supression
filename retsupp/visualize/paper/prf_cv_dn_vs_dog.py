"""DN vs DoG pRFs per ROI, from the existing leave-one-run-set-out results (no refits).

Input: ``notes/data/prf_cv_roi_summary.tsv`` (``retsupp/modeling/cv/aggregate_prf_cv_roi.py``): per subject,
fold, ROI, voxel selection and pRF model, the median held-out R² over the ROI's voxels. A subject's value
per ROI is the mean of that median over the three folds, as in Fig. 3b.

Models: 1 Gaussian, 2 DoG, 5 DN (canonical HRF); 3, 4, 6 the same with a flexible HRF per voxel.

Per ROI: median over subjects of each model's held-out R²; per-subject ΔR² = DN − DoG, its median with a
bootstrapped 95% CI (10,000 resamples of subjects); the number of subjects with DN > DoG; two-sided
Wilcoxon signed-rank p, Benjamini-Hochberg across the nine ROIs (per voxel set and HRF). Pooled: each
subject's median over the nine ROIs; also the ``all`` row of the summary (one median over the voxels of
all ROIs together, eleven ROIs including SPL1 and FEF), which is where the earlier "30/30" came from.

Output: ``notes/data/prf_cv_dn_vs_dog_per_roi_2026-11.md``.

Usage::

    python -m retsupp.visualize.paper.prf_cv_dn_vs_dog
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
HRFS = {'Canonical HRF': {'Gaussian': 1, 'DoG': 2, 'DN': 5}, 'Flexible HRF per voxel': {'Gaussian': 3, 'DoG': 4, 'DN': 6}}
SETS = {'Reliable voxels': 'reliable', 'All voxels': 'all'}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile(np.median(rng.choice(x, (n, len(x))), axis=1), [2.5, 97.5])


def fmt(x, d=3):
    return f'{x:.{d}f}'.replace('-', '−')


def pval(p):
    return '< .001' if p < .001 else f'{p:.3f}'.lstrip('0')


def rows(s, models):
    """s: subject × (roi, model) table of subject values."""
    out = []
    for roi in ROIS + ['Pooled (median over 9 ROIs)', 'Pooled (all-ROI voxels, 11 ROIs)']:
        if roi.startswith('Pooled (median'):
            v = {m: s.xs(i, axis=1, level='model')[ROIS].median(axis=1) for m, i in models.items()}
        elif roi.startswith('Pooled (all'):
            v = {m: s[('all', i)] for m, i in models.items()}
        else:
            v = {m: s[(roi, i)] for m, i in models.items()}
        diff = (v['DN'] - v['DoG']).dropna().to_numpy()
        lo, hi = boot_ci(diff)
        out.append(dict(roi=roi, **{m: np.nanmedian(v[m]) for m in models}, d=np.median(diff), lo=lo, hi=hi,
                        n_pos=int((diff > 0).sum()), n=len(diff), p=stats.wilcoxon(diff).pvalue))
    t = pd.DataFrame(out)
    k = t.roi.isin(ROIS)
    t['q'] = np.nan
    t.loc[k, 'q'] = stats.false_discovery_control(t.loc[k, 'p'])
    return t


def main(table, out):
    d = pd.read_csv(table, sep='\t')
    lines = ['# DN vs DoG pRFs per ROI (2026-11-06)', '',
             'Held-out R² from the leave-one-run-set-out fits (3 folds; `notes/data/prf_cv_roi_summary.tsv`), no refits. '
             'Subject value per ROI: the fold mean of the median held-out R² over the ROI\'s voxels. ΔR² = DN − DoG per '
             'subject; median with bootstrapped 95% CI; Wilcoxon signed-rank (two-sided), q: Benjamini-Hochberg across '
             'the nine ROIs. "Pooled (median over 9 ROIs)": each subject\'s median over the nine ROIs. "Pooled (all-ROI '
             'voxels)": the summary\'s `all` row, one median over the voxels of all eleven ROIs together (including SPL1 '
             'and FEF), the basis of the earlier "DN > DoG in 30/30". Script: '
             '`retsupp/visualize/paper/prf_cv_dn_vs_dog.py`.', '']
    for set_name, sel in SETS.items():
        lines += [f'## {set_name}', '']
        for hrf, models in HRFS.items():
            x = d[(d.selection == sel) & d.model.isin(models.values())]
            s = x.groupby(['subject', 'roi', 'model']).median_r2.mean().unstack(['roi', 'model'])
            t = rows(s, models)
            lines += [f'**{hrf}**', '',
                      '| ROI | Gaussian | DoG | DN | ΔR² DN − DoG, median [95% CI] | DN > DoG | p | q |',
                      '|---|---|---|---|---|---|---|---|']
            for _, r in t.iterrows():
                q = '' if np.isnan(r.q) else pval(r.q)
                lines.append(f'| {r.roi} | {fmt(r.Gaussian)} | {fmt(r.DoG)} | {fmt(r.DN)} | {fmt(r.d, 4)} [{fmt(r.lo, 4)}, '
                             f'{fmt(r.hi, 4)}] | {r.n_pos}/{r.n} | {pval(r.p)} | {q} |')
            lines.append('')
    open(out, 'w').write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', default='notes/data/prf_cv_roi_summary.tsv')
    ap.add_argument('--out', default='notes/data/prf_cv_dn_vs_dog_per_roi_2026-11.md')
    a = ap.parse_args()
    main(a.table, a.out)
