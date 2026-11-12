"""Side project (hippocampus): reduce the cross-validated region pRF fits to per-voxel and per-(subject, region) tables.

Reads ``derivatives/side_hpc_prf_cv/sub-*/..._voxels.tsv.gz`` (both variants) and writes

- ``--voxels-out``: one row per subject × region × variant × model × voxel: cross-validated R² pooled over the
  three folds (1 − ΣSSE/ΣSST of the held-out runs), mean x, y, σ over folds, hemisphere, and the fold-to-fold
  spread of the centre;
- ``--summary-out``: one row per subject × region × model: median cvR² (true, scramble), fraction of voxels with
  cvR² > 0, the threshold (95th percentile of the scramble cvR² in that subject and region), the fraction of true
  voxels above it ("pRF voxels"; 5 % expected without visual-field-specific signal), and among pRF voxels the
  contralateral fraction and the fold-to-fold agreement of x and y (Pearson r between folds, mean over fold pairs).

Run on a compute node (many files)::

    python -m retsupp.side_hippocampus.collect_prf_cv --voxels-out ... --summary-out ...
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.side_hippocampus.fit_prf_cv_region import OUT_SUBDIR


def fold_agreement(d, col):
    w = d.pivot(index='voxel', columns='fold', values=col)
    rs = [np.corrcoef(w[a], w[b])[0, 1] for a, b in itertools.combinations(w.columns, 2) if len(w) > 2]
    return float(np.nanmean(rs)) if rs else np.nan


def main(bids_folder, voxels_out, summary_out):
    files = sorted((Path(bids_folder) / 'derivatives' / OUT_SUBDIR).glob('sub-*/*_voxels.tsv.gz'))
    print(len(files), 'files')
    df = pd.concat([pd.read_csv(f, sep='\t', dtype={'variant': str}) for f in files], ignore_index=True)
    key = ['subject', 'region', 'variant', 'model', 'voxel']
    g = df.groupby(key)
    vox = g[['sse_test', 'sst_test']].sum()
    vox['cvr2'] = 1 - vox.sse_test / vox.sst_test
    vox = vox.join(g[['x', 'y', 'sd']].mean()).join(g.hemi.first())
    vox = vox.join(g.x.std().rename('x_sd_folds')).join(g.y.std().rename('y_sd_folds')).reset_index()
    vox['contra'] = np.where(vox.hemi == 'L', vox.x > 0, vox.x < 0)
    vox.to_csv(voxels_out, sep='\t', index=False)

    rows = []
    for (s, r, m), d in vox.groupby(['subject', 'region', 'model']):
        t, n = d[d.variant == 'true'], d[d.variant == 'scramble']
        if t.empty or n.empty:
            continue
        thr = np.nanpercentile(n.cvr2, 95)
        sel_t, sel_n = t[t.cvr2 > thr], n[n.cvr2 > thr]
        raw = df[(df.subject == s) & (df.region == r) & (df.model == m)]
        rows.append(dict(subject=s, region=r, model=m, n_voxels=len(t), median_cvr2_true=t.cvr2.median(),
                         median_cvr2_scramble=n.cvr2.median(), frac_pos_true=(t.cvr2 > 0).mean(),
                         frac_pos_scramble=(n.cvr2 > 0).mean(), thr95_scramble=thr, frac_above_thr=len(sel_t) / len(t),
                         n_prf=len(sel_t), contra_prf=sel_t.contra.mean() if len(sel_t) else np.nan,
                         contra_prf_scramble=sel_n.contra.mean() if len(sel_n) else np.nan,
                         contra_all=t.contra.mean(),
                         ecc_prf=np.hypot(sel_t.x, sel_t.y).median() if len(sel_t) else np.nan,
                         sd_prf=sel_t.sd.median() if len(sel_t) else np.nan,
                         x_agree=fold_agreement(raw[(raw.variant == 'true') & raw.voxel.isin(sel_t.voxel)], 'x'),
                         y_agree=fold_agreement(raw[(raw.variant == 'true') & raw.voxel.isin(sel_t.voxel)], 'y'),
                         x_agree_all=fold_agreement(raw[raw.variant == 'true'], 'x'),
                         x_agree_all_scramble=fold_agreement(raw[raw.variant == 'scramble'], 'x')))
    pd.DataFrame(rows).to_csv(summary_out, sep='\t', index=False)
    print(pd.DataFrame(rows).groupby(['region', 'model']).mean(numeric_only=True).round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--voxels-out', required=True)
    ap.add_argument('--summary-out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.voxels_out, a.summary_out)
