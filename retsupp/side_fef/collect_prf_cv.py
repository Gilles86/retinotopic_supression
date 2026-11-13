"""Side project (FEF / sPCS): reduce the cross-validated ROI pRF fits to per-voxel and per-(subject, ROI) tables.

Same reduction as ``side_hippocampus.collect_prf_cv`` (cvR² pooled over the three folds, scramble 95th percentile per
subject x ROI, fraction of voxels above it, contralateral fraction among them, ...), over
``derivatives/side_fef_prf_cv`` plus the two cortical controls of the hippocampus project
(``side_hpc_prf_cv``: lateral occipital, precentral; same method, same participants).

Also writes ``--hrf-out``: per subject x ROI the region HRF of the DoG + region-HRF fits on the true stimulus
(mean over the three folds), in the format of ``modeling/roi_hrf.load_table`` (``subject, roi, hrf_delay,
hrf_dispersion``), for the gain-field fits.

Run on a compute node::

    python -m retsupp.side_fef.collect_prf_cv --voxels-out ... --summary-out ... --hrf-out ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.side_fef.fit_prf_cv_roi import OUT_SUBDIR
from retsupp.side_hippocampus.collect_prf_cv import fold_agreement

HPC_CONTROLS = ('lateral_occipital', 'precentral')


def main(bids_folder, voxels_out, summary_out, hrf_out, prf_voxels_out):
    der = Path(bids_folder) / 'derivatives'
    files = sorted((der / OUT_SUBDIR).glob('sub-*/*_voxels.tsv.gz'))
    files += [f for r in HPC_CONTROLS for f in sorted((der / 'side_hpc_prf_cv').glob(f'sub-*/*_region-{r}_*.tsv.gz'))]
    print(len(files), 'files')
    df = pd.concat([pd.read_csv(f, sep='\t', dtype={'variant': str}) for f in files], ignore_index=True)
    df = df[df.subject.between(3, 30)]
    hrf = (df[(df.model == 'dog_roihrf') & (df.variant == 'true')].groupby(['subject', 'region', 'fold'])
           [['hrf_delay', 'hrf_dispersion']].first().groupby(['subject', 'region']).mean().reset_index()
           .rename(columns={'region': 'roi'}))
    hrf.to_csv(hrf_out, sep='\t', index=False)
    key = ['subject', 'region', 'variant', 'model', 'voxel']
    g = df.groupby(key)
    vox = g[['sse_test', 'sst_test']].sum()
    vox['cvr2'] = 1 - vox.sse_test / vox.sst_test
    vox = vox.join(g[['x', 'y', 'sd']].mean()).join(g.hemi.first())
    vox = vox.join(g.x.std().rename('x_sd_folds')).join(g.y.std().rename('y_sd_folds')).reset_index()
    vox['contra'] = np.where(vox.hemi == 'L', vox.x > 0, vox.x < 0)
    vox.to_csv(voxels_out, sep='\t', index=False)

    rows, prf = [], []
    for (s, r, m), d in vox.groupby(['subject', 'region', 'model']):
        t, n = d[d.variant == 'true'], d[d.variant == 'scramble']
        if t.empty or n.empty:
            continue
        thr = np.nanpercentile(n.cvr2, 95)
        sel_t, sel_n = t[t.cvr2 > thr], n[n.cvr2 > thr]
        prf.append(sel_t)
        raw = df[(df.subject == s) & (df.region == r) & (df.model == m)]
        rows.append(dict(subject=s, region=r, model=m, n_voxels=len(t), median_cvr2_true=t.cvr2.median(),
                         median_cvr2_scramble=n.cvr2.median(), frac_pos_true=(t.cvr2 > 0).mean(),
                         frac_pos_scramble=(n.cvr2 > 0).mean(), thr95_scramble=thr, frac_above_thr=len(sel_t) / len(t),
                         n_prf=len(sel_t), contra_prf=sel_t.contra.mean() if len(sel_t) else np.nan,
                         contra_prf_scramble=sel_n.contra.mean() if len(sel_n) else np.nan,
                         contra_all=t.contra.mean(),
                         ecc_prf=np.hypot(sel_t.x, sel_t.y).median() if len(sel_t) else np.nan,
                         sd_prf=sel_t.sd.median() if len(sel_t) else np.nan,
                         upper_prf=(sel_t.y > 0).mean() if len(sel_t) else np.nan,
                         x_agree=fold_agreement(raw[(raw.variant == 'true') & raw.voxel.isin(sel_t.voxel)], 'x'),
                         y_agree=fold_agreement(raw[(raw.variant == 'true') & raw.voxel.isin(sel_t.voxel)], 'y')))
    pd.concat(prf).to_csv(prf_voxels_out, sep='\t', index=False)
    out = pd.DataFrame(rows)
    out.to_csv(summary_out, sep='\t', index=False)
    print(out.groupby(['region', 'model']).median(numeric_only=True).round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--voxels-out', required=True)
    ap.add_argument('--summary-out', required=True)
    ap.add_argument('--hrf-out', required=True)
    ap.add_argument('--prf-voxels-out', required=True, help='true-stimulus voxels above the scramble 95th pct')
    a = ap.parse_args()
    main(a.bids_folder, a.voxels_out, a.summary_out, a.hrf_out, a.prf_voxels_out)
