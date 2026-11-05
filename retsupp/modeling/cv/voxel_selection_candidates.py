"""Voxel-selection candidates per subject x ROI, and each ROI's HRF in m4.

Every ROI voxel gets its hemisphere, m4 parameters, ``p_signal`` and the model-free cross-run
reliability of ``prf_cv_reliability`` (mean over the three folds, as a stand-in for a full-data
value). Candidates:

- ``psignal_top500``: the current rule, p_signal > .5 and the top 500 by R².
- ``psignal_top500_reliable``: the same, restricted to reliability > 0.
- ``r2_top500``: R² >= .1, top 500 by R².
- ``reliable``: reliability > 0, nothing else.
- ``reliable_top500``: reliability > 0, the top 500 by reliability.

Outputs ``notes/data/voxel_selection_candidates_2026-11.tsv`` (subject x ROI x candidate) and
``notes/data/hrf_by_roi_2026-11.tsv``.

Usage::

    python -m retsupp.modeling.cv.voxel_selection_candidates --bids-folder /data/ds-retsupp
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
MIN_HRF_R2 = .1


def one(args):
    s, bids = args
    from retsupp.utils.data import Subject
    sub = Subject(s, bids)
    masker = sub.get_bold_mask(return_masker=True)
    masker.fit()
    pars = pd.DataFrame(sub.get_prf_parameters_volume(model=4, return_images=False))
    psig = masker.transform(f'{bids}/derivatives/prf/model4/sub-{s:02d}/sub-{s:02d}_desc-p_signal.nii.gz').ravel()
    rel = pd.concat([pd.read_csv(f'{bids}/derivatives/prf_cv_roi_reliability/sub-{s:02d}/sub-{s:02d}_fold-{k}_reliability.tsv.gz',
                                 sep='\t') for k in range(3)])
    rel = rel.groupby(['voxel_index', 'roi']).reliability.mean().reset_index()
    d = rel.join(pars[['x', 'y', 'sd', 'r2', 'hrf_delay', 'hrf_dispersion']], on='voxel_index')
    d['p_signal'] = psig[d.voxel_index]
    left = np.zeros(len(pars), bool)
    for roi in ROIS:
        left |= masker.transform(sub.get_retinotopic_roi(f'{roi}_L', bold_space=True)).ravel() > 0
    d['hemi'] = np.where(left[d.voxel_index], 'L', 'R')
    d['contra'] = np.where(d.hemi == 'L', d.x > 0, d.x < 0)
    rows, hrf = [], []
    for roi, g in d.groupby('roi'):
        real = g[(g.r2 > 0) & g.sd.notna()]
        cands = {
            'psignal_top500': real[real.p_signal > .5].nlargest(500, 'r2'),
            'r2_top500': real[real.r2 >= .1].nlargest(500, 'r2'),
            'reliable': real[real.reliability > 0],
            'reliable_top500': real[real.reliability > 0].nlargest(500, 'reliability'),
        }
        cands['psignal_top500_reliable'] = cands['psignal_top500'].query('reliability > 0')
        for name, c in cands.items():
            rows.append(dict(subject=s, roi=roi, candidate=name, n=len(c), contra=c.contra.mean() * 100 if len(c) else np.nan,
                             below_r2_1=(c.r2 < .1).mean() * 100 if len(c) else np.nan, median_r2=c.r2.median()))
        good = real[real.r2 >= MIN_HRF_R2]
        hrf.append(dict(subject=s, roi=roi, n=len(good), hrf_delay=good.hrf_delay.median(), hrf_dispersion=good.hrf_dispersion.median(),
                        rho_delay_sd=good[['hrf_delay', 'sd']].corr('spearman').iloc[0, 1] if len(good) > 20 else np.nan,
                        rho_dispersion_sd=good[['hrf_dispersion', 'sd']].corr('spearman').iloc[0, 1] if len(good) > 20 else np.nan))
    print(s, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(hrf)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/data/ds-retsupp')
    ap.add_argument('--n-procs', type=int, default=4)
    a = ap.parse_args()
    with mp.Pool(a.n_procs) as pool:
        out = pool.map(one, [(s, a.bids_folder) for s in range(1, 31)])
    pd.concat(o[0] for o in out).to_csv('notes/data/voxel_selection_candidates_2026-11.tsv', sep='\t', index=False)
    pd.concat(o[1] for o in out).to_csv('notes/data/hrf_by_roi_2026-11.tsv', sep='\t', index=False)
