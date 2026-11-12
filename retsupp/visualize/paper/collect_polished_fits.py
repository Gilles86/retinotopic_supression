"""Clamped (pRFs frozen at the m0 fit) versus polished (pRFs refitted together with the gain fields) ladder fits.

For each subject x ROI and model (default m4_TDS, m7_full), reads the fold-0 parameters of both roots and writes

* ``<out>_shared.tsv``: one row per subject x ROI x model x fit (clamped / polished) with the shared terms
  (gains, widths, item scale), dS = g_HP − g_LP, dD = g_HP_dyn − g_LP_dyn, the training SSE (from the pickle's info)
* ``<out>_prf_shift.tsv``: per subject x ROI x model, the median and 90th percentile of the per-voxel pRF change
  (centre shift in degrees, relative change in sd and amplitude), overall and for voxels with a pRF centre within 2°
  of an item ring location (all four, and the HP location of each run separately is not possible: one pRF per voxel)

    python -m retsupp.visualize.paper.collect_polished_fits \\
        --root af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf --out notes/data/prf_refit_2026-11-12
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = ['sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'item_scale']
RING = 4.0 / np.sqrt(2) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])


def load(pkl):
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    pars = d['fit_pars_per_fold'][0]
    info = (d.get('optimizer_info') or [None])[0]
    return pars.reset_index(drop=True), info, np.asarray(d.get('voxel_ids', []))


def main(bids, root, models, out, subjects):
    deriv = Path(bids) / 'derivatives'
    croot, proot = deriv / root, deriv / f'{root}_polished'
    shared_rows, shift_rows = [], []
    for model in models:
        for pkl in sorted((proot / model).glob('sub-*/*_cv-fits.pkl')):
            sub = int(pkl.name[4:6])
            if sub not in subjects:
                continue
            roi = pkl.name.split('_roi-')[1].split('_cv-fits')[0]
            cpkl = croot / model / pkl.parent.name / pkl.name
            if not cpkl.exists():
                continue
            pp, pinfo, pids = load(pkl)
            cp, cinfo, cids = load(cpkl)
            assert len(pp) == len(cp) and (len(pids) == 0 or (pids == cids).all())
            for tag, p, info in (('clamped', cp, cinfo), ('polished', pp, pinfo)):
                row = dict(subject=sub, roi=roi, model=model, fit=tag, n_voxels=len(p))
                row.update({k: float(p[k].iloc[0]) for k in SHARED if k in p})
                row['dS'] = row['g_HP'] - row['g_LP']
                row['dD'] = row['g_HP_dyn'] - row['g_LP_dyn']
                if isinstance(info, dict):
                    for k in ('sse_clamped', 'sse_final', 'seconds', 'nit', 'train_sse_total'):
                        if k in info:
                            row[k] = info[k]
                shared_rows.append(row)
            shift = np.hypot(pp.x - cp.x, pp.y - cp.y)
            dsd = (pp.sd - cp.sd) / cp.sd
            damp = (pp.amplitude - cp.amplitude) / cp.amplitude.abs().clip(lower=1e-9)
            near = np.min(np.hypot(cp.x.values[:, None] - RING[:, 0], cp.y.values[:, None] - RING[:, 1]), 1) < 2
            for sel, mask in (('all', np.ones(len(cp), bool)), ('near_items', near)):
                if mask.sum() == 0:
                    continue
                shift_rows.append(dict(subject=sub, roi=roi, model=model, voxels=sel, n=int(mask.sum()),
                                       shift_median=float(np.median(shift[mask])),
                                       shift_p90=float(np.percentile(shift[mask], 90)),
                                       dsd_median=float(np.median(dsd[mask])),
                                       damp_median=float(np.median(damp[mask]))))
    pd.DataFrame(shared_rows).to_csv(f'{out}_shared.tsv', sep='\t', index=False)
    pd.DataFrame(shift_rows).to_csv(f'{out}_prf_shift.tsv', sep='\t', index=False)
    print(f'{len(shared_rows) // 2} subject x ROI x model pairs -> {out}_shared.tsv, {out}_prf_shift.tsv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--root', default='af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf')
    ap.add_argument('--models', nargs='+', default=['m4_TDS', 'm7_full'])
    ap.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 31)))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.root, a.models, a.out, set(a.subjects))
