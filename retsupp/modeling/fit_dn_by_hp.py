"""DN pRF parameters fitted separately per HP condition (exploratory, no cross-validation).

Question: do divisive-normalization parameters (not only the gain) differ when a voxel's pRF lies near
the high-probability (HP) distractor location? The HP location rotates between blocks of three runs, so
every voxel is seen under all four HP conditions and serves as its own control.

Per subject, for the voxels of the nine analysis ROIs with cross-run reliability > 0 (mean over the
``prf_cv_roi_reliability`` folds) and BOLD SD <= 20 % (over all runs), as in the gain fits:

1. ``all``: the DN pRF model with a fixed HRF per subject × ROI is refit on all runs, warm-started from
   the mean over the three folds of the model-16 fits (``prf_cv_roi_sharedhrf``, ``fit_prf_cv_roi_hrf``).
   The HRF is that of model 16, averaged over the folds (each fold has one HRF per ROI).
2. One fit per HP condition, on the (usually three) runs of that condition only, warm-started from
   ``all``. Every pRF and DN parameter is free; the HRF stays fixed. Paradigm: bar + search items
   (``'full'`` in ``fit_prf_cv.load_runs``), as for model 16.

Output, never overwritten:
``derivatives/<output_subdir>/sub-XX/sub-XX_dn_by_hp.tsv.gz`` with one row per voxel × fit
(``condition`` = ``all`` or the HP location), all parameters, ``r2`` (on the runs of that fit),
``r2_allfit`` (the ``all`` parameters on the same runs), ``n_runs``, ``reliability`` and ``bold_sd``.

Usage::

    python -m retsupp.modeling.fit_dn_by_hp 3 --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

from retsupp.modeling.cv.fit_prf_cv import load_runs
from retsupp.modeling.cv.fit_prf_cv_roi import DERIVED, roi_voxels, sse_sst
from retsupp.modeling.fit_prf import HRF, MODEL_CFG, build_model_factory, gd_fit_scheduled
from retsupp.utils.data import Subject
from retsupp.utils.sentinels import assert_gpu_available_if_expected

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'VO', 'LO', 'TO']
BASE_MODEL, PARENT = 16, 6          # DN, one fixed HRF per ROI; its flexible-HRF class
STAGE = [(HRF, 0.001, 1500, 1e-5)]  # everything but the HRF free, as in fit_prf_cv_roi_hrf
MAX_BOLD_SD = 20.


def load_condition(sub, masker, vox, runs, resolution):
    """BOLD (only the selected voxels) and paradigm of a list of runs, loaded run by run to save memory."""
    bold, par, grid = [], [], None
    for r in runs:
        b, p, grid = load_runs(sub, masker, resolution, 'full', [r])
        bold.append(b[:, vox])
        par.append(p)
    return np.vstack(bold), np.vstack(par), grid


def r2_of(factory, bold, par, pars):
    sse, sst = sse_sst(bold, np.asarray(factory(bold, par).predict(parameters=pars)))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sst > 0, 1 - sse / sst, np.nan)


def main(subject, bids_folder, resolution=50, voxel_chunk_size=10000, sd_min=0.2,
         base_subdir='prf_cv_roi_sharedhrf', rel_subdir='prf_cv_roi_reliability',
         output_subdir='prf_dn_by_hp_2026-11', max_voxels=None):
    assert_gpu_available_if_expected()
    deriv = Path(bids_folder) / 'derivatives'
    out_fn = deriv / output_subdir / f'sub-{subject:02d}' / f'sub-{subject:02d}_dn_by_hp.tsv.gz'
    if out_fn.exists():
        print(f'{out_fn} exists, nothing to do')
        return
    t0 = time.time()
    sub = Subject(subject, Path(bids_folder))
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask(session=1, run=sub.get_runs(1)[0]))
    masker.fit()
    vox_all, roi_all = roi_voxels(sub, masker)

    # Warm start and HRF: model 16, mean over the folds, per voxel.
    base_dir = deriv / base_subdir / f'sub-{subject:02d}'
    folds = []
    for k in range(3):
        cv = pd.read_csv(base_dir / f'sub-{subject:02d}_fold-{k}_model-{BASE_MODEL}_cvr2.tsv.gz', sep='\t')
        if not np.array_equal(cv.voxel_index.to_numpy(), vox_all):
            raise ValueError(f'fold {k}: model-{BASE_MODEL} voxels differ from the ROI set')
        p = pd.read_pickle(base_dir / f'sub-{subject:02d}_fold-{k}_model-{BASE_MODEL}_pars.pkl.gz')
        folds.append(p.drop(columns=DERIVED, errors='ignore').reset_index(drop=True))
    labels = list(folds[0].columns)
    init_all = pd.DataFrame(np.nanmean(np.stack([f[labels].to_numpy(float) for f in folds]), 0), columns=labels)

    rel = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
        (deriv / rel_subdir / f'sub-{subject:02d}').glob(f'sub-{subject:02d}_fold-*_reliability.tsv.gz'))])
    if rel.groupby('voxel_index').size().nunique() != 1 or rel.voxel_index.nunique() != len(vox_all):
        raise ValueError('reliability folds do not cover the ROI voxels')
    rel = rel.groupby('voxel_index').reliability.mean().reindex(vox_all).to_numpy()

    # Runs per HP condition; load each condition once, then the BOLD SD over all runs.
    groups = sub.get_runs_by_hp()
    cand = np.isin(roi_all, ROIS) & (rel > 0) & init_all.notna().all(axis=1).to_numpy()
    data, grid = {}, None
    for hp, runs in sorted(groups.items()):
        b, p, grid = load_condition(sub, masker, vox_all[cand], runs, resolution)
        data[hp] = (b, p, len(runs))
    bold_sd = np.vstack([d[0] for d in data.values()]).std(0)
    keep = bold_sd <= MAX_BOLD_SD
    if max_voxels:   # canary only: a subset, spread over the ROIs
        keep &= np.isin(np.arange(keep.size), np.linspace(0, keep.size - 1, max_voxels).astype(int))
    idx = np.flatnonzero(cand)[keep]
    vox, roi, rel, bold_sd = vox_all[idx], roi_all[idx], rel[idx], bold_sd[keep]
    data = {hp: (b[:, keep], p, n) for hp, (b, p, n) in data.items()}
    init_all = init_all.iloc[idx].reset_index(drop=True)
    print(f'{len(vox)} voxels (reliability > 0, BOLD SD <= {MAX_BOLD_SD:g}): '
          + ', '.join(f'{r} {np.sum(roi == r)}' for r in ROIS) + f' | loaded in {time.time() - t0:.0f} s')
    print('runs per HP: ' + ', '.join(f'{hp} {groups[hp]}' for hp in sorted(groups)))

    factory = build_model_factory(MODEL_CFG[PARENT], grid, sd_min)
    rows = []

    def table(cond, pars, r2, r2_allfit, n_runs):
        t = pars[labels].copy()
        t.insert(0, 'condition', cond)
        t.insert(0, 'roi', roi)
        t.insert(0, 'voxel_index', vox)
        t.insert(0, 'subject', subject)
        t['r2'], t['r2_allfit'], t['n_runs'] = r2, r2_allfit, n_runs
        t['reliability'], t['bold_sd'] = rel, bold_sd
        return t

    bold_all = np.vstack([d[0] for d in data.values()])
    par_all = np.vstack([d[1] for d in data.values()])
    fit_all = gd_fit_scheduled(factory, bold_all, par_all, init_all, voxel_chunk_size, STAGE)
    fit_all = fit_all.drop(columns=DERIVED, errors='ignore').reset_index(drop=True)
    r2_all = r2_of(factory, bold_all, par_all, fit_all)
    rows.append(table('all', fit_all, r2_all, r2_all, sum(d[2] for d in data.values())))
    print(f'all runs: median R² {np.nanmedian(r2_all):.4f} ({time.time() - t0:.0f} s)', flush=True)
    del bold_all, par_all

    for hp, (b, p, n) in data.items():
        fit = gd_fit_scheduled(factory, b, p, fit_all, voxel_chunk_size, STAGE)
        fit = fit.drop(columns=DERIVED, errors='ignore').reset_index(drop=True)
        r2, r2_ref = r2_of(factory, b, p, fit), r2_of(factory, b, p, fit_all)
        rows.append(table(hp, fit, r2, r2_ref, n))
        print(f'{hp} ({n} runs): median R² {np.nanmedian(r2):.4f}, all-runs parameters {np.nanmedian(r2_ref):.4f} '
              f'({time.time() - t0:.0f} s)', flush=True)

    out_fn.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_fn, sep='\t', index=False, float_format='%.6g')
    print(f'wrote {out_fn} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--resolution', type=int, default=50)
    ap.add_argument('--voxel-chunk-size', type=int, default=10000)
    ap.add_argument('--sd-min', type=float, default=0.2)
    ap.add_argument('--output-subdir', default='prf_dn_by_hp_2026-11')
    ap.add_argument('--max-voxels', type=int, default=None, help='Canary: fit only this many voxels.')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.resolution, a.voxel_chunk_size, a.sd_min,
         output_subdir=a.output_subdir, max_voxels=a.max_voxels)
