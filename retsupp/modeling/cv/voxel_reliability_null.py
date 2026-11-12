"""Model-free voxel reliability with a circular-shift null, for voxel selection without p_signal.

The bar sequence is the same in every run, so a voxel that follows the stimulus has a timecourse that repeats
across runs. Per voxel (all ROI-union voxels, cleaned BOLD, all runs of both sessions, 258 volumes each):

- ``r``: mean over runs of corr(run, mean of the other runs).
- Null: the same statistic with each run's other-runs mean circularly shifted by a random 30-228 TRs
  (one shift per run), 100 shift sets; ``p = (#null >= r + 1) / 101``.
- ``sd``: mean over runs of the BOLD SD (percent signal).

No encoding model is involved, so selecting on ``r`` and ``p`` cannot favour any gain-field model.
Descriptive columns: ``rel_loo`` (mean over the ``prf_cv_reliability`` folds), and the whole-brain DoG pRF
(model 4) ``x``, ``y``, ``r2``, ``p_signal``.

Output (never overwritten):
``derivatives/prf_cv_roi_reliability_null/sub-XX/sub-XX_reliability_null.tsv.gz``, one row per voxel,
``voxel_index`` = index into the BOLD mask (the ladder's ``voxel_ids``). Used by
``fit_cv_ladder_nested.py --selection-table``.

``--folds``: the same statistic per cross-validation fold of ``fit_prf_cv.make_fold_assignments`` (3 folds, the
folds of the pRF fits and of ``decode/decode_final.py``), computed on that fold's TRAINING runs only, so a held-out
(decoded) run never influences its voxel selection. ``sd`` is then the mean BOLD SD over the training runs, the
null seeded by (subject, fold). Output (never overwritten):
``derivatives/prf_cv_roi_reliability_null_folds/sub-XX/sub-XX_fold-K_reliability_null.tsv.gz`` with
``voxel_index``, ``roi``, ``r``, ``null_mean``, ``null_95``, ``p``, ``sd``, ``n_train_runs``. Used by
``decode_final.py --selection relnull``.

Usage::

    python -m retsupp.modeling.cv.voxel_reliability_null 3 --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
    python -m retsupp.modeling.cv.voxel_reliability_null 3 --folds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

from retsupp.modeling.cv.fit_prf_cv_roi import roi_voxels
from retsupp.utils.data import Subject

T_RUN = 258
N_NULL = 100
SHIFT_RANGE = (30, 228)   # TRs; integers drawn in [30, 228)
OUT_SUBDIR = 'prf_cv_roi_reliability_null'
FOLDS_SUBDIR = 'prf_cv_roi_reliability_null_folds'


def out_path(bids_folder, subject):
    return (Path(bids_folder) / 'derivatives' / OUT_SUBDIR / f'sub-{subject:02d}'
            / f'sub-{subject:02d}_reliability_null.tsv.gz')


def fold_path(bids_folder, subject, fold):
    return (Path(bids_folder) / 'derivatives' / FOLDS_SUBDIR / f'sub-{subject:02d}'
            / f'sub-{subject:02d}_fold-{fold}_reliability_null.tsv.gz')


def _z(a):
    sd = a.std(0)
    return (a - a.mean(0)) / np.where(sd > 0, sd, np.nan)


def reliability_null(runs: np.ndarray, rng: np.random.Generator, n_null=N_NULL):
    """(R, T, V) runs -> r (V,), null (V, n_null)."""
    n = runs.shape[0]
    total = runs.sum(0)
    zr = [_z(runs[r]) for r in range(n)]
    zp = [_z((total - runs[r]) / (n - 1)) for r in range(n)]

    def corr(shifts=None):
        out = 0
        for r in range(n):
            p = zp[r] if shifts is None else np.roll(zp[r], shifts[r], axis=0)
            out = out + (zr[r] * p).mean(0)
        return out / n

    r_obs = corr()
    null = np.stack([corr(rng.integers(*SHIFT_RANGE, size=n)) for _ in range(n_null)], 1)
    return r_obs, null


def main(subject, bids_folder):
    bids_folder = Path(bids_folder)
    out = out_path(bids_folder, subject)
    if out.exists():
        print(f'{out} exists, nothing to do')
        return
    der = bids_folder / 'derivatives'
    sub = Subject(subject, bids_folder)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask(session=1, run=sub.get_runs(1)[0]))
    masker.fit()
    vox, roi = roi_voxels(sub, masker)
    runs = []
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            fn = (der / 'cleaned' / f'sub-{subject:02d}' / f'ses-{ses}' / 'func'
                  / f'sub-{subject:02d}_ses-{ses}_task-search_desc-cleaned_run-{run}_bold.nii.gz')
            runs.append(masker.transform(fn).astype(np.float32)[:T_RUN, vox])
    runs = np.stack(runs)
    sd = runs.std(1).mean(0)
    # Seeded by subject, so a rerun reproduces the null exactly.
    r_obs, null = reliability_null(runs, np.random.default_rng(subject))
    p = ((null >= r_obs[:, None]).sum(1) + 1) / (N_NULL + 1)

    rel = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(
        (der / 'prf_cv_roi_reliability' / f'sub-{subject:02d}').glob('*_reliability.tsv.gz'))])
    rel = rel.groupby('voxel_index').reliability.mean()

    def prf(k):
        fn = der / 'prf' / 'model4' / f'sub-{subject:02d}' / f'sub-{subject:02d}_desc-{k}.nii.gz'
        return masker.transform(str(fn)).ravel()[vox]

    df = pd.DataFrame(dict(sub=subject, voxel_index=vox, roi=roi, r=r_obs, null_mean=null.mean(1),
                           null_95=np.quantile(null, .95, axis=1), p=p, rel_loo=rel.reindex(vox).to_numpy(),
                           sd=sd, x=prf('x'), y=prf('y'), r2=prf('r2'), p_signal=prf('p_signal')))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    print(f'sub-{subject:02d}: {len(df)} voxels -> {out}')


def main_folds(subject, bids_folder, n_folds=3):
    """Per fold: r, null, p and SD on the fold's training runs only (see the module docstring)."""
    from retsupp.modeling.cv.fit_prf_cv import make_fold_assignments
    bids_folder = Path(bids_folder)
    todo = [k for k in range(n_folds) if not fold_path(bids_folder, subject, k).exists()]
    if not todo:
        print('All fold outputs exist, nothing to do')
        return
    der = bids_folder / 'derivatives'
    sub = Subject(subject, bids_folder)
    folds = make_fold_assignments(sub, n_folds=n_folds)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask(session=1, run=sub.get_runs(1)[0]))
    masker.fit()
    vox, roi = roi_voxels(sub, masker)
    bold = {}
    for ses, run in sorted(r for f in folds for r in f):
        fn = (der / 'cleaned' / f'sub-{subject:02d}' / f'ses-{ses}' / 'func'
              / f'sub-{subject:02d}_ses-{ses}_task-search_desc-cleaned_run-{run}_bold.nii.gz')
        bold[ses, run] = masker.transform(fn).astype(np.float32)[:T_RUN, vox]
    for k in todo:
        train = sorted(r for j, f in enumerate(folds) if j != k for r in f)
        runs = np.stack([bold[r] for r in train])
        r_obs, null = reliability_null(runs, np.random.default_rng([subject, k]))
        p = ((null >= r_obs[:, None]).sum(1) + 1) / (N_NULL + 1)
        df = pd.DataFrame(dict(sub=subject, fold=k, voxel_index=vox, roi=roi, r=r_obs, null_mean=null.mean(1),
                               null_95=np.quantile(null, .95, axis=1), p=p, sd=runs.std(1).mean(0),
                               n_train_runs=len(train)))
        out = fold_path(bids_folder, subject, k)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep='\t', index=False)
        print(f'sub-{subject:02d} fold {k}: {len(train)} training runs, {len(df)} voxels; r > .1 & p <= .01: '
              f'{((df.r > .1) & (df.p <= .01)).sum()}, r > .2 & p <= .01: {((df.r > .2) & (df.p <= .01)).sum()} -> {out}',
              flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--folds', action='store_true', help='per CV fold, on the training runs only')
    a = ap.parse_args()
    if a.folds:
        main_folds(a.subject, a.bids_folder)
    else:
        main(a.subject, a.bids_folder)
