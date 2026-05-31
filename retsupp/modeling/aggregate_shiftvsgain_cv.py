"""Aggregate the cross-validated shift-vs-gain comparison.

Reads BOTH arms of ``derivatives/af_prf_cv_shiftvsgain/`` —

    af_prf_cv_shiftvsgain/shift/sub-XX/sub-XX_roi-{ROI}_cv-fits.pkl
    af_prf_cv_shiftvsgain/gain/sub-XX/sub-XX_roi-{ROI}_cv-fits.pkl

— and, per (subject, ROI), aligns the two arms by ``voxel_mask_indices``
(they MUST be identical — both arms use the same ``select_roi_voxels`` +
``--max-voxels`` ranking; the aggregator asserts this and skips with a
loud warning if they ever differ).

Per voxel: ΔCV-R² = shift − gain, where each arm's per-voxel CV-R² is the
``nanmean`` over the (up to) 4 leave-one-condition-out folds.  Then:

* per (subject, ROI): mean/median per-arm CV-R² and median ΔCV-R²,
* per ROI (across subjects): median-of-(per-subject-median-Δ) and a
  sign test (how many subjects have median-Δ > 0).

Writes a long-format TSV::

    derivatives/af_prf_cv_shiftvsgain/delta_cv_r2.tsv

with one row per (subject, ROI).

Usage
-----
``python -m retsupp.modeling.aggregate_shiftvsgain_cv \\
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp``

(Tiny: globs pickles + numpy reductions; safe to run locally once the
pickles are rsync'd back, or on a login node in a pinch.)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _load(pkl: Path) -> dict:
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def _per_voxel_cv_r2(payload: dict) -> np.ndarray:
    """nanmean over folds of cv_r2_per_fold -> (n_voxels,)."""
    folds = [np.asarray(a, dtype=np.float64) for a in payload['cv_r2_per_fold']]
    stack = np.vstack(folds)               # (4, V)
    with np.errstate(invalid='ignore'):
        return np.nanmean(stack, axis=0)   # (V,)


def collect(bids_folder: Path) -> pd.DataFrame:
    root = bids_folder / 'derivatives' / 'af_prf_cv_shiftvsgain'
    shift_root = root / 'shift'
    gain_root = root / 'gain'

    rows = []
    shift_pkls = sorted(shift_root.glob('sub-*/sub-*_roi-*_cv-fits.pkl'))
    print(f'Found {len(shift_pkls)} shift pickles under {shift_root}')

    for shift_pkl in shift_pkls:
        rel = shift_pkl.relative_to(shift_root)
        gain_pkl = gain_root / rel
        if not gain_pkl.exists():
            print(f'  -> no gain twin for {rel}, skipping')
            continue

        s = _load(shift_pkl)
        g = _load(gain_pkl)

        subject = int(s['subject'])
        roi = s['roi']

        s_idx = np.asarray(s['voxel_mask_indices'])
        g_idx = np.asarray(g['voxel_mask_indices'])
        if s_idx.shape != g_idx.shape or not np.array_equal(s_idx, g_idx):
            print(f'  !! VOXEL MASK MISMATCH sub-{subject:02d} {roi}: '
                  f'shift n={s_idx.size}, gain n={g_idx.size} '
                  f'(equal_arrays={np.array_equal(s_idx, g_idx) if s_idx.shape == g_idx.shape else False}). '
                  'SKIPPING — arms are NOT comparable for this cell.')
            continue

        s_r2 = _per_voxel_cv_r2(s)          # (V,)
        g_r2 = _per_voxel_cv_r2(g)
        delta = s_r2 - g_r2                 # shift - gain

        with np.errstate(invalid='ignore'):
            rows.append(dict(
                subject=subject,
                roi=roi,
                n_voxels=int(s_idx.size),
                shift_cv_r2_mean=float(np.nanmean(s_r2)),
                shift_cv_r2_median=float(np.nanmedian(s_r2)),
                gain_cv_r2_mean=float(np.nanmean(g_r2)),
                gain_cv_r2_median=float(np.nanmedian(g_r2)),
                delta_cv_r2_mean=float(np.nanmean(delta)),
                delta_cv_r2_median=float(np.nanmedian(delta)),
            ))

    return pd.DataFrame(rows)


def per_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Across-subjects per-ROI summary + sign test on per-subject median-Δ."""
    out = []
    for roi, sub_df in df.groupby('roi'):
        med = sub_df['delta_cv_r2_median'].values
        med = med[np.isfinite(med)]
        n = med.size
        n_pos = int(np.sum(med > 0))
        # Two-sided sign test (binomial) on #subjects with median-Δ > 0.
        if n > 0:
            p_sign = stats.binomtest(n_pos, n, 0.5).pvalue
        else:
            p_sign = np.nan
        out.append(dict(
            roi=roi,
            n_subjects=n,
            shift_cv_r2_median=float(np.nanmedian(sub_df['shift_cv_r2_median'])),
            gain_cv_r2_median=float(np.nanmedian(sub_df['gain_cv_r2_median'])),
            delta_cv_r2_median=float(np.nanmedian(med)) if n else np.nan,
            n_shift_better=n_pos,
            sign_test_p=float(p_sign) if np.isfinite(p_sign) else np.nan,
        ))
    return pd.DataFrame(out)


def main(bids_folder: str):
    bids_folder = Path(bids_folder)
    df = collect(bids_folder)
    if df.empty:
        print('No matched (shift, gain) cells found — nothing to aggregate.')
        return

    out_tsv = (bids_folder / 'derivatives' / 'af_prf_cv_shiftvsgain'
               / 'delta_cv_r2.tsv')
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(['roi', 'subject']).reset_index(drop=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'\nWrote per-(subject, ROI) table: {out_tsv}')

    roi_df = per_roi_summary(df)
    # Order ROIs by the canonical hierarchy if present.
    roi_order = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO',
                 'IPS', 'SPL1', 'FEF']
    roi_df['__o'] = roi_df['roi'].apply(
        lambda r: roi_order.index(r) if r in roi_order else 999)
    roi_df = roi_df.sort_values('__o').drop(columns='__o').reset_index(drop=True)

    print('\nPer-ROI summary (shift - gain; positive = shift wins):')
    with pd.option_context('display.float_format', lambda v: f'{v: .4f}'):
        print(roi_df.to_string(index=False))

    roi_tsv = (bids_folder / 'derivatives' / 'af_prf_cv_shiftvsgain'
               / 'delta_cv_r2_per_roi.tsv')
    roi_df.to_csv(roi_tsv, sep='\t', index=False)
    print(f'\nWrote per-ROI summary: {roi_tsv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    args = parser.parse_args()
    main(bids_folder=args.bids_folder)
