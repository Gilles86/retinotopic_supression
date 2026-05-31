"""Aggregate the cross-validated 4-level shift-vs-gain comparison.

Four nested levels, all on IDENTICAL voxels / folds / HRF, differing only
in mechanism:

* **null**   — mean-response (MODEL-FREE). For each fold, predict the
               held-out condition's BOLD with the mean over TRAINING runs'
               BOLD timeseries (per voxel). No fitting. Defines the signal
               voxels + the condition-blind noise ceiling. Computed HERE,
               in this aggregator, from the same fold geometry the fitting
               arms use (so this script LOADS BOLD — run it cluster-side).
* **model0** — fixed PRF, NO attention = the gain harness's all-zero-gains
               cell (``af_prf_cv_shiftvsgain/model0``). The common baseline
               both gain and shift are tested against.
* **gain**   — CV-v2 all gains free (``.../gain``).
* **shift**  — Klein 6-σ shift (``.../shift``).

The three fitted arms are read from their per-(subject, ROI, model) TSVs
(``sub-XX_roi-{ROI}_cv-r2.tsv``), NOT pickles. Their ``voxel_id``s MUST be
identical across all three arms (same selector + ranking); we ASSERT this
and fail loudly otherwise.

Outputs (all TSV, under ``derivatives/af_prf_cv_shiftvsgain/``):

* ``cv4_per_voxel.tsv`` — one row per (subject, roi, voxel_id) with
  ``cv_r2_{null,model0,gain,shift}`` and ``is_signal``.
* ``cv4_per_roi.tsv``   — per-ROI medians of each level + Δ(gain−model0),
  Δ(shift−model0), Δ(shift−gain), Δ(gain−null), Δ(shift−null), each
  restricted to signal voxels and to the gated set, plus a binomial
  sign test across subjects on Δ(shift−gain).
* ``cv4_threshold_sweep.tsv`` — median Δ(shift−gain) as a function of an
  increasing null-CV-R² inclusion cutoff.

Inclusion gate for the gain-vs-shift verdict: signal voxel AND
``max(gain, shift) > model0`` (attention helps at all). The gate is on the
COMMON model0 — never on the gain−shift difference.

Usage
-----
``python -m retsupp.modeling.aggregate_shiftvsgain_cv \\
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp``

This LOADS BOLD to compute the null level, so run it on a compute node
(``srun``) cluster-side, not on the login node.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.modeling.cv_helpers import (
    held_out_condition,
    per_voxel_r2,
)
from retsupp.modeling.fit_af_prf_cv_v2 import (
    build_data_and_paradigm,
    split_by_condition_with_target,
)
from retsupp.utils.data import Subject

ARMS = ['model0', 'gain', 'shift']

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO',
             'IPS', 'SPL1', 'FEF']

# Null-CV-R² inclusion cutoffs swept in the threshold-sweep table.
SWEEP_CUTOFFS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# Loading fitted arms from TSV.
# ---------------------------------------------------------------------------

def _arm_per_voxel_cv_r2(tsv: Path) -> pd.DataFrame:
    """Read one arm's cv-r2.tsv -> per-voxel nanmean-over-folds cv_r2.

    Returns a DataFrame indexed by voxel_id with a single ``cv_r2`` column
    (mean over the up-to-4 folds, ignoring NaN/skipped folds).
    """
    df = pd.read_csv(tsv, sep='\t')
    agg = (df.groupby('voxel_id')['cv_r2']
             .apply(lambda s: np.nanmean(s.to_numpy(dtype=np.float64)))
             .rename('cv_r2'))
    return agg.to_frame()


# ---------------------------------------------------------------------------
# Null (mean-response) level — MODEL-FREE, computed from BOLD here.
# ---------------------------------------------------------------------------

def compute_null_cv_r2(
    sub: Subject, voxel_ids: np.ndarray, *,
    resolution: int = 50, grid_radius: float = 5.0,
) -> np.ndarray:
    """Per-voxel null CV-R² (mean over folds), MODEL-FREE.

    For each leave-one-condition-out fold: prediction = mean over TRAINING
    runs' BOLD timeseries (per voxel), broadcast across the held-out run's
    timepoints. Scored with the same ``per_voxel_r2`` on the held-out BOLD.

    The bar sweep is a fixed protocol across runs, so the cross-run mean is
    time-aligned (after the standard 258-TR crop already done by
    ``build_data_and_paradigm``), making this a valid condition-blind
    noise ceiling.

    ``voxel_ids`` are the BOLD-masker flat indices the fitted arms used;
    we slice the loaded BOLD to those columns so the null aligns by
    voxel_id.
    """
    (bold, paradigm, condition_indicator,
     dynamic_indicator, target_indicator,
     _grid, _masker, run_meta) = build_data_and_paradigm(
        sub, resolution=resolution, grid_radius=grid_radius)

    voxel_ids = np.asarray(voxel_ids)
    bold_sel = bold[:, voxel_ids]   # (T_total, n_vox)
    n_vox = voxel_ids.size

    fold_r2 = []
    for cv_fold in range(4):
        held_out = held_out_condition(cv_fold)
        try:
            split = split_by_condition_with_target(
                bold=bold_sel, paradigm=paradigm,
                condition_indicator=condition_indicator,
                dynamic_indicator=dynamic_indicator,
                target_indicator=target_indicator,
                run_meta=run_meta, held_out_cond=held_out)
        except RuntimeError:
            fold_r2.append(np.full(n_vox, np.nan, dtype=np.float64))
            continue

        train_bold = split['train']['bold']     # (T_train, n_vox)
        held_bold = split['held']['bold']        # (T_held, n_vox)

        # MODEL-FREE prediction: per-voxel mean over ALL training timepoints
        # (= the cross-run mean response of the bar protocol), broadcast.
        train_mean = train_bold.mean(axis=0, keepdims=True)   # (1, n_vox)
        pred = np.repeat(train_mean, held_bold.shape[0], axis=0)
        fold_r2.append(per_voxel_r2(held_bold, pred).astype(np.float64))

    stack = np.vstack(fold_r2)        # (4, n_vox)
    with np.errstate(invalid='ignore'):
        return np.nanmean(stack, axis=0)


# ---------------------------------------------------------------------------
# Collect per-voxel table.
# ---------------------------------------------------------------------------

def collect(bids_folder: Path) -> pd.DataFrame:
    root = bids_folder / 'derivatives' / 'af_prf_cv_shiftvsgain'

    # Drive iteration off the SHIFT arm (one cv-r2.tsv per (subject, ROI)).
    shift_root = root / 'shift'
    shift_tsvs = sorted(shift_root.glob('sub-*/sub-*_roi-*_cv-r2.tsv'))
    print(f'Found {len(shift_tsvs)} shift cv-r2 TSVs under {shift_root}')

    rows = []
    for shift_tsv in shift_tsvs:
        rel = shift_tsv.relative_to(shift_root)
        # Parse subject + roi from the filename.
        name = shift_tsv.name  # sub-XX_roi-{ROI}_cv-r2.tsv
        subject = int(name.split('_')[0].split('-')[1])
        roi = name.split('roi-')[1].split('_cv-r2')[0]

        arm_tsvs = {arm: (root / arm / rel) for arm in ARMS}
        missing = [arm for arm, p in arm_tsvs.items() if not p.exists()]
        if missing:
            print(f'  -> sub-{subject:02d} {roi}: missing arms {missing}, '
                  'skipping')
            continue

        arm_r2 = {arm: _arm_per_voxel_cv_r2(arm_tsvs[arm]) for arm in ARMS}

        # ASSERT identical voxel_ids across all three fitted arms.
        ref_ids = arm_r2['shift'].index.to_numpy()
        ok = True
        for arm in ARMS:
            ids = arm_r2[arm].index.to_numpy()
            if ids.shape != ref_ids.shape or not np.array_equal(ids, ref_ids):
                print(f'  !! VOXEL-ID MISMATCH sub-{subject:02d} {roi}: '
                      f"arm '{arm}' n={ids.size} vs shift n={ref_ids.size}. "
                      'SKIPPING — arms NOT comparable.')
                ok = False
                break
        if not ok:
            continue

        voxel_ids = ref_ids

        # Null level: model-free, from BOLD (slices to these voxel_ids).
        sub = Subject(subject, bids_folder)
        try:
            null_r2 = compute_null_cv_r2(sub, voxel_ids)
        except Exception as e:                       # noqa: BLE001
            print(f'  !! null computation failed sub-{subject:02d} {roi}: '
                  f'{e}. Skipping.')
            continue

        block = pd.DataFrame({
            'subject': subject,
            'roi': roi,
            'voxel_id': voxel_ids,
            'cv_r2_null': null_r2,
            'cv_r2_model0': arm_r2['model0']['cv_r2'].to_numpy(),
            'cv_r2_gain': arm_r2['gain']['cv_r2'].to_numpy(),
            'cv_r2_shift': arm_r2['shift']['cv_r2'].to_numpy(),
        })
        # Signal voxel = null mean-over-folds > 0 (continuous null also
        # emitted so a stricter cutoff can be swept downstream).
        block['is_signal'] = block['cv_r2_null'] > 0.0
        rows.append(block)
        print(f'  sub-{subject:02d} {roi}: n_vox={voxel_ids.size}, '
              f'n_signal={int(block["is_signal"].sum())}, '
              f'null_med={np.nanmedian(null_r2):.4f}')

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-ROI summary.
# ---------------------------------------------------------------------------

def _gate_mask(g: pd.DataFrame) -> np.ndarray:
    """Inclusion gate: signal AND max(gain, shift) > model0."""
    helps = np.maximum(g['cv_r2_gain'].to_numpy(),
                       g['cv_r2_shift'].to_numpy()) > g['cv_r2_model0'].to_numpy()
    return g['is_signal'].to_numpy() & helps


def per_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for roi, roi_df in df.groupby('roi'):
        # Per-subject median Δ(shift−gain) within the gated set, for the
        # across-subject sign test.
        per_sub_delta = []
        for _subject, sg in roi_df.groupby('subject'):
            gate = _gate_mask(sg)
            if gate.sum() == 0:
                continue
            d = (sg['cv_r2_shift'].to_numpy()[gate]
                 - sg['cv_r2_gain'].to_numpy()[gate])
            per_sub_delta.append(np.nanmedian(d))
        per_sub_delta = np.asarray(per_sub_delta, dtype=np.float64)
        per_sub_delta = per_sub_delta[np.isfinite(per_sub_delta)]
        n_sub = per_sub_delta.size
        n_pos = int(np.sum(per_sub_delta > 0))
        p_sign = (stats.binomtest(n_pos, n_sub, 0.5).pvalue
                  if n_sub > 0 else np.nan)

        sig = roi_df[roi_df['is_signal']]
        gate_all = _gate_mask(roi_df)
        gated = roi_df[gate_all]

        def med(frame, col):
            v = frame[col].to_numpy(dtype=np.float64)
            return float(np.nanmedian(v)) if v.size else np.nan

        out.append(dict(
            roi=roi,
            n_subjects=int(roi_df['subject'].nunique()),
            n_voxels=int(roi_df.shape[0]),
            n_signal=int(sig.shape[0]),
            n_gated=int(gated.shape[0]),
            # Medians of each level over SIGNAL voxels.
            null_median_sig=med(sig, 'cv_r2_null'),
            model0_median_sig=med(sig, 'cv_r2_model0'),
            gain_median_sig=med(sig, 'cv_r2_gain'),
            shift_median_sig=med(sig, 'cv_r2_shift'),
            # Deltas over SIGNAL voxels.
            d_gain_model0_sig=med(sig, 'cv_r2_gain') - med(sig, 'cv_r2_model0'),
            d_shift_model0_sig=med(sig, 'cv_r2_shift') - med(sig, 'cv_r2_model0'),
            d_shift_gain_sig=med(sig, 'cv_r2_shift') - med(sig, 'cv_r2_gain'),
            d_gain_null_sig=med(sig, 'cv_r2_gain') - med(sig, 'cv_r2_null'),
            d_shift_null_sig=med(sig, 'cv_r2_shift') - med(sig, 'cv_r2_null'),
            # Deltas over the GATED set (signal AND attention-helps).
            d_gain_model0_gated=med(gated, 'cv_r2_gain') - med(gated, 'cv_r2_model0'),
            d_shift_model0_gated=med(gated, 'cv_r2_shift') - med(gated, 'cv_r2_model0'),
            d_shift_gain_gated=med(gated, 'cv_r2_shift') - med(gated, 'cv_r2_gain'),
            d_gain_null_gated=med(gated, 'cv_r2_gain') - med(gated, 'cv_r2_null'),
            d_shift_null_gated=med(gated, 'cv_r2_shift') - med(gated, 'cv_r2_null'),
            # Across-subject sign test on per-subject median Δ(shift−gain)
            # within the gated set.
            n_sub_tested=n_sub,
            n_shift_better=n_pos,
            sign_test_p=float(p_sign) if np.isfinite(p_sign) else np.nan,
        ))
    out_df = pd.DataFrame(out)
    out_df['__o'] = out_df['roi'].apply(
        lambda r: ROI_ORDER.index(r) if r in ROI_ORDER else 999)
    return out_df.sort_values('__o').drop(columns='__o').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Threshold sweep.
# ---------------------------------------------------------------------------

def threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Median Δ(shift−gain) vs increasing null-CV-R² inclusion cutoff.

    For each (roi, cutoff): keep voxels with cv_r2_null > cutoff AND the
    attention-helps gate (max(gain, shift) > model0), then report the
    median Δ(shift−gain) and the voxel count.
    """
    helps = np.maximum(df['cv_r2_gain'].to_numpy(),
                       df['cv_r2_shift'].to_numpy()) > df['cv_r2_model0'].to_numpy()
    delta = df['cv_r2_shift'].to_numpy() - df['cv_r2_gain'].to_numpy()
    null = df['cv_r2_null'].to_numpy()

    rows = []
    for roi, roi_df in df.groupby('roi'):
        mask_roi = (df['roi'] == roi).to_numpy()
        for cutoff in SWEEP_CUTOFFS:
            keep = mask_roi & helps & (null > cutoff)
            d = delta[keep]
            d = d[np.isfinite(d)]
            rows.append(dict(
                roi=roi,
                null_cutoff=cutoff,
                n_voxels=int(d.size),
                d_shift_gain_median=(float(np.median(d)) if d.size else np.nan),
            ))
    # also an ALL-ROI pooled row set
    for cutoff in SWEEP_CUTOFFS:
        keep = helps & (null > cutoff)
        d = delta[keep]
        d = d[np.isfinite(d)]
        rows.append(dict(
            roi='ALL',
            null_cutoff=cutoff,
            n_voxels=int(d.size),
            d_shift_gain_median=(float(np.median(d)) if d.size else np.nan),
        ))
    sweep = pd.DataFrame(rows)
    sweep['__o'] = sweep['roi'].apply(
        lambda r: ROI_ORDER.index(r) if r in ROI_ORDER else 999)
    return (sweep.sort_values(['__o', 'null_cutoff'])
                 .drop(columns='__o').reset_index(drop=True))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(bids_folder: str):
    bids_folder = Path(bids_folder)
    out_root = bids_folder / 'derivatives' / 'af_prf_cv_shiftvsgain'
    out_root.mkdir(parents=True, exist_ok=True)

    df = collect(bids_folder)
    if df.empty:
        print('No matched (model0, gain, shift) cells found — nothing to '
              'aggregate.')
        return

    per_vox_tsv = out_root / 'cv4_per_voxel.tsv'
    df = df.sort_values(['roi', 'subject', 'voxel_id']).reset_index(drop=True)
    df.to_csv(per_vox_tsv, sep='\t', index=False)
    print(f'\nWrote per-voxel table: {per_vox_tsv}  ({df.shape[0]} rows)')

    roi_df = per_roi_summary(df)
    roi_tsv = out_root / 'cv4_per_roi.tsv'
    roi_df.to_csv(roi_tsv, sep='\t', index=False)
    print(f'Wrote per-ROI summary: {roi_tsv}')
    with pd.option_context('display.float_format', lambda v: f'{v: .4f}',
                           'display.max_columns', None, 'display.width', 200):
        print('\nPer-ROI summary (signal-voxel medians + deltas):')
        cols = ['roi', 'n_signal', 'n_gated', 'null_median_sig',
                'model0_median_sig', 'gain_median_sig', 'shift_median_sig',
                'd_shift_gain_gated', 'n_shift_better', 'n_sub_tested',
                'sign_test_p']
        print(roi_df[cols].to_string(index=False))

    sweep_df = threshold_sweep(df)
    sweep_tsv = out_root / 'cv4_threshold_sweep.tsv'
    sweep_df.to_csv(sweep_tsv, sep='\t', index=False)
    print(f'\nWrote threshold-sweep table: {sweep_tsv}')
    with pd.option_context('display.float_format', lambda v: f'{v: .4f}'):
        print(sweep_df[sweep_df['roi'] == 'ALL'].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    args = parser.parse_args()
    main(bids_folder=args.bids_folder)
