"""Helpers for leave-one-condition-out cross-validation of AF + PRF fits.

The non-CV fit scripts (``fit_af_prf_braincoder.py``,
``fit_dynamic_af_braincoder.py``, ``fit_dog_af_prf_braincoder.py``,
``fit_dog_dynamic_af_braincoder.py``) build, for each subject, a
concatenated stack of:

* ``bold``                — (T_total, V)
* ``paradigm``            — (T_total, G)
* ``condition_indicator`` — (T_total, n_C)
* (optional) ``dynamic_indicator`` — (T_total, n_C)

across the 12 (session, run) chunks. Each chunk has its single HP
condition flagged in ``condition_indicator``.

For CV we need to:

1. Split the stack into "train" rows (3 conditions) and "held-out"
   rows (1 condition) using the run-level HP labels.
2. Refit the model on the train rows only.
3. Construct a fresh model instance with the held-out paradigm /
   condition_indicator / dynamic_indicator (the braincoder model
   stores ``condition_indicator`` and ``dynamic_indicator`` as
   instance attributes — they are not predict-time arguments).
4. Compute per-voxel CV-R² on the held-out BOLD.

This module owns those split + scoring utilities so all CV variants
share a single reference implementation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


@dataclass
class RunMeta:
    """Per-run-chunk bookkeeping for the concatenated CV stack."""
    session: int
    run: int
    hp: str            # one of CONDITIONS
    start: int         # row offset of this run in the concatenated stack
    n_T: int           # number of timepoints contributed by this run

    @property
    def stop(self) -> int:
        return self.start + self.n_T


def held_out_condition(cv_fold: int) -> str:
    """Return the HP-condition string corresponding to a CV fold index."""
    if cv_fold not in (0, 1, 2, 3):
        raise ValueError(
            f'cv_fold must be one of 0, 1, 2, 3 (got {cv_fold!r}).')
    return CONDITIONS[cv_fold]


def split_by_condition(
    bold: np.ndarray,
    paradigm: np.ndarray,
    condition_indicator: np.ndarray,
    run_meta: list[RunMeta],
    held_out_cond: str,
    dynamic_indicator: Optional[np.ndarray] = None,
):
    """Slice the concatenated stack into train vs held-out tensors.

    Parameters
    ----------
    bold : (T_total, V) ndarray
    paradigm : (T_total, G) ndarray
    condition_indicator : (T_total, n_C) ndarray
    run_meta : list of RunMeta
        One entry per run chunk; ``sum(rm.n_T for rm in run_meta) ==
        T_total`` is required.
    held_out_cond : str
        One of CONDITIONS — the HP string for the held-out fold.
    dynamic_indicator : (T_total, n_C) ndarray, optional
        Same shape as condition_indicator; only set for the dynamic-AF
        models.

    Returns
    -------
    dict with keys:
        train : dict
            bold, paradigm, condition_indicator, [dynamic_indicator],
            run_meta (subset of input).
        held : dict
            bold, paradigm, condition_indicator, [dynamic_indicator],
            run_meta (subset of input).
    """
    if held_out_cond not in CONDITIONS:
        raise ValueError(
            f'held_out_cond must be in {CONDITIONS!r} (got '
            f'{held_out_cond!r}).')

    expected_T = sum(rm.n_T for rm in run_meta)
    if bold.shape[0] != expected_T:
        raise ValueError(
            f'run_meta sums to {expected_T} rows but bold has '
            f'{bold.shape[0]} rows.')

    train_rows = []
    held_rows = []
    train_run_meta: list[RunMeta] = []
    held_run_meta: list[RunMeta] = []

    train_offset = 0
    held_offset = 0
    for rm in run_meta:
        rows = np.arange(rm.start, rm.stop)
        if rm.hp == held_out_cond:
            held_rows.append(rows)
            held_run_meta.append(RunMeta(
                session=rm.session, run=rm.run, hp=rm.hp,
                start=held_offset, n_T=rm.n_T,
            ))
            held_offset += rm.n_T
        else:
            train_rows.append(rows)
            train_run_meta.append(RunMeta(
                session=rm.session, run=rm.run, hp=rm.hp,
                start=train_offset, n_T=rm.n_T,
            ))
            train_offset += rm.n_T

    if not train_rows:
        raise RuntimeError(
            f'No training runs left after holding out {held_out_cond!r}.')
    if not held_rows:
        raise RuntimeError(
            f'No held-out runs found for condition {held_out_cond!r}. '
            f'(Subject may not have any runs with this HP.)')

    train_idx = np.concatenate(train_rows)
    held_idx = np.concatenate(held_rows)

    def take(arr):
        return None if arr is None else arr[train_idx]

    train = dict(
        bold=bold[train_idx],
        paradigm=paradigm[train_idx],
        condition_indicator=condition_indicator[train_idx],
        dynamic_indicator=take(dynamic_indicator),
        run_meta=train_run_meta,
    )
    held = dict(
        bold=bold[held_idx],
        paradigm=paradigm[held_idx],
        condition_indicator=condition_indicator[held_idx],
        dynamic_indicator=(None if dynamic_indicator is None
                           else dynamic_indicator[held_idx]),
        run_meta=held_run_meta,
    )
    return {'train': train, 'held': held}


def per_voxel_r2(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Per-voxel coefficient of determination on (T, V) arrays.

    R² = 1 − Σ_t (y_t − ŷ_t)² / Σ_t (y_t − ȳ)²,  per voxel.

    No clipping (returns negative values when the model fits worse than
    the held-out per-voxel mean — that is a meaningful CV-R² sign).
    """
    if observed.shape != predicted.shape:
        raise ValueError(
            f'observed shape {observed.shape} != predicted shape '
            f'{predicted.shape}.')
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    ss_res = np.sum((obs - pred) ** 2, axis=0)
    ss_tot = np.sum((obs - obs.mean(axis=0, keepdims=True)) ** 2, axis=0)
    # Avoid 0/0 -> nan; treat zero-variance voxels as R²=0.
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(ss_tot > 0, r2, 0.0)
    return r2.astype(np.float32)


def restrict_to_train_condition(
    bold: np.ndarray,
    paradigm: np.ndarray,
    condition_indicator: np.ndarray,
    run_meta: list[RunMeta],
    held_out_cond_idx: int,
    dynamic_indicator: Optional[np.ndarray] = None,
):
    """Public façade matching the spec — wraps ``split_by_condition``.

    Returns a single dict ``{'train': ..., 'held': ...}`` with train-only
    views of the data plus the held-out portions.
    """
    held_out_cond = held_out_condition(held_out_cond_idx)
    return split_by_condition(
        bold=bold, paradigm=paradigm,
        condition_indicator=condition_indicator,
        run_meta=run_meta,
        held_out_cond=held_out_cond,
        dynamic_indicator=dynamic_indicator,
    )


def write_cv_tsvs(
    out_dir: Path,
    *,
    subject: int,
    roi: str,
    model: str,
    voxel_ids: np.ndarray,
    cv_r2_per_fold: list,
    train_r2_per_fold: list,
    fit_pars_per_fold: list,
    shared_pars_per_fold: list,
    shared_par_labels: list[str],
    per_voxel_par_cols: list[str],
    train_run_meta_per_fold: list,
    held_run_meta_per_fold: list,
    selector: str,
    p_signal_thr: float,
    extra_meta: Optional[dict] = None,
) -> dict[str, Path]:
    """Write the canonical 3-file CV output for one (subject, ROI, model).

    Produces, under ``out_dir``::

        sub-XX_roi-{ROI}_cv-r2.tsv      tidy long, one row per (fold, voxel)
        sub-XX_roi-{ROI}_cv-params.tsv  tidy long, per-voxel + broadcast shared
        sub-XX_roi-{ROI}_meta.json      fold run assignments, n_vox, selector

    ``voxel_ids`` are the BOLD-masker flat indices (so every CV arm —
    null/model0/gain/shift — aligns exactly by voxel_id).  ``cv_r2_per_fold``
    and ``train_r2_per_fold`` are length-4 lists of per-voxel arrays
    (folds where the held-out condition was absent may be ``None`` or a
    NaN array; both are handled).  ``fit_pars_per_fold`` is a length-4 list
    of per-fold DataFrames (rows aligned to ``voxel_ids``) or ``None`` for
    skipped folds.  Shared parameters are broadcast onto every voxel row of
    the params table (a single fit produces one shared value per fold).

    Returns a dict mapping ``{'r2', 'params', 'meta'}`` -> written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_ids = np.asarray(voxel_ids)
    n_vox = voxel_ids.size
    prefix = f'sub-{subject:02d}_roi-{roi}'

    # --- cv-r2.tsv : [subject, roi, model, fold, voxel_id, cv_r2, train_r2]
    r2_rows = []
    n_folds = len(cv_r2_per_fold)
    for fold in range(n_folds):
        cv = cv_r2_per_fold[fold]
        tr = train_r2_per_fold[fold]
        cv = (np.full(n_vox, np.nan) if cv is None
              else np.asarray(cv, dtype=np.float64).ravel())
        tr = (np.full(n_vox, np.nan) if tr is None
              else np.asarray(tr, dtype=np.float64).ravel())
        r2_rows.append(pd.DataFrame(dict(
            subject=subject, roi=roi, model=model, fold=fold,
            voxel_id=voxel_ids, cv_r2=cv, train_r2=tr,
        )))
    r2_df = pd.concat(r2_rows, ignore_index=True)
    r2_path = out_dir / f'{prefix}_cv-r2.tsv'
    r2_df.to_csv(r2_path, sep='\t', index=False)

    # --- cv-params.tsv : per-voxel params + broadcast shared params.
    par_rows = []
    for fold in range(n_folds):
        fit_pars = fit_pars_per_fold[fold]
        shared = shared_pars_per_fold[fold]
        if fit_pars is None:
            block = pd.DataFrame({
                'subject': subject, 'roi': roi, 'model': model,
                'fold': fold, 'voxel_id': voxel_ids,
            })
            for c in per_voxel_par_cols:
                block[c] = np.nan
            for c in shared_par_labels:
                block[c] = np.nan
        else:
            fp = fit_pars.reset_index(drop=True)
            block = pd.DataFrame({
                'subject': subject, 'roi': roi, 'model': model,
                'fold': fold, 'voxel_id': voxel_ids,
            })
            for c in per_voxel_par_cols:
                block[c] = (fp[c].to_numpy() if c in fp.columns
                            else np.nan)
            for c in shared_par_labels:
                if shared is not None and c in shared:
                    block[c] = shared[c]
                elif c in fp.columns:
                    block[c] = fp[c].iloc[0]
                else:
                    block[c] = np.nan
        par_rows.append(block)
    par_df = pd.concat(par_rows, ignore_index=True)
    par_path = out_dir / f'{prefix}_cv-params.tsv'
    par_df.to_csv(par_path, sep='\t', index=False)

    # --- meta.json
    meta = dict(
        subject=subject, roi=roi, model=model,
        n_vox=int(n_vox),
        selector=selector,
        p_signal_thr=p_signal_thr,
        per_voxel_par_cols=list(per_voxel_par_cols),
        shared_par_labels=list(shared_par_labels),
        cv_folds=list(range(n_folds)),
        cv_held_out_conditions=list(CONDITIONS[:n_folds]),
        train_run_meta_per_fold=train_run_meta_per_fold,
        held_run_meta_per_fold=held_run_meta_per_fold,
    )
    if extra_meta:
        meta.update(extra_meta)
    meta_path = out_dir / f'{prefix}_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return {'r2': r2_path, 'params': par_path, 'meta': meta_path}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def summarize_split(run_meta: list[RunMeta], held_out_cond: str) -> str:
    """Build a human-readable string listing train vs held-out runs."""
    train_runs = [(rm.session, rm.run, rm.hp) for rm in run_meta
                  if rm.hp != held_out_cond]
    held_runs = [(rm.session, rm.run, rm.hp) for rm in run_meta
                 if rm.hp == held_out_cond]
    lines = [
        f'CV split: held-out HP = {held_out_cond!r}',
        f'  Train runs ({len(train_runs)}):',
    ]
    for ses, run, hp in train_runs:
        lines.append(f'    ses-{ses} run-{run}  HP={hp}')
    lines.append(f'  Held-out runs ({len(held_runs)}):')
    for ses, run, hp in held_runs:
        lines.append(f'    ses-{ses} run-{run}  HP={hp}')
    return '\n'.join(lines)
