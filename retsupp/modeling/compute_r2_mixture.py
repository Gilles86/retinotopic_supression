"""Per-(subject, ROI) 2-component Gaussian mixture on **logit(R²)**.

The mixture math (fit / threshold / plot) lives in
:mod:`braincoder.utils.stats`. This module wraps it with BIDS-aware
paths: per-(subject, model, ROI) cached JSON sidecar of mixture
params, a per-voxel posterior NIfTI, and a per-subject diagnostic PDF
under ``derivatives/prf_diagnostics/r2_mixture/``.

Cache schema: ``derivatives/{prf_base_dir}/model{N}/sub-XX/
sub-XX_desc-p_signal.json`` with one entry per ROI plus ``'BRAIN'``
(the whole-brain mixture). Each entry carries ``mixture: 'gmm_logit'``
+ Gaussian-on-logit parameters (see
:func:`braincoder.utils.stats.fit_r2_mixture`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nilearn import maskers

from braincoder.utils.stats import (
    fit_r2_mixture, r2_fdr_threshold as _r2_fdr_threshold_from_fit,
)
from retsupp.utils.data import Subject

ROI_DEFAULTS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']


def _is_gmm_logit(info: dict | None) -> bool:
    """True iff ``info`` is a new-format (logit-Gaussian) mixture entry."""
    return bool(info) and ('noise_mu' in info or info.get('mixture') == 'gmm_logit')


def fit_one_roi(r2: np.ndarray) -> dict:
    """Fit the mixture on one ROI's R²; return per-voxel posterior + fit dict.

    Thin wrapper around :func:`braincoder.utils.stats.fit_and_classify`
    that returns ``{'p_signal', 'fit', 'reason'}`` (the keys retsupp
    callers expect).
    """
    from braincoder.utils.stats import fit_and_classify
    out = fit_and_classify(r2)
    return {'p_signal': out['p_signal'],
            'fit': out['fit'],
            'reason': out['reason']}


def get_or_fit_brain_mixture(subject: int, model: int,
                               bids_folder: str | Path,
                               prf_base_dir: str = 'prf',
                               force: bool = False,
                               roi: str = 'BRAIN') -> dict:
    """Return the cached whole-mask mixture entry for ``(subject, model, roi)``,
    fitting it on cache miss. ``roi`` ∈ ``{'BRAIN', 'GM'}``.

    On miss (or ``force=True``, or stale Beta-format cache) calls
    :func:`retsupp.modeling.run_r2_mixture_all.run_one` which writes:
      - ``derivatives/{prf_base_dir}/model{N}/sub-XX/sub-XX_desc-p_signal.json``
      - ``derivatives/{prf_base_dir}/model{N}/sub-XX/sub-XX_desc-p_signal.nii.gz``
      - ``derivatives/prf_diagnostics/r2_mixture/model{N}/sub-XX_r2_mixture.pdf``
    """
    if roi not in ('BRAIN', 'GM'):
        raise ValueError(f"get_or_fit_brain_mixture expects roi in {{BRAIN, GM}}, got {roi!r}")
    bids_folder = Path(bids_folder)
    sidecar = (bids_folder / 'derivatives' / prf_base_dir
               / f'model{model}' / f'sub-{subject:02d}'
               / f'sub-{subject:02d}_desc-p_signal.json')
    if not force and sidecar.exists():
        with open(sidecar) as fh:
            summary = json.load(fh)
        info = summary.get(roi)
        if _is_gmm_logit(info):
            return info
    from retsupp.modeling.run_r2_mixture_all import run_one
    plots_dir = (bids_folder / 'derivatives' / 'prf_diagnostics'
                 / 'r2_mixture' / f'model{model}')
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"  fitting R² logit-GMM mixture for sub-{subject:02d} model {model} "
          f"(writing cache: {sidecar.name})")
    summary = run_one(subject, model, bids_folder,
                      rois=ROI_DEFAULTS, plots_dir=plots_dir)
    if summary is None:
        raise FileNotFoundError(
            f"No R² NIfTI for sub-{subject:02d} model {model} under "
            f"derivatives/{prf_base_dir}/model{model}/.")
    info = summary.get(roi)
    if not _is_gmm_logit(info):
        raise RuntimeError(
            f"{roi} mixture fit failed for sub-{subject:02d} model {model}: "
            f"{info.get('reason') if info else f'no {roi} entry'}")
    return info


def r2_fdr_threshold_inline(r2: np.ndarray, alpha: float = 0.05) -> float:
    """Pooled-data (no caching) tail-FDR R² threshold via logit-GMM.

    Thin alias for :func:`braincoder.utils.stats.r2_fdr_threshold` —
    kept here so retsupp callers can import everything from one place.
    """
    try:
        return _r2_fdr_threshold_from_fit(r2, alpha=alpha)
    except ValueError:
        return float('inf')


def r2_fdr_threshold(subject: int, model: int,
                      bids_folder: str | Path,
                      alpha: float = 0.05,
                      roi: str = 'BRAIN',
                      prf_base_dir: str = 'prf',
                      force: bool = False) -> float:
    """Per-(subject, model, ROI) tail-FDR R² threshold via logit-GMM.

    ``roi='BRAIN'`` (default) → whole-brain threshold, lazy-fitted via
    :func:`get_or_fit_brain_mixture` (which also writes the diagnostic
    PDF + per-voxel p_signal NIfTI). Any other ROI name → cached
    per-ROI mixture from ``desc-p_signal.json``; if absent (or in the
    legacy Beta format), fit it on-the-fly from the ROI's R² values
    (via :meth:`Subject.get_prf_roi_pars`) and append to the sidecar.
    """
    bids_folder = Path(bids_folder)
    if roi in ('BRAIN', 'GM'):
        info = get_or_fit_brain_mixture(subject, model, bids_folder,
                                          prf_base_dir=prf_base_dir,
                                          force=force, roi=roi)
        return _r2_fdr_threshold_from_fit(info, alpha=alpha)

    sidecar = (bids_folder / 'derivatives' / prf_base_dir
               / f'model{model}' / f'sub-{subject:02d}'
               / f'sub-{subject:02d}_desc-p_signal.json')
    if not force and sidecar.exists():
        with open(sidecar) as fh:
            summary = json.load(fh)
        info = summary.get(roi)
        if _is_gmm_logit(info):
            return _r2_fdr_threshold_from_fit(info, alpha=alpha)

    sub = Subject(subject, bids_folder)
    df = sub.get_prf_roi_pars(roi=roi, model=model)
    real = (df['r2'] > 0) & df['sd'].notna() & (df['r2'] < 0.99)
    if real.sum() < 50:
        return float('inf')
    out = fit_one_roi(df.loc[real, 'r2'].to_numpy())
    if out['fit'] is None:
        return float('inf')
    info = out['fit']
    summary = {}
    if sidecar.exists():
        with open(sidecar) as fh:
            summary = json.load(fh)
    summary.setdefault('model', model)
    summary[roi] = info
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"  fit + cached ROI={roi} mixture into {sidecar.name}")
    return _r2_fdr_threshold_from_fit(info, alpha=alpha)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--model', type=int, default=1)
    p.add_argument('--rois', nargs='+', default=ROI_DEFAULTS)
    p.add_argument('--prf-base-dir', default='prf',
                   help='Subdir under derivatives/ (e.g. prf, prf_bar).')
    args = p.parse_args()

    bids = Path(args.bids_folder)
    sub = Subject(args.subject, bids)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask())
    masker.fit()

    r2_path = (bids / 'derivatives' / args.prf_base_dir
               / f'model{args.model}' / f'sub-{args.subject:02d}'
               / f'sub-{args.subject:02d}_desc-r2.nii.gz')
    r2 = masker.transform(str(r2_path)).flatten()

    p_signal_all = np.full(r2.size, np.nan, dtype=np.float32)
    summary = {'model': args.model}
    for roi in args.rois:
        try:
            roi_img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
            roi_mask = masker.transform(roi_img).flatten().astype(bool)
        except Exception as e:
            summary[roi] = {'reason': f'roi load failed: {e}'}
            continue
        idx = np.where(roi_mask & np.isfinite(r2)
                       & (r2 > 0) & (r2 < 0.99))[0]
        if len(idx) < 50:
            summary[roi] = {'reason': f'only {len(idx)} usable voxels'}
            continue
        out = fit_one_roi(r2[idx])
        if out['fit'] is not None:
            p_signal_all[idx] = out['p_signal']
            summary[roi] = out['fit']
        else:
            summary[roi] = {'reason': out['reason']}

    out_dir = (bids / 'derivatives' / args.prf_base_dir
               / f'model{args.model}' / f'sub-{args.subject:02d}')
    # Float32 wrap — see CLAUDE.md §"NIfTI dtype trap".
    nii = masker.inverse_transform(p_signal_all)
    nii.set_data_dtype(np.float32)
    nii.header.set_slope_inter(slope=1, inter=0)
    out_nii = out_dir / f'sub-{args.subject:02d}_desc-p_signal.nii.gz'
    nii.to_filename(str(out_nii))
    out_json = out_dir / f'sub-{args.subject:02d}_desc-p_signal.json'
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'Wrote {out_nii}')
    print(f'Wrote {out_json}')
    for roi, info in summary.items():
        if isinstance(info, dict) and 'signal_mu' in info:
            print(f'  {roi}: signal μ_R²={info["signal_mean_r2"]:.3f} '
                  f'noise μ_R²={info["noise_mean_r2"]:.3f} '
                  f'(w_signal={info["signal_weight"]:.2f})')
        elif isinstance(info, dict) and 'reason' in info:
            print(f'  {roi}: skipped ({info["reason"]})')


if __name__ == '__main__':
    main()
