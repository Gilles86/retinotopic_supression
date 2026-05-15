"""Decode one ``(subject, ROI, run)`` into a per-TR decoded paradigm.

One SLURM array task = one (sub, roi, run). See
``notes/decoding_runwise_plan.md`` for the surrounding plan.

Pipeline
--------
1. Load m4 PRF parameters for the ROI and select voxels by the
   per-(subject, ROI) **p_signal mixture threshold** (posterior=0.5).
   Voxels are kept on r2 > threshold alone -- no sd, no ecc filtering.
2. Slice cleaned BOLD for the requested run out of the per-subject
   ``cleaned_bold_cache`` npz (concat run order is
   ``[(1, r) for r in get_runs(1)] + [(2, r) for r in get_runs(2)]``).
   Falls back to per-run NIfTI when the cache is absent.
3. Build a DoG-with-flex-HRF encoding model on the bar + 8-distractor
   paradigm and fit residual covariance ``omega`` (ResidualFitter).
4. Invert with ``braincoder.optimize.StimulusFitter`` (l2_norm,
   learning_rate, max_n_iterations from CLI; the encoder's HRF is
   applied so the decoded paradigm is at the neural timescale).
5. Persist ``(T, R, R)`` decoded tensor + provenance to
   ``derivatives/decoded_paradigm/model{M}/sub-XX/...``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject
from retsupp.decode.decoder import (
    PRF_PARS_BY_MODEL,
    FLEX_HRF_BY_MODEL,
    _bc,
    load_prf_pars,
    build_paradigm_and_grid,
    make_prf_model,
)


def cache_path_full(sub: Subject, resolution: int) -> Path:
    return (sub.bids_folder / 'derivatives' / 'cleaned_bold_cache'
            / f'sub-{sub.subject_id:02d}'
            / f'sub-{sub.subject_id:02d}_kind-full_res-{resolution}.npz')


def concat_run_index(sub: Subject, session: int, run: int) -> int:
    """Index of ``(session, run)`` in the concat order used by the cache."""
    order = []
    for s in (1, 2):
        for r in sub.get_runs(s):
            order.append((s, r))
    try:
        return order.index((session, run))
    except ValueError:
        raise ValueError(
            f'sub-{sub.subject_id:02d} has no run ses-{session} run-{run}. '
            f'Available: {order}')


def load_run_bold(sub: Subject, session: int, run: int, masker,
                  resolution: int = 50) -> np.ndarray:
    """Cleaned BOLD for one run, cropped to 258 TRs, in BOLD-mask order.

    Prefers the cached npz; falls back to per-run cleaned NIfTI when the
    cache is absent.
    """
    cache = cache_path_full(sub, resolution=resolution)
    if cache.exists():
        with np.load(cache) as d:
            bold = d['bold']
        idx = concat_run_index(sub, session, run)
        start = idx * 258
        out = bold[start:start + 258].astype(np.float32)
        if out.shape[0] != 258:
            raise RuntimeError(
                f'Cache slice for sub-{sub.subject_id:02d} ses-{session} '
                f'run-{run} returned {out.shape[0]} TRs; expected 258.')
        return out

    # Fallback: read the per-run NIfTI directly via the masker.
    fn = (sub.bids_folder / 'derivatives' / 'cleaned'
          / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
          / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
            f'desc-cleaned_run-{run}_bold.nii.gz')
    return masker.transform(str(fn)).astype(np.float32)[:258]


def select_psignal_voxels(sub: Subject, roi: str, prf: dict, masker,
                           model: int = 4, posterior: float = 0.5):
    """Indices into the BOLD-mask flattened array for ROI voxels passing
    ``r2 > p_signal_threshold(roi)``.

    No sd, eccentricity, or hand-tuned r2 cutoffs. Per user instruction
    (2026-05-15): rely solely on the FDR/p_signal mixture posterior.
    Voxels with non-finite PRF parameters are dropped on top of that
    (those are stale NaN sentinels from ``mark_invalid_fits``, not
    candidates for the encoding model).
    """
    r2_thresh = sub.get_r2_threshold(model=model, roi=roi,
                                       posterior=posterior)
    if not np.isfinite(r2_thresh):
        raise RuntimeError(
            f'No p_signal threshold for sub-{sub.subject_id:02d} '
            f'roi={roi} model={model}. Run compute_r2_mixture first.')

    roi_img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
    roi_flat = masker.transform(roi_img).flatten().astype(bool)
    pars = PRF_PARS_BY_MODEL[model]
    finite = np.all(np.stack([np.isfinite(prf[p]) for p in pars]), axis=0)
    keep = roi_flat & finite & (prf['r2'] > r2_thresh)
    return np.where(keep)[0], float(r2_thresh)


def decode_runwise(sub: Subject, session: int, run: int, roi: str, *,
                    model: int = 4,
                    resolution: int = 50,
                    posterior: float = 0.5,
                    l2_norm: float = 1.0,
                    learning_rate: float = 0.01,
                    max_n_iterations: int = 1000,
                    min_n_iterations: int = 200,
                    resid_max_iter: int = 300,
                    residual_method: str = 'gauss',
                    progressbar: bool = False,
                    verbose: bool = True):
    """Decode one run; returns dict of arrays ready to be written to npz."""
    SPMHRFModel, PRFCls, ResidualFitter, StimulusFitter = _bc(model)

    t_total = time.time()
    if verbose:
        print(f'[decode_runwise] sub-{sub.subject_id:02d} ses-{session} '
              f'run-{run} roi={roi} model={model} res={resolution} '
              f'l2={l2_norm} posterior={posterior}', flush=True)

    t0 = time.time()
    prf, masker = load_prf_pars(sub, model=model)
    masker.fit()
    if verbose:
        print(f'  [{time.time() - t0:5.1f}s] PRF params loaded', flush=True)

    t0 = time.time()
    voxel_idx, r2_thresh = select_psignal_voxels(
        sub, roi=roi, prf=prf, masker=masker, model=model, posterior=posterior,
    )
    if verbose:
        print(f'  [{time.time() - t0:5.1f}s] {len(voxel_idx)} voxels kept '
              f'(p_signal thresh r2 > {r2_thresh:.3f})', flush=True)
    if len(voxel_idx) < 10:
        raise RuntimeError(
            f'Too few voxels for sub-{sub.subject_id:02d} roi={roi}: '
            f'{len(voxel_idx)} after p_signal>{posterior}.')

    pars = PRF_PARS_BY_MODEL[model]
    pars_df = pd.DataFrame({p: prf[p][voxel_idx] for p in pars})

    t0 = time.time()
    bold = load_run_bold(sub, session=session, run=run, masker=masker,
                          resolution=resolution)[:, voxel_idx]
    if verbose:
        print(f'  [{time.time() - t0:5.1f}s] BOLD loaded: {bold.shape}',
              flush=True)

    t0 = time.time()
    paradigm, grid = build_paradigm_and_grid(sub, session=session, run=run,
                                              resolution=resolution)
    if verbose:
        print(f'  [{time.time() - t0:5.1f}s] paradigm built: {paradigm.shape}',
              flush=True)

    bold_df = pd.DataFrame(bold,
                            index=pd.Index(np.arange(bold.shape[0]),
                                            name='frame'))
    bold_df.columns.name = 'voxel'
    paradigm_df = pd.DataFrame(paradigm,
                                index=pd.Index(np.arange(paradigm.shape[0]),
                                                name='frame'))

    t0 = time.time()
    prf_model = make_prf_model(model, grid, paradigm, pars_df, data=bold_df)
    rf = ResidualFitter(model=prf_model, data=bold_df, paradigm=paradigm_df,
                         parameters=pars_df.astype(np.float32))
    omega, dof = rf.fit(max_n_iterations=resid_max_iter,
                          method=residual_method,
                          progressbar=progressbar)
    if verbose:
        dof_msg = (f', t-likelihood dof={float(dof):.2f}'
                   if dof is not None
                   else ', Gaussian likelihood')
        print(f'  [{time.time() - t0:5.1f}s] ResidualFitter done{dof_msg}',
              flush=True)

    # Rebuild model on filtered data; StimulusFitter writes to model.parameters.
    prf_model = make_prf_model(model, grid, paradigm, pars_df, data=bold_df)

    t0 = time.time()
    sf = StimulusFitter(model=prf_model, data=bold_df, omega=omega,
                         parameters=pars_df.astype(np.float32),
                         dof=dof)
    decoded = sf.fit(l2_norm=l2_norm, learning_rate=learning_rate,
                      max_n_iterations=max_n_iterations,
                      min_n_iterations=min_n_iterations,
                      progressbar=progressbar)
    if verbose:
        print(f'  [{time.time() - t0:5.1f}s] StimulusFitter done',
              flush=True)

    T = decoded.shape[0]
    res = int(np.sqrt(grid.shape[0]))
    if res * res != grid.shape[0]:
        raise RuntimeError(f'Grid not square: G={grid.shape[0]}')
    decoded_arr = decoded.values.reshape(T, res, res).astype(np.float32)

    if verbose:
        print(f'[decode_runwise] total wall-clock '
              f'{time.time() - t_total:.1f}s', flush=True)

    extent = [float(grid[:, 0].min()), float(grid[:, 0].max()),
              float(grid[:, 1].min()), float(grid[:, 1].max())]
    hp_loc = sub.get_hpd_locations().get((session, run))

    return {
        'decoded': decoded_arr,
        'paradigm': paradigm.reshape(T, res, res).astype(np.float32),
        'grid': grid.astype(np.float32),
        'extent': np.array(extent, dtype=np.float32),
        'hp_location': np.array(str(hp_loc)),
        'n_voxels_used': np.array(len(voxel_idx)),
        'r2_threshold': np.array(r2_thresh, dtype=np.float32),
        'subject': np.array(sub.subject_id),
        'roi': np.array(roi),
        'session': np.array(session),
        'run': np.array(run),
        'model': np.array(model),
        'l2_norm': np.array(l2_norm, dtype=np.float32),
        'posterior': np.array(posterior, dtype=np.float32),
        'resolution': np.array(resolution),
    }


def output_path(bids_folder: Path, subject: int, model: int, roi: str,
                session: int, run: int) -> Path:
    return (Path(bids_folder) / 'derivatives' / 'decoded_paradigm'
            / f'model{model}' / f'sub-{subject:02d}'
            / f'sub-{subject:02d}_ses-{session}_run-{run}_'
              f'roi-{roi}_desc-decoded.npz')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject', type=int)
    p.add_argument('--session', type=int, required=True)
    p.add_argument('--run', type=int, required=True)
    p.add_argument('--roi', required=True)
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    p.add_argument('--model', type=int, default=4)
    p.add_argument('--resolution', type=int, default=50)
    p.add_argument('--posterior', type=float, default=0.5)
    p.add_argument('--residual-method', choices=['gauss', 't'], default='gauss',
                   help='ResidualFitter likelihood; "t" fits dof and uses '
                        'Student-t residuals in StimulusFitter (more robust '
                        'to outlier voxels).')
    p.add_argument('--l2-norm', type=float, default=1.0)
    p.add_argument('--learning-rate', type=float, default=0.01)
    p.add_argument('--max-n-iterations', type=int, default=1000)
    p.add_argument('--min-n-iterations', type=int, default=200)
    p.add_argument('--resid-max-iter', type=int, default=300)
    p.add_argument('--progressbar', action='store_true',
                   help='Show iter-level progress (off by default for cleaner SLURM logs).')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing output npz.')
    p.add_argument('--output-path', default=None,
                   help='Override the default derivatives/decoded_paradigm/... '
                        'output path. Useful for L2 sweeps.')
    a = p.parse_args()

    if a.output_path:
        out = Path(a.output_path)
    else:
        out = output_path(Path(a.bids_folder), a.subject, a.model, a.roi,
                           a.session, a.run)
    if out.exists() and not a.force:
        print(f'[decode_runwise] output exists, skipping: {out}', flush=True)
        return

    sub = Subject(a.subject, bids_folder=a.bids_folder)
    result = decode_runwise(
        sub, session=a.session, run=a.run, roi=a.roi,
        model=a.model, resolution=a.resolution, posterior=a.posterior,
        l2_norm=a.l2_norm, learning_rate=a.learning_rate,
        max_n_iterations=a.max_n_iterations,
        min_n_iterations=a.min_n_iterations,
        resid_max_iter=a.resid_max_iter,
        residual_method=a.residual_method,
        progressbar=a.progressbar,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    print(f'[decode_runwise] wrote {out}  '
          f'(decoded shape {result["decoded"].shape}, '
          f'{int(result["n_voxels_used"])} voxels)', flush=True)


if __name__ == '__main__':
    main()
