"""Mean-fit baseline PRF using the FULL paradigm (bar + distractor disks).

Mirrors :mod:`retsupp.modeling.fit_prf` but with the FULL paradigm: each
run gets its own ``Subject.get_stimulus_with_distractors`` time-course
(extended grid that contains the 4° ring distractors), and per-run
cleaned BOLD is concatenated and fit jointly.

Compared to :mod:`fit_prf_extended` (which is a similar idea but already
exists), this script:
- supports voxel chunking so the fit fits in GPU RAM at high voxel
  counts (whole-brain ~280 K voxels × 80²-grid would otherwise OOM on
  L4); see :func:`fit_in_chunks`.
- writes to ``derivatives/prf_distractors/model{N}/sub-XX/`` using the
  same parameter NIfTI naming as ``derivatives/prf/model{N}/`` so the
  ``Subject.get_prf_parameters_volume`` accessor can be pointed at it
  with a path swap.

Usage
-----
``python -m retsupp.modeling.fit_prf_distractors 5 --model 4``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, maskers
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import (
    DifferenceOfGaussiansPRF2DWithHRF,
    DivisiveNormalizationGaussianPRF2DWithHRF,
    GaussianPRF2DWithHRF,
)
from braincoder.optimize import ParameterFitter

from retsupp.utils.data import Subject


def _log_gpu_mem(tag: str = '') -> None:
    """Log peak GPU memory (TF). Best-effort; no-op if TF/GPU unavailable."""
    try:
        import tensorflow as tf  # local import to keep CPU runs cheap
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return
        info = tf.config.experimental.get_memory_info('GPU:0')
        peak_gb = info.get('peak', 0) / (1024 ** 3)
        cur_gb = info.get('current', 0) / (1024 ** 3)
        print(f'    [GPU mem{(" " + tag) if tag else ""}: '
              f'current={cur_gb:.2f} GB, peak={peak_gb:.2f} GB]')
        # Reset peak so each chunk reports its own peak.
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception as e:
        print(f'    [GPU mem log failed: {e}]')


def build_data_and_paradigm(sub: Subject, brain_masker, resolution: int,
                            grid_radius: float, distractor_radius: float,
                            max_distractor_duration: float):
    """Concatenate cleaned BOLD across runs + per-run extended paradigm.

    Pattern matches ``fit_af_prf_braincoder.py``'s ``--paradigm-type full``
    branch (lines 60-194): per-run paradigm + per-run BOLD, vstacked.
    """
    data_chunks = []
    paradigm_chunks = []
    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    n_T_run = 258
    for session, run in tqdm(session_runs, desc='Loading runs'):
        bold_fn = (
            sub.bids_folder / 'derivatives' / 'cleaned'
            / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
            / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
              f'desc-cleaned_run-{run}_bold.nii.gz'
        )
        data_run = brain_masker.transform(bold_fn).astype(np.float32)
        if data_run.shape[0] > n_T_run:
            data_run = data_run[:n_T_run]
        elif data_run.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - data_run.shape[0], data_run.shape[1]),
                           dtype=np.float32)
            data_run = np.vstack([data_run, pad])
        # Demean per run (== adds a per-run intercept). Cleaned BOLD has
        # already had cosine drifts regressed; this just kills the run-
        # level offset that would otherwise leak into the amplitude fit.
        data_run = data_run - data_run.mean(axis=0, keepdims=True)
        data_chunks.append(data_run)

        para_run = sub.get_stimulus_with_distractors(
            session=session, run=run,
            resolution=resolution, grid_radius=grid_radius,
            distractor_radius=distractor_radius,
            max_distractor_duration=max_distractor_duration,
        ).astype(np.float32)
        paradigm_chunks.append(para_run)

    data = np.concatenate(data_chunks, axis=0)
    paradigm = np.concatenate(paradigm_chunks, axis=0)
    return data, paradigm


def fit_in_chunks(model_cls, model_kwargs, data, paradigm, init_pars,
                  max_n_iterations: int, learning_rate: float,
                  voxel_chunk_size: int):
    """Run ParameterFitter.fit on consecutive chunks of voxels.

    Each chunk gets its own model instance (which is cheap — just wires
    up the shared paradigm + grid into a tf.Module) and ParameterFitter.
    Returns a tuple (concatenated_pars_df, concatenated_pred, concatenated_r2).

    Parameters
    ----------
    model_cls : class
        e.g. ``GaussianPRF2DWithHRF``.
    model_kwargs : dict
        Constructor kwargs reused per chunk (paradigm, grid_coordinates,
        hrf_model, flexible_hrf_parameters, ...).
    data : ndarray (T, V)
    paradigm : ndarray (T, G)
    init_pars : DataFrame (V, n_pars)
    voxel_chunk_size : int
        Number of voxels per chunk. If V <= chunk_size, runs as a single
        chunk.
    """
    n_voxels = data.shape[1]
    if voxel_chunk_size <= 0 or n_voxels <= voxel_chunk_size:
        model = model_cls(**model_kwargs)
        fitter = ParameterFitter(model, data, paradigm)
        fit_pars = fitter.fit(
            init_pars=init_pars,
            max_n_iterations=max_n_iterations,
            learning_rate=learning_rate,
        )
        r2 = fitter.get_rsq(fit_pars)
        pred = fitter.predictions
        return fit_pars, pred, r2

    pars_chunks = []
    pred_chunks = []
    r2_chunks = []
    n_chunks = (n_voxels + voxel_chunk_size - 1) // voxel_chunk_size
    print(f'  voxel chunking: {n_voxels} voxels -> {n_chunks} chunks of '
          f'<= {voxel_chunk_size}')
    for ci in range(n_chunks):
        lo = ci * voxel_chunk_size
        hi = min(lo + voxel_chunk_size, n_voxels)
        print(f'  -- chunk {ci + 1}/{n_chunks}: voxels [{lo}:{hi}]')
        data_chunk = data[:, lo:hi]
        init_chunk = init_pars.iloc[lo:hi].reset_index(drop=True)
        # ParameterFitter expects a DataFrame for `data` so the column
        # index aligns with init_pars index.
        data_chunk_df = pd.DataFrame(data_chunk)
        model = model_cls(**model_kwargs)
        fitter = ParameterFitter(model, data_chunk_df, paradigm)
        fit_pars_chunk = fitter.fit(
            init_pars=init_chunk,
            max_n_iterations=max_n_iterations,
            learning_rate=learning_rate,
        )
        r2_chunk = fitter.get_rsq(fit_pars_chunk)
        pred_chunk = fitter.predictions
        pars_chunks.append(fit_pars_chunk.reset_index(drop=True))
        pred_chunks.append(np.asarray(pred_chunk))
        r2_arr = r2_chunk.values if hasattr(r2_chunk, 'values') \
            else np.asarray(r2_chunk)
        r2_chunks.append(r2_arr)
        _log_gpu_mem(tag=f'after chunk {ci + 1}/{n_chunks}')
    fit_pars = pd.concat(pars_chunks, ignore_index=True)
    pred = np.concatenate(pred_chunks, axis=1)
    r2 = np.concatenate(r2_chunks, axis=0)
    return fit_pars, pred, r2


def main(subject, model_label, bids_folder='/data/ds-retsupp',
         grid_r2_thr=0.05, max_n_iterations=2000, debug=False,
         resolution=50, grid_radius=5.0, distractor_radius=0.4,
         max_distractor_duration=1.5, voxel_chunk_size=1000):

    print(f"Fitting DISTRACTOR PRF model: {model_label}  (sub-{subject:02d})")

    bids_folder = Path(bids_folder)
    if debug:
        target_dir = (bids_folder / 'derivatives' / 'prf_distractors.debug'
                      / f'model{model_label}' / f'sub-{subject:02d}')
    else:
        target_dir = (bids_folder / 'derivatives' / 'prf_distractors'
                      / f'model{model_label}' / f'sub-{subject:02d}')
    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder)

    bold_mask = sub.get_bold_mask()
    if debug:
        bold_mask = image.math_img(
            'np.where(mask.astype(bool) & (np.random.rand(*mask.shape) < 0.01), 1, 0)',
            mask=bold_mask,
        )
        max_n_iterations = 100
        resolution = 30

    brain_masker = maskers.NiftiMasker(mask_img=bold_mask)
    brain_masker.fit()

    # --- Load BOLD + per-run extended paradigm. ---
    data, paradigm = build_data_and_paradigm(
        sub, brain_masker,
        resolution=resolution, grid_radius=grid_radius,
        distractor_radius=distractor_radius,
        max_distractor_duration=max_distractor_duration,
    )
    print(f'  concat: data {data.shape}, paradigm {paradigm.shape}')

    grid_coordinates_2d = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius,
    )
    grid_coordinates = np.stack(
        (grid_coordinates_2d[0].ravel(), grid_coordinates_2d[1].ravel()),
        axis=1,
    ).astype(np.float32)
    paradigm = paradigm.reshape((paradigm.shape[0], -1))

    # --- Grid fit (Gaussian) on full voxel set. ---
    # The grid step uses braincoder's internal parameter-grid chunking
    # (memory_limit-driven), so all voxels can stay on-device for the
    # grid pass. Only the gradient-descent step needs voxel chunking.
    hrf_model = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
    prf_model = GaussianPRF2DWithHRF(
        grid_coordinates, paradigm, hrf_model=hrf_model,
    )

    grid_x = np.linspace(-grid_radius, grid_radius, 16)
    grid_y = np.linspace(-grid_radius, grid_radius, 16)
    grid_sd = np.linspace(0.5, 4.0, 8)
    grid_amplitude = [1]
    grid_baseline = [0]

    fitter = ParameterFitter(prf_model, data, paradigm)
    grid_pars = fitter.fit_grid(
        grid_x, grid_y, grid_sd, grid_baseline, grid_amplitude,
        use_correlation_cost=True,
    )
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars, l2_alpha=0.001)
    grid_pars['theta'] = np.arctan2(grid_pars['y'], grid_pars['x'])
    grid_pars['ecc'] = np.sqrt(grid_pars['x'] ** 2 + grid_pars['y'] ** 2)
    grid_pars['r2'] = fitter.get_rsq(grid_pars)
    r2_grid = grid_pars['r2'].values

    print(pd.Series(r2_grid).describe())
    print(
        f"Voxels passing grid r2 > {grid_r2_thr}: "
        f"{(r2_grid > grid_r2_thr).sum()} / {len(r2_grid)} "
        f"({(r2_grid > grid_r2_thr).mean() * 100:.2f}%)"
    )

    r2_mask = image.math_img(
        f'r2 > {grid_r2_thr}',
        r2=brain_masker.inverse_transform(grid_pars['r2']),
    )
    r2_masker = maskers.NiftiMasker(mask_img=r2_mask)
    r2_masker.fit()

    # Take suprathreshold subset for the GD step. We do this by re-
    # masking the (T, V_full) data array with the suprathreshold mask
    # so we don't have to write/read NIfTIs.
    keep_mask = r2_grid > grid_r2_thr
    data_thr = data[:, keep_mask]
    grid_pars_thr = grid_pars[keep_mask].reset_index(drop=True)

    # --- Gradient-descent fit, chunked over voxels. ---
    # The gradient pass holds the full (V_chunk, G) PRF kernel matrix on
    # device along with (T, V_chunk) data and gradients => RAM scales
    # linearly with V_chunk. Splitting V into chunks of <= voxel_chunk_size
    # keeps peak GPU memory bounded.
    if model_label == 1:
        model_cls = GaussianPRF2DWithHRF
        model_kwargs = dict(
            grid_coordinates=grid_coordinates, paradigm=paradigm,
            hrf_model=hrf_model,
        )
        init_pars_full = grid_pars_thr.copy()

    elif model_label == 2:
        model_cls = DifferenceOfGaussiansPRF2DWithHRF
        model_kwargs = dict(
            grid_coordinates=grid_coordinates, paradigm=paradigm,
            hrf_model=hrf_model, flexible_hrf_parameters=False,
        )
        init_pars_full = grid_pars_thr.copy()
        init_pars_full['srf_amplitude'] = 5e-2
        init_pars_full['srf_size'] = 2.0

    elif model_label == 3:
        model_cls = GaussianPRF2DWithHRF
        model_kwargs = dict(
            grid_coordinates=grid_coordinates, paradigm=paradigm,
            hrf_model=hrf_model, flexible_hrf_parameters=True,
        )
        init_pars_full = grid_pars_thr.copy()
        init_pars_full['hrf_delay'] = 4.5
        init_pars_full['hrf_dispersion'] = 0.75

    elif model_label == 4:
        model_cls = DifferenceOfGaussiansPRF2DWithHRF
        model_kwargs = dict(
            grid_coordinates=grid_coordinates, paradigm=paradigm,
            hrf_model=hrf_model, flexible_hrf_parameters=True,
        )
        init_pars_full = grid_pars_thr.copy()
        init_pars_full['srf_amplitude'] = 1e-3
        init_pars_full['srf_size'] = 2.0
        init_pars_full['hrf_delay'] = 4.5
        init_pars_full['hrf_dispersion'] = 0.75

    elif model_label == 6:
        model_cls = DivisiveNormalizationGaussianPRF2DWithHRF
        model_kwargs = dict(
            grid_coordinates=grid_coordinates, paradigm=paradigm,
            hrf_model=hrf_model, flexible_hrf_parameters=True,
        )
        init_pars_full = grid_pars_thr.copy()
        init_pars_full['rf_amplitude'] = grid_pars_thr['amplitude']
        init_pars_full['srf_amplitude'] = 1e-2
        init_pars_full['srf_size'] = 2.0
        init_pars_full['neural_baseline'] = 1.0
        init_pars_full['surround_baseline'] = 1.0
        init_pars_full['bold_baseline'] = 0.0
        init_pars_full['hrf_delay'] = 4.5
        init_pars_full['hrf_dispersion'] = 0.75
    else:
        raise ValueError(f'Unknown model label: {model_label}')

    final_pars, pred, r2 = fit_in_chunks(
        model_cls, model_kwargs,
        data_thr, paradigm, init_pars_full,
        max_n_iterations=max_n_iterations,
        learning_rate=0.005,
        voxel_chunk_size=voxel_chunk_size,
    )

    final_pars['theta'] = np.arctan2(final_pars['y'], final_pars['x'])
    final_pars['ecc'] = np.sqrt(final_pars['x'] ** 2 + final_pars['y'] ** 2)
    final_pars['r2'] = r2

    # --- Write parameter maps. Float32 wrap — see CLAUDE.md §"NIfTI dtype trap".
    for par in final_pars.columns:
        final_par_img = r2_masker.inverse_transform(final_pars[par])
        if par in grid_pars.columns:
            grid_par_img = brain_masker.inverse_transform(grid_pars[par])
        else:
            grid_par_img = image.math_img(
                'np.zeros_like(img)', img=final_par_img,
            )
        r2_grid_img = brain_masker.inverse_transform(r2_grid)
        par_img = image.math_img(
            f'np.where(r2 > {grid_r2_thr}, final_pars, grid_pars)',
            r2=r2_grid_img, final_pars=final_par_img,
            grid_pars=grid_par_img,
        )
        par_img.set_data_dtype(np.float32)
        par_img.header.set_slope_inter(slope=1, inter=0)
        par_img.to_filename(target_dir / f'sub-{subject:02d}_desc-{par}.nii.gz')

    pred_img = r2_masker.inverse_transform(pred)
    pred_img.set_data_dtype(np.float32)
    pred_img.header.set_slope_inter(slope=1, inter=0)
    pred_img.to_filename(target_dir / f'sub-{subject:02d}_desc-pred.nii.gz')

    print(f"Wrote outputs to {target_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int, help='Subject ID')
    p.add_argument('--model', type=int, default=1)
    p.add_argument('--r2_thr', default=0.04, type=float,
                   help='R2 threshold for grid fit')
    p.add_argument('--resolution', default=50, type=int,
                   help='Resolution of stimulus grid (per side)')
    p.add_argument('--grid_radius', default=5.0, type=float,
                   help='Half-width of stimulus grid in degrees')
    p.add_argument('--distractor_radius', default=0.4, type=float,
                   help='Distractor disk radius in degrees')
    p.add_argument('--max_distractor_duration', default=1.5, type=float,
                   help='Max on-window length per trial in seconds')
    p.add_argument('--max_n_iterations', default=4000, type=int)
    p.add_argument('--voxel_chunk_size', default=1000, type=int,
                   help='Voxels per GD chunk (set 0 to disable chunking). '
                        '1000 fits in L4 (24 GB) at resolution=50; lower '
                        'further for resolution=80 or higher.')
    p.add_argument('--bids_folder', default='/data/ds-retsupp')
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    main(
        args.subject, model_label=args.model, bids_folder=args.bids_folder,
        grid_r2_thr=args.r2_thr, debug=args.debug,
        resolution=args.resolution, grid_radius=args.grid_radius,
        distractor_radius=args.distractor_radius,
        max_distractor_duration=args.max_distractor_duration,
        max_n_iterations=args.max_n_iterations,
        voxel_chunk_size=args.voxel_chunk_size,
    )
