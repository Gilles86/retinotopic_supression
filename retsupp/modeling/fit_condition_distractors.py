"""Conditionwise PRF fits using the FULL paradigm (bar + distractor disks).

Mirrors :mod:`retsupp.modeling.fit_condition` but with per-run distractor
information baked into each TR's stimulus, on the wider extended grid.
For each HP condition we concatenate that condition's runs (BOLD + per-
run extended paradigm) and re-fit the PRF, initialized from a baseline
mean fit (default ``--init-model 4`` from ``derivatives/prf/model4``).

Output
------
``derivatives/prf_conditionfit_distractors/model{N}/sub-XX/`` with one
NIfTI per (condition, parameter), matching the naming used by
:mod:`fit_condition`.

Voxel chunking
--------------
The gradient-descent step is chunked over voxels (default 5000 per
chunk) so the (V_chunk, G) PRF kernel + (T, V_chunk) data fit in GPU
RAM at whole-brain voxel counts.

HP assignment
-------------
Uses :meth:`Subject.get_hpd_locations`, NOT the older buggy
``get_distractor_mapping_old``.

Usage
-----
``python -m retsupp.modeling.fit_condition_distractors 5 --model 4``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import (
    DifferenceOfGaussiansPRF2DWithHRF,
    DivisiveNormalizationGaussianPRF2DWithHRF,
    GaussianPRF2DWithHRF,
)
from braincoder.optimize import ParameterFitter

from retsupp.utils.data import Subject

CONDITIONS_HP = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


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
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception as e:
        print(f'    [GPU mem log failed: {e}]')


def build_model(model_label, grid_coordinates, paradigm, hrf_model):
    """Instantiate a PRF model class matching the integer label."""
    if model_label == 1:
        return GaussianPRF2DWithHRF(grid_coordinates, paradigm,
                                    hrf_model=hrf_model)
    if model_label == 2:
        return DifferenceOfGaussiansPRF2DWithHRF(
            grid_coordinates, paradigm, hrf_model=hrf_model,
        )
    if model_label == 3:
        return GaussianPRF2DWithHRF(
            grid_coordinates, paradigm, hrf_model=hrf_model,
            flexible_hrf_parameters=True,
        )
    if model_label == 4:
        return DifferenceOfGaussiansPRF2DWithHRF(
            grid_coordinates, paradigm, hrf_model=hrf_model,
            flexible_hrf_parameters=True,
        )
    if model_label == 6:
        return DivisiveNormalizationGaussianPRF2DWithHRF(
            grid_coordinates, paradigm, hrf_model=hrf_model,
            flexible_hrf_parameters=True,
        )
    raise ValueError(f'Unknown model label: {model_label}')


def fit_in_chunks(prf_model_factory, data, paradigm, init_pars,
                  max_n_iterations: int, learning_rate: float,
                  voxel_chunk_size: int):
    """Run refine_baseline_and_amplitude + fit on consecutive voxel chunks.

    ``prf_model_factory`` is a no-argument callable returning a freshly
    constructed PRF model (paradigm + grid baked in).  Returns
    (concatenated_pars_df, concatenated_pred (T, V), concatenated_r2 (V,)).
    """
    n_voxels = data.shape[1]
    if voxel_chunk_size <= 0 or n_voxels <= voxel_chunk_size:
        prf_model = prf_model_factory()
        data_df = pd.DataFrame(data)
        fitter = ParameterFitter(prf_model, data_df, paradigm)
        refined = fitter.refine_baseline_and_amplitude(
            init_pars.reset_index(drop=True), l2_alpha=1e-3,
        )
        fit_pars = fitter.fit(
            init_pars=refined,
            max_n_iterations=max_n_iterations,
            learning_rate=learning_rate,
        )
        r2 = fitter.get_rsq(fit_pars)
        pred = fitter.predictions
        r2_arr = r2.values if hasattr(r2, 'values') else np.asarray(r2)
        return fit_pars.reset_index(drop=True), np.asarray(pred), r2_arr

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
        data_chunk_df = pd.DataFrame(data[:, lo:hi])
        init_chunk = init_pars.iloc[lo:hi].reset_index(drop=True)
        prf_model = prf_model_factory()
        fitter = ParameterFitter(prf_model, data_chunk_df, paradigm)
        refined = fitter.refine_baseline_and_amplitude(init_chunk,
                                                      l2_alpha=1e-3)
        fit_pars_chunk = fitter.fit(
            init_pars=refined,
            max_n_iterations=max_n_iterations,
            learning_rate=learning_rate,
        )
        r2_chunk = fitter.get_rsq(fit_pars_chunk)
        pred_chunk = fitter.predictions
        pars_chunks.append(fit_pars_chunk.reset_index(drop=True))
        pred_chunks.append(np.asarray(pred_chunk))
        r2_arr = (r2_chunk.values if hasattr(r2_chunk, 'values')
                  else np.asarray(r2_chunk))
        r2_chunks.append(r2_arr)
        _log_gpu_mem(tag=f'after chunk {ci + 1}/{n_chunks}')
    fit_pars = pd.concat(pars_chunks, ignore_index=True)
    pred = np.concatenate(pred_chunks, axis=1)
    r2 = np.concatenate(r2_chunks, axis=0)
    return fit_pars, pred, r2


def init_pars_for_model(model_label, init_pars_baseline):
    """Take a baseline-fit DataFrame (from sub.get_prf_parameters_volume)
    and return an init for the requested model with extra parameters
    filled to sensible defaults.

    The baseline (init_model, e.g. model 4) and the target model don't
    have to match; we only require it has at minimum
    ``x, y, sd, amplitude, baseline``.
    """
    base = init_pars_baseline[['x', 'y', 'sd', 'baseline', 'amplitude']].copy()
    if model_label == 1:
        return base
    if model_label == 2:
        out = base.copy()
        out['srf_amplitude'] = 5e-2
        out['srf_size'] = 2.0
        return out
    if model_label == 3:
        out = base.copy()
        out['hrf_delay'] = 4.5
        out['hrf_dispersion'] = 0.75
        return out
    if model_label == 4:
        out = base.copy()
        out['srf_amplitude'] = 1e-3
        out['srf_size'] = 2.0
        out['hrf_delay'] = 4.5
        out['hrf_dispersion'] = 0.75
        return out
    if model_label == 6:
        out = base.copy()
        out['rf_amplitude'] = base['amplitude']
        out['srf_amplitude'] = 1e-2
        out['srf_size'] = 2.0
        out['neural_baseline'] = 1.0
        out['surround_baseline'] = 1.0
        out['bold_baseline'] = 0.0
        out['hrf_delay'] = 4.5
        out['hrf_dispersion'] = 0.75
        return out
    raise ValueError(f'Unknown model label: {model_label}')


def main(subject, model_label=1, bids_folder='/data/ds-retsupp',
         init_model=4, max_n_iterations=2000, debug=False,
         resolution=50, r2_thr=0.04,
         grid_radius=5.0, distractor_radius=0.4,
         max_distractor_duration=1.5, voxel_chunk_size=1000,
         learning_rate=0.005):
    print(f"Conditionwise FULL-paradigm PRF fit | sub-{subject:02d} | "
          f"model {model_label} | init from model {init_model}")
    bids_folder = Path(bids_folder)
    target_dir = (bids_folder / 'derivatives'
                  / 'prf_conditionfit_distractors'
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

    brain_masker = input_data.NiftiMasker(mask_img=bold_mask)
    brain_masker.fit()

    # 1) Init from baseline mean-model fit. Filter by r²>thr to avoid
    #    fitting noise voxels.
    init_full = sub.get_prf_parameters_volume(model=init_model,
                                              return_images=False)
    if not isinstance(init_full, pd.DataFrame):
        init_full = pd.DataFrame(init_full)

    if 'r2' not in init_full.columns:
        raise RuntimeError("Init parameters DataFrame missing 'r2' column.")
    keep = init_full['r2'].values > r2_thr
    n_keep = int(keep.sum())
    print(f'  init r²>{r2_thr}: {n_keep} / {keep.size} voxels')
    if n_keep == 0:
        raise RuntimeError(f"No voxels above r2 threshold {r2_thr}.")
    init_baseline = init_full.loc[keep].reset_index(drop=True)
    voxel_idx_global = np.where(keep)[0]

    # 2) Build extended paradigm + grid (shared across conditions).
    grid_coords_2d = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius,
    )
    grid_coordinates = np.stack(
        (grid_coords_2d[0].ravel(), grid_coords_2d[1].ravel()), axis=1,
    ).astype(np.float32)
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)

    # 3) Load BOLD + per-run paradigm; group by HP condition.
    hp_per_run = sub.get_hpd_locations()
    print(f'  HP per run: {hp_per_run}')

    bold_stacks = {c: [] for c in CONDITIONS_HP}
    paradigm_stacks = {c: [] for c in CONDITIONS_HP}
    n_T_run = 258

    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    for session, run in tqdm(session_runs, desc='Loading runs'):
        hp = hp_per_run.get((session, run))
        if hp not in CONDITIONS_HP:
            print(f'  ses-{session}_run-{run}: HP={hp!r}, skipping')
            continue
        bold_fn = (
            bids_folder / 'derivatives' / 'cleaned'
            / f'sub-{subject:02d}' / f'ses-{session}' / 'func'
            / f'sub-{subject:02d}_ses-{session}_task-search_'
              f'desc-cleaned_run-{run}_bold.nii.gz'
        )
        data_run = brain_masker.transform(bold_fn).astype(np.float32)
        if data_run.shape[0] > n_T_run:
            data_run = data_run[:n_T_run]
        elif data_run.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - data_run.shape[0], data_run.shape[1]),
                           dtype=np.float32)
            data_run = np.vstack([data_run, pad])
        # Demean per run to absorb run-level intercept.
        data_run = data_run - data_run.mean(axis=0, keepdims=True)
        # Restrict to the kept voxels.
        data_run_kept = data_run[:, keep]

        para_run = sub.get_stimulus_with_distractors(
            session=session, run=run,
            resolution=resolution, grid_radius=grid_radius,
            distractor_radius=distractor_radius,
            max_distractor_duration=max_distractor_duration,
        ).astype(np.float32)
        if para_run.shape[0] > n_T_run:
            para_run = para_run[:n_T_run]
        elif para_run.shape[0] < n_T_run:
            pad_p = np.zeros((n_T_run - para_run.shape[0],
                              para_run.shape[1], para_run.shape[2]),
                             dtype=np.float32)
            para_run = np.vstack([para_run, pad_p])
        para_run_flat = para_run.reshape((para_run.shape[0], -1))

        bold_stacks[hp].append(data_run_kept)
        paradigm_stacks[hp].append(para_run_flat)

    # 4) Fit per condition: concatenate runs of that HP, GD-fit chunked.
    for cond in CONDITIONS_HP:
        if not bold_stacks[cond]:
            print(f'  HP={cond}: no runs, skipping')
            continue
        bold_cond = np.concatenate(bold_stacks[cond], axis=0)
        paradigm_cond = np.concatenate(paradigm_stacks[cond], axis=0)
        print(f'  HP={cond}: bold {bold_cond.shape}, '
              f'paradigm {paradigm_cond.shape}')

        init_for_model = init_pars_for_model(model_label, init_baseline)

        def factory(paradigm=paradigm_cond):
            return build_model(model_label, grid_coordinates,
                               paradigm, hrf_model)

        fit_pars, pred, r2 = fit_in_chunks(
            factory, bold_cond, paradigm_cond, init_for_model,
            max_n_iterations=max_n_iterations,
            learning_rate=learning_rate,
            voxel_chunk_size=voxel_chunk_size,
        )

        fit_pars['theta'] = np.arctan2(fit_pars['y'], fit_pars['x'])
        fit_pars['ecc'] = np.sqrt(fit_pars['x'] ** 2 + fit_pars['y'] ** 2)
        fit_pars['r2'] = r2

        # Save NIfTIs (one per parameter) using the base masker so
        # downstream code can read them like the bar-only conditionfit.
        # Float32 wrap — see CLAUDE.md §"NIfTI dtype trap".
        n_full = int(brain_masker.mask_img_.get_fdata().sum())
        for par in fit_pars.columns:
            arr = np.zeros(n_full, dtype=np.float32)
            arr[voxel_idx_global] = fit_pars[par].values
            par_img = brain_masker.inverse_transform(arr)
            par_img.set_data_dtype(np.float32)
            par_img.header.set_slope_inter(slope=1, inter=0)
            par_img.to_filename(
                target_dir / f'sub-{subject:02d}_cond-{cond}_desc-{par}.nii.gz'
            )

        pred_full = np.zeros((pred.shape[0], n_full), dtype=np.float32)
        pred_full[:, voxel_idx_global] = pred
        pred_img = brain_masker.inverse_transform(pred_full)
        pred_img.set_data_dtype(np.float32)
        pred_img.header.set_slope_inter(slope=1, inter=0)
        pred_img.to_filename(
            target_dir / f'sub-{subject:02d}_cond-{cond}_desc-pred.nii.gz'
        )

        print(f'  HP={cond}: mean R²={np.nanmean(r2):.3f}, '
              f'wrote {target_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int, help='Subject ID')
    p.add_argument('--model', type=int, default=1,
                   help='Target PRF model label (1, 2, 3, 4, 6).')
    p.add_argument('--init-model', dest='init_model', type=int, default=4,
                   help='Mean-fit PRF model used to initialize x/y/sd/'
                        'amplitude/baseline (default 4 = DoG+HRF).')
    p.add_argument('--bids_folder', default='/data/ds-retsupp')
    p.add_argument('--max_n_iterations', default=4000, type=int)
    p.add_argument('--resolution', default=50, type=int)
    p.add_argument('--r2_thr', default=0.04, type=float,
                   help='Threshold on baseline-fit R² for which voxels to fit.')
    p.add_argument('--grid_radius', default=5.0, type=float)
    p.add_argument('--distractor_radius', default=0.4, type=float)
    p.add_argument('--max_distractor_duration', default=1.5, type=float)
    p.add_argument('--voxel_chunk_size', default=1000, type=int,
                   help='Voxels per GD chunk (set 0 to disable chunking). '
                        '1000 fits in L4 (24 GB) at resolution=50; lower '
                        'further for resolution=80 or higher.')
    p.add_argument('--learning_rate', default=0.005, type=float)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    main(
        args.subject, model_label=args.model, bids_folder=args.bids_folder,
        init_model=args.init_model,
        max_n_iterations=args.max_n_iterations,
        debug=args.debug, resolution=args.resolution, r2_thr=args.r2_thr,
        grid_radius=args.grid_radius,
        distractor_radius=args.distractor_radius,
        max_distractor_duration=args.max_distractor_duration,
        voxel_chunk_size=args.voxel_chunk_size,
        learning_rate=args.learning_rate,
    )
