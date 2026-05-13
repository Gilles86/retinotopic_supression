"""Fit a PRF model with the EXTENDED design matrix that includes the
search-task distractors as additional stimulus inputs.

This is "cluster plan #2": we use ``Subject.get_stimulus_with_distractors``
which returns a (T, R, R) stimulus on a wider 5°-radius grid that
contains both the bar (inside the bar aperture) and four distractor
disks at the 4° eccentricity ring whenever a target trial with a
non-NaN ``distractor_location`` was active.

Parameter maps end up under
``derivatives/prf_extended/model{N}/sub-{XX}/`` to keep them separate
from the bar-only fits in ``derivatives/prf/``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, maskers

from braincoder.models import (
    GaussianPRF2DWithHRF,
    DifferenceOfGaussiansPRF2DWithHRF,
    DivisiveNormalizationGaussianPRF2DWithHRF,
)
from braincoder.optimize import ParameterFitter
from braincoder.hrf import SPMHRFModel

from retsupp.utils.data import Subject


def main(subject, model_label, bids_folder='/data/ds-retsupp',
         grid_r2_thr=0.05, max_n_iterations=2000, debug=False,
         resolution=80, grid_radius=5.0, distractor_radius=0.4,
         max_distractor_duration=1.5):

    print(f"Fitting EXTENDED PRF model: {model_label} (sub-{subject:02d})")

    bids_folder = Path(bids_folder)
    if debug:
        target_dir = (bids_folder / 'derivatives' / 'prf_extended.debug'
                      / f'model{model_label}' / f'sub-{subject:02d}')
    else:
        target_dir = (bids_folder / 'derivatives' / 'prf_extended'
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

    brain_masker = maskers.NiftiMasker(mask_img=bold_mask)

    if debug:
        resolution = 40

    # --- Concatenated per-run BOLD + per-run extended paradigm. ---
    # Critical: the extended design has run-specific distractor patterns
    # (HP differs per block, trial-by-trial randomization). A mean-BOLD
    # fit with one run's design would mismatch. Instead we concatenate
    # cleaned BOLD + per-run designs and demean each run to remove
    # run-level offsets (proper handling for multi-run continuous fit).
    data_list = []
    paradigm_list = []
    for session in [1, 2]:
        for run in sub.get_runs(session):
            bold_run = sub.get_bold(session=session, run=run, type='cleaned')
            data_run = brain_masker.fit_transform(bold_run)
            if data_run.shape[0] != 258:
                data_run = data_run[:258]
            # Demean per run (== add a per-run intercept to the design).
            # The cleaned BOLD already has cosine drift removed.
            data_run = data_run - data_run.mean(axis=0, keepdims=True)
            data_list.append(data_run.astype(np.float32))

            para_run = sub.get_stimulus_with_distractors(
                session=session, run=run,
                resolution=resolution, grid_radius=grid_radius,
                distractor_radius=distractor_radius,
                max_distractor_duration=max_distractor_duration,
            ).astype(np.float32)
            paradigm_list.append(para_run)
    data = np.concatenate(data_list, axis=0)
    paradigm = np.concatenate(paradigm_list, axis=0)
    print(f"  concatenated: {len(data_list)} runs → {data.shape[0]} TRs, "
          f"paradigm shape {paradigm.shape}")

    grid_coordinates_2d = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1,
        grid_radius=grid_radius,
    )
    grid_coordinates = np.stack(
        (grid_coordinates_2d[0].ravel(), grid_coordinates_2d[1].ravel()),
        axis=1,
    ).astype(np.float32)
    paradigm = paradigm.reshape((paradigm.shape[0], -1))

    # --- Set up the PRF model + grid. ---
    hrf_model = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
    prf_model = GaussianPRF2DWithHRF(
        grid_coordinates, paradigm, hrf_model=hrf_model,
    )

    # Wider grid than the bar-only fits, since voxels can now sit
    # outside the bar aperture (the distractors live at ~4° ecc).
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

    # --- Gradient-descent fit on the suprathreshold subset. ---
    r2_masker = maskers.NiftiMasker(mask_img=r2_mask)
    data_thr = r2_masker.fit_transform(mean_ts)
    grid_pars_thr = grid_pars[r2_grid > grid_r2_thr]
    fitter = ParameterFitter(prf_model, data_thr, paradigm)
    gd_pars = fitter.fit(
        init_pars=grid_pars_thr, max_n_iterations=max_n_iterations,
        learning_rate=0.005,
    )
    pred = fitter.predictions
    r2 = fitter.get_rsq(gd_pars)

    if model_label == 1:
        final_pars = gd_pars.copy()

    elif model_label == 2:
        model_dog = DifferenceOfGaussiansPRF2DWithHRF(
            data=data_thr, paradigm=paradigm, hrf_model=hrf_model,
            grid_coordinates=grid_coordinates,
            flexible_hrf_parameters=False,
        )
        pars_dog_init = gd_pars.copy()
        pars_dog_init['srf_amplitude'] = 5e-2
        pars_dog_init['srf_size'] = 2.0

        par_fitter_dog = ParameterFitter(
            model=model_dog, data=data_thr, paradigm=paradigm,
        )
        pars_dog = par_fitter_dog.fit(
            init_pars=pars_dog_init,
            max_n_iterations=max_n_iterations, learning_rate=0.005,
        )
        pars_dog['theta'] = np.arctan2(pars_dog['y'], pars_dog['x'])
        pars_dog['ecc'] = np.sqrt(pars_dog['x'] ** 2 + pars_dog['y'] ** 2)
        pars_dog['r2'] = par_fitter_dog.get_rsq(pars_dog)

        final_pars = pars_dog.copy()
        pred = par_fitter_dog.predictions
        r2 = par_fitter_dog.get_rsq(final_pars)

    elif model_label == 3:
        gaussian_hrf_model = GaussianPRF2DWithHRF(
            grid_coordinates, paradigm, hrf_model=hrf_model,
            flexible_hrf_parameters=True,
        )
        gaussian_hrf_init_pars = gd_pars.copy()
        gaussian_hrf_init_pars['hrf_delay'] = 4.5
        gaussian_hrf_init_pars['hrf_dispersion'] = 0.75
        gaussian_hrf_fitter = ParameterFitter(
            model=gaussian_hrf_model, data=data_thr, paradigm=paradigm,
        )
        gaussian_hrf_pars = gaussian_hrf_fitter.fit(
            init_pars=gaussian_hrf_init_pars,
            max_n_iterations=max_n_iterations, learning_rate=0.005,
        )
        r2 = gaussian_hrf_fitter.get_rsq(gaussian_hrf_pars)
        pred = gaussian_hrf_fitter.predictions
        final_pars = gaussian_hrf_pars.copy()

    elif model_label == 4:
        dog_hrf_model = DifferenceOfGaussiansPRF2DWithHRF(
            data=data_thr, paradigm=paradigm, hrf_model=hrf_model,
            grid_coordinates=grid_coordinates,
            flexible_hrf_parameters=True,
        )
        dog_hrf_init_pars = gd_pars.copy()
        dog_hrf_init_pars['srf_amplitude'] = 1e-3
        dog_hrf_init_pars['srf_size'] = 2.0
        dog_hrf_init_pars['hrf_delay'] = 4.5
        dog_hrf_init_pars['hrf_dispersion'] = 0.75

        dog_hrf_fitter = ParameterFitter(
            model=dog_hrf_model, data=data_thr, paradigm=paradigm,
        )
        dog_hrf_pars = dog_hrf_fitter.fit(
            init_pars=dog_hrf_init_pars,
            max_n_iterations=max_n_iterations, learning_rate=0.005,
        )
        r2 = dog_hrf_fitter.get_rsq(dog_hrf_pars)
        pred = dog_hrf_fitter.predictions
        final_pars = dog_hrf_pars.copy()

    elif model_label == 6:
        dn_model = DivisiveNormalizationGaussianPRF2DWithHRF(
            data=data_thr, paradigm=paradigm, hrf_model=hrf_model,
            grid_coordinates=grid_coordinates,
            flexible_hrf_parameters=True,
        )
        dn_init_pars = gd_pars.copy()
        dn_init_pars['rf_amplitude'] = gd_pars['amplitude']
        dn_init_pars['srf_amplitude'] = 1e-2
        dn_init_pars['srf_size'] = 2.0
        dn_init_pars['neural_baseline'] = 1.0
        dn_init_pars['surround_baseline'] = 1.0
        dn_init_pars['bold_baseline'] = 0.0
        dn_init_pars['hrf_delay'] = 4.5
        dn_init_pars['hrf_dispersion'] = 0.75

        dn_fitter = ParameterFitter(
            model=dn_model, data=data_thr, paradigm=paradigm,
        )
        dn_pars = dn_fitter.fit(
            init_pars=dn_init_pars,
            max_n_iterations=max_n_iterations, learning_rate=0.005,
        )
        r2 = dn_fitter.get_rsq(dn_pars)
        pred = dn_fitter.predictions
        final_pars = dn_pars.copy()
    else:
        raise ValueError(f'Unknown model label: {model_label}')

    final_pars['theta'] = np.arctan2(final_pars['y'], final_pars['x'])
    final_pars['ecc'] = np.sqrt(final_pars['x'] ** 2 + final_pars['y'] ** 2)
    final_pars['r2'] = r2

    # --- Write parameter maps. Float32 wrap — see CLAUDE.md §"NIfTI dtype trap".
    for par in final_pars.columns:
        final_par_img = r2_masker.inverse_transform(final_pars[par])
        if par in grid_pars.columns:
            grid_par_img = brain_masker.inverse_transform(grid_pars[par])
        else:
            grid_par_img = image.math_img('np.zeros_like(img)',
                                          img=final_par_img)
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
    p = argparse.ArgumentParser(description="Fit extended-design PRF model")
    p.add_argument('subject', type=int, help='Subject ID')
    p.add_argument('--model', type=int, default=1)
    p.add_argument('--r2_thr', default=0.04, type=float,
                   help='R2 threshold for grid fit')
    p.add_argument('--resolution', default=80, type=int,
                   help='Resolution of stimulus grid (per side)')
    p.add_argument('--grid_radius', default=5.0, type=float,
                   help='Half-width of stimulus grid in degrees')
    p.add_argument('--distractor_radius', default=0.4, type=float,
                   help='Distractor disk radius in degrees')
    p.add_argument('--max_distractor_duration', default=1.5, type=float,
                   help='Max on-window length per trial in seconds')
    p.add_argument('--max_n_iterations', default=4000, type=int)
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
    )
