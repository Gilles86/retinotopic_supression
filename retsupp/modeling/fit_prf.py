import argparse
from nilearn import input_data, image
from retsupp.utils.data import Subject
from pathlib import Path

from braincoder.models import GaussianPRF2DWithHRF, DifferenceOfGaussiansPRF2DWithHRF
from braincoder.optimize import ParameterFitter
from braincoder.hrf import SPMHRFModel
import numpy as np
import pandas as pd

def main(subject, model_label, bids_folder='/data/ds-retsupp', grid_r2_thr=0.05, max_n_iterations=2000, debug=False, resolution=50):

    print(f"Fitting model: {model_label}")

    bids_folder = Path(bids_folder)
    if debug:
        target_dir = bids_folder / 'derivatives' / 'prf.debug' / f'model{model_label}' / f'sub-{subject:02d}'
    else:
        target_dir = bids_folder / 'derivatives' / 'prf' / f'model{model_label}' / f'sub-{subject:02d}'

    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder)

    bold_mask = sub.get_bold_mask()

    if debug:
        bold_mask = image.math_img('np.where(mask.astype(bool) & (np.random.rand(*mask.shape) < 0.01), 1, 0)', mask=bold_mask)
        max_n_iterations = 100

    mean_ts = bids_folder / 'derivatives' / 'mean_signal' / f'sub-{subject:02d}'/ f'sub-{subject:02d}_desc-mean_bold.nii.gz'

    brain_masker = input_data.NiftiMasker(mask_img=bold_mask)

    data = brain_masker.fit_transform(mean_ts)

    if debug:
        resolution = 20

    paradigm = sub.get_stimulus(resolution=resolution).astype(np.float32)
    grid_coordinates = sub.get_grid_coordinates(resolution=resolution)
    grid_coordinates = np.stack((grid_coordinates[0].ravel(), grid_coordinates[1].ravel()), axis=1).astype(np.float32)
    paradigm = paradigm.reshape((paradigm.shape[0], -1))


    # Load the PRF model

    hrf_model = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
    prf_model = GaussianPRF2DWithHRF(grid_coordinates, paradigm, hrf_model=hrf_model)

    grid_x = np.linspace(-3, 3, 12)
    grid_y = np.linspace(-3, 3, 12)
    grid_sd = np.linspace(1., 4., 8)
    # grid_x = np.linspace(-4, 4, 3)
    # grid_y = np.linspace(-4, 4, 3)
    # grid_sd = np.linspace(.33, 6, 3)
    grid_amplitude = [1]
    grid_baseline = [0]

    # Grid fit
    fitter = ParameterFitter(prf_model, data, paradigm,)
    grid_pars = fitter.fit_grid(grid_x, grid_y, grid_sd, grid_baseline, grid_amplitude, use_correlation_cost=True)

    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars, l2_alpha=0.001)

    grid_pars['theta'] = np.arctan2(grid_pars['y'], grid_pars['x'])
    grid_pars['ecc'] = np.sqrt(grid_pars['x']**2 + grid_pars['y']**2)
    grid_pars['r2'] = fitter.get_rsq(grid_pars)
    r2_grid = grid_pars['r2'].values

    print(pd.Series(r2_grid).describe())
    print(f'Percentage of selected voxels (>r2_thr): {(r2_grid > grid_r2_thr).sum()} / {len(r2_grid)} ({(r2_grid > grid_r2_thr).mean() * 100:.2f}%)')

    r2_mask = image.math_img(f'r2 > {grid_r2_thr}', r2=brain_masker.inverse_transform(grid_pars['r2']))

    # GD fit
    r2_masker = input_data.NiftiMasker(mask_img=r2_mask)
    data = r2_masker.fit_transform(mean_ts)
    grid_pars_thr = grid_pars[r2_grid > grid_r2_thr]
    fitter = ParameterFitter(prf_model, data, paradigm)
    gd_pars = fitter.fit(init_pars=grid_pars_thr, max_n_iterations=max_n_iterations, learning_rate=0.005,)
    pred = fitter.predictions
    r2 = fitter.get_rsq(gd_pars)

    if model_label == 1:
        final_pars = gd_pars.copy()


    elif model_label == 2: # DOG
        print('yo')
        model_dog = DifferenceOfGaussiansPRF2DWithHRF(data=data, paradigm=paradigm, hrf_model=hrf_model,
                                                grid_coordinates=grid_coordinates, flexible_hrf_parameters=False)

        pars_dog_init = gd_pars.copy()
        # This is the relative amplitude of the inhibitory receptive field
        # compared to the excitatory one.
        pars_dog_init['srf_amplitude'] = grid_pars['srf_amplitude'] = 1e-3

        # This is the relative size of the inhibitory receptive field
        # compared to the excitatory one.
        pars_dog_init['srf_size'] = grid_pars['srf_size'] = 2.

        # Let's set up a new parameterfitter 
        par_fitter_dog = ParameterFitter(model=model_dog, data=data, paradigm=paradigm)

        # Note how, for now, we are not optimizing the HRF parameters.
        pars_dog = par_fitter_dog.fit(init_pars=pars_dog_init, max_n_iterations=max_n_iterations, learning_rate=0.005,)

        pars_dog['theta'] = np.arctan2(pars_dog['y'], pars_dog['x'])
        pars_dog['ecc'] = np.sqrt(pars_dog['x']**2 + pars_dog['y']**2)
        pars_dog['r2'] = fitter.get_rsq(pars_dog)

        final_pars = pars_dog.copy()
        pred = par_fitter_dog.predictions
        r2 = par_fitter_dog.get_rsq(final_pars)


    elif model_label == 3:
        gaussian_hrf_model = GaussianPRF2DWithHRF(grid_coordinates, paradigm, hrf_model=hrf_model, flexible_hrf_parameters=True)

        gaussian_hrf_init_pars = gd_pars.copy() 

        gaussian_hrf_init_pars['hrf_delay'] = 4.5
        gaussian_hrf_init_pars['hrf_dispersion'] = .75

        gaussian_hrf_fitter = ParameterFitter(model=gaussian_hrf_model, data=data, paradigm=paradigm)

        gaussian_hrf_pars = gaussian_hrf_fitter.fit(init_pars=gaussian_hrf_init_pars, max_n_iterations=max_n_iterations, learning_rate=0.005)

        r2 = gaussian_hrf_fitter.get_rsq(gaussian_hrf_pars)
        pred = gaussian_hrf_fitter.predictions

        final_pars = gaussian_hrf_pars.copy()

    elif model_label == 4:
        dog_hrf_model = DifferenceOfGaussiansPRF2DWithHRF(data=data, paradigm=paradigm, hrf_model=hrf_model,
                                                grid_coordinates=grid_coordinates, flexible_hrf_parameters=True)
        
        dog_hrf_init_pars = gd_pars.copy()

        dog_hrf_init_pars['srf_amplitude'] = grid_pars['srf_amplitude'] = 1e-3
        dog_hrf_init_pars['srf_size'] = grid_pars['srf_size'] = 2.

        dog_hrf_init_pars['hrf_delay'] = 4.5
        dog_hrf_init_pars['hrf_dispersion'] = .75

        dog_hrf_fitter = ParameterFitter(model=dog_hrf_model, data=data, paradigm=paradigm)

        dog_hrf_pars = dog_hrf_fitter.fit(init_pars=dog_hrf_init_pars, max_n_iterations=max_n_iterations, learning_rate=0.005)

        r2 = dog_hrf_fitter.get_rsq(dog_hrf_pars)

        pred = dog_hrf_fitter.predictions

        final_pars = dog_hrf_pars.copy()

    else:
        raise ValueError(f'Unknown model label: {model_label}')


    final_pars['theta'] = np.arctan2(final_pars['y'], final_pars['x'])
    final_pars['ecc'] = np.sqrt(final_pars['x']**2 + final_pars['y']**2)
    final_pars['r2'] = r2

    for par in final_pars.columns:
        final_par_img = r2_masker.inverse_transform(final_pars[par])
        
        if par in grid_pars.columns:
            grid_par_img = brain_masker.inverse_transform(grid_pars[par])
        else:
            grid_par_img = image.math_img('np.zeros_like(img)', img=final_par_img)

        r2_grid_img = brain_masker.inverse_transform(r2_grid)

        par_img = image.math_img(f'np.where(r2 > {grid_r2_thr}, final_pars, grid_pars)', 
                                r2=r2_grid_img, final_pars=final_par_img, grid_pars=grid_par_img)

        par_img.to_filename(target_dir / f'sub-{subject:02d}_desc-{par}.nii.gz')


    pred_img = r2_masker.inverse_transform(pred)
    pred_img.to_filename(target_dir / f'sub-{subject:02d}_desc-pred.nii.gz')

if __name__ == "__main__":
    # Parse command line arguments
    argument_parser = argparse.ArgumentParser(description="Fit PRF model")
    argument_parser.add_argument('subject', type=int, help='Subject ID')
    argument_parser.add_argument('--model', type=int, default=1)
    argument_parser.add_argument('--r2_thr', default=0.04, type=float, help='R2 threshold for grid fit')
    argument_parser.add_argument('--resolution', default=40, type=int, help='Resolution of PRF stimulus')
    argument_parser.add_argument('--max_n_iterations', default=4000, type=int, help='Number of iterations for each optimisation step')

    argument_parser.add_argument('--bids_folder', default='/data/ds-retsupp', help='BIDS folder path')
    argument_parser.add_argument('--debug', action='store_true', help='Enable debug mode')  

    args = argument_parser.parse_args()

    main(args.subject, model_label=args.model, bids_folder=args.bids_folder, grid_r2_thr=args.r2_thr, 
         debug=args.debug, resolution=args.resolution, max_n_iterations=args.max_n_iterations)
