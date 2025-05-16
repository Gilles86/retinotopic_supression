import argparse
from nilearn import input_data, image
from retsupp.utils.data import Subject
from pathlib import Path

from braincoder.models import GaussianPRF2DWithHRF, DifferenceOfGaussiansPRF2DWithHRF
from braincoder.optimize import ParameterFitter
from braincoder.hrf import SPMHRFModel
import numpy as np

def main(subject, model_label, bids_folder='/data/ds-retsupp', grid_r2_thr=0.2, max_n_iterations=1000):

    print(f"Fitting model: {model_label}")

    bids_folder = Path(bids_folder)
    target_dir = bids_folder / 'derivatives' / 'prf' / f'model{model_label}' / f'sub-{subject:02d}'
    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder)

    bold_mask = sub.get_bold_mask()
    mean_ts = bids_folder / 'derivatives' / 'mean_signal' / f'sub-{subject:02d}'/ f'sub-{subject:02d}_desc-mean_bold.nii.gz'

    brain_masker = input_data.NiftiMasker(mask_img=bold_mask)

    data = brain_masker.fit_transform(mean_ts)

    paradigm = sub.get_stimulus(resolution=25).astype(np.float32)
    grid_coordinates = sub.get_grid_coordinates(resolution=25)
    grid_coordinates = np.stack((grid_coordinates[0].ravel(), grid_coordinates[1].ravel()), axis=1).astype(np.float32)
    paradigm = paradigm.reshape((paradigm.shape[0], -1))


    # Load the PRF model

    hrf_model = SPMHRFModel(tr=1.6)
    prf_model = GaussianPRF2DWithHRF(grid_coordinates, paradigm, hrf_model=hrf_model)

    grid_x = np.linspace(-3, 3, 12)
    grid_y = np.linspace(-3, 3, 12)
    grid_sd = np.linspace(.33, 3, 8)
    # grid_x = np.linspace(-4, 4, 3)
    # grid_y = np.linspace(-4, 4, 3)
    # grid_sd = np.linspace(.33, 6, 3)
    grid_amplitude = [1]
    grid_baseline = [0]

    # Grid fit
    fitter = ParameterFitter(prf_model, data, paradigm,)
    grid_pars = fitter.fit_grid(grid_x, grid_y, grid_sd, grid_baseline, grid_amplitude, use_correlation_cost=True)
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)
    grid_pars['theta'] = np.arctan2(grid_pars['y'], grid_pars['x'])
    grid_pars['ecc'] = np.sqrt(grid_pars['x']**2 + grid_pars['y']**2)
    grid_pars['r2'] = fitter.get_rsq(grid_pars)
    r2_grid = grid_pars['r2'].values

    r2_mask = image.math_img(f'r2 > {grid_r2_thr}', r2=brain_masker.inverse_transform(grid_pars['r2']))

    # GD fit
    r2_masker = input_data.NiftiMasker(mask_img=r2_mask)
    data = r2_masker.fit_transform(mean_ts)
    grid_pars_thr = grid_pars[r2_grid > grid_r2_thr]
    fitter = ParameterFitter(prf_model, data, paradigm)
    gd_pars = fitter.fit(init_pars=grid_pars_thr, max_n_iterations=max_n_iterations)

    if model_label == 1:
        gd_pars['theta'] = np.arctan2(gd_pars['y'], gd_pars['x'])
        gd_pars['ecc'] = np.sqrt(gd_pars['x']**2 + gd_pars['y']**2)
        gd_pars['r2'] = fitter.get_rsq(gd_pars)

        # save the results
        for par in gd_pars.columns:
            gd_par_img = r2_masker.inverse_transform(gd_pars[par])
            grid_par_img = brain_masker.inverse_transform(grid_pars[par])
            r2_grid_img = brain_masker.inverse_transform(r2_grid)

            par_img = image.math_img(f'np.where(r2 > {grid_r2_thr}, gd_pars, grid_pars)', 
                                    r2=r2_grid_img, gd_pars=gd_par_img, grid_pars=grid_par_img)

            par_img.to_filename(target_dir / f'sub-{subject:02d}_desc-{par}.nii.gz')

    elif model_label == 2:
        print('yo')
        model_dog = DifferenceOfGaussiansPRF2DWithHRF(data=data, paradigm=paradigm, hrf_model=hrf_model,
                                                grid_coordinates=grid_coordinates, flexible_hrf_parameters=False)

        pars_dog_init = gd_pars.copy()
        # This is the relative amplitude of the inhibitory receptive field
        # compared to the excitatory one.
        pars_dog_init['srf_amplitude'] = grid_pars['srf_amplitude'] = 0.1

        # This is the relative size of the inhibitory receptive field
        # compared to the excitatory one.
        pars_dog_init['srf_size'] = grid_pars['srf_size'] = 2.

        # Let's set up a new parameterfitter 
        par_fitter_dog = ParameterFitter(model=model_dog, data=data, paradigm=paradigm)

        # Note how, for now, we are not optimizing the HRF parameters.
        pars_dog = par_fitter_dog.fit(init_pars=pars_dog_init, max_n_iterations=max_n_iterations)

        pars_dog['theta'] = np.arctan2(pars_dog['y'], pars_dog['x'])
        pars_dog['ecc'] = np.sqrt(pars_dog['x']**2 + pars_dog['y']**2)
        pars_dog['r2'] = fitter.get_rsq(pars_dog)

        for par in pars_dog.columns:
            pars_dog_img = r2_masker.inverse_transform(pars_dog[par])
            grid_par_img = brain_masker.inverse_transform(grid_pars[par])
            r2_grid_img = brain_masker.inverse_transform(r2_grid)

            par_img = image.math_img(f'np.where(r2 > {grid_r2_thr}, gd_pars, grid_pars)', 
                                    r2=r2_grid_img, gd_pars=pars_dog_img, grid_pars=grid_par_img)

            par_img.to_filename(target_dir / f'sub-{subject:02d}_desc-{par}.nii.gz')



if __name__ == "__main__":
    # Parse command line arguments
    argument_parser = argparse.ArgumentParser(description="Fit PRF model")
    argument_parser.add_argument('subject', type=int, help='Subject ID')
    argument_parser.add_argument('--model', type=int, default=1)
    argument_parser.add_argument('--bids_folder', default='/data/ds-retsupp', help='BIDS folder path')


    args = argument_parser.parse_args()

    main(args.subject, model_label=args.model, bids_folder=args.bids_folder)

