import argparse
from pathlib import Path
from nilearn import input_data, image
import numpy as np
import pandas as pd
from retsupp.utils.data import Subject
from braincoder.models import GaussianPRF2DWithHRF, DifferenceOfGaussiansPRF2DWithHRF
from braincoder.optimize import ParameterFitter
from braincoder.hrf import SPMHRFModel
from tqdm.contrib.itertools import product

def main(subject, model_label=1, bids_folder='/data/ds-retsupp', max_n_iterations=2000, debug=False, resolution=50, r2_thr=0.04):
    print(f"Fitting PRF model for subject {subject} across all sessions by condition.")
    bids_folder = Path(bids_folder)
    target_dir = bids_folder / 'derivatives' / 'prf_conditionfit' / f'model{model_label}' / f'sub-{subject:02d}'
    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder)

    # Load cleaned BOLD file for all sessions and runs and build DataFrame
    bold_mask = sub.get_bold_mask()
    brain_masker = input_data.NiftiMasker(mask_img=bold_mask)
    all_dfs = []

    print("Loading and masking BOLD data...")

    for session, run in product([1, 2], range(1, 7)):
        bold_fn = bids_folder / 'derivatives' / 'cleaned' / f'sub-{subject:02d}' / f'ses-{session}' / 'func' / f'sub-{subject:02d}_ses-{session}_task-search_desc-cleaned_run-{run}_bold.nii.gz'
        data = brain_masker.fit_transform(bold_fn)
        if data.shape[0] != 258:
            print(f"WARNING: Data for session {session} run {run} has {data.shape[0]} timepoints, expected 258. Cropping.")
            data = data[:258]
        df = pd.DataFrame(data)
        df['session'] = session
        df['run'] = run
        df['timepoint'] = np.arange(df.shape[0])
        all_dfs.append(df)

    data_df = pd.concat(all_dfs, ignore_index=True)

    # Get distractor mapping for this subject
    distractor_map = sub.get_distractor_mapping()

    # Map condition for each (session, run)
    data_df['condition'] = data_df.apply(lambda row: distractor_map[(row['session'], row['run'])], axis=1)

    # Set multi-index (session, run, timepoint)
    data_df.set_index(['session', 'run', 'timepoint'], inplace=True)

    print(data_df)

    # Get stimulus and grid coordinates (use first session/run for simplicity)
    paradigm = sub.get_stimulus(session=1, run=1, resolution=resolution).astype(np.float32)
    grid_coordinates = sub.get_grid_coordinates(resolution=resolution, session=1, run=1)
    grid_coordinates = np.stack((grid_coordinates[0].ravel(), grid_coordinates[1].ravel()), axis=1).astype(np.float32)
    paradigm = paradigm.reshape((paradigm.shape[0], -1))

    # Get subject's PRF parameters as initial values
    prf_pars = sub.get_prf_parameters_volume(model=model_label, return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)

    print(prf_pars)
    mask_r2 = prf_pars['r2'] > r2_thr
    if mask_r2.sum() == 0:
        print(f"No voxels above r2 threshold {r2_thr} for condition {cond}.")
        raise Exception("No voxels above r2 threshold.")

    # Group by condition and fit PRF model to mean time series
    for cond, cond_df in data_df.groupby(['condition']):
        print(f'Fitting condition {cond}...')

        mean_ts = cond_df.drop(columns=['condition']).groupby('timepoint').mean()  # mean time series for voxels
        print(mean_ts)
        print(mask_r2)

        mean_ts = mean_ts.loc[:,mask_r2]
        init_pars = prf_pars.loc[mask_r2]

        hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=4.5, dispersion=0.75)

        if model_label == 1:
            prf_model = GaussianPRF2DWithHRF(grid_coordinates, paradigm, hrf_model=hrf_model)
        elif model_label == 2:
            prf_model = DifferenceOfGaussiansPRF2DWithHRF(grid_coordinates, paradigm, hrf_model=hrf_model)
        elif model_label == 3:
            prf_model = GaussianPRF2DWithHRF(grid_coordinates, paradigm, hrf_model=hrf_model, flexible_hrf_parameters=True)
        elif model_label == 4:
            prf_model = DifferenceOfGaussiansPRF2DWithHRF(grid_coordinates, paradigm, hrf_model=hrf_model, flexible_hrf_parameters=True)
        else:
            raise ValueError(f'Unknown model label: {model_label}')

        fitter = ParameterFitter(prf_model, mean_ts, paradigm)
        refined_pars = fitter.refine_baseline_and_amplitude(init_pars, l2_alpha=0.001)
        fit_pars = fitter.fit(init_pars=refined_pars, max_n_iterations=max_n_iterations, learning_rate=0.005)
        r2 = fitter.get_rsq(fit_pars)
        pred = fitter.predictions

        fit_pars['theta'] = np.arctan2(fit_pars['y'], fit_pars['x'])
        fit_pars['ecc'] = np.sqrt(fit_pars['x']**2 + fit_pars['y']**2)
        fit_pars['r2'] = r2

        # Save parameter maps for this condition
        n_voxels = brain_masker.mask_img_.get_fdata().sum().astype(int)
        for par in fit_pars.columns:
            arr = np.zeros(n_voxels)
            arr[mask_r2.values] = fit_pars[par].values
            par_img = brain_masker.inverse_transform(arr)
            par_img.to_filename(target_dir / f'sub-{subject:02d}_cond-{cond}_desc-{par}.nii.gz')

        # Save prediction (only fitted voxels, others set to 0)
        pred_full = np.zeros((pred.shape[0], n_voxels))
        pred_full[:, mask_r2.values] = pred
        pred_img = brain_masker.inverse_transform(pred_full)
        pred_img.to_filename(target_dir / f'sub-{subject:02d}_cond-{cond}_desc-pred.nii.gz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit PRF model to mean time series for each condition across all sessions.")
    parser.add_argument('subject', type=int, help='Subject ID')
    parser.add_argument('--model', type=int, default=1, help='Model label (1=Gaussian, 2=DoG, 3=Gaussian+HRF, 4=DoG+HRF)')
    parser.add_argument('--bids_folder', default='/data/ds-retsupp', help='BIDS folder path')
    parser.add_argument('--max_n_iterations', default=4000, type=int, help='Max number of iterations')
    parser.add_argument('--resolution', default=50, type=int, help='Stimulus resolution')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--r2_thr', default=0.04, type=float, help='R2 threshold for output masking')
    args = parser.parse_args()
    main(args.subject, model_label=args.model, bids_folder=args.bids_folder, max_n_iterations=args.max_n_iterations, debug=args.debug, resolution=args.resolution, r2_thr=args.r2_thr)
