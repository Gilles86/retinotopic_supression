"""Cross-validated comparison of bar-only vs extended-design PRF fits.

For one subject, hold out one run, fit both PRF models on the remaining
runs, and score per-voxel R^2 on the held-out run.  Outputs a single TSV
with one row per voxel and columns:

    subject, session_holdout, run_holdout, voxel,
    r2_bar_train, r2_bar_test,
    r2_ext_train, r2_ext_test,
    delta_r2_test,                # r2_ext_test - r2_bar_test
    x_bar, y_bar, ecc_bar, sd_bar,
    x_ext, y_ext, ecc_ext, sd_ext

The expected pattern: voxels with PRF centers near a ring (ecc ~ 3-4°)
should have ``delta_r2_test > 0`` (extended model wins), foveal voxels
should have ``delta_r2_test ~ 0`` or slightly negative (bar-only wins
because the extended design just adds noise to the fovea).

Outputs go to ``derivatives/prf_extended_cv/sub-XX/`` so they can be
pooled across subjects without overwriting.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, maskers

from braincoder.models import GaussianPRF2DWithHRF
from braincoder.optimize import ParameterFitter
from braincoder.hrf import SPMHRFModel

from retsupp.utils.data import Subject


def _build_paradigm_bar(sub, sessions_runs, resolution):
    """Concatenate bar-only stimulus across the given (session, run) pairs."""
    paradigms = []
    for ses, run in sessions_runs:
        s = sub.get_stimulus(session=ses, run=run, resolution=resolution)
        paradigms.append(s.astype(np.float32))
    return np.concatenate(paradigms, axis=0)


def _build_paradigm_extended(sub, sessions_runs, resolution, grid_radius,
                             distractor_radius, max_distractor_duration):
    paradigms = []
    for ses, run in sessions_runs:
        s = sub.get_stimulus_with_distractors(
            session=ses, run=run, resolution=resolution,
            grid_radius=grid_radius,
            distractor_radius=distractor_radius,
            max_distractor_duration=max_distractor_duration,
        )
        paradigms.append(s.astype(np.float32))
    return np.concatenate(paradigms, axis=0)


def _load_concatenated_data(sub, sessions_runs, brain_masker, n_volumes=258):
    """Load and concatenate cleaned BOLD across (ses, run) pairs."""
    arrays = []
    for ses, run in sessions_runs:
        bold_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                   / f'sub-{sub.subject_id:02d}' / f'ses-{ses}' / 'func'
                   / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                     f'desc-cleaned_run-{run}_bold.nii.gz')
        d = brain_masker.transform(str(bold_fn))
        if d.shape[0] != n_volumes:
            print(f"WARNING: ses-{ses} run-{run} has {d.shape[0]} TRs; "
                  f"cropping to {n_volumes}.")
            d = d[:n_volumes]
        arrays.append(d.astype(np.float32))
    return np.concatenate(arrays, axis=0)


def _fit_one(grid_coordinates_2d, paradigm_train, paradigm_test,
             data_train, data_test, grid_radius, max_n_iterations):
    """Return train / test R^2 plus fitted parameters."""
    grid_coordinates = np.stack(
        (grid_coordinates_2d[0].ravel(), grid_coordinates_2d[1].ravel()),
        axis=1,
    ).astype(np.float32)
    para_train = paradigm_train.reshape((paradigm_train.shape[0], -1))
    para_test = paradigm_test.reshape((paradigm_test.shape[0], -1))

    hrf_model = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
    train_model = GaussianPRF2DWithHRF(
        grid_coordinates, para_train, hrf_model=hrf_model,
    )

    grid_x = np.linspace(-grid_radius, grid_radius, 14)
    grid_y = np.linspace(-grid_radius, grid_radius, 14)
    grid_sd = np.linspace(0.5, 4.0, 8)
    grid_amplitude = [1.0]
    grid_baseline = [0.0]

    fitter = ParameterFitter(train_model, data_train, para_train)
    grid_pars = fitter.fit_grid(
        grid_x, grid_y, grid_sd, grid_baseline, grid_amplitude,
        use_correlation_cost=True,
    )
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars, l2_alpha=0.001)
    fit_pars = fitter.fit(
        init_pars=grid_pars,
        max_n_iterations=max_n_iterations, learning_rate=0.005,
    )
    r2_train = fitter.get_rsq(fit_pars).values

    # Score on held-out paradigm.
    test_model = GaussianPRF2DWithHRF(
        grid_coordinates, para_test, hrf_model=hrf_model,
    )
    test_fitter = ParameterFitter(test_model, data_test, para_test)
    r2_test = test_fitter.get_rsq(fit_pars).values

    fit_pars = fit_pars.copy()
    fit_pars['ecc'] = np.sqrt(fit_pars['x'] ** 2 + fit_pars['y'] ** 2)
    fit_pars['theta'] = np.arctan2(fit_pars['y'], fit_pars['x'])
    return fit_pars, r2_train, r2_test


def main(subject, holdout_session, holdout_run,
         bids_folder='/data/ds-retsupp', resolution=60,
         grid_radius=5.0, distractor_radius=0.4,
         max_distractor_duration=1.5,
         max_n_iterations=2000, r2_thr=0.04, debug=False):

    bids_folder = Path(bids_folder)
    target_dir = (bids_folder / 'derivatives' / 'prf_extended_cv'
                  / f'sub-{subject:02d}')
    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder)

    # Build train / test (session, run) lists.
    train, test = [], []
    for ses in [1, 2]:
        for run in sub.get_runs(session=ses):
            if (ses, run) == (holdout_session, holdout_run):
                test.append((ses, run))
            else:
                train.append((ses, run))
    if not test:
        raise ValueError(
            f"holdout (ses={holdout_session}, run={holdout_run}) is not in "
            f"this subject's run list."
        )
    print(f"Train pairs ({len(train)}): {train}")
    print(f"Test pair: {test}")

    # Mask & data for ALL runs we'll touch (shared brain mask).
    bold_mask = sub.get_bold_mask()
    if debug:
        bold_mask = image.math_img(
            'np.where(mask.astype(bool) & (np.random.rand(*mask.shape) < 0.01), 1, 0)',
            mask=bold_mask,
        )
        max_n_iterations = 80
        resolution = 30
    brain_masker = maskers.NiftiMasker(mask_img=bold_mask)
    brain_masker.fit()

    print("Loading BOLD time series ...")
    data_train = _load_concatenated_data(sub, train, brain_masker)
    data_test = _load_concatenated_data(sub, test, brain_masker)

    # Pre-screen voxels by an R^2 threshold using the bar paradigm grid
    # fit (cheap) so we don't burn GPU on noise voxels.
    print("Building paradigms ...")
    para_train_bar = _build_paradigm_bar(sub, train, resolution)
    para_test_bar = _build_paradigm_bar(sub, test, resolution)
    para_train_ext = _build_paradigm_extended(
        sub, train, resolution=resolution, grid_radius=grid_radius,
        distractor_radius=distractor_radius,
        max_distractor_duration=max_distractor_duration,
    )
    para_test_ext = _build_paradigm_extended(
        sub, test, resolution=resolution, grid_radius=grid_radius,
        distractor_radius=distractor_radius,
        max_distractor_duration=max_distractor_duration,
    )

    grid_coords_bar = sub.get_grid_coordinates(
        resolution=resolution, session=train[0][0], run=train[0][1],
    )
    grid_coords_ext = sub.get_extended_grid_coordinates(
        resolution=resolution, session=train[0][0], run=train[0][1],
        grid_radius=grid_radius,
    )

    # --- Cheap pre-screen: bar-only grid fit, keep voxels above r2_thr.
    grid_coordinates_bar = np.stack(
        (grid_coords_bar[0].ravel(), grid_coords_bar[1].ravel()), axis=1,
    ).astype(np.float32)
    para_screen = para_train_bar.reshape((para_train_bar.shape[0], -1))

    hrf_model = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
    screen_model = GaussianPRF2DWithHRF(
        grid_coordinates_bar, para_screen, hrf_model=hrf_model,
    )
    screen_fitter = ParameterFitter(screen_model, data_train, para_screen)
    screen_grid = screen_fitter.fit_grid(
        np.linspace(-3, 3, 12), np.linspace(-3, 3, 12),
        np.linspace(1.0, 4.0, 8), [0], [1],
        use_correlation_cost=True,
    )
    screen_grid = screen_fitter.refine_baseline_and_amplitude(
        screen_grid, l2_alpha=0.001,
    )
    screen_r2 = screen_fitter.get_rsq(screen_grid).values
    keep = screen_r2 > r2_thr
    print(f"Voxels passing pre-screen r2>{r2_thr}: {keep.sum()} / {len(keep)} "
          f"({keep.mean()*100:.2f}%)")

    data_train_kept = data_train[:, keep]
    data_test_kept = data_test[:, keep]

    # --- Bar-only fit on kept voxels. ---
    print("Fitting BAR-ONLY model ...")
    bar_pars, bar_r2_train, bar_r2_test = _fit_one(
        grid_coords_bar, para_train_bar, para_test_bar,
        data_train_kept, data_test_kept,
        grid_radius=3.5, max_n_iterations=max_n_iterations,
    )

    # --- Extended fit on kept voxels. ---
    print("Fitting EXTENDED-DESIGN model ...")
    ext_pars, ext_r2_train, ext_r2_test = _fit_one(
        grid_coords_ext, para_train_ext, para_test_ext,
        data_train_kept, data_test_kept,
        grid_radius=grid_radius, max_n_iterations=max_n_iterations,
    )

    # --- Build output table. ---
    voxel_ids = np.where(keep)[0]
    out = pd.DataFrame({
        'subject': subject,
        'session_holdout': holdout_session,
        'run_holdout': holdout_run,
        'voxel': voxel_ids,
        'r2_bar_train': bar_r2_train,
        'r2_bar_test': bar_r2_test,
        'r2_ext_train': ext_r2_train,
        'r2_ext_test': ext_r2_test,
        'delta_r2_test': ext_r2_test - bar_r2_test,
        'x_bar': bar_pars['x'].values,
        'y_bar': bar_pars['y'].values,
        'sd_bar': bar_pars['sd'].values,
        'ecc_bar': bar_pars['ecc'].values,
        'x_ext': ext_pars['x'].values,
        'y_ext': ext_pars['y'].values,
        'sd_ext': ext_pars['sd'].values,
        'ecc_ext': ext_pars['ecc'].values,
    })

    fn = (target_dir
          / f'sub-{subject:02d}_ses-{holdout_session}_run-{holdout_run}_'
            f'cv_compare.tsv')
    out.to_csv(fn, sep='\t', index=False)
    print(f"Wrote {fn}")
    print(out[['delta_r2_test', 'r2_bar_test', 'r2_ext_test']].describe())


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Cross-validated PRF comparison: bar-only vs extended"
    )
    p.add_argument('subject', type=int)
    p.add_argument('--holdout_session', type=int, default=2)
    p.add_argument('--holdout_run', type=int, default=6)
    p.add_argument('--bids_folder', default='/data/ds-retsupp')
    p.add_argument('--resolution', type=int, default=60)
    p.add_argument('--grid_radius', type=float, default=5.0)
    p.add_argument('--distractor_radius', type=float, default=0.4)
    p.add_argument('--max_distractor_duration', type=float, default=1.5)
    p.add_argument('--max_n_iterations', type=int, default=2000)
    p.add_argument('--r2_thr', type=float, default=0.04)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()
    main(
        args.subject, args.holdout_session, args.holdout_run,
        bids_folder=args.bids_folder, resolution=args.resolution,
        grid_radius=args.grid_radius,
        distractor_radius=args.distractor_radius,
        max_distractor_duration=args.max_distractor_duration,
        max_n_iterations=args.max_n_iterations, r2_thr=args.r2_thr,
        debug=args.debug,
    )
