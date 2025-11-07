import numpy as np
import pandas as pd
from nilearn import image
import argparse
from retsupp.utils.data import Subject
from tqdm.contrib.itertools import product
from pathlib import Path

def main(subject, mean_prf=True, bids_folder='/data/ds-retsupp'):

    sub = Subject(subject, bids_folder=bids_folder)
    sessions = [1, 2]
    runs = list(range(1, 7))

    for session, run in product(sessions, runs):
        target_dir = Path(bids_folder) / 'derivatives' / 'prf_regressed_out' / f'sub-{subject:02d}' / f'ses-{session}' / 'func'
        target_dir.mkdir(parents=True, exist_ok=True)

        # instantiate Subject for this session/run

        # Use the same masker for both bold and PRF predictions so voxels align
        bold = sub.get_bold(session=session, run=run, type='cleaned')
        bold = image.index_img(bold, slice(258))  # limit to first 258 frames

        masker = sub.get_bold_mask(return_masker=True)

        # Extract time series: shape (T, V)
        bold_ts = masker.fit_transform(bold)

        # Get PRF predictions as a 4D nifti and extract using same masker
        if mean_prf:
            prf_img = sub.get_prf_predictions(model=4, type='mean', return_image=True)
        else:
            prf_img = sub.get_prf_predictions(model=4, type='run', session=session, run=run, return_image=True)

        prf_ts = masker.transform(prf_img)  # shape (T, V) or (T, K)

        # Normalize shapes
        if prf_ts.ndim == 1:
            prf_ts = prf_ts[:, None]

        if prf_ts.shape[0] != bold_ts.shape[0]:
            raise ValueError(f"Time dimension mismatch: PRF {prf_ts.shape[0]} vs BOLD {bold_ts.shape[0]}")

        T, V = bold_ts.shape

        # Case A: prf_ts has same number of columns as voxels -> a single voxel-wise predictor per voxel (T, V)
        if prf_ts.shape == bold_ts.shape:
            x = prf_ts  # (T, V)
            y = bold_ts  # (T, V)

            # vectorized simple linear regression (one predictor + intercept) per voxel
            x_mean = x.mean(axis=0)
            y_mean = y.mean(axis=0)

            num = ((x - x_mean) * (y - y_mean)).sum(axis=0)   # (V,)
            den = ((x - x_mean) ** 2).sum(axis=0)            # (V,)
            beta = num / (den + 1e-12)
            intercept = y_mean - beta * x_mean

            predicted = x * beta[np.newaxis, :] + intercept[np.newaxis, :]
            cleaned_ts = y - predicted

        else:
            # Case B: prf_ts has K regressors (T, K) shared across voxels
            # Build design matrix A: (T, K+1) with intercept
            A = np.concatenate([prf_ts, np.ones((T, 1))], axis=1)  # (T, K+1)

            # Solve least squares for all voxels at once: A @ coeffs = bold_ts
            coeffs, *_ = np.linalg.lstsq(A, bold_ts, rcond=None)  # coeffs: (K+1, V)
            predicted = A @ coeffs  # (T, V)
            cleaned_ts = bold_ts - predicted

        # Reconstruct 4D image from cleaned time series and save
        out_img = masker.inverse_transform(cleaned_ts)  # nifti image in original space
        out_fname = target_dir / f"sub-{subject:02d}_ses-{session}_run-{run}_task-prf_cleaned_regressed.nii.gz"
        out_img.to_filename(out_fname)
        print(f"Saved regressed image to {out_fname}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int, default=None)
    parser.add_argument('--bids_folder', default='/data/ds-retsupp')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)