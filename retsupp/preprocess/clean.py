import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from nilearn import image
from retsupp.utils.data import Subject

def interpolate_outliers(data, confounds):
    """
    Interpolate frames flagged as outliers in the confounds DataFrame.

    Args:
        data: 4D fMRI data (Niimg-like object)
        confounds: DataFrame with motion_outlier_XX columns

    Returns:
        Interpolated 4D fMRI data (Niimg-like object)
        Confounds DataFrame with outlier columns removed
    """
    # Extract outlier columns
    outlier_cols = [col for col in confounds.columns if col.startswith('motion_outlier')]
    if not outlier_cols:
        return data, confounds  # No outliers to interpolate

    # Find indices of outlier frames (where any outlier column is 1)
    outlier_indices = confounds[outlier_cols].any(axis=1)
    outlier_frame_indices = np.where(outlier_indices)[0]

    # Load data and set outliers to NaN
    data_array = data.get_fdata()
    data_array[..., outlier_frame_indices] = np.nan

    # Reshape for interpolation: (voxels, time)
    original_shape = data_array.shape
    voxel_time_series = data_array.reshape(-1, original_shape[-1])

    # Interpolate NaNs using pandas
    df = pd.DataFrame(voxel_time_series).T
    interpolated_df = df.interpolate(axis=0, limit_direction='both')

    # Reshape back and create a new image
    interpolated_data = interpolated_df.T.values.reshape(original_shape)
    interpolated_img = image.new_img_like(data, interpolated_data)

    # Remove outlier columns from confounds
    cleaned_confounds = confounds.drop(columns=outlier_cols)

    return interpolated_img, cleaned_confounds

def main(subject, session, bids_folder='/data/ds-retsupp'):
    sub = Subject(subject, bids_folder)
    runs = sub.get_runs(session)
    print(f"Subject {subject} session {session} runs: {runs}")

    bids_folder = Path(bids_folder)
    target_dir = bids_folder / 'derivatives' / 'cleaned' / f'sub-{subject:02d}' / f'ses-{session}' / 'func'
    target_dir.mkdir(parents=True, exist_ok=True)

    for run in tqdm(runs, desc=f"Processing subject {subject} session {session}"):
        data = sub.get_bold(session, run, type='fmriprep')
        confounds = sub.get_confounds(session, run)

        # Interpolate outliers and clean confounds
        interpolated_data, cleaned_confounds = interpolate_outliers(data, confounds)

        # Clean data with nilearn (standardize, regress confounds)
        cleaned_data = image.clean_img(
            interpolated_data,
            confounds=cleaned_confounds,
            standardize='psc',
        )

        # Save cleaned data
        target_fn = target_dir / f'sub-{subject:02d}_ses-{session}_task-search_desc-cleaned_run-{run}_bold.nii.gz'
        cleaned_data.to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess RETSUP data.")
    parser.add_argument('subject', type=int, help='Subject ID')
    parser.add_argument('session', type=int, help='Session number')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-retsupp', help='BIDS folder path')
    args = parser.parse_args()
    main(args.subject, args.session, args.bids_folder)
