import argparse
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np
from itertools import product
from tqdm import tqdm
from pathlib import Path
from retsupp.utils.data import Subject
from nilearn import image


def main(subject_id, bids_folder):
    sub = Subject(f'{subject_id:02d}', bids_folder=bids_folder)

    design_matrices = []
    data = []

    TR = 1.6
    frame_times = np.arange(TR/2., TR * 258, TR)

    mask = sub.get_bold_mask()
    fmri_glm = FirstLevelModel(minimize_memory=True, mask_img=mask)

    session_runs = [(session, run) for session in [1, 2] for run in sub.get_runs(session)]

    for session, run in tqdm(session_runs):

        output_dir = bids_folder / 'derivatives' / 'eccentric_glm' / f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'func'
        output_dir.mkdir(parents=True, exist_ok=True)

        onsets = sub.get_onsets(session, run)
        onsets = onsets.query('event_type == "target"')

        if 'HPL_distractor' not in onsets.columns:

            hpl_distractor = sub.get_hpd_locations()[session, run]
            hpl_distractor = {v: k for k, v in sub.location_mapping.items()}[hpl_distractor]
            onsets['HPL_distractor'] = hpl_distractor

        # onsets['trial_type'] = onsets.apply(lambda row: f'distractor_{row.distractor_location}_hpl_{row.HPL_distractor}', axis=1)
        onsets['trial_type'] = onsets.apply(lambda row: f'distractor_{row.distractor_location}', axis=1)
        onsets = onsets[['onset', 'duration', 'trial_type']]

        dm = make_first_level_design_matrix(frame_times, [onsets], drift_model=None)

        bold = sub.get_bold(session=session, run=run, type='prf_regressed_out')

        bold = image.index_img(bold, slice(258))
        r = fmri_glm.fit(bold, design_matrices=dm)

        for distractor in [1.0, 3.0, 5.0, 7.0, 10.0]:
            contrast_name = f'distractor_{distractor}'
            beta = r.compute_contrast(contrast_name, output_type='effect_size')
            zmap = r.compute_contrast(contrast_name, output_type='z_score')

            beta.to_filename(output_dir / f'sub-{subject_id:02d}_ses-{session:02d}_run-{run:02d}_desc-{contrast_name}_beta.nii.gz')
            zmap.to_filename(output_dir / f'sub-{subject_id:02d}_ses-{session:02d}_run-{run:02d}_desc-{contrast_name}_zmap.nii.gz')

        data.append(bold)
        design_matrices.append(dm)


    results_all = fmri_glm.fit(data, design_matrices=design_matrices)

    output_dir = bids_folder / 'derivatives' / 'eccentric_glm' / f'sub-{subject_id:02d}'  / 'func'
    output_dir.mkdir(parents=True, exist_ok=True)

    for distractor in [1.0, 3.0, 5.0, 7.0, 10.0]:
        contrast_name = f'distractor_{distractor}'
        beta = results_all.compute_contrast(contrast_name, output_type='effect_size')
        zmap = results_all.compute_contrast(contrast_name, output_type='z_score')

        beta.to_filename(output_dir / f'sub-{subject_id:02d}_desc-{contrast_name}_beta.nii.gz')
        zmap.to_filename(output_dir / f'sub-{subject_id:02d}_desc-{contrast_name}_zmap.nii.gz')


    zmap_all = results_all.compute_contrast([[1, 1, 1, 1]] * 12, output_type='z_score')
    zmap_all.to_filename(output_dir / f'sub-{subject_id:02d}_desc-all_distractors_zmap.nii.gz')

    # Save contrasts for all runs/sessions combined
    zmap_all.to_filename(output_dir / f'sub-{subject_id:02d}_desc-all_distractors_zmap.nii.gz')

    for i, distractor in enumerate([1.0, 3.0, 5.0, 7.0]):
        contrast_name = f'distractor_{distractor}'
        beta = results_all.compute_contrast(contrast_name, output_type='z_score')
        beta.to_filename(output_dir / f'sub-{subject_id:02d}_desc-{contrast_name}_zmap.nii.gz')

        contrast_vs = [[1 if j == i else -1/3. for j in range(4)] for _ in range(12)]
        zmap = results_all.compute_contrast(contrast_vs, output_type='z_score')
        zmap.to_filename(output_dir / f'sub-{subject_id:02d}_desc-{contrast_name}_vs_rest_zmap.nii.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit GLM to eccentricity task data.')
    parser.add_argument('subject_id', type=int, help='Subject ID to process.')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-retsupp', help='Path to BIDS folder.')
    args = parser.parse_args()

    main(args.subject_id, Path(args.bids_folder))