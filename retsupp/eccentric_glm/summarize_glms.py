import pandas as pd
from tqdm import tqdm
import argparse
from retsupp.utils.data import Subject
from pathlib import Path
from nilearn import image
import numpy as np

def main(subject_id, bids_folder='/data/ds-retsupp'):

    target_dir = Path(bids_folder) / 'derivatives' / 'eccentric_glm_summary'
    target_dir.mkdir(parents=True, exist_ok=True)

    # Define parameters
    rois = ['V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1', 'TO2', 'V3A', 'V3B']
    hemis = ['L', 'R']
    quadrant_map = {'L': ['lower_right', 'upper_right'], 'R': ['lower_left', 'upper_left']}
    distractors = [1.0, 3.0, 5.0, 7.0]

    # Initialize
    sub = Subject(subject_id, bids_folder=bids_folder)

    # Build atlas
    atlas = {}
    for roi in rois:
        for hemi in hemis:
            for quadrant in quadrant_map[hemi]:
                roi_key = f'{roi}_{hemi}_{quadrant}'
                atlas[roi_key] = sub.get_eccentric_roi(roi=f'{roi}_{hemi}', quadrant=quadrant, bold_space=True)

    # Process data with tqdm
    df = []
    for session, run, distractor in tqdm(
        [(s, r, d) for s in [1, 2] for r in sub.get_runs(s) for d in distractors],
        desc="Processing sessions/runs/distractors",
        total=len([1, 2]) * len(sub.get_runs(1)) * len(distractors)  # Adjust if runs vary by session
    ):
        beta_path = (
            sub.bids_folder / 'derivatives' / 'eccentric_glm' /
            f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'func' /
            f'sub-{subject_id:02d}_ses-{session:02d}_run-{run:02d}_desc-distractor_{distractor:.1f}_beta.nii.gz'
        )

        zmap_path = beta_path.with_name(beta_path.name.replace('beta', 'zmap'))

        beta = image.load_img(beta_path)
        zmap = image.load_img(zmap_path)

        for roi_key in atlas.keys():
            try:
                mean_beta = atlas[roi_key].fit_transform(beta).mean()
                mean_z = atlas[roi_key].fit_transform(zmap).mean()
            except ValueError:
                mean_beta = np.nan
            df.append({
                'subject_id': subject_id,
                'session': session,
                'run': run,
                'distractor': distractor,
                'roi': roi_key,
                'mean_beta': mean_beta,
                'mean_z': mean_z
            })

    # Convert to DataFrame
    df = pd.DataFrame(df)
    target_fn = target_dir / f'sub-{subject_id:02d}_eccentric_glm_summary.tsv'
    df.to_csv(target_fn, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize eccentric GLM results.')
    parser.add_argument('subject', type=int, help='Subject ID')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-retsupp', help='Path to BIDS folder')
    args = parser.parse_args()

    main(subject_id=args.subject, bids_folder=args.bids_folder)