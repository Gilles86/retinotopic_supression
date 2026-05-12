import pandas as pd
from tqdm import tqdm
import argparse
from retsupp.utils.data import Subject
from pathlib import Path
from nilearn import image
import numpy as np
from joblib import Parallel, delayed

def process_roi(roi_key, beta_img, zmap_img, zmap_all_distractors_img, atlas):
    try:
        beta = atlas[roi_key].fit_transform(beta_img)
        mean_beta = beta.mean()
        mean_z = atlas[roi_key].fit_transform(zmap_img).mean()
        zmap_all_distractors_roi = atlas[roi_key].fit_transform(zmap_all_distractors_img)
        mean_beta_filtered = beta[zmap_all_distractors_roi > 2.3].mean() if len(beta[zmap_all_distractors_roi > 2.3]) > 0 else np.nan
    except ValueError:
        mean_beta, mean_z, mean_beta_filtered = np.nan, np.nan, np.nan
    return {
        'roi': roi_key,
        'mean_beta': mean_beta,
        'mean_beta_filtered': mean_beta_filtered,
        'mean_z': mean_z
    }

def main(subject_id, bids_folder='/data/ds-retsupp'):
    target_dir = Path(bids_folder) / 'derivatives' / 'eccentric_glm_summary'
    target_dir.mkdir(parents=True, exist_ok=True)

    rois = ['V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1', 'TO2', 'V3A', 'V3B']
    hemis = ['L', 'R']
    quadrant_map = {'L': ['lower_right', 'upper_right'], 'R': ['lower_left', 'upper_left']}
    distractors = [1.0, 3.0, 5.0, 7.0]

    sub = Subject(subject_id, bids_folder=bids_folder)
    atlas = {
        f'{roi}_{hemi}_{quadrant}': sub.get_eccentric_roi(roi=f'{roi}_{hemi}', quadrant=quadrant, bold_space=True)
        for roi in rois for hemi in hemis for quadrant in quadrant_map[hemi]
    }

    zmap_all_distractors = (
        sub.bids_folder / 'derivatives' / 'eccentric_glm' /
        f'sub-{subject_id:02d}' / 'func' /
        f'sub-{subject_id:02d}_desc-all_distractors_zmap.nii.gz'
    )
    zmap_all_distractors_img = image.load_img(zmap_all_distractors)

    # Preload images with tqdm
    beta_images = {}
    zmap_images = {}
    for session in [1, 2]:
        for run in tqdm(sub.get_runs(session), desc=f"Preloading session {session} runs", leave=False):
            for distractor in tqdm(distractors, desc="Preloading distractors", leave=False):
                beta_path = (
                    sub.bids_folder / 'derivatives' / 'eccentric_glm' /
                    f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'func' /
                    f'sub-{subject_id:02d}_ses-{session:02d}_run-{run:02d}_desc-distractor_{distractor:.1f}_beta.nii.gz'
                )
                zmap_path = beta_path.with_name(beta_path.name.replace('beta', 'zmap'))
                beta_images[(session, run, distractor)] = image.load_img(beta_path)
                zmap_images[(session, run, distractor)] = image.load_img(zmap_path)

    results = []
    for session, run, distractor in tqdm(
        [(s, r, d) for s in [1, 2] for r in sub.get_runs(s) for d in distractors],
        desc="Processing sessions/runs/distractors",
        total=len([1, 2]) * len(sub.get_runs(1)) * len(distractors)
    ):
        beta_img = beta_images[(session, run, distractor)]
        zmap_img = zmap_images[(session, run, distractor)]
        roi_results = Parallel(n_jobs=-1)(
            delayed(process_roi)(roi_key, beta_img, zmap_img, zmap_all_distractors_img, atlas)
            for roi_key in atlas.keys()
        )
        for roi_result in roi_results:
            results.append((
                subject_id, session, run, distractor,
                roi_result['roi'], roi_result['mean_beta'],
                roi_result['mean_beta_filtered'], roi_result['mean_z']
            ))

    df = pd.DataFrame(results, columns=['subject_id', 'session', 'run', 'distractor', 'roi', 'mean_beta', 'mean_beta_filtered', 'mean_z'])
    target_fn = target_dir / f'sub-{subject_id:02d}_eccentric_glm_summary.tsv'
    df.to_csv(target_fn, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize eccentric GLM results.')
    parser.add_argument('subject', type=int, help='Subject ID')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-retsupp', help='Path to BIDS folder')
    args = parser.parse_args()
    main(subject_id=args.subject, bids_folder=args.bids_folder)
