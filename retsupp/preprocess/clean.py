import argparse
from retsupp.utils.data import Subject
from tqdm import tqdm
from pathlib import Path
from nilearn import image


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

        cleaned_data = image.clean_img(data, confounds=confounds, standardize='psc')   

        target_fn = target_dir / f'sub-{subject:02d}_ses-{session}_task-search_desc-cleaned_run-{run}_bold.nii.gz'

        cleaned_data.to_filename(target_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess RETSUP data.")
    parser.add_argument('subject', type=int, help='Subject ID')
    parser.add_argument('session', type=int, help='Session number')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-retsupp', help='BIDS folder path')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder) 