import argparse
from retsupp.utils.data import Subject
from tqdm import tqdm
from pathlib import Path
from nilearn import image
import numpy as np


def main(subject, bids_folder='/data/ds-retsupp', n_frames=258):

    sub = Subject(subject, bids_folder)

    bids_folder = Path(bids_folder)
    target_dir = bids_folder / 'derivatives' / 'mean_signal' / f'sub-{subject:02d}'
    target_dir.mkdir(parents=True, exist_ok=True)

    im0 = image.index_img(sub.get_bold(1, 1, type='cleaned'), slice(n_frames))
    im0 = image.new_img_like(im0, np.zeros(im0.shape))

    n = 0 
    for session in [1, 2]:
        for run in sub.get_runs(session):
            print(f'Loading session {session} run {run}')
            d = sub.get_bold(session, run, type='cleaned')
            print(d.shape)
            d = image.index_img(d, slice(258))
            # data.append(d.get_fdata())
            # print(d.shape)
            im0 = image.math_img('img + d', img=im0, d=d)
            n += 1

    target_fn = target_dir / f'sub-{subject:02d}_desc-mean_bold.nii.gz'
    im0 = image.math_img(f'img / {n}', img=im0)
    im0.to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess RETSUP data.")
    parser.add_argument('subject', type=int, help='Subject ID')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-retsupp', help='BIDS folder path')
    args = parser.parse_args()

    main(args.subject, args.bids_folder) 