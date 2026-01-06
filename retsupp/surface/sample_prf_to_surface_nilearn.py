import argparse
import os.path as op
from retsupp.utils.data import Subject
from nilearn import surface, image
import nibabel as nb
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform
import numpy as np
from pathlib import Path

def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder):

        subjects_dir = op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer')

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = in_file.replace('fsnative', 'fsaverage')
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = 'fsaverage'
        sxfm.inputs.hemi = fs_hemi

        r = sxfm.run()
        return r

def main(subject, model, bids_folder):

    sub = Subject(subject, bids_folder=bids_folder)
    surfinfo = sub.get_surf_info()

    prf_pars_volume = sub.get_prf_parameters_volume(model=model, type='mean')


    par_keys = prf_pars_volume.index
    prf_pars_volume = image.concat_imgs([prf_pars_volume[k] for k in par_keys])

    bids_folder = Path(bids_folder)

    target_dir = bids_folder / 'derivatives' / 'prf' / f'model{model}' / f'sub-{subject:02d}'
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f'Writing to {target_dir}')

    for hemi in ['L', 'R']:
        samples = surface.vol_to_surf(prf_pars_volume, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
        samples = samples.astype(np.float32)
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        for ix, par in enumerate(par_keys):
            if par in par_keys:
                im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples[:, ix])])
                
                target_fn =  op.join(target_dir, f'sub-{subject:02d}_desc-{par}.optim.nilearn_space-fsnative_hemi-{hemi}.func.gii')

                nb.save(im, target_fn)

                transform_fsaverage(target_fn, fs_hemi, f'sub-{subject:02d}', bids_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int, default=None)
    parser.add_argument('--model', default=1, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-retsupp')
    args = parser.parse_args()

    main(args.subject, args.model, bids_folder=args.bids_folder)
