import numpy as np
import argparse
from cortex import freesurfer
from cortex.xfm import Transform
from nitransforms.linear import Affine
import os.path as op


def main(subject, bids_folder):


    freesurfer.import_subj(f'sub-{subject}', 
            pycortex_subject=f'retsupp.sub-{subject}',
            freesurfer_subject_dir=op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer'))

    if int(subject) < 3:
        t1w = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat',
                f'sub-{subject}_desc-preproc_T1w.nii.gz')

        fsnative2t1w = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat',
                f'sub-{subject}_from-fsnative_to-T1w_mode-image_xfm.txt')

        epi = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'ses-1', 'func',
                f'sub-{subject}_ses-1_task-search_rec-NORDIC_run-1_space-T1w_boldref.nii.gz')

        fsnative2t1w = Affine.from_filename(fsnative2t1w, fmt='itk',
                reference=t1w)

        fsnative2t1w.to_filename(op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat',
                f'sub-{subject}_from-fsnative_to-T1w_mode-image_xfm.fsl'),
                fmt='fsl')

        pycortex_transform = Transform.from_fsl(op.join(bids_folder, 'derivatives',
                'fmriprep',
                f'sub-{subject}', 'anat',
                f'sub-{subject}_from-fsnative_to-T1w_mode-image_xfm.fsl'),
                epi, t1w)
        
    else:
        t1w = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'ses-1', 'anat',
                      f'sub-{subject}_ses-1_desc-preproc_T1w.nii.gz')

        fsnative2t1w = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'ses-1', 'anat',
                                   f'sub-{subject}_ses-1_from-fsnative_to-T1w_mode-image_xfm.txt')
        
        epi = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'ses-1', 'func',
                            f'sub-{subject}_ses-1_task-search_rec-NORDIC_run-1_space-T1w_boldref.nii.gz')
        
        fsnative2t1w = Affine.from_filename(fsnative2t1w, fmt='itk',
                                                reference=t1w)
        fsnative2t1w.to_filename(op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'ses-1', 'anat',
                                                f'sub-{subject}_ses-1_from-fsnative_to-T1w_mode-image_xfm.fsl'),
                                                fmt='fsl')

        pycortex_transform = Transform.from_fsl(op.join(bids_folder, 'derivatives',
                'fmriprep',
                f'sub-{subject}', 'ses-1', 'anat',
                f'sub-{subject}_ses-1_from-fsnative_to-T1w_mode-image_xfm.fsl'),
                epi, t1w)


    pycortex_transform.save(f'retsupp.sub-{subject}', 'epi', xfmtype='coord')

    identity_transform = Transform(np.identity(4), epi).save(f'retsupp.sub-{subject}', 'epi.identity')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('--bids_folder', default='/data/ds-retsupp')
    args = parser.parse_args()
    main(args.subject, args.bids_folder)
