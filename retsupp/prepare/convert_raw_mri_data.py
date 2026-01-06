import argparse
from pathlib import Path
import nibabel as nib
from nilearn import image
import re
import json
from tqdm import tqdm
from utils import create_preprocessed_t1w
import shutil

default_json = {
  "Manufacturer": "Philips",
  "ManufacturersModelName": "Ingenia",
  "MagneticFieldStrength": 7,
  "ReceiveCoilName": ["SENSE-Head-7T-Ant", "SENSE-Head-7T-Post"],
  "FlipAngle": 66,
  "EchoTime": 0.024,
  "RepetitionTime": 1.6,
  "SliceThickness": 1.7,
  "VoxelSize": [1.68, 1.68, 1.7],
  "FieldOfView": [215, 215, 137.7],
  "MultibandAccelerationFactor": 3,
  "ParallelReductionFactorInPlane": 2.4,
  "FatSuppressionTechnique": "SPIR",
  "MRAcquisitionType": "3D",
  "PulseSequenceType": "EPI",
  "PartialFourier": 0.796610177,
  "PhaseEncodingDirection":"j",
  "TotalScanDuration": "07:05.6",
  "TotalReadoutTime": 0.027578312,
  'RepetitionTime':1.6
}

def main(subject, session, bids_dir):
    """
    Convert raw MRI data to NIfTI format and organize it according to BIDS specifications.

    Parameters:
    subject (int): Subject ID (e.g., 01)
    session (int): Session ID (e.g., 01)
    bids_dir (str): Path to BIDS directory
    """
    # Placeholder for the conversion logic
    print(f"Converting raw MRI data for subject {subject}, session {session}...")
    print(f"Saving to BIDS directory: {bids_dir}")

    bids_dir = Path(bids_dir)
    sourcedata = bids_dir / 'sourcedata'

    # source_dir = sourcedata / 'mri' / f'sub-{subject:02d}' / f'ses-{session}'
    
    sourde_dir_candidates = [sourcedata / 'mri' / f'sub-{subject:02d}_ses{session}',
                             sourcedata / 'mri' / f'sub-{subject:02d}_ses-{session}']
    
    source_dir = None

    for candidate in sourde_dir_candidates:
        if candidate.exists():
            source_dir = candidate
            break

    if source_dir is None:
        raise FileNotFoundError(f'No source directory found for subject {subject}, session {session} in {sourcedata}')

    target_dir = bids_dir / f'sub-{subject:02d}' / f'ses-{session}'
    target_dir.mkdir(parents=True, exist_ok=True)

    # # Anatomical data
    print('Converting anatomical data...')
    (target_dir / 'anat').mkdir(parents=True, exist_ok=True)

    mp2rage_candidates = list(source_dir.glob('*MP2_085mm*_1.PAR'))
    print(f'Found {len(mp2rage_candidates)} T1w candidates')

    if len(mp2rage_candidates) == 1:
        mp2rage = list(mp2rage_candidates)[0]
        print('Storing raw MP2RAGE data...')

        # # Convert to NIfTI
        mp2rage = nib.load(mp2rage)
        mp2rage_nii = nib.Nifti1Image(mp2rage.dataobj, mp2rage.affine)
        mp2rage_uni1 = image.index_img(mp2rage_nii, 0)
        mp2rage_uni2 = image.index_img(mp2rage_nii, 1)

        mp2rage_uni1.to_filename(target_dir / 'anat' / f'sub-{subject:02d}_ses-{session}_inv-1_MP2RAGE.nii.gz')
        mp2rage_uni2.to_filename(target_dir / 'anat' / f'sub-{subject:02d}_ses-{session}_inv-2_MP2RAGE.nii.gz')

    # #T1w
    t1w_candidates = list(source_dir.glob('*MP2_085mm*_3.PAR'))
    print(source_dir)
    print(f'Found {len(t1w_candidates)} T1w candidates')

    assert(len(list(t1w_candidates)) in [0, 1]), f'Expected 1 T1w candidate, found {len(list(t1w_candidates))}'
    if len(t1w_candidates) == 1:
        t1w = list(t1w_candidates)[0]

        # # Convert to NIfTI
        img = nib.load(t1w)
        t1w_nii = nib.Nifti1Image(img.dataobj, img.affine)
        t1w_nii = image.index_img(t1w_nii, 0)
        t1w_nii = image.math_img('np.where(img == img.max(), 0, img)', img=t1w_nii)
        print(t1w_nii.get_fdata().min())

        # Rescale the image to 0-4092
        t1w_nii = image.math_img('img - img.min()', img=t1w_nii)

        print(t1w_nii.get_fdata().min())
        t1w_nii.to_filename(target_dir / 'anat' / f'sub-{subject:02d}_ses-{session}_T1w.nii.gz')

        # Create preprocessing workflow
        wf = create_preprocessed_t1w(subject, session, bids_folder=bids_dir)
        wf.run()

    # # T2w
    t2w_candidates = list(source_dir.glob('*T2w*.PAR'))
    print(f'Found {len(t2w_candidates)} T2w candidates')
    
    if len(t2w_candidates) == 0:
        print('No T2w data found, skipping T2w conversion.')
        
    elif len(t2w_candidates) == 1:

        t2w = list(t2w_candidates)[0]
        print('Storing raw T2w data...')
        # # Convert to NIfTI
        t2w_nii = image.load_img(t2w)
        t2w_nii = nib.Nifti1Image(t2w_nii.dataobj, t2w_nii.affine)
        t2w_nii = image.math_img('np.where(img == img.max(), 0, img)', img=t2w_nii)
        t2w_nii.to_filename(target_dir / 'anat' / f'sub-{subject:02d}_ses-{session}_T2w.nii.gz')
    else:
        raise ValueError(f'Expected 0/1 T2w candidate, found {len(t2w_candidates)}')

    print('Anatomical data converted and saved.')

    # Functional data
    print('Converting functional data...')

    # Find functional data
    (target_dir / 'func').mkdir(parents=True, exist_ok=True)

    # Find all functional runs
    func_candidates = list(source_dir.glob('*_func-bold_task_run-*.PAR'))

    for func_candidate in tqdm(func_candidates):
        # Convert to NIfTI
        try:
            func_nii = image.load_img(func_candidate)
            func_nii = nib.Nifti1Image(func_nii.dataobj, func_nii.affine)
            func_nii_magnitude = image.index_img(func_nii, slice(0, int(func_nii.shape[-1]/2)))
            func_nii_phase = image.index_img(func_nii, slice(int(func_nii.shape[-1]/2), func_nii.shape[-1]))

            reg = re.compile(r'run-(?P<run>[0-9]+)')
            run = int(reg.search(func_candidate.name).group('run'))

            # # Save magnitude and phase images
            func_nii_magnitude.to_filename(target_dir / 'func' / f'sub-{subject:02d}_ses-{session}_task-search_run-{run}_part-mag_bold.nii.gz')
            func_nii_phase.to_filename(target_dir / 'func' / f'sub-{subject:02d}_ses-{session}_task-search_run-{run}_part-phase_bold.nii.gz')

            with open(target_dir / 'func' / f'sub-{subject:02d}_ses-{session}_task-search_run-{run}_part-mag_bold.json', 'w') as f:
                json.dump(default_json, f, indent=2)

            with open(target_dir / 'func' / f'sub-{subject:02d}_ses-{session}_task-search_run-{run}_rec-NORDIC_bold.json', 'w') as f:
                json.dump(default_json, f, indent=2)
        except Exception as e:
            print(f'Error processing {func_candidate}: {e}')

    # B0 maps
    print('Converting B0 maps...')
    (target_dir / 'fmap').mkdir(parents=True, exist_ok=True)

    # Find all B- maps
    bmap_candidates = list(source_dir.glob('*fmap-B0*.PAR'))

    print(bmap_candidates)

    n_bmap_candidates = len(bmap_candidates)

    assert n_bmap_candidates in [2], f'Expected 1 or 2 B0 candidates, found {n_bmap_candidates}'

    for ix, bmap in enumerate(bmap_candidates):
        bmap_nii = image.load_img(bmap)
        bmap_nii = nib.Nifti1Image(bmap_nii.dataobj, bmap_nii.affine)

        bmap_magnitude = image.index_img(bmap_nii, 0)
        bmap_phase = image.index_img(bmap_nii, 1)

        json_bmap = {'EchoTimeDifference': 0.001,
                     'EchoTime1':0.0045,
                     'EchoTime2':0.0055}

        if n_bmap_candidates == 2:
            json_bmap['IntendedFor'] = [f'ses-{session}/func/sub-{subject:02d}_ses-{session}_task-search_run-{run}_rec-NORDIC_bold.nii.gz' for run in range(1 + ix*3, 4 + ix*3)]
        elif n_bmap_candidates == 1:
            json_bmap['Intended;or'] = [f'ses-{session}/func/sub-{subject:02d}_ses-{session}_task-search_run-{run}_rec-NORDIC_bold.nii.gz' for run in range(1, 6)]
        elif n_bmap_candidates == 3:
            json_bmap['IntendedFor'] = [f'ses-{session}/func/sub-{subject:02d}_ses-{session}_task-search_run-{run}_rec-NORDIC_bold.nii.gz' for run in range(1 + ix*2, 3 + ix*2)]

        bmap_magnitude.to_filename(target_dir / 'fmap' / f'sub-{subject:02d}_ses-{session}_run-{ix+1}_magnitude1.nii.gz')
        bmap_phase.to_filename(target_dir / 'fmap' / f'sub-{subject:02d}_ses-{session}_run-{ix+1}_phasediff.nii.gz')

        with open(target_dir / 'fmap' / f'sub-{subject:02d}_ses-{session}_run-{ix+1}_magnitude1.json', 'w') as f:
            json.dump(json_bmap, f, indent=2)

        with open(target_dir / 'fmap' / f'sub-{subject:02d}_ses-{session}_run-{ix+1}_phasediff.json', 'w') as f:
            json.dump(json_bmap, f, indent=2)


if __name__ == "__main__":
    # Parse command line arguments
    argument_parser = argparse.ArgumentParser(description="Convert raw MRI data to NIfTI format.")
    argument_parser.add_argument('subject', type=int, help='Subject ID (e.g., 01)')
    argument_parser.add_argument('session', type=int, help='Session ID (e.g., 01)')
    argument_parser.add_argument('--bids_dir', type=str, help='Path to BIDS directory', default='/data/ds-retsupp')

    args = argument_parser.parse_args()

    # Call the main function with parsed arguments
    main(args.subject, args.session, args.bids_dir)

