#!/usr/bin/env python
import argparse
import subprocess
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from retsupp.utils import Subject

def main(subject_id, bids_dir="/data/ds-retsupp", model=4):
    # Load subject and pRF parameters
    sub = Subject(subject_id, bids_folder=bids_dir)
    pars = sub.get_prf_parameters_surface(model)
    y, x = pars['y'], pars['x']

    # Compute polar angle for each hemisphere
    pars['lh_angle'] = np.clip(-np.degrees(np.arctan2(y, x)) + 90, 0, 180)
    pars['rh_angle'] = np.clip(-np.degrees(np.arctan2(y, -x)) + 90, 0, 180)
    pars.loc['L', 'angle'] = pars['lh_angle']
    pars.loc['R', 'angle'] = pars['rh_angle']

    # Set up output directories
    bids_dir = Path(bids_dir)
    freesurfer_dir = bids_dir / "derivatives" / "fmriprep" / "sourcedata" / "freesurfer"
    subject_dir = freesurfer_dir / f"sub-{sub.subject_id:02d}"
    surf_dir = subject_dir / "surf"
    surf_dir.mkdir(parents=True, exist_ok=True)

    # Map (parameter → filename suffix)
    param_map = {
        "angle": "prf_angle",
        "ecc": "prf_eccen",
        "r2": "prf_vexpl",
        "sd": "prf_radius"
    }

    # Write out mgz files and collect filenames
    hemi_files = {}
    for hemi, prefix in zip(["L", "R"], ['lh', 'rh']):
        hemi_files[prefix] = {}
        for param, suffix in param_map.items():
            data = np.asarray(pars.loc[hemi, param], dtype=np.float32)[np.newaxis, np.newaxis, :]
            data[~np.isfinite(data)] = 0
            img = nib.MGHImage(data, affine=np.eye(4))
            out_file = surf_dir / f"{prefix}.{suffix}.mgz"
            nib.save(img, out_file)
            hemi_files[prefix][suffix] = str(out_file)

    # Set the SUBJECTS_DIR environment variable while preserving the existing environment
    env = os.environ.copy()
    env['SUBJECTS_DIR'] = str(freesurfer_dir)

    # Define the neuropythy command
    cmd = [
        'python', '-m', 'neuropythy', 'register_retinotopy', f"sub-{sub.subject_id:02d}",
        '--lh-eccen', hemi_files['lh']['prf_eccen'],
        '--lh-angle', hemi_files['lh']['prf_angle'],
        '--lh-weight', hemi_files['lh']['prf_vexpl'],
        '--lh-radius', hemi_files['lh']['prf_radius'],
        '--rh-eccen', hemi_files['rh']['prf_eccen'],
        '--rh-angle', hemi_files['rh']['prf_angle'],
        '--rh-weight', hemi_files['rh']['prf_vexpl'],
        '--rh-radius', hemi_files['rh']['prf_radius'],
        '--verbose'
    ]

    print("Starting retinotopy registration...")
    print(' '.join(cmd))  # Print the command for transparency

    # Run the command with live output
    result = subprocess.run(cmd, env=env, check=True)

    print("\nRegistration complete! Results saved in:")
    print(f"Surface files: {surf_dir}/")
    print(f"Volumetric files: {subject_dir}/mri/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register retinotopy for a subject.")
    parser.add_argument("subject_id", type=int, help="Subject ID (integer, e.g., 3)")
    parser.add_argument("--bids_dir", type=str, default="/data/ds-retsupp", help="BIDS root directory")
    args = parser.parse_args()
    main(args.subject_id, args.bids_dir)
