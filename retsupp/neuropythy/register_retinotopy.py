#!/usr/bin/env python
import argparse
import subprocess
import os
import numpy as np
# neuropythy 0.12.16 uses the deprecated `np.complex` alias (numpy<1.20).
# Restore it before importing neuropythy so register_retinotopy can run
# under modern numpy (the retsupp_neuropythy env ships 1.26.x).
if not hasattr(np, 'complex'):
    np.complex = complex
import nibabel as nib
from pathlib import Path
from retsupp.utils import Subject

def main(subject_id, bids_dir="/data/ds-retsupp", model=4, fs_subject=None):
    # Load subject and pRF parameters
    sub = Subject(subject_id, bids_folder=bids_dir)
    pars = sub.get_prf_parameters_surface(model)
    y, x = pars['y'], pars['x']

    # Compute polar angle for each hemisphere
    pars['lh_angle'] = np.clip(-np.degrees(np.arctan2(y, x)) + 90, 0, 180)
    pars['rh_angle'] = np.clip(-np.degrees(np.arctan2(y, -x)) + 90, 0, 180)
    pars.loc['L', 'angle'] = pars['lh_angle']
    pars.loc['R', 'angle'] = pars['rh_angle']

    # Set up output directories.  ``fs_subject`` allows pointing at an
    # alternate freesurfer subject dir (e.g. ``sub-16_ses-1`` when the
    # canonical ``sub-16`` recon-all has a different vertex count than
    # the prf .gii inputs).  When given, both the prf .mgz inputs and
    # neuropythy's registration run against this alternate dir; the
    # caller is responsible for copying the resulting ``inferred_*.mgz``
    # back into the canonical sub-XX/mri/ afterwards (safe because the
    # underlying ``orig.mgz`` is byte-identical between recon runs on
    # the same input).
    fs_subject = fs_subject or f"sub-{sub.subject_id:02d}"
    bids_dir = Path(bids_dir)
    freesurfer_dir = bids_dir / "derivatives" / "fmriprep" / "sourcedata" / "freesurfer"
    subject_dir = freesurfer_dir / fs_subject
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
    os.environ['SUBJECTS_DIR'] = str(freesurfer_dir)

    # Call neuropythy in-process (was subprocess; subprocess loses the
    # np.complex monkey-patch above and fails under numpy>=1.20).
    argv = [
        'register_retinotopy', fs_subject,
        '--lh-eccen', hemi_files['lh']['prf_eccen'],
        '--lh-angle', hemi_files['lh']['prf_angle'],
        '--lh-weight', hemi_files['lh']['prf_vexpl'],
        '--lh-radius', hemi_files['lh']['prf_radius'],
        '--rh-eccen', hemi_files['rh']['prf_eccen'],
        '--rh-angle', hemi_files['rh']['prf_angle'],
        '--rh-weight', hemi_files['rh']['prf_vexpl'],
        '--rh-radius', hemi_files['rh']['prf_radius'],
        '--verbose',
    ]
    print("Starting retinotopy registration...")
    print('neuropythy ' + ' '.join(argv))

    from neuropythy.commands import register_retinotopy as rr
    rr.main(argv[1:])

    print("\nRegistration complete! Results saved in:")
    print(f"Surface files: {surf_dir}/")
    print(f"Volumetric files: {subject_dir}/mri/")

    # Snapshot per-model: copy inferred_*.mgz to
    # derivatives/neuropythy/model{N}/sub-XX/{surf,mri}/ so multiple
    # model runs don't clobber each other and a loader can pick the
    # model of interest. Canonical freesurfer location keeps the most
    # recent run (model-4 by default, downstream code reads from there).
    import shutil
    archive_root = bids_dir / 'derivatives' / 'neuropythy' / \
        f'model{model}' / f'sub-{sub.subject_id:02d}'
    arch_surf = archive_root / 'surf'
    arch_mri = archive_root / 'mri'
    arch_surf.mkdir(parents=True, exist_ok=True)
    arch_mri.mkdir(parents=True, exist_ok=True)
    mri_dir = subject_dir / 'mri'

    n_copied = 0
    for f in surf_dir.glob('*.inferred_*'):
        shutil.copy2(f, arch_surf / f.name)
        n_copied += 1
    for f in mri_dir.glob('inferred_*.mgz'):
        shutil.copy2(f, arch_mri / f.name)
        n_copied += 1
    print(f"Snapshot {n_copied} files to {archive_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register retinotopy for a subject.")
    parser.add_argument("subject_id", type=int, help="Subject ID (integer, e.g., 3)")
    parser.add_argument("--bids_dir", type=str, default="/data/ds-retsupp", help="BIDS root directory")
    parser.add_argument("--model", type=int, default=4,
                        help="PRF model number to source surface files from (default 4). "
                             "Use 1 when only Gaussian m1 fits exist (e.g. recovery subjects).")
    parser.add_argument("--fs-subject", type=str, default=None,
                        help="Override the freesurfer subject id used as SUBJECTS_DIR/<subject> "
                             "(default sub-XX). Use for recovery subjects where the canonical "
                             "freesurfer recon has a different vertex count than the .gii "
                             "surface PRF fits; the matching recon may live under sub-XX_ses-1.")
    args = parser.parse_args()
    main(args.subject_id, args.bids_dir, model=args.model, fs_subject=args.fs_subject)
