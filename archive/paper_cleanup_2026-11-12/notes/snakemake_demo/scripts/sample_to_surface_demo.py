"""Demo surface-sampling script for the Snakemake POC.

Resamples PRF NIfTIs from
``derivatives/prf.<suffix>/model{N}/sub-XX/`` to fsnative + fsaverage
.gii files in the same suffix-aware folder. Mirrors the logic of
``retsupp/surface/sample_prf_to_surface_nilearn.py`` but reads inputs
directly from a path you pass (rather than going through
``Subject.get_prf_parameters_volume`` which hardcodes ``prf/``).

Kept under ``notes/snakemake_demo/`` to avoid changing canonical
analysis code while the Snakemake demo is exploratory.

Usage:
    python sample_to_surface_demo.py SUBJECT --model M \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
        --output-suffix snakemake_demo
"""
from __future__ import annotations

import argparse
import os.path as op
from pathlib import Path

import nibabel as nb
import numpy as np
from nilearn import image, surface
from nipype.interfaces.freesurfer import SurfaceTransform

from retsupp.utils.data import Subject
from retsupp.modeling.fit_prf import MODEL_CFG


def _param_labels(model: int):
    """Pull the columns Subject.get_prf_parameter_labels would return.

    fit_prf.py writes one NIfTI per output column of the merged
    chunks. We can recover the list of columns by globbing the
    on-disk NIfTIs in the output dir — cleaner than re-deriving the
    model-specific column list, and resilient to the merge step
    having added derived cols like ``theta`` and ``ecc``.
    """
    raise NotImplementedError  # use glob, see _glob_params below


def _glob_params(model_dir: Path, subject: int):
    """Return the list of parameter names that have merged NIfTIs on disk."""
    files = sorted(model_dir.glob(
        f'sub-{subject:02d}_desc-*.nii.gz'))
    pars = []
    for f in files:
        # Strip the .optim.nilearn_space-* derivatives that the
        # surface step itself writes back; we only want the volume
        # parameter NIfTIs as inputs.
        name = f.name
        if '.optim.' in name:
            continue
        # sub-XX_desc-PARAM.nii.gz
        par = name.split('_desc-')[1].split('.nii.gz')[0]
        pars.append(par)
    return pars


def _transform_fsaverage(in_file: str, fs_hemi: str, source_subject: str,
                         bids_folder: str | Path):
    subjects_dir = op.join(str(bids_folder), 'derivatives', 'fmriprep',
                            'sourcedata', 'freesurfer')
    sxfm = SurfaceTransform(subjects_dir=subjects_dir)
    sxfm.inputs.source_file = in_file
    sxfm.inputs.out_file = in_file.replace('fsnative', 'fsaverage')
    sxfm.inputs.source_subject = source_subject
    sxfm.inputs.target_subject = 'fsaverage'
    sxfm.inputs.hemi = fs_hemi
    try:
        return sxfm.run()
    except Exception as e:
        print(f"  WARN: fsaverage transform failed for {in_file}: {e}")
        return None


def main(subject: int, model: int, bids_folder: str,
         output_suffix: str = ''):
    bids = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids)
    surfinfo = sub.get_surf_info()

    base_dir = 'prf' if not output_suffix else f'prf.{output_suffix}'
    model_dir = (bids / 'derivatives' / base_dir
                 / f'model{model}' / f'sub-{subject:02d}')
    if not model_dir.exists():
        raise FileNotFoundError(f"No merged PRF dir at {model_dir}")

    par_keys = _glob_params(model_dir, subject)
    if not par_keys:
        raise FileNotFoundError(
            f"No sub-XX_desc-*.nii.gz volume params in {model_dir}")

    print(f"Found {len(par_keys)} PRF params: {par_keys}")
    imgs = [nb.load(str(model_dir / f'sub-{subject:02d}_desc-{p}.nii.gz'))
            for p in par_keys]
    concat = image.concat_imgs(imgs)

    print(f'Writing surface .gii to {model_dir}')
    for hemi in ['L', 'R']:
        samples = surface.vol_to_surf(
            concat,
            surfinfo[hemi]['outer'],
            inner_mesh=surfinfo[hemi]['inner'])
        samples = samples.astype(np.float32)
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        for ix, par in enumerate(par_keys):
            im = nb.gifti.GiftiImage(
                darrays=[nb.gifti.GiftiDataArray(samples[:, ix])])
            target_fn = op.join(
                str(model_dir),
                f'sub-{subject:02d}_desc-{par}.optim.nilearn_'
                f'space-fsnative_hemi-{hemi}.func.gii')
            nb.save(im, target_fn)
            _transform_fsaverage(target_fn, fs_hemi,
                                  f'sub-{subject:02d}', bids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--model', default=1, type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--output-suffix', default='',
                        help='Read/write under derivatives/prf.<suffix>/ '
                             'instead of derivatives/prf/.')
    args = parser.parse_args()
    main(args.subject, args.model,
         bids_folder=args.bids_folder,
         output_suffix=args.output_suffix)
