"""Side project (hippocampus): cleaned BOLD of FreeSurfer aparc+aseg regions, all runs, one NPZ per subject.

Regions (left, right labels): hippocampus 17/53, entorhinal 1006/2006, parahippocampal 1016/2016, amygdala 18/54,
lateral occipital 1011/2011 (positive control), precentral 1024/2024 (negative control). aparc+aseg is resampled
(nearest neighbour) to the BOLD mask of ses-1 run-1; only voxels inside the BOLD mask are kept.

Output (never overwritten): ``derivatives/side_hpc_region_ts/sub-XX/sub-XX_desc-cleaned_region_ts.npz`` with
``bold`` (n_runs, 258, V) float32 cleaned BOLD, ``runs`` (n_runs, 2) session/run, ``region``, ``hemi``, ``ijk``
(V, 3) voxel coordinates in BOLD space, ``paradigm`` (n_runs, 258, P) bar + search-item stimulus as in
``retsupp.modeling.cv.fit_prf_cv.load_runs`` (kind 'full', resolution 50, grid radius 5), ``grid`` (P, 2).

Usage::

    python -m retsupp.side_hippocampus.extract_region_ts 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from retsupp.utils.data import Subject

REGIONS = {'hippocampus': (17, 53), 'entorhinal': (1006, 2006), 'parahippocampal': (1016, 2016),
           'amygdala': (18, 54), 'lateral_occipital': (1011, 2011), 'precentral': (1024, 2024)}
OUT_SUBDIR = 'side_hpc_region_ts'


def main(subject, bids_folder, resolution=50):
    bids = Path(bids_folder)
    out = bids / 'derivatives' / OUT_SUBDIR / f'sub-{subject:02d}' / f'sub-{subject:02d}_desc-cleaned_region_ts.npz'
    if out.exists():
        print('exists', out)
        return
    sub = Subject(subject, bids)
    mask = sub.get_bold_mask(session=1, run=sub.get_runs(1)[0])
    aseg = nib.load(bids / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer' / f'sub-{subject:02d}' / 'mri'
                    / 'aparc+aseg.mgz')
    seg = np.asarray(resample_to_img(aseg, mask, interpolation='nearest').dataobj).astype(int)
    inmask = np.asarray(mask.dataobj) > 0
    region = np.full(seg.shape, '', dtype=object)
    hemi = np.full(seg.shape, '', dtype=object)
    for name, (left, right) in REGIONS.items():
        for h, lab in (('L', left), ('R', right)):
            m = (seg == lab) & inmask
            region[m], hemi[m] = name, h
    ijk = np.argwhere(region != '')
    idx = tuple(ijk.T)
    bold, runs, pars = [], [], []
    grid = None
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            fn = (bids / 'derivatives' / 'cleaned' / f'sub-{subject:02d}' / f'ses-{ses}' / 'func'
                  / f'sub-{subject:02d}_ses-{ses}_task-search_desc-cleaned_run-{run}_bold.nii.gz')
            img = nib.load(fn)
            assert img.shape[:3] == seg.shape, (img.shape, seg.shape)
            bold.append(np.asarray(img.dataobj)[idx][:, :258].T.astype(np.float32))
            par = sub.get_stimulus_with_distractors(session=ses, run=run, resolution=resolution, grid_radius=5.0,
                                                    distractor_shape='rectangle', distractor_long_side=1.5,
                                                    distractor_short_side=0.375).astype(np.float32)
            pars.append(par.reshape(par.shape[0], -1)[:258])
            if grid is None:
                gx, gy = sub.get_extended_grid_coordinates(resolution=resolution, session=ses, run=run, grid_radius=5.0)
                grid = np.stack((gx.ravel(), gy.ravel()), axis=1).astype(np.float32)
            runs.append((ses, run))
            print(ses, run, bold[-1].shape, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, bold=np.stack(bold), runs=np.array(runs), region=region[idx].astype(str), hemi=hemi[idx].astype(str),
             ijk=ijk, paradigm=np.stack(pars), grid=grid)
    print('wrote', out, {r: int((region[idx] == r).sum()) for r in REGIONS})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    a = ap.parse_args()
    main(a.subject, a.bids_folder)
