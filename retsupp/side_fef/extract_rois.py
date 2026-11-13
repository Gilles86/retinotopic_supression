"""Side project (FEF / sPCS): frontal ROIs in each participant's BOLD space, their cleaned BOLD, and a selection table.

Per participant (array index = subject):

1. **Group ROIs projected to BOLD space** (fsaverage -> fsnative -> volume). The fsaverage masks come from
   ``define_group_roi.py`` (``frontal``, ``sPCS``, ``iPCS``). Every fsnative vertex takes the label of its nearest
   fsaverage vertex on ``?h.sphere.reg``. Each vertex is then sampled at six depths from white to pial surface
   (fmriprep GIfTIs, T1w space = BOLD space) and every sample votes for the BOLD voxel it falls in; a voxel takes the
   label with most votes ("none" = cortex outside the ROI), so a voxel mostly in the opposite bank of the sulcus is
   not included. Only voxels inside the BOLD mask are kept. Hemisphere = hemisphere of the votes.
2. **Per-participant sPCS** (``sPCSind``): voxels of the projected generous ``frontal`` mask with a cross-validated
   positive pRF in that participant: signed Gaussian pRF of the pRF-atlas side project (``side_prf_atlas_signed``),
   session-split CV correlation beating its circular-shift null (``cv_p < .05``, ``cv_r > 0``) and amplitude > 0.
3. **Controls**: the Wang-atlas ``FEF`` (the paper's atlas label), FreeSurfer lateral occipital (positive control,
   as in the hippocampus side project) and ``frontal_rest`` = ``frontal`` minus ``sPCS`` and ``iPCS`` (a large
   frontal region without the group pRF clusters).

Outputs (never overwritten), all under ``derivatives/``:

- ``side_fef_rois/sub-XX/sub-XX_desc-<roi>_mask.nii.gz``: float32, 1 = left, 2 = right hemisphere, 0 elsewhere
  (scl_slope cleared). Read by ``fit_cv_ladder_nested.py --roi-mask-dir``.
- ``side_fef_region_ts/sub-XX/sub-XX_desc-cleaned_region_ts.npz``: ``bold`` (n_runs, 258, V) cleaned BOLD of the
  union of all ROI voxels, ``member`` (V, n_roi) bool, ``rois``, ``hemi``, ``ijk``, ``voxel_index`` (index into the
  BOLD mask, the ladder's ``voxel_ids``), ``runs``, and the paper's bar + search-item ``paradigm`` and ``grid``
  (as ``side_hippocampus.extract_region_ts``).
- ``side_fef_selection/sub-XX/sub-XX_reliability_null.tsv.gz``: whole-brain model-free cross-run reliability ``r``,
  circular-shift ``p`` (100 shift sets) and ``sd``, taken from ``side_prf_atlas_signed`` (``rel_r``, ``rel_p``,
  ``bold_sd``; the statistic of ``cv/voxel_reliability_null.py``, which covers only the atlas ROIs). Same format
  as that table, for ``fit_cv_ladder_nested.py --selection-table``.

Usage::

    python -m retsupp.side_fef.extract_rois 3 --group-rois notes/side_fef/data/fsaverage_rois.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from scipy.spatial import cKDTree

from retsupp.utils.data import Subject

ROIS_SUBDIR, TS_SUBDIR, SEL_SUBDIR = 'side_fef_rois', 'side_fef_region_ts', 'side_fef_selection'
GROUP_ROIS = ('sPCS', 'iPCS')
ALL_ROIS = ('sPCS', 'iPCS', 'sPCSind', 'FEFatlas', 'frontal_rest', 'lateral_occipital')
DEPTHS = np.linspace(0, 1, 6)


def fs_sphere(fs_root, subject, fh, n):
    for cand in (f'sub-{subject:02d}', f'sub-{subject:02d}_ses-1'):
        f = fs_root / cand / 'surf' / f'{fh}.sphere.reg'
        if f.exists():
            sph = nib.freesurfer.read_geometry(str(f))[0]
            if len(sph) == n:
                return sph
    raise RuntimeError(f'no sphere.reg with {n} vertices for sub-{subject:02d} {fh}')


def project_group_rois(sub, fs_root, group, mask_img, mask_bool):
    """fsaverage label maps -> BOLD-voxel labels by depth-sampled majority vote. Returns {roi: (labelvol)}."""
    surf = sub.get_surf_info()
    inv = np.linalg.inv(mask_img.affine)
    names = ['none', 'frontal_only'] + list(GROUP_ROIS)
    shape = mask_bool.shape
    out = {r: np.zeros(shape, np.int8) for r in ('frontal',) + GROUP_ROIS}
    fsa_root = fs_root / 'fsaverage' / 'surf'
    for hi, (hemi, fh) in enumerate((('L', 'lh'), ('R', 'rh'))):
        white = nib.load(str(surf[hemi]['inner'])).agg_data('NIFTI_INTENT_POINTSET')
        pial = nib.load(str(surf[hemi]['outer'])).agg_data('NIFTI_INTENT_POINTSET')
        sph = fs_sphere(fs_root, sub.subject_id, fh, len(white))
        fsa = nib.freesurfer.read_geometry(str(fsa_root / f'{fh}.sphere.reg'))[0]
        _, nn = cKDTree(fsa).query(sph)                        # fsnative vertex -> nearest fsaverage vertex
        lab = np.zeros(len(white), np.int64)                     # 0 none, 1 frontal only, 2.. group ROIs
        lab[group['frontal'][hi][nn]] = 1
        for k, r in enumerate(GROUP_ROIS):
            lab[group[r][hi][nn]] = 2 + k
        votes = np.zeros((int(np.prod(shape)), len(names)), np.int32)
        for d in DEPTHS:
            p = white + d * (pial - white)
            ijk = np.rint(nib.affines.apply_affine(inv, p)).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(shape)), 1)
            lin = np.ravel_multi_index(tuple(ijk[ok].T), shape)
            np.add.at(votes, (lin, lab[ok]), 1)
        hit = votes.sum(1) > 0
        win = np.where(hit, votes.argmax(1), -1).reshape(shape)
        frontal_any = (votes[:, 1:].sum(1) > votes[:, 0]).reshape(shape)
        code = hi + 1
        out['frontal'][frontal_any & mask_bool & (out['frontal'] == 0)] = code
        for k, r in enumerate(GROUP_ROIS):
            out[r][(win == 2 + k) & mask_bool & (out[r] == 0)] = code
    return out


def write_label(vol, mask_img, path):
    img = nib.Nifti1Image(vol.astype(np.float32), mask_img.affine)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.to_filename(str(path))


def main(subject, bids_folder, group_rois, resolution=50):
    bids = Path(bids_folder)
    der = bids / 'derivatives'
    s = f'sub-{subject:02d}'
    ts_out = der / TS_SUBDIR / s / f'{s}_desc-cleaned_region_ts.npz'
    roi_dir = der / ROIS_SUBDIR / s
    sel_out = der / SEL_SUBDIR / s / f'{s}_reliability_null.tsv.gz'
    if ts_out.exists():
        print('exists', ts_out)
        return
    sub = Subject(subject, bids)
    fs_root = der / 'fmriprep' / 'sourcedata' / 'freesurfer'
    mask_img = sub.get_bold_mask(session=1, run=sub.get_runs(1)[0])
    mask_bool = np.asarray(mask_img.dataobj) > 0
    group = dict(np.load(group_rois))
    labs = project_group_rois(sub, fs_root, group, mask_img, mask_bool)

    sig = der / 'side_prf_atlas_signed' / s

    def signed(k):
        img = nib.load(str(sig / f'{s}_desc-{k}_bold.nii.gz'))
        assert img.shape == mask_bool.shape and np.allclose(img.affine, mask_img.affine), k
        return np.asarray(img.dataobj, np.float32)

    cv_ok = (signed('cv_p') < .05) & (signed('cv_r') > 0) & (signed('amplitude') > 0)
    labs['sPCSind'] = np.where(cv_ok, labs['frontal'], 0).astype(np.int8)
    labs['frontal_rest'] = np.where((labs['sPCS'] == 0) & (labs['iPCS'] == 0), labs['frontal'], 0).astype(np.int8)
    fef = np.zeros(mask_bool.shape, np.int8)
    for code, h in ((1, 'L'), (2, 'R')):
        m = np.asarray(sub.get_retinotopic_roi(roi=f'FEF_{h}', bold_space=True).dataobj) > 0
        fef[m & mask_bool] = code
    labs['FEFatlas'] = fef
    aseg = nib.load(str(fs_root / s / 'mri' / 'aparc+aseg.mgz'))
    seg = np.asarray(resample_to_img(aseg, mask_img, interpolation='nearest').dataobj).astype(int)
    lo = np.zeros(mask_bool.shape, np.int8)
    lo[(seg == 1011) & mask_bool] = 1
    lo[(seg == 2011) & mask_bool] = 2
    labs['lateral_occipital'] = lo

    roi_dir.mkdir(parents=True, exist_ok=True)
    for r in ('frontal',) + ALL_ROIS:
        f = roi_dir / f'{s}_desc-{r}_mask.nii.gz'
        if not f.exists():
            write_label(labs[r], mask_img, f)
    print({r: [int((labs[r] == c).sum()) for c in (1, 2)] for r in ('frontal',) + ALL_ROIS}, flush=True)

    if not sel_out.exists():
        lin = np.flatnonzero(mask_bool.ravel())
        tab = pd.DataFrame(dict(voxel_index=np.arange(lin.size), r=signed('rel_r')[mask_bool],
                                p=signed('rel_p')[mask_bool], sd=signed('bold_sd')[mask_bool]))
        tab = tab[np.isfinite(tab.r)]
        sel_out.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(sel_out, sep='\t', index=False)
        print('wrote', sel_out, len(tab))

    union = np.zeros(mask_bool.shape, bool)
    for r in ALL_ROIS:
        union |= labs[r] > 0
    ijk = np.argwhere(union)
    idx = tuple(ijk.T)
    member = np.stack([labs[r][idx] > 0 for r in ALL_ROIS], 1)
    hemi_code = np.zeros(len(ijk), np.int8)
    for r in ALL_ROIS:
        hemi_code = np.where(hemi_code == 0, labs[r][idx], hemi_code)
    hemi = np.where(hemi_code == 1, 'L', 'R')
    lin_index = -np.ones(mask_bool.shape, np.int64)
    lin_index[mask_bool] = np.arange(mask_bool.sum())
    voxel_index = lin_index[idx]
    bold, runs, pars, grid = [], [], [], None
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            fn = (der / 'cleaned' / s / f'ses-{ses}' / 'func'
                  / f'{s}_ses-{ses}_task-search_desc-cleaned_run-{run}_bold.nii.gz')
            img = nib.load(str(fn))
            assert img.shape[:3] == mask_bool.shape, (img.shape, mask_bool.shape)
            bold.append(np.asarray(img.dataobj)[idx][:, :258].T.astype(np.float32))
            par = sub.get_stimulus_with_distractors(session=ses, run=run, resolution=resolution, grid_radius=5.0,
                                                    distractor_shape='rectangle', distractor_long_side=1.5,
                                                    distractor_short_side=0.375).astype(np.float32)
            pars.append(par.reshape(par.shape[0], -1)[:258])
            if grid is None:
                gx, gy = sub.get_extended_grid_coordinates(resolution=resolution, session=ses, run=run,
                                                           grid_radius=5.0)
                grid = np.stack((gx.ravel(), gy.ravel()), axis=1).astype(np.float32)
            runs.append((ses, run))
            print(ses, run, bold[-1].shape, flush=True)
    ts_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ts_out, bold=np.stack(bold), runs=np.array(runs), rois=np.array(ALL_ROIS), member=member, hemi=hemi,
             ijk=ijk, voxel_index=voxel_index, paradigm=np.stack(pars), grid=grid)
    print('wrote', ts_out, dict(zip(ALL_ROIS, member.sum(0).tolist())))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--group-rois', required=True)
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.group_rois)
