"""Side project (FEF / sPCS): leave-one-participant-out group ROIs, so that a participant's ROI is not selected on
that participant's own data.

The group sPCS / iPCS of ``define_group_roi.py`` are defined on the median CV R² of all 28 participants, including
the one being tested, which favours that participant's noise in the selected vertices. Here, for participant s, the
group map is the median over the OTHER 27 participants (``side_prf_atlas_signed/sub-XX_space-fsaverage_maps.npz``,
``sign(cv_r) * cv_r**2``, as ``side_prf_atlas/group_sums.py``), the same cluster rule is applied
(``define_group_roi.define``), and the result is projected to s's BOLD space as in ``extract_rois.py``.

Outputs (never overwritten): ``side_fef_rois/sub-XX/sub-XX_desc-{sPCSloso,iPCSloso}_mask.nii.gz``,
``side_fef_rois/sub-XX/sub-XX_desc-loso_fsaverage.npz`` (the vertex masks) and
``side_fef_region_ts/sub-XX/sub-XX_desc-cleanedloso_region_ts.npz`` (same layout as the main time-series file).

Usage::

    python -m retsupp.side_fef.loso_rois 3
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from retsupp.side_fef.define_group_roi import define
from retsupp.side_fef.extract_rois import ROIS_SUBDIR, TS_SUBDIR, extract_ts, project_group_rois, write_label
from retsupp.utils.data import Subject

SUBJECTS = range(3, 31)
LOSO_ROIS = ('sPCSloso', 'iPCSloso')


def main(subject, bids_folder):
    der = Path(bids_folder) / 'derivatives'
    s = f'sub-{subject:02d}'
    ts_out = der / TS_SUBDIR / s / f'{s}_desc-cleanedloso_region_ts.npz'
    if ts_out.exists():
        print('exists', ts_out)
        return
    maps = []
    for o in SUBJECTS:
        if o == subject:
            continue
        cv = np.load(der / 'side_prf_atlas_signed' / f'sub-{o:02d}' / f'sub-{o:02d}_space-fsaverage_maps.npz')['cv_r']
        maps.append(np.where(np.isfinite(cv), np.sign(cv) * cv ** 2, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = np.nanmedian(np.stack(maps), 0)
    fs_root = der / 'fmriprep' / 'sourcedata' / 'freesurfer'
    group = define(q, fs_root / 'fsaverage')
    print(f'{len(maps)} other participants; vertices (L, R):', {k: v.sum(1).tolist() for k, v in group.items()})
    roi_dir = der / ROIS_SUBDIR / s
    roi_dir.mkdir(parents=True, exist_ok=True)
    fsa_out = roi_dir / f'{s}_desc-loso_fsaverage.npz'
    if not fsa_out.exists():
        np.savez_compressed(fsa_out, q=q.astype(np.float32), **group)
    sub = Subject(subject, Path(bids_folder))
    mask_img = sub.get_bold_mask(session=1, run=sub.get_runs(1)[0])
    mask_bool = np.asarray(mask_img.dataobj) > 0
    labs = project_group_rois(sub, fs_root, group, mask_img, mask_bool)
    labs = {'sPCSloso': labs['sPCS'], 'iPCSloso': labs['iPCS']}
    for r in LOSO_ROIS:
        f = roi_dir / f'{s}_desc-{r}_mask.nii.gz'
        if not f.exists():
            write_label(labs[r], mask_img, f)
    print({r: [int((labs[r] == c).sum()) for c in (1, 2)] for r in LOSO_ROIS}, flush=True)
    extract_ts(sub, der, labs, LOSO_ROIS, mask_bool, ts_out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    a = ap.parse_args()
    main(a.subject, a.bids_folder)
