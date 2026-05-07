"""Project Wang 2015 surface labels (lh/rh.wang15_mplbl.mgz) to T1w
volume MGZ using neuropythy's Python API. CLI of `neuropythy atlas
--volume-export --image-template …` had broken option-parsing in our
version; this is the cleanest workaround.

Output: <freesurfer>/sub-XX/mri/wang15_atlas.mgz, in T1w/orig space
matching `inferred_varea.mgz`. After this, `Subject.get_retinotopic_atlas`
can be extended to read it for IPS/SPL/FEF labels.

Usage:
  python -m retsupp.neuropythy.wang_to_volume 5
  python -m retsupp.neuropythy.wang_to_volume --all
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import neuropythy as ny
import nibabel as nib
import numpy as np


def project_one(subject: int, fs_dir: Path, overwrite: bool = False) -> Path | None:
    sub_dir = fs_dir / f'sub-{subject:02d}'
    out = sub_dir / 'mri' / 'wang15_atlas.mgz'
    if out.exists() and not overwrite:
        print(f'  sub-{subject:02d}: {out} exists, skipping')
        return out
    lh = sub_dir / 'surf' / 'lh.wang15_mplbl.mgz'
    rh = sub_dir / 'surf' / 'rh.wang15_mplbl.mgz'
    if not (lh.exists() and rh.exists()):
        print(f'  sub-{subject:02d}: missing wang15 surf files, skipping')
        return None

    sub = ny.freesurfer_subject(str(sub_dir))
    lh_arr = nib.load(str(lh)).get_fdata().astype(np.int32).flatten()
    rh_arr = nib.load(str(rh)).get_fdata().astype(np.int32).flatten()
    template_path = sub_dir / 'mri' / 'orig.mgz'
    template = nib.load(str(template_path))

    # IMPORTANT: pass a ZEROS template (same shape/affine as orig) so
    # voxels outside the cortical ribbon are 0, not the T1w intensity.
    zeros_template = nib.MGHImage(
        np.zeros(template.shape, dtype=np.int32),
        affine=template.affine, header=template.header,
    )
    vol_img = sub.cortex_to_image(
        (lh_arr, rh_arr), zeros_template,
        method='nearest', dtype=np.int32, fill=0,
    )
    vol = np.asarray(vol_img.dataobj).astype(np.int32)
    img = nib.MGHImage(vol, affine=template.affine, header=template.header)
    img.to_filename(str(out))
    print(f'  sub-{subject:02d}: wrote {out} '
          f'(shape={vol.shape}, n_labelled_voxels={(vol > 0).sum()})')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', nargs='?', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    fs_dir = Path(args.bids_folder) / 'derivatives' / 'fmriprep' \
              / 'sourcedata' / 'freesurfer'
    os.environ['SUBJECTS_DIR'] = str(fs_dir)

    if args.all:
        subs = list(range(1, 31))
    elif args.subject is not None:
        subs = [args.subject]
    else:
        raise SystemExit('Pass a subject ID or --all.')
    for s in subs:
        try:
            project_one(s, fs_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f'  sub-{s:02d}: ERROR — {e}')


if __name__ == '__main__':
    main()
