"""Infer the Wang 2015 maximum-probability retinotopic atlas per subject.

The default neuropythy `register_retinotopy` command produces the
Benson-14 inferred-varea atlas which only labels V1, V2, V3, hV4, V3A,
V3B, LO1, LO2, TO1, TO2, VO1, VO2 (12 ROIs). We're missing IPS, SPL1,
FEF — all part of the Wang 2015 max-prob atlas.

This script runs `python -m neuropythy atlas <sub> --atlases wang15`
which interpolates the Wang atlas onto the subject's freesurfer
surface AND projects to volume.

After running, you'll have new files in:
  derivatives/fmriprep/sourcedata/freesurfer/sub-XX/mri/wang15_atlas.mgz
  derivatives/fmriprep/sourcedata/freesurfer/sub-XX/surf/{lh,rh}.wang15_atlas.mgz

Wang 2015 label map:
   1: V1v   2: V1d   3: V2v   4: V2d   5: V3v   6: V3d
   7: hV4   8: VO1   9: VO2  10: PHC1 11: PHC2
  12: TO2  13: TO1  14: LO2  15: LO1  16: V3B  17: V3A
  18: IPS0 19: IPS1 20: IPS2 21: IPS3 22: IPS4 23: IPS5
  24: SPL1 25: FEF

Usage:
  python -m retsupp.neuropythy.infer_wang_atlas 5 --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main(subject: int, bids_folder: str = '/data/ds-retsupp'):
    bids_folder = Path(bids_folder)
    fs_dir = bids_folder / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer'
    sub_dir = fs_dir / f'sub-{subject:02d}'
    if not sub_dir.exists():
        raise FileNotFoundError(f'No freesurfer dir for sub-{subject:02d}: {sub_dir}')

    env = os.environ.copy()
    env['SUBJECTS_DIR'] = str(fs_dir)

    cmd = [
        'python', '-m', 'neuropythy', 'atlas', f'sub-{subject:02d}',
        '--atlases', 'wang15',
        '--volume-export',
        '--surface-export',
        '--output-path', str(sub_dir / 'surf'),
        '--create-directory',
        '--verbose',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, env=env, check=True)
    print(f'\nWang atlas written for sub-{subject:02d}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    args = parser.parse_args()
    main(args.subject, bids_folder=args.bids_folder)
