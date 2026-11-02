"""Where does the huge variance of some voxels come from?

For every voxel used in the attention-field fits of one subject, per run:
raw mean and SD (fMRIPrep), and SD after cleaning (percent signal change).
If a voxel's variance comes from dividing by a small mean, ``sd_clean`` tracks
``100 * sd_raw / mean_raw``, and a voxel whose mean collapses in some runs has
its variance in those runs only.

Output (never overwritten):
``derivatives/outlier_voxels/sub-XX_outlier_voxels_per_run.tsv.gz``

Usage::

    python -m retsupp.modeling.diagnose_outlier_voxels 7 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

from retsupp.utils.data import Subject

ROIS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO', 'IPS', 'SPL1', 'FEF']
BASE = 'derivatives/af_prf_cv_ladder_nested_full_bounded/m0_prf'


def main(subject, bids_folder):
    bids_folder = Path(bids_folder)
    out = bids_folder / 'derivatives' / 'outlier_voxels' / \
        f'sub-{subject:02d}_outlier_voxels_per_run.tsv.gz'
    if out.exists():
        print(f'{out} exists, nothing to do')
        return
    sub = Subject(subject, bids_folder)
    labels = []
    for roi in ROIS:
        pkl = bids_folder / BASE / f'sub-{subject:02d}' / \
            f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
        with open(pkl, 'rb') as f:
            ids = pickle.load(f)['voxel_ids']
        labels.append(pd.DataFrame(dict(roi=roi, voxel_id=ids)))
    labels = pd.concat(labels, ignore_index=True)
    ids = labels.voxel_id.values

    mask = sub.get_bold_mask()
    masker = maskers.NiftiMasker(mask_img=mask).fit(mask)
    rows = []
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            raw = masker.transform(sub.get_bold(ses, run, type='fmriprep'))[:, ids]
            clean = masker.transform(sub.get_bold(ses, run, type='cleaned'))[:, ids]
            rows.append(labels.assign(
                session=ses, run=run, mean_raw=raw.mean(0), sd_raw=raw.std(0),
                sd_clean=clean.std(0), max_abs_clean=np.abs(clean).max(0)))
            print(f'ses-{ses} run-{run} done', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out, sep='\t', index=False)
    print(f'Saved {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    a = ap.parse_args()
    main(a.subject, a.bids_folder)
