"""Merge per-chunk NPZ files from chunked fit_prf runs into final NIfTIs.

For (subject, model), reads
``derivatives/prf/model{N}/sub-XX/chunks/chunk-XXXX-of-YYYY.npz`` files,
concatenates by voxel index, inverse-transforms via the BOLD-mask
masker, and writes the canonical
``derivatives/prf/model{N}/sub-XX/sub-XX_desc-{par}.nii.gz`` outputs.

Usage:
    python -m retsupp.modeling.merge_prf_chunks 3 --model 1 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

from retsupp.utils.data import Subject


def main(subject: int, model: int, bids_folder: str = '/data/ds-retsupp',
         paradigm_kind: str = 'full', keep_chunks: bool = False):
    bids = Path(bids_folder)
    sub = Subject(subject, bids)
    base_dir = 'prf_bar' if paradigm_kind == 'bar' else 'prf'
    src_dir = (bids / 'derivatives' / base_dir / f'model{model}'
               / f'sub-{subject:02d}' / 'chunks')
    out_dir = (bids / 'derivatives' / base_dir / f'model{model}'
               / f'sub-{subject:02d}')

    chunks = sorted(src_dir.glob('chunk-*-of-*.npz'))
    if not chunks:
        raise FileNotFoundError(f"No chunk NPZ files in {src_dir}")
    total = int(chunks[0].name.split('-of-')[1].split('.npz')[0])
    if len(chunks) != total:
        raise RuntimeError(
            f"{len(chunks)} chunk files found but expected {total}. "
            f"Some chunks failed?")

    voxel_idx_all, parframes = [], []
    for f in chunks:
        d = np.load(f, allow_pickle=True)
        voxel_idx_all.append(d['voxel_indices'])
        cols = list(d['columns'])
        df = pd.DataFrame({c: d[f'col_{c}'] for c in cols})
        parframes.append(df)
    vox = np.concatenate(voxel_idx_all)
    pars = pd.concat(parframes, axis=0).reset_index(drop=True)
    if not np.array_equal(np.sort(vox), np.arange(len(vox))):
        raise RuntimeError("Voxel indices not contiguous 0..N-1; chunks "
                           "are missing or duplicated.")
    pars.index = vox
    pars = pars.sort_index()

    if 'x' in pars and 'y' in pars:
        pars['theta'] = np.arctan2(pars['y'], pars['x'])
        pars['ecc'] = np.sqrt(pars['x'] ** 2 + pars['y'] ** 2)

    first_run = sub.get_runs(1)[0]
    masker = maskers.NiftiMasker(
        mask_img=sub.get_bold_mask(session=1, run=first_run))
    masker.fit()
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in pars.columns:
        img = masker.inverse_transform(pars[col].values)
        # The BOLD mask is uint8 (datatype=2). Without an explicit cast
        # here, nilearn lets nibabel inherit that dtype + auto-pick a
        # scl_slope to fit the output range, quantizing per-voxel values
        # to 256 bins (with outliers up to ~50000, real values around 0
        # get crushed into 1-2 bins — the merged amplitude looks like
        # ~30 unique values across the whole brain). Force float32.
        img.set_data_dtype(np.float32)
        img.header.set_slope_inter(slope=1, inter=0)
        img.to_filename(out_dir / f'sub-{subject:02d}_desc-{col}.nii.gz')
    print(f"Merged {len(chunks)} chunks -> {out_dir} ({len(pars.columns)} params)")

    if not keep_chunks:
        for f in chunks:
            f.unlink()
        try:
            src_dir.rmdir()
        except OSError:
            pass
        print(f"Removed chunks/ subdir.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--model', type=int, required=True)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--paradigm-kind', choices=['full', 'bar'], default='full')
    p.add_argument('--keep-chunks', action='store_true',
                   help="Don't delete chunks/ after merging.")
    a = p.parse_args()
    main(a.subject, a.model, bids_folder=a.bids_folder,
         paradigm_kind=a.paradigm_kind, keep_chunks=a.keep_chunks)
