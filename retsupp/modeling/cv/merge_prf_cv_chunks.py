"""Merge per-chunk CV r2 NPZs into a single (sub, model, fold) NIfTI.

Companion to ``retsupp/modeling/cv/fit_prf_cv.py``. Reads
``derivatives/prf_cv/model{N}/fold-{K}/sub-XX/chunks/chunk-XXXX-of-YYYY.npz``,
concatenates by voxel index, inverse-transforms via the BOLD-mask
masker, and writes the canonical
``derivatives/prf_cv/model{N}/fold-{K}/sub-XX/sub-XX_fold-{K}_desc-r2_test.nii.gz``.

Usage:
    python -m retsupp.modeling.cv.merge_prf_cv_chunks 3 \\
        --model 4 --fold 0 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from nilearn import maskers

from retsupp.utils.data import Subject


def main(subject: int, model: int, fold: int,
         bids_folder: str = '/data/ds-retsupp',
         keep_chunks: bool = False):
    bids = Path(bids_folder)
    sub = Subject(subject, bids)
    src_dir = (bids / 'derivatives' / 'prf_cv' / f'model{model}'
               / f'fold-{fold}' / f'sub-{subject:02d}' / 'chunks')
    out_dir = src_dir.parent

    all_chunks = sorted(src_dir.glob('chunk-*-of-*.npz'))
    if not all_chunks:
        raise FileNotFoundError(f"No chunk NPZ files in {src_dir}")

    # Pick the most populated -of-MMMM family (defends against stale
    # chunks from earlier sweeps with a different N_CHUNKS).
    families = Counter(p.name.split('-of-')[1].split('.npz')[0]
                       for p in all_chunks)
    canonical_family = families.most_common(1)[0][0]
    canonical = sorted(src_dir.glob(f'chunk-*-of-{canonical_family}.npz'))
    if len(families) > 1:
        ignored = sum(c for f, c in families.items() if f != canonical_family)
        print(f"  WARN: {ignored} stale chunk(s) ignored "
              f"(families={dict(families)}, canonical=of-{canonical_family})")
    n_expected = int(canonical_family)
    if len(canonical) != n_expected:
        raise RuntimeError(
            f"Incomplete chunk set in {src_dir}: have {len(canonical)} "
            f"of {n_expected} expected (of-{canonical_family})")

    # Concatenate voxel-indexed r2 values.
    first = np.load(canonical[0])
    n_total_vox = int(first['n_total_vox'])
    full_r2 = np.zeros(n_total_vox, dtype=np.float32)
    for cp in canonical:
        d = np.load(cp)
        full_r2[d['vox_idx']] = d['r2']
    print(f"  merged {len(canonical)} chunks → r2 shape ({n_total_vox},)")
    print(f"  test R² stats: median={np.median(full_r2):.3f} "
          f"p90={np.quantile(full_r2, 0.9):.3f} "
          f"frac>0.1={np.mean(full_r2 > 0.1):.3f}")

    # Inverse-transform + write NIfTI. Always float32, clear scl_slope.
    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()
    img = masker.inverse_transform(full_r2)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    out_path = (out_dir
                / f'sub-{subject:02d}_fold-{fold}_desc-r2_test.nii.gz')
    img.to_filename(out_path)
    print(f"wrote {out_path}")

    if not keep_chunks:
        for cp in canonical:
            cp.unlink()
        # remove json sidecars if any
        for jp in src_dir.glob('chunk-*.json'):
            jp.unlink()
        if not any(src_dir.iterdir()):
            src_dir.rmdir()
        print(f"  cleaned up chunks/ ({len(canonical)} npz removed)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('subject', type=int)
    p.add_argument('--model', type=int, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--keep-chunks', action='store_true')
    args = p.parse_args()
    main(args.subject, args.model, args.fold,
         bids_folder=args.bids_folder, keep_chunks=args.keep_chunks)
