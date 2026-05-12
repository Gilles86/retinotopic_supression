"""Build per-subject NPZ cache of cleaned BOLD + paradigm tensor.

For each subject, loads all (session, run) cleaned BOLDs, masker-
transforms them, builds the paradigm tensor, and writes the result to

    derivatives/cleaned_bold_cache/sub-XX/sub-XX_kind-{full|bar}_res-RR.npz

Downstream chunked fits (fit_prf_chunked.sh etc.) hit this cache via
``load_concatenated`` and skip the ~60-120s gzip decompression on
every task — turning per-chunk overhead from 2-3 min into ~30s.

The "cleaned" in the directory name is load-bearing: the cache stores
cleaned BOLD (motion outliers interpolated, confounds regressed by
preprocess/clean.py), NOT fmriprep-raw or other variants.

CLI::

    python build_cleaned_bold_cache.py SUB --bids-folder PATH \\
        [--paradigm-kind {full,bar}] [--resolution 50]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from nilearn import maskers

from retsupp.utils.data import Subject
from retsupp.modeling.fit_prf import load_concatenated


def cache_path(bids_folder: Path, subject: int,
                paradigm_kind: str, resolution: int) -> Path:
    """Return the canonical NPZ path for this (subject, kind, resolution)."""
    return (Path(bids_folder) / 'derivatives' / 'cleaned_bold_cache'
            / f'sub-{subject:02d}'
            / f'sub-{subject:02d}_kind-{paradigm_kind}_res-{resolution}.npz')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--bids-folder', required=True)
    p.add_argument('--paradigm-kind', choices=['full', 'bar'], default='full')
    p.add_argument('--resolution', type=int, default=50)
    p.add_argument('--force', action='store_true',
                   help='Rebuild even if the cache file already exists.')
    a = p.parse_args()

    t0 = time.time()
    bids = Path(a.bids_folder)
    sub = Subject(a.subject, bids)
    out = cache_path(bids, a.subject, a.paradigm_kind, a.resolution)

    if out.exists() and not a.force:
        print(f'cache already exists: {out}  '
              f'({os.path.getsize(out)/1e6:.1f} MB) — pass --force to rebuild')
        return

    first_run = sub.get_runs(1)[0]
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask(
        session=1, run=first_run))
    masker.fit()

    data, paradigm, grid_coords = load_concatenated(
        sub, masker, a.resolution, a.paradigm_kind)
    print(f'[{time.time()-t0:.1f}s] loaded: bold {data.shape}  '
          f'paradigm {paradigm.shape}  grid {grid_coords.shape}')

    out.parent.mkdir(parents=True, exist_ok=True)
    # np.savez (no underscore) is *uncompressed* npz — the archive is a
    # zip with stored-only entries. Read speed is essentially the disk
    # bandwidth. np.savez_compressed would DEFLATE-compress and slow
    # loading significantly. This cache is the per-task hot path, so
    # we deliberately want uncompressed.
    np.savez(out, bold=data.astype(np.float32),
             paradigm=paradigm.astype(np.float32),
             grid_coords=grid_coords.astype(np.float32))
    print(f'[{time.time()-t0:.1f}s] wrote {out}  '
          f'({os.path.getsize(out)/1e6:.1f} MB, uncompressed)')


if __name__ == '__main__':
    main()
