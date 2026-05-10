"""Per-(subject, ROI) batch driver for StimulusFitter decoding.

Loops over all (session, run) pairs for a subject, runs ``decode_run``
once per run, samples at the four ring positions, and writes a single
TSV with one row per (frame, ring_location):

    derivatives/decode/sub-XX/sub-XX_roi-{ROI}_decoded_drive.tsv

Designed to be called as a single SLURM array task per (subject, ROI).
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject
from retsupp.decode.decoder import (
    decode_run, sample_at_ring_positions, label_hp_lp,
)


def run_one(subject: int, roi: str, *, bids_folder: str,
            resolution: int = 30, max_voxels: int = 200,
            l2_norm: float = 0.01, learning_rate: float = 0.01,
            max_n_iterations: int = 1000, resid_max_iter: int = 500,
            disk_radius: float = 0.4,
            sd_min: float = 0.5, r2_min: float = 0.05,
            out_dir: Path | None = None,
            progressbar: bool = False) -> Path:
    sub = Subject(subject, bids_folder=bids_folder)
    hp_for_runs = sub.get_hpd_locations()

    out_dir = (Path(out_dir) if out_dir
               else Path(bids_folder) / 'derivatives' / 'decode'
                    / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / (f'sub-{subject:02d}_roi-{roi}_decoded_drive.tsv')

    t_total = time.time()
    rows = []
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            t0 = time.time()
            try:
                decoded, grid, _, _, _, _ = decode_run(
                    sub, session=ses, run=run, roi=roi,
                    resolution=resolution, max_voxels=max_voxels,
                    sd_min=sd_min, r2_min=r2_min,
                    l2_norm=l2_norm, learning_rate=learning_rate,
                    max_n_iterations=max_n_iterations,
                    resid_max_iter=resid_max_iter,
                    progressbar=progressbar, verbose=True)
            except Exception as e:
                print(f'  [sub-{subject:02d} {roi} ses-{ses} run-{run}] '
                      f'FAILED: {e}', flush=True)
                traceback.print_exc()
                continue
            per_ring = sample_at_ring_positions(
                decoded, grid, ring_disk_radius=disk_radius)
            hp_loc = hp_for_runs[(ses, run)]
            per_ring = label_hp_lp(per_ring, hp_loc)
            per_ring['subject'] = subject
            per_ring['roi'] = roi
            per_ring['session'] = ses
            per_ring['run'] = run
            rows.append(per_ring)
            print(f'  ok ses-{ses} run-{run} hp={hp_loc} in '
                  f'{time.time() - t0:.1f}s '
                  f'(rows={len(per_ring)})', flush=True)

    if not rows:
        raise RuntimeError(f'No runs decoded for sub-{subject:02d} {roi}')

    df = pd.concat(rows, axis=0, ignore_index=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'\nWrote {len(df)} rows  ({time.time() - t_total:.1f}s)  '
          f'-> {out_tsv}', flush=True)
    return out_tsv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--roi', required=True)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--resolution', type=int, default=30)
    p.add_argument('--max-voxels', type=int, default=200)
    p.add_argument('--l2-norm', type=float, default=0.01)
    p.add_argument('--learning-rate', type=float, default=0.01)
    p.add_argument('--max-n-iterations', type=int, default=1000)
    p.add_argument('--resid-max-iter', type=int, default=500)
    p.add_argument('--disk-radius', type=float, default=0.4)
    p.add_argument('--sd-min', type=float, default=0.5)
    p.add_argument('--r2-min', type=float, default=0.05)
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--progressbar', action='store_true')
    args = p.parse_args()
    run_one(args.subject, args.roi,
            bids_folder=args.bids_folder,
            resolution=args.resolution, max_voxels=args.max_voxels,
            l2_norm=args.l2_norm, learning_rate=args.learning_rate,
            max_n_iterations=args.max_n_iterations,
            resid_max_iter=args.resid_max_iter,
            disk_radius=args.disk_radius,
            sd_min=args.sd_min, r2_min=args.r2_min,
            out_dir=args.out_dir,
            progressbar=args.progressbar)


if __name__ == '__main__':
    main()
