"""Extract per-(subject, ROI, hp_cond, dir_tag, t) mean BOLD traces.

Cluster-side analogue to the slow part of ``plot_voxel_traces_group.py``.
For ONE subject, runs the full matched-voxel + bar-event time-locking
pipeline, and writes a small TSV with the aggregated traces. Plotting
then reads these TSVs and renders the figure in seconds.

TSV columns (long format):
  subject, roi, hp_cond, dir_tag, t_offset, mean_bold, n_events,
  n_voxels

  hp_cond ∈ {close, lateral, opposite}
  dir_tag ∈ {all, toward, away, orth, toVq, awayVq}
  t_offset ∈ {-10, ..., 10}  TRs relative to bar centre passing RF.

Per-subject TSV is ~3024 rows × ~50 B ≈ 150 KB → no gzip needed; if it
ever climbs past 10 MB we switch to ``.tsv.gz``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import maskers

from retsupp.utils.data import Subject
from retsupp.visualize.plot_voxel_traces_group import (
    aggregate_subject, HP_CONDS,
)

DIR_TAGS = ('all', 'toward', 'away', 'orth', 'toVq', 'awayVq')
ROI_DEFAULTS = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'VO', 'LO', 'TO']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--rois', nargs='+', default=ROI_DEFAULTS)
    p.add_argument('--prf-model', type=int, default=4)
    p.add_argument('--sd-min', type=float, default=0.2)
    p.add_argument('--sd-max', type=float, default=3.5)
    p.add_argument('--r2-min', type=float, default=0.10)
    p.add_argument('--margin-kind', choices=['fwhm', 'sd', 'mass'],
                   default='mass')
    p.add_argument('--quadrant-mass', type=float, default=0.5)
    p.add_argument('--min-voxels', type=int, default=5)
    p.add_argument('--max-bar-dist', type=float, default=0.5)
    p.add_argument('--window-TRs', type=int, default=21)
    p.add_argument('--plot-upsample-dt', type=float, default=0.0,
                   help='No-op here (kept for arg-compatibility with '
                        'aggregate_subject).')
    p.add_argument('--out-dir', type=Path,
                   default=Path('/data/ds-retsupp/derivatives/'
                                  'voxel_traces_cache'))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub = Subject(args.subject, args.bids_folder)
    masker = maskers.NiftiMasker(mask_img=sub.get_bold_mask())
    masker.fit()
    print(f'sub-{args.subject:02d} | rois={args.rois}')
    res = aggregate_subject(sub, masker, args)
    if res is None:
        print(f'  no data for sub-{args.subject:02d}')
        return
    win = res['win']
    half = win // 2
    t_axis = np.arange(win) - half

    rows = []
    for roi, rd in res['rois'].items():
        n_vox = int(rd.get('n_voxels', 0))
        for (hp_cond, dir_tag), d in rd['agg'].items():
            if d is None:
                continue
            mean_trace = d['mean']
            n_ev = int(d['n'])
            for ti, t_offset in enumerate(t_axis):
                rows.append({
                    'subject': args.subject,
                    'roi': roi,
                    'hp_cond': hp_cond,
                    'dir_tag': dir_tag,
                    't_offset': int(t_offset),
                    'mean_bold': float(mean_trace[ti]),
                    'n_events': n_ev,
                    'n_voxels': n_vox,
                })

    if not rows:
        print(f'  no rows produced for sub-{args.subject:02d}')
        return
    df = pd.DataFrame(rows)
    out_fn = args.out_dir / f'sub-{args.subject:02d}_voxel_traces.tsv'
    df.to_csv(out_fn, sep='\t', index=False)
    print(f'  wrote {out_fn}  ({len(df)} rows, '
          f'{df.roi.nunique()} ROIs)')


if __name__ == '__main__':
    main()
