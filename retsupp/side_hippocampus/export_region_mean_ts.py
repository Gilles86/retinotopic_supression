"""Side project (hippocampus): region-mean cleaned BOLD per run, from the ``side_hpc_region_ts`` NPZs.

All voxels of each anatomical region (no selection on the data), averaged per volume, for the left hemisphere, the
right hemisphere and both. Written as one long table for all subjects:
``derivatives/side_hpc_region_mean_ts/region_mean_ts.tsv.gz`` (subject, region, hemi, session, run, tr_index, signal,
n_voxels). Never overwritten.

Usage (compute node)::

    python -m retsupp.side_hippocampus.export_region_mean_ts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.side_hippocampus.extract_region_ts import OUT_SUBDIR as TS_SUBDIR, REGIONS

OUT_SUBDIR = 'side_hpc_region_mean_ts'


def main(bids_folder, subjects):
    deriv = Path(bids_folder) / 'derivatives'
    out = deriv / OUT_SUBDIR / 'region_mean_ts.tsv.gz'
    if out.exists():
        print('exists', out)
        return
    rows = []
    for s in subjects:
        z = np.load(deriv / TS_SUBDIR / f'sub-{s:02d}' / f'sub-{s:02d}_desc-cleaned_region_ts.npz')
        bold, reg, hemi, runs = z['bold'], z['region'], z['hemi'], z['runs']
        for r in REGIONS:
            for h in ('L', 'R', 'both'):
                m = (reg == r) & ((hemi == h) if h != 'both' else True)
                x = bold[:, :, m]
                x = x[:, :, (x.std(1) > 0).all(0)]
                sig = x.mean(2)                                     # (runs, T)
                for k, (ses, run) in enumerate(runs):
                    rows.append(pd.DataFrame(dict(subject=s, region=r, hemi=h, session=int(ses), run=int(run),
                                                  tr_index=np.arange(sig.shape[1]), signal=sig[k],
                                                  n_voxels=x.shape[2])))
        print('sub', s, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out, sep='\t', index=False, float_format='%.6g')
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--subjects', type=int, nargs='+', default=list(range(3, 31)))
    a = ap.parse_args()
    main(a.bids_folder, a.subjects)
