"""Sub-02 V1 smoke test for ``StimulusFitter``-based stimulus decoding.

Run::

    ~/mambaforge/envs/retsupp/bin/python -m retsupp.decode.smoke_test \\
        --bids-folder /data/ds-retsupp \\
        --resolution 30 --max-voxels 200 \\
        --max-n-iterations 1000

Outputs
-------
* ``notes/figures/decoded_smoke_sub-02_V1.pdf`` -- 12-panel grid of
  decoded stimulus maps over the run, with the (centre of bar) overlay
  if the bar is visible at that frame.
* ``notes/data/decoded_smoke_sub-02_V1_ring.tsv`` -- per-frame disk
  averages at the 4 ring positions plus HP/LP labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from retsupp.utils.data import Subject
from retsupp.decode.decoder import (
    decode_run, sample_at_ring_positions, label_hp_lp,
)


def main(bids_folder: str = '/data/ds-retsupp',
         subject: int = 2, roi: str = 'V1',
         session: int = 1, run: int = 1,
         resolution: int = 30, max_voxels: int = 200,
         l2_norm: float = 0.01, learning_rate: float = 0.01,
         max_n_iterations: int = 1000,
         resid_max_iter: int = 500,
         out_fig: str = 'notes/figures/decoded_smoke_sub-02_V1.pdf',
         out_tsv: str = 'notes/data/decoded_smoke_sub-02_V1_ring.tsv',
         repo_root: str | None = None):
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    out_fig_p = repo / out_fig
    out_tsv_p = repo / out_tsv
    out_fig_p.parent.mkdir(parents=True, exist_ok=True)
    out_tsv_p.parent.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder=bids_folder)

    print(f'== Smoke test: sub-{subject:02d} {roi} ses-{session} run-{run} ==')

    decoded, grid, voxel_idx, omega, pars_df, bold = decode_run(
        sub, session=session, run=run, roi=roi,
        resolution=resolution, max_voxels=max_voxels,
        l2_norm=l2_norm, learning_rate=learning_rate,
        max_n_iterations=max_n_iterations,
        resid_max_iter=resid_max_iter,
        progressbar=True, verbose=True)

    # Figure: 12 frames spanning the run.
    n_frames = decoded.shape[0]
    pick = np.linspace(0, n_frames - 1, 12).astype(int)
    res = int(np.sqrt(grid.shape[0]))
    assert res * res == grid.shape[0], 'Grid not square'

    extent = [grid[:, 0].min(), grid[:, 0].max(),
              grid[:, 1].min(), grid[:, 1].max()]

    decoded_arr = decoded.values.reshape(n_frames, res, res)
    # Stimulus needs the same reshape; rebuild to overlay.
    par = sub.get_stimulus(session=session, run=run,
                           resolution=resolution).astype(np.float32)
    # par shape: (T, R, R). decoded shape: (T, R*R). The stimulus
    # function returns (T, R, R) indexed [t, y_row, x_col] where y_row=0
    # is top of array. The grid coordinates put y=+max at row 0 too, so
    # we just need imshow with origin='lower' to align with x,y.
    par = par[:n_frames]

    vmax = float(np.quantile(decoded_arr, 0.99))
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5),
                             sharex=True, sharey=True)
    for ax, t in zip(axes.flat, pick):
        ax.imshow(decoded_arr[t], extent=extent, origin='lower',
                  cmap='magma', vmin=0, vmax=vmax)
        # True bar in cyan contour.
        if par[t].max() > 0:
            ax.contour(par[t], levels=[0.5], colors='cyan',
                       linewidths=1.0, origin='lower', extent=extent)
        # Ring positions in white circles.
        ecc = 4.0
        for cx, cy in [(+ecc/np.sqrt(2), +ecc/np.sqrt(2)),
                       (-ecc/np.sqrt(2), +ecc/np.sqrt(2)),
                       (+ecc/np.sqrt(2), -ecc/np.sqrt(2)),
                       (-ecc/np.sqrt(2), -ecc/np.sqrt(2))]:
            ax.plot(cx, cy, 'wo', mfc='none', mec='white', ms=10, mew=1.0)
        ax.set_title(f'TR {t}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f'StimulusFitter decoded stimulus  sub-{subject:02d}  {roi}  '
        f'ses-{session} run-{run}\n(cyan=true bar, white circles=ring positions)',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_fig_p, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote: {out_fig_p}')

    # Per-frame ring-position drives.
    per_ring = sample_at_ring_positions(decoded, grid, ring_disk_radius=0.4)
    hp_for_runs = sub.get_hpd_locations()
    hp_loc = hp_for_runs[(session, run)]
    print(f'  HP location for ses-{session} run-{run}: {hp_loc}')
    per_ring = label_hp_lp(per_ring, hp_loc)
    per_ring['subject'] = subject
    per_ring['roi'] = roi
    per_ring['session'] = session
    per_ring['run'] = run
    per_ring.to_csv(out_tsv_p, sep='\t', index=False)
    print(f'Wrote: {out_tsv_p}')

    # Quick sanity summary.
    summary = (per_ring.groupby('hp_role')['decoded']
                       .agg(['mean', 'std', 'count']))
    print('\nMean decoded drive at ring positions (one run):')
    print(summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subject', type=int, default=2)
    p.add_argument('--roi', default='V1')
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--run', type=int, default=1)
    p.add_argument('--resolution', type=int, default=30)
    p.add_argument('--max-voxels', type=int, default=200)
    p.add_argument('--l2-norm', type=float, default=0.01)
    p.add_argument('--learning-rate', type=float, default=0.01)
    p.add_argument('--max-n-iterations', type=int, default=1000)
    p.add_argument('--resid-max-iter', type=int, default=500)
    a = p.parse_args()
    main(bids_folder=a.bids_folder, subject=a.subject, roi=a.roi,
         session=a.session, run=a.run, resolution=a.resolution,
         max_voxels=a.max_voxels, l2_norm=a.l2_norm,
         learning_rate=a.learning_rate,
         max_n_iterations=a.max_n_iterations,
         resid_max_iter=a.resid_max_iter)
