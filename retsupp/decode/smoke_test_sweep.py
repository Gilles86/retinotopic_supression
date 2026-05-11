"""Hyperparameter sweep for StimulusFitter on a single (subject, ROI, run).

Sweeps (l2_norm, learning_rate) on a grid with voxel selection fixed to a
tight set (PRF eccentricity <= 3.0 deg, r2 >= 0.1, top-100 by r2). The
default cell grid is l2 in {0.01, 0.05, 0.1, 0.5} x lr in {0.01, 0.05, 0.1}.

For each cell:
- Runs ``decode_run`` and saves the decoded tensor as an npz under
  ``notes/data/decode_sweep/<tag>/l2-{l2}_lr-{lr}.npz``.
- Computes summary metrics into ``metrics.tsv`` in the same directory:
    max, min, p99, p1, std            -- detect blow-ups
    signal, noise, snr (signal-noise) -- decoded value at stim vs off pixels
    corr                              -- corr(decoded, paradigm) across all (t, pixel)
    frac_pos_off                      -- fraction of off-pixels with decoded > 0.1

- Writes a grid summary PDF (rows = l2_norm, cols = learning_rate) showing
  the decoded map at the frame with maximum bar coverage, with the true
  paradigm (bar + distractor rectangles) as a cyan contour overlay.

Run::

    ~/mambaforge/envs/retsupp/bin/python -m retsupp.decode.smoke_test_sweep \\
        --bids-folder /data/ds-retsupp
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from retsupp.utils.data import Subject
from retsupp.decode.decoder import decode_run


VOXEL_SD_MIN = 0.05
VOXEL_R2_MIN = 0.1
VOXEL_ECC_MAX = 3.0
VOXEL_MAX = 100


def compute_metrics(decoded_arr: np.ndarray, par: np.ndarray) -> dict:
    """decoded_arr: (T, R, R) decoded intensity; par: (T, R, R) paradigm in [0,1]."""
    stim_mask = par > 0.5
    off_mask = ~stim_mask
    if stim_mask.sum() == 0:
        signal = np.nan
    else:
        signal = float(decoded_arr[stim_mask].mean())
    if off_mask.sum() == 0:
        noise = np.nan
    else:
        noise = float(decoded_arr[off_mask].mean())
    flat_d = decoded_arr.reshape(-1)
    flat_p = par.reshape(-1)
    if flat_p.std() > 0:
        corr = float(np.corrcoef(flat_d, flat_p)[0, 1])
    else:
        corr = float('nan')
    return dict(
        max=float(decoded_arr.max()),
        min=float(decoded_arr.min()),
        p99=float(np.quantile(decoded_arr, 0.99)),
        p1=float(np.quantile(decoded_arr, 0.01)),
        std=float(decoded_arr.std()),
        signal=signal,
        noise=noise,
        snr=signal - noise,
        corr=corr,
        frac_pos_off=float((decoded_arr[off_mask] > 0.1).mean())
            if off_mask.sum() > 0 else float('nan'),
    )


def build_paradigm_for_metrics(sub: Subject, session: int, run: int,
                               resolution: int) -> np.ndarray:
    """Same paradigm as decode_run uses internally, on the extended (5deg) grid."""
    par = sub.get_stimulus_with_distractors(
        session=session, run=run, resolution=resolution,
        grid_radius=5.0, distractor_shape='rectangle',
        distractor_long_side=1.5, distractor_short_side=0.375,
    ).astype(np.float32)
    return par  # (T, R, R)


def pick_max_bar_frame(par: np.ndarray) -> int:
    """Frame with max paradigm coverage (most pixels on)."""
    return int(par.reshape(par.shape[0], -1).sum(axis=1).argmax())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--subject', type=int, default=2)
    p.add_argument('--roi', default='V1')
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--run', type=int, default=1)
    p.add_argument('--resolution', type=int, default=30)
    p.add_argument('--l2-norms', type=float, nargs='+',
                   default=[0.01, 0.05, 0.1, 0.5])
    p.add_argument('--learning-rates', type=float, nargs='+',
                   default=[0.01, 0.05, 0.1])
    p.add_argument('--max-n-iterations', type=int, default=1000)
    p.add_argument('--resid-max-iter', type=int, default=300)
    p.add_argument('--repo-root', type=Path,
                   default=Path(__file__).resolve().parents[2])
    args = p.parse_args()

    sub = Subject(args.subject, bids_folder=args.bids_folder)
    tag = (f'sub-{args.subject:02d}_{args.roi}_'
           f'ses-{args.session}_run-{args.run}')
    data_dir = args.repo_root / 'notes' / 'data' / 'decode_sweep' / tag
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.repo_root / 'notes' / 'figures' / 'decode_sweep'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f'{tag}.pdf'
    metrics_path = data_dir / 'metrics.tsv'

    par = build_paradigm_for_metrics(sub, args.session, args.run,
                                     args.resolution)
    print(f'Paradigm: {par.shape}', flush=True)
    print(f'Voxel selection: ecc<={VOXEL_ECC_MAX}, r2>={VOXEL_R2_MIN}, '
          f'top-{VOXEL_MAX}', flush=True)

    cells = list(itertools.product(args.l2_norms, args.learning_rates))
    print(f'Sweep cells: {len(cells)} '
          f'(l2={args.l2_norms} x lr={args.learning_rates})\n', flush=True)

    rows = []
    decoded_per_cell: dict[tuple[float, float], np.ndarray] = {}
    grid_saved = None
    for i, (l2, lr) in enumerate(cells):
        cell_tag = f'l2-{l2}_lr-{lr}'
        print(f'\n=== [{i + 1}/{len(cells)}] {cell_tag} ===', flush=True)
        t0 = time.time()
        decoded, grid, voxel_idx, omega, pars_df, bold = decode_run(
            sub, session=args.session, run=args.run, roi=args.roi,
            resolution=args.resolution, max_voxels=VOXEL_MAX,
            sd_min=VOXEL_SD_MIN, r2_min=VOXEL_R2_MIN,
            ecc_max=VOXEL_ECC_MAX,
            l2_norm=l2, learning_rate=lr,
            max_n_iterations=args.max_n_iterations,
            resid_max_iter=args.resid_max_iter,
            progressbar=False, verbose=True)
        dt = time.time() - t0
        T = decoded.shape[0]
        R = int(np.sqrt(grid.shape[0]))
        assert R * R == grid.shape[0], 'Grid not square'
        decoded_arr = decoded.values.reshape(T, R, R)
        par_t = par[:T]

        m = compute_metrics(decoded_arr, par_t)
        m.update(dict(l2=l2, lr=lr, n_voxels=int(len(voxel_idx)),
                      seconds=round(dt, 1)))
        rows.append(m)
        print('  metrics:',
              {k: round(v, 4) for k, v in m.items()
               if isinstance(v, float) and not np.isnan(v)})

        np.savez_compressed(
            data_dir / f'{cell_tag}.npz',
            decoded=decoded_arr.astype(np.float32),
            paradigm=par_t.astype(np.float32),
            grid=grid.astype(np.float32),
        )
        decoded_per_cell[(l2, lr)] = decoded_arr
        grid_saved = grid

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[['l2', 'lr', 'n_voxels', 'seconds',
                              'max', 'min', 'p99', 'p1', 'std',
                              'signal', 'noise', 'snr', 'corr',
                              'frac_pos_off']]
    metrics_df.to_csv(metrics_path, sep='\t', index=False)
    print(f'\nWrote: {metrics_path}')
    with pd.option_context('display.width', 200,
                           'display.max_columns', None):
        print(metrics_df.round(3))

    # Grid summary PDF: panels = (l2 row, lr col), showing max-bar-coverage frame.
    frame = pick_max_bar_frame(par)
    print(f'\nShowing frame {frame} (max paradigm coverage).')
    extent = [float(grid_saved[:, 0].min()), float(grid_saved[:, 0].max()),
              float(grid_saved[:, 1].min()), float(grid_saved[:, 1].max())]
    nrow = len(args.l2_norms)
    ncol = len(args.learning_rates)
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(3.4 * ncol, 3.4 * nrow),
                              sharex=True, sharey=True, squeeze=False)
    for i, l2 in enumerate(args.l2_norms):
        for j, lr in enumerate(args.learning_rates):
            ax = axes[i, j]
            arr = decoded_per_cell[(l2, lr)][frame]
            # Choose vmax from the *cell* itself so blow-ups stay visible,
            # but cap to 5x the paradigm scale so a single runaway cell
            # doesn't compress the others to invisibility.
            vmax = max(float(np.quantile(arr, 0.99)), 0.2)
            ax.imshow(arr, extent=extent, origin='lower',
                      cmap='magma', vmin=0, vmax=vmax)
            ax.contour(par[frame], levels=[0.5], colors='cyan',
                       linewidths=0.9, origin='lower', extent=extent)
            mrow = metrics_df[(metrics_df.l2 == l2)
                              & (metrics_df.lr == lr)].iloc[0]
            ax.set_title(
                f'L2={l2}, lr={lr}\n'
                f'Max={mrow["max"]:.2f}  P99={mrow["p99"]:.2f}\n'
                f'SNR={mrow["snr"]:.3f}  Corr={mrow["corr"]:.2f}',
                fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f'Decode sweep  {tag}  (frame={frame}, '
                  f'cyan=true paradigm)',
                  fontsize=13, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote: {fig_path}')


if __name__ == '__main__':
    main()
