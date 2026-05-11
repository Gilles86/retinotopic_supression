"""Hyperparameter sweep for StimulusFitter on a single (subject, ROI, run).

Sweeps (l2_norm, learning_rate) on a grid with voxel selection fixed to a
tight set (PRF eccentricity <= 3.0 deg, r2 >= 0.1, top-100 by r2). The
default cell grid is l2 in {0.01, 0.05, 0.1, 0.5} x lr in {0.01, 0.05, 0.1}.

This script is split into two modes:

1. **Cell mode** (default): decode one or more (l2, lr) cells.
   Pass ``--l2-norms`` and ``--learning-rates`` to control which cells run.
   For each cell, writes:
       <data_dir>/l2-{l2}_lr-{lr}.npz          -- decoded + paradigm + grid
       <data_dir>/l2-{l2}_lr-{lr}.tsv          -- single-row metrics

2. **Aggregate mode** (``--aggregate-only``): scan ``<data_dir>`` for
   per-cell TSVs and npzs, concatenate to ``metrics.tsv``, and render the
   grid summary PDF ``<fig_dir>/<tag>.pdf``.

The split lets SLURM array jobs run one cell per task (12-way parallelism)
and a separate aggregator step build the final figure.

Run examples::

    # Single cell (one SLURM array task)
    python -m retsupp.decode.smoke_test_sweep \\
        --subject 2 --roi V1 --session 1 --run 1 \\
        --l2-norms 0.05 --learning-rates 0.01

    # Aggregate after all cells are done
    python -m retsupp.decode.smoke_test_sweep \\
        --subject 2 --roi V1 --session 1 --run 1 --aggregate-only

    # Local sequential sweep (no SLURM)
    python -m retsupp.decode.smoke_test_sweep --bids-folder /data/ds-retsupp
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

METRIC_COLS = ['l2', 'lr', 'n_voxels', 'seconds',
               'max', 'min', 'p99', 'p1', 'std',
               'signal', 'noise', 'snr', 'corr',
               'frac_pos_off']


def cell_tag(l2: float, lr: float) -> str:
    return f'l2-{l2}_lr-{lr}'


def compute_metrics(decoded_arr: np.ndarray, par: np.ndarray) -> dict:
    """decoded_arr: (T, R, R) decoded intensity; par: (T, R, R) paradigm in [0,1]."""
    stim_mask = par > 0.5
    off_mask = ~stim_mask
    signal = (float(decoded_arr[stim_mask].mean()) if stim_mask.sum() > 0
              else float('nan'))
    noise = (float(decoded_arr[off_mask].mean()) if off_mask.sum() > 0
             else float('nan'))
    flat_d = decoded_arr.reshape(-1)
    flat_p = par.reshape(-1)
    corr = (float(np.corrcoef(flat_d, flat_p)[0, 1])
            if flat_p.std() > 0 else float('nan'))
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
        frac_pos_off=(float((decoded_arr[off_mask] > 0.1).mean())
                      if off_mask.sum() > 0 else float('nan')),
    )


def build_paradigm_for_metrics(sub: Subject, session: int, run: int,
                               resolution: int) -> np.ndarray:
    """Same paradigm as decode_run uses internally (on the extended 5deg grid)."""
    par = sub.get_stimulus_with_distractors(
        session=session, run=run, resolution=resolution,
        grid_radius=5.0, distractor_shape='rectangle',
        distractor_long_side=1.5, distractor_short_side=0.375,
    ).astype(np.float32)
    return par  # (T, R, R)


def pick_max_bar_frame(par: np.ndarray) -> int:
    """Frame with max paradigm coverage (most pixels on)."""
    return int(par.reshape(par.shape[0], -1).sum(axis=1).argmax())


def run_one_cell(sub: Subject, *, roi: str, session: int, run: int,
                 resolution: int, l2: float, lr: float,
                 max_n_iterations: int, resid_max_iter: int,
                 par: np.ndarray, data_dir: Path) -> dict:
    """Decode one (l2, lr) cell and persist npz + 1-row TSV. Returns the row."""
    tag = cell_tag(l2, lr)
    print(f'\n=== {tag} ===', flush=True)
    t0 = time.time()
    decoded, grid, voxel_idx, omega, pars_df, bold = decode_run(
        sub, session=session, run=run, roi=roi,
        resolution=resolution, max_voxels=VOXEL_MAX,
        sd_min=VOXEL_SD_MIN, r2_min=VOXEL_R2_MIN,
        ecc_max=VOXEL_ECC_MAX,
        l2_norm=l2, learning_rate=lr,
        max_n_iterations=max_n_iterations,
        resid_max_iter=resid_max_iter,
        progressbar=False, verbose=True)
    dt = time.time() - t0
    T = decoded.shape[0]
    R = int(np.sqrt(grid.shape[0]))
    assert R * R == grid.shape[0], 'Grid not square'
    decoded_arr = decoded.values.reshape(T, R, R)
    par_t = par[:T]

    m = compute_metrics(decoded_arr, par_t)
    m.update(dict(l2=l2, lr=lr,
                  n_voxels=int(len(voxel_idx)),
                  seconds=round(dt, 1)))
    print('  metrics:',
          {k: round(v, 4) for k, v in m.items()
           if isinstance(v, float) and not np.isnan(v)})

    np.savez_compressed(
        data_dir / f'{tag}.npz',
        decoded=decoded_arr.astype(np.float32),
        paradigm=par_t.astype(np.float32),
        grid=grid.astype(np.float32),
    )
    pd.DataFrame([m])[METRIC_COLS].to_csv(
        data_dir / f'{tag}.tsv', sep='\t', index=False)
    return m


def aggregate(data_dir: Path, fig_path: Path, par: np.ndarray, tag_str: str,
              l2_norms: list[float], learning_rates: list[float]) -> None:
    """Concatenate per-cell TSVs + npzs into metrics.tsv and grid PDF.

    Missing cells (no TSV on disk) are skipped with a warning; aggregation
    still proceeds with whatever is present, but the grid PDF panel for
    missing cells is left blank.
    """
    rows = []
    decoded_per_cell: dict[tuple[float, float], np.ndarray] = {}
    grid_saved = None
    missing = []
    for l2, lr in itertools.product(l2_norms, learning_rates):
        tsv = data_dir / f'{cell_tag(l2, lr)}.tsv'
        npz = data_dir / f'{cell_tag(l2, lr)}.npz'
        if not tsv.exists() or not npz.exists():
            missing.append((l2, lr))
            continue
        rows.append(pd.read_csv(tsv, sep='\t'))
        with np.load(npz) as f:
            decoded_per_cell[(l2, lr)] = f['decoded']
            grid_saved = f['grid']
    if missing:
        print(f'  WARN: {len(missing)} cells missing from disk: '
              f'{missing}', flush=True)
    if not rows:
        raise RuntimeError(f'No per-cell results found in {data_dir}')

    metrics_df = pd.concat(rows, axis=0, ignore_index=True)
    metrics_df = metrics_df[METRIC_COLS].sort_values(['l2', 'lr'])
    metrics_path = data_dir / 'metrics.tsv'
    metrics_df.to_csv(metrics_path, sep='\t', index=False)
    print(f'Wrote: {metrics_path}')
    with pd.option_context('display.width', 200,
                           'display.max_columns', None):
        print(metrics_df.round(3))

    if grid_saved is None:
        print('No npz cells available; skipping grid PDF.')
        return

    frame = pick_max_bar_frame(par)
    print(f'\nShowing frame {frame} (max paradigm coverage).')
    extent = [float(grid_saved[:, 0].min()), float(grid_saved[:, 0].max()),
              float(grid_saved[:, 1].min()), float(grid_saved[:, 1].max())]
    nrow = len(l2_norms)
    ncol = len(learning_rates)
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(3.4 * ncol, 3.4 * nrow),
                              sharex=True, sharey=True, squeeze=False)
    for i, l2 in enumerate(l2_norms):
        for j, lr in enumerate(learning_rates):
            ax = axes[i, j]
            if (l2, lr) not in decoded_per_cell:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            arr = decoded_per_cell[(l2, lr)][frame]
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
    fig.suptitle(f'Decode sweep  {tag_str}  (frame={frame}, '
                  f'cyan=true paradigm)',
                  fontsize=13, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote: {fig_path}')


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
    p.add_argument('--aggregate-only', action='store_true',
                   help='Skip running cells; read existing per-cell TSVs/npzs '
                        'from <data_dir> and build metrics.tsv + grid PDF.')
    p.add_argument('--full-grid', action='store_true',
                   help='When aggregating, use the *default* full grid '
                        '(0.01/0.05/0.1/0.5 x 0.01/0.05/0.1) for the figure '
                        'layout regardless of --l2-norms / --learning-rates.')
    args = p.parse_args()

    sub = Subject(args.subject, bids_folder=args.bids_folder)
    tag = (f'sub-{args.subject:02d}_{args.roi}_'
           f'ses-{args.session}_run-{args.run}')
    data_dir = args.repo_root / 'notes' / 'data' / 'decode_sweep' / tag
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_path = (args.repo_root / 'notes' / 'figures' / 'decode_sweep'
                / f'{tag}.pdf')

    par = build_paradigm_for_metrics(sub, args.session, args.run,
                                     args.resolution)
    print(f'Paradigm: {par.shape}', flush=True)

    if args.full_grid:
        agg_l2 = [0.01, 0.05, 0.1, 0.5]
        agg_lr = [0.01, 0.05, 0.1]
    else:
        agg_l2 = list(args.l2_norms)
        agg_lr = list(args.learning_rates)

    if args.aggregate_only:
        print(f'Aggregating over l2={agg_l2}, lr={agg_lr}', flush=True)
        aggregate(data_dir, fig_path, par, tag, agg_l2, agg_lr)
        return

    print(f'Voxel selection: ecc<={VOXEL_ECC_MAX}, r2>={VOXEL_R2_MIN}, '
          f'top-{VOXEL_MAX}', flush=True)
    cells = list(itertools.product(args.l2_norms, args.learning_rates))
    print(f'Sweep cells: {len(cells)} '
          f'(l2={args.l2_norms} x lr={args.learning_rates})\n', flush=True)

    for i, (l2, lr) in enumerate(cells):
        print(f'\n--- [{i + 1}/{len(cells)}] ---', flush=True)
        run_one_cell(sub, roi=args.roi, session=args.session, run=args.run,
                     resolution=args.resolution, l2=l2, lr=lr,
                     max_n_iterations=args.max_n_iterations,
                     resid_max_iter=args.resid_max_iter,
                     par=par, data_dir=data_dir)

    # If the local run covered the requested grid, also produce aggregation.
    print('\nAggregating completed cells...', flush=True)
    aggregate(data_dir, fig_path, par, tag, agg_l2, agg_lr)


if __name__ == '__main__':
    main()
