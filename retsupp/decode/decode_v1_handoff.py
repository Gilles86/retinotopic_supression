"""V1 stimulus decode from the m4 handoff NPZ for one HP condition.

Reads ``derivatives/v1_decode_handoffs/sub-{NN}_m4_V1.npz`` (m4 DoG +
flex-HRF PRF params + cleaned BOLD across all 12 runs, voxel-aligned),
averages BOLD across the runs of one HP-distractor condition (typically
3 runs), builds the matching averaged bar-only paradigm via
:meth:`Subject.get_bar_stimulus`, and decodes the per-TR stimulus with
braincoder's StimulusFitter.

Averaging before decoding (rather than decoding then averaging) is the
right move: HRF convolution couples TRs in the StimulusFitter cost, so
per-condition decodes need self-contained HRF-coupled BOLD.

Usage::

    python -m retsupp.decode.decode_v1_handoff --subject 15 --hp upper_right

Output::

    notes/data/v1_decode/sub-{NN}/decoded_hp-{HP}[_vox{N}].npz
        decoded   (T, R, R)   one HP condition's averaged decode
        paradigm  (T, R, R)   matched averaged paradigm
        grid      (G, 2)
        voxel_idx (V_sel,)    brain-mask flat indices (for back-projection)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.utils.data import (
    Subject, select_well_fit_voxels, validate_prf_parameters,
)


M4_PARS = ['x', 'y', 'sd', 'baseline', 'amplitude',
           'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion']
M4_SD_MIN = 0.2
N_TR_PER_RUN = 258
HP_CHOICES = ('upper_right', 'upper_left', 'lower_left', 'lower_right')


def _run_position_in_concat(sub: Subject) -> dict[tuple[int, int], int]:
    """(session, run) → row index in the concatenated 12-run BOLD."""
    return {sr: i for i, sr in enumerate(
        (ses, run) for ses in (1, 2) for run in sub.get_runs(ses))}


def decode(*, subject: int, hp: str, bids_folder: str,
           l2_norm: float, learning_rate: float,
           max_n_iterations: int, min_n_iterations: int,
           resid_max_iter: int, max_voxels: int | None,
           out_path: Path):
    from braincoder.hrf import SPMHRFModel
    from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
    from braincoder.optimize import ResidualFitter, StimulusFitter

    npz_path = (Path(bids_folder) / 'derivatives' / 'v1_decode_handoffs'
                / f'sub-{subject:02d}_m4_V1.npz')
    print(f'Loading {npz_path}', flush=True)
    with np.load(npz_path, allow_pickle=True) as f:
        npz = {k: f[k] for k in f.keys()}

    resolution = int(npz['resolution'])
    grid_radius = float(npz['grid_radius'])
    tr = float(npz['tr'])
    bold = npz['bold'].astype(np.float32)
    T_all, V = bold.shape
    if T_all % N_TR_PER_RUN != 0:
        raise RuntimeError(f'T={T_all} not divisible by {N_TR_PER_RUN}.')
    n_runs = T_all // N_TR_PER_RUN
    bold_runs = bold.reshape(n_runs, N_TR_PER_RUN, V)
    print(f'  V1 voxels: {V}  runs: {n_runs}  TRs/run: {N_TR_PER_RUN}  '
          f'res: {resolution}  grid_radius: {grid_radius}°', flush=True)

    sub = Subject(subject, bids_folder=bids_folder)
    hp_runs = sub.get_runs_by_hp(hp)
    run_pos = _run_position_in_concat(sub)
    concat_idxs = [run_pos[sr] for sr in hp_runs]
    print(f'  HP={hp}: {hp_runs}  (concat rows {concat_idxs})', flush=True)

    # Canonical voxel filter, n_timepoints = TRs in the averaged chunk.
    pars_all = pd.DataFrame({p: npz[p] for p in M4_PARS}).astype(np.float32)
    pars_all['r2'] = npz['r2'].astype(np.float32)
    sel, fdr_thr = select_well_fit_voxels(
        pars_all, n_params=len(M4_PARS), n_timepoints=N_TR_PER_RUN,
        aperture_radius=float(npz['aperture_radius']))
    if max_voxels is not None and len(sel) > max_voxels:
        sel = sel.sort_values('r2', ascending=False).head(max_voxels)
    npz_rows = sel.index.to_numpy()
    sel = sel.reset_index(drop=True)
    print(f'  Selected voxels: {len(sel)} / {V}  '
          f'(FDR r²>{fdr_thr:.3f}, mass>0.5, 0.3≤sd≤4)', flush=True)
    if len(sel) < 20:
        raise RuntimeError(f'Too few voxels selected: {len(sel)}')

    pars_df = sel[M4_PARS].astype(np.float32)
    validate_prf_parameters(pars_df, sd_min=M4_SD_MIN, model_label='m4',
                             source=str(npz_path))

    # Average BOLD and paradigm across this HP's runs.
    bold_avg = bold_runs[concat_idxs][:, :, npz_rows].mean(axis=0)
    par_runs = np.stack([
        sub.get_bar_stimulus(session=ses, run=run,
                              resolution=resolution, grid_radius=grid_radius)
        for ses, run in hp_runs
    ], axis=0)
    paradigm_avg = par_runs.mean(axis=0).astype(np.float32)
    T = paradigm_avg.shape[0]
    print(f'  averaged BOLD: {bold_avg.shape}  '
          f'averaged paradigm: {paradigm_avg.shape}', flush=True)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    paradigm_flat = paradigm_avg.reshape(T, -1).astype(np.float32)
    bold_df = pd.DataFrame(bold_avg.astype(np.float32),
                            index=pd.Index(np.arange(T), name='frame'))
    bold_df.columns.name = 'voxel'
    paradigm_df = pd.DataFrame(paradigm_flat,
                                index=pd.Index(np.arange(T), name='frame'))

    hrf = SPMHRFModel(tr=tr, delay=4.5, dispersion=0.75)

    def mk_model():
        return DifferenceOfGaussiansPRF2DWithHRF(
            grid_coordinates=grid, paradigm=paradigm_flat,
            hrf_model=hrf, data=bold_df, parameters=pars_df,
            flexible_hrf_parameters=True, sd_min=M4_SD_MIN)

    print('Fitting residual covariance...', flush=True)
    t0 = time.time()
    rf = ResidualFitter(model=mk_model(), data=bold_df, paradigm=paradigm_df,
                        parameters=pars_df)
    omega, _ = rf.fit(max_n_iterations=resid_max_iter, progressbar=True)
    print(f'  omega: {omega.shape}  ({time.time() - t0:.0f}s)', flush=True)

    print('Decoding stimulus...', flush=True)
    t0 = time.time()
    sf = StimulusFitter(model=mk_model(), data=bold_df, omega=omega,
                        parameters=pars_df)
    decoded = sf.fit(l2_norm=l2_norm, learning_rate=learning_rate,
                     max_n_iterations=max_n_iterations,
                     min_n_iterations=min_n_iterations, progressbar=True)
    decoded_arr = decoded.values.reshape(T, resolution, resolution).astype(np.float32)
    print(f'  decoded: {decoded_arr.shape}  ({time.time() - t0:.0f}s)',
          flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        decoded=decoded_arr,
        paradigm=paradigm_avg,
        grid=grid,
        voxel_idx=npz['voxel_idx'][npz_rows],
        hp=np.array(hp),
        runs=np.array(hp_runs),
        fdr_thr=np.float32(fdr_thr),
        l2_norm=np.float32(l2_norm), learning_rate=np.float32(learning_rate),
        resolution=np.int32(resolution),
        grid_radius=np.float32(grid_radius),
        tr=np.float32(tr),
    )
    print(f'Wrote: {out_path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--hp', required=True, choices=list(HP_CHOICES))
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--l2-norm', type=float, default=0.01)
    p.add_argument('--learning-rate', type=float, default=0.5)
    p.add_argument('--max-n-iterations', type=int, default=1000)
    p.add_argument('--min-n-iterations', type=int, default=200)
    p.add_argument('--resid-max-iter', type=int, default=300)
    p.add_argument('--max-voxels', type=int, default=None,
                   help='Optional cap on # voxels after FDR (top-N by r²).')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()

    tag = f'hp-{args.hp}'
    if args.max_voxels is not None:
        tag = f'{tag}_vox{args.max_voxels}'
    out_path = args.out or (Path(__file__).resolve().parents[2]
                            / 'notes' / 'data' / 'v1_decode'
                            / f'sub-{args.subject:02d}'
                            / f'decoded_{tag}.npz')
    decode(subject=args.subject, hp=args.hp, bids_folder=args.bids_folder,
           l2_norm=args.l2_norm, learning_rate=args.learning_rate,
           max_n_iterations=args.max_n_iterations,
           min_n_iterations=args.min_n_iterations,
           resid_max_iter=args.resid_max_iter,
           max_voxels=args.max_voxels, out_path=out_path)


if __name__ == '__main__':
    main()
