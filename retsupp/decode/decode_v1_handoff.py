"""V1 stimulus decode from the m4 handoff NPZ, averaged per HP condition.

For each of the 4 HP-distractor conditions, we average the cleaned BOLD
across that condition's runs (3 runs per HP for the canonical 12-run
subjects), build the matching averaged bar-only paradigm, and decode the
per-TR stimulus with braincoder's StimulusFitter. Averaging before
decoding (rather than decoding then averaging) is the right move because
the model convolves the stimulus with the HRF — the per-TR decodes are
coupled.

Output::

    notes/data/v1_decode/sub-{NN}/decoded_per_hp.npz
        decoded   (4, 258, R, R)  one movie per HP condition
        paradigm  (4, 258, R, R)  matched averaged paradigm
        grid      (G, 2)
        hp_labels (4,)  ['upper_right', 'upper_left', 'lower_left', 'lower_right']
        voxel_idx (V_sel,)        brain-mask flat indices of selected voxels
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
HP_LABELS = ('upper_right', 'upper_left', 'lower_left', 'lower_right')


def _bar_run(sub: Subject, session: int, run: int,
             resolution: int, grid_radius: float) -> np.ndarray:
    """Bar-only stimulus for one run on a ±grid_radius° grid.

    Mirrors the bar pass in ``Subject.get_stimulus_with_distractors``
    (which already supports the extended grid) but skips the distractor
    pass.
    """
    settings = sub.get_experimental_settings(session, run)
    tr = sub.get_tr(session, run)
    n_volumes = sub.get_n_volumes(session, run)
    frametimes = np.arange(tr / 2., tr * n_volumes + tr / 2., tr)

    bar_aperture = settings['radius_bar_aperture']
    bar_width = settings['bar_width']
    speed = settings['speed']
    fov_size = settings['fov_size']

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=session, run=run,
        grid_radius=grid_radius)
    aperture = np.sqrt(gx ** 2 + gy ** 2) <= bar_aperture

    onsets = sub.get_onsets(session, run)
    bar_events = onsets[onsets['event_type'].apply(lambda s: s.startswith('bar'))]

    stim = np.zeros((len(frametimes), resolution, resolution), dtype=np.float32)
    ori, pos = 0, -fov_size - bar_width

    for i, t in enumerate(frametimes):
        if t < bar_events['onset'].min():
            continue
        state_row = bar_events[bar_events['onset'] < t].iloc[-1]
        state = state_row['event_type']
        dt = t - state_row['onset']

        if state in ('bar_rest', 'bar_break'):
            continue
        if state == 'bar_right':
            ori, pos = 0, -bar_aperture - bar_width / 2 + dt * speed
        elif state == 'bar_left':
            ori, pos = 0, bar_aperture + bar_width / 2 - dt * speed
        elif state == 'bar_up':
            ori, pos = 90, -bar_aperture - bar_width / 2 + dt * speed
        elif state == 'bar_down':
            ori, pos = 90, bar_aperture + bar_width / 2 - dt * speed
        else:
            continue

        frame = np.zeros_like(gx, dtype=np.float32)
        if ori == 0:
            frame[np.abs(gx - pos) < bar_width / 2] = 1.0
        else:
            frame[np.abs(gy - pos) < bar_width / 2] = 1.0
        stim[i] = frame * aperture
    return stim


def _group_runs_by_hp(sub: Subject):
    """Return ``({hp: [run_pos_in_concat, ...]}, [(ses, run), ...])``.

    ``run_pos_in_concat`` is the row index after reshaping concatenated
    BOLD to (n_runs, 258, V) — matches the iteration order
    ses-1 run-1..6, ses-2 run-1..6 used when building the handoff NPZ.
    """
    hpd = sub.get_hpd_locations()
    run_index = [(ses, run) for ses in (1, 2) for run in sub.get_runs(ses)]
    hp_groups: dict[str, list[int]] = {}
    for i, sr in enumerate(run_index):
        hp_groups.setdefault(hpd[sr], []).append(i)
    return hp_groups, run_index


def _fit_one_hp(bold_avg: np.ndarray, paradigm_avg: np.ndarray,
                pars_df: pd.DataFrame, grid: np.ndarray, *,
                tr: float, l2_norm: float, learning_rate: float,
                max_n_iterations: int, min_n_iterations: int,
                resid_max_iter: int):
    """Run ResidualFitter + StimulusFitter for one (T=258, V) BOLD chunk."""
    from braincoder.hrf import SPMHRFModel
    from braincoder.models import DifferenceOfGaussiansPRF2DWithHRF
    from braincoder.optimize import ResidualFitter, StimulusFitter

    T = bold_avg.shape[0]
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

    rf = ResidualFitter(model=mk_model(), data=bold_df, paradigm=paradigm_df,
                        parameters=pars_df)
    omega, _ = rf.fit(max_n_iterations=resid_max_iter, progressbar=True)

    sf = StimulusFitter(model=mk_model(), data=bold_df, omega=omega,
                        parameters=pars_df)
    decoded = sf.fit(l2_norm=l2_norm, learning_rate=learning_rate,
                     max_n_iterations=max_n_iterations,
                     min_n_iterations=min_n_iterations, progressbar=True)
    return decoded.values.reshape(T, paradigm_avg.shape[1],
                                   paradigm_avg.shape[2]).astype(np.float32)


def decode(*, subject: int, bids_folder: str,
           l2_norm: float, learning_rate: float,
           max_n_iterations: int, min_n_iterations: int,
           resid_max_iter: int, max_voxels: int | None,
           out_path: Path):
    npz_path = (Path(bids_folder) / 'derivatives' / 'v1_decode_handoffs'
                / f'sub-{subject:02d}_m4_V1.npz')
    print(f'Loading {npz_path}', flush=True)
    with np.load(npz_path, allow_pickle=True) as f:
        npz = {k: f[k] for k in f.keys()}

    resolution = int(npz['resolution'])
    grid_radius = float(npz['grid_radius'])
    tr = float(npz['tr'])
    bold = npz['bold'].astype(np.float32)  # (T_all, V)
    T_all, V = bold.shape
    if T_all % N_TR_PER_RUN != 0:
        raise RuntimeError(f'T={T_all} not divisible by {N_TR_PER_RUN}; '
                            f'this subject may have run-count anomalies.')
    n_runs = T_all // N_TR_PER_RUN
    bold_runs = bold.reshape(n_runs, N_TR_PER_RUN, V)
    print(f'  V1 voxels: {V}  runs: {n_runs}  TRs/run: {N_TR_PER_RUN}  '
          f'res: {resolution}  grid_radius: {grid_radius}°', flush=True)

    # Voxel selection — canonical filter, n_timepoints = TRs in averaged chunk.
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

    # Build per-run paradigms and group runs by HP.
    sub = Subject(subject, bids_folder=bids_folder)
    hp_groups, run_index = _group_runs_by_hp(sub)
    print(f'  HP groups:')
    for hp in HP_LABELS:
        runs = [run_index[i] for i in hp_groups[hp]]
        print(f'    {hp:13s}  {runs}')

    print('Building per-run paradigms...', flush=True)
    par_runs = np.stack([
        _bar_run(sub, ses, run, resolution=resolution, grid_radius=grid_radius)
        for ses, run in run_index
    ], axis=0)  # (n_runs, 258, R, R)
    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    # Decode each HP condition on the averaged BOLD + paradigm.
    bold_runs_sel = bold_runs[:, :, npz_rows]  # (n_runs, 258, V_sel)
    decoded_per_hp = np.zeros((len(HP_LABELS), N_TR_PER_RUN,
                                resolution, resolution), dtype=np.float32)
    paradigm_per_hp = np.zeros_like(decoded_per_hp)

    t0_all = time.time()
    for k, hp in enumerate(HP_LABELS):
        idxs = hp_groups[hp]
        bold_avg = bold_runs_sel[idxs].mean(axis=0)        # (258, V_sel)
        par_avg = par_runs[idxs].mean(axis=0)              # (258, R, R)
        print(f'\n=== HP={hp}  (n={len(idxs)} runs averaged) ===', flush=True)
        t0 = time.time()
        decoded_per_hp[k] = _fit_one_hp(
            bold_avg, par_avg, pars_df, grid,
            tr=tr, l2_norm=l2_norm, learning_rate=learning_rate,
            max_n_iterations=max_n_iterations,
            min_n_iterations=min_n_iterations,
            resid_max_iter=resid_max_iter)
        paradigm_per_hp[k] = par_avg
        print(f'  [{hp}] decode done in {time.time() - t0:.0f}s', flush=True)
    print(f'\nAll 4 HP decodes done in {time.time() - t0_all:.0f}s', flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        decoded=decoded_per_hp,
        paradigm=paradigm_per_hp,
        grid=grid,
        hp_labels=np.array(HP_LABELS),
        voxel_idx=npz['voxel_idx'][npz_rows],
        fdr_thr=np.float32(fdr_thr),
        l2_norm=np.float32(l2_norm), learning_rate=np.float32(learning_rate),
        resolution=np.int32(resolution),
        grid_radius=np.float32(grid_radius),
        tr=np.float32(tr),
    )
    print(f'\nWrote: {out_path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, required=True)
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

    out_path = args.out or (Path(__file__).resolve().parents[2]
                            / 'notes' / 'data' / 'v1_decode'
                            / f'sub-{args.subject:02d}' / 'decoded_per_hp.npz')
    decode(subject=args.subject, bids_folder=args.bids_folder,
           l2_norm=args.l2_norm, learning_rate=args.learning_rate,
           max_n_iterations=args.max_n_iterations,
           min_n_iterations=args.min_n_iterations,
           resid_max_iter=args.resid_max_iter,
           max_voxels=args.max_voxels, out_path=out_path)


if __name__ == '__main__':
    main()
