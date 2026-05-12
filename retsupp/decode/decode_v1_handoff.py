"""V1 stimulus decode from the m4 handoff NPZ — one single run per call.

Reads ``derivatives/v1_decode_handoffs/sub-{NN}_m4_V1.npz`` (m4 DoG +
flex-HRF PRF params + cleaned BOLD across all 12 runs, voxel-aligned),
selects the rows for one (session, run) from the concatenated BOLD,
builds the matching bar-only paradigm via :meth:`Subject.get_bar_stimulus`,
and decodes the per-TR stimulus with braincoder's StimulusFitter.

Per-run decoding keeps trialwise structure available for later analyses
(e.g. distractor-locked modulation); per-HP / per-session averages can
be reconstructed post-hoc by averaging the per-run NPZs.

Voxel filter is **top-N by r² only** — the new warmstart m4 fits enforce
``sd_min=0.2`` at the model level, so phantom-like degenerate fits don't
exist and FDR / mass-in-aperture / σ-bound guards are unnecessary.

Usage::

    python -m retsupp.decode.decode_v1_handoff --subject 15 --session 1 --run 1

Output::

    notes/data/v1_decode/sub-{NN}/decoded_ses-{S}_run-{R}[_vox{N}].npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject, validate_prf_parameters


PARS_BY_MODEL = {
    4: ['x', 'y', 'sd', 'baseline', 'amplitude',
        'srf_amplitude', 'srf_size', 'hrf_delay', 'hrf_dispersion'],
    6: ['x', 'y', 'sd',
        'rf_amplitude', 'srf_amplitude', 'srf_size',
        'neural_baseline', 'surround_baseline', 'bold_baseline',
        'hrf_delay', 'hrf_dispersion'],
}
SD_MIN_BY_MODEL = {4: 0.2, 6: 0.2}
N_TR_PER_RUN = 258


def _make_model(model: int, *, grid_coordinates, paradigm, hrf_model,
                data, parameters, sd_min: float):
    """Build the braincoder PRF model corresponding to ``model``."""
    from braincoder.models import (
        DifferenceOfGaussiansPRF2DWithHRF,
        DivisiveNormalizationGaussianPRF2DWithHRF,
    )
    cls = {4: DifferenceOfGaussiansPRF2DWithHRF,
           6: DivisiveNormalizationGaussianPRF2DWithHRF}[model]
    return cls(grid_coordinates=grid_coordinates, paradigm=paradigm,
               hrf_model=hrf_model, data=data, parameters=parameters,
               flexible_hrf_parameters=True, sd_min=sd_min)


def _run_index_in_concat(sub: Subject, session: int, run: int) -> int:
    """Row index of (session, run) in the 12-run concatenated BOLD.

    Order: ses-1 run-1..n, ses-2 run-1..n (matches the handoff NPZ build).
    """
    idx = 0
    for ses in (1, 2):
        for r in sub.get_runs(ses):
            if (ses, r) == (session, run):
                return idx
            idx += 1
    raise KeyError(f'(ses={session}, run={run}) not in concat order for '
                    f'sub-{sub.subject_id:02d}')


def decode(*, subject: int, session: int, run: int, model: int,
           bids_folder: str,
           l2_norm: float, learning_rate: float,
           max_n_iterations: int, min_n_iterations: int,
           resid_max_iter: int, max_voxels: int,
           noise_dist: str,
           out_path: Path):
    from braincoder.hrf import SPMHRFModel
    from braincoder.optimize import ResidualFitter, StimulusFitter

    pars_list = PARS_BY_MODEL[model]
    sd_min = SD_MIN_BY_MODEL[model]
    npz_path = (Path(bids_folder) / 'derivatives' / 'v1_decode_handoffs'
                / f'sub-{subject:02d}_m{model}_V1.npz')
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
    print(f'  V1 voxels: {V}  runs in concat: {n_runs}  TRs/run: {N_TR_PER_RUN}',
          flush=True)

    sub = Subject(subject, bids_folder=bids_folder)
    row = _run_index_in_concat(sub, session, run)
    print(f'  ses-{session} run-{run}: concat row {row}', flush=True)

    # Voxel filter: top-N by r² with finite parameters. No FDR / σ / mass
    # guards — the m4 warmstart fits have sd_min=0.2 built in, no phantoms.
    pars_all = pd.DataFrame({p: npz[p] for p in pars_list}).astype(np.float32)
    pars_all['r2'] = npz['r2'].astype(np.float32)
    finite = np.all(np.isfinite(pars_all.values), axis=1) & (pars_all.r2 > 0)
    pars_all = pars_all[finite]
    sel = pars_all.sort_values('r2', ascending=False).head(max_voxels)
    npz_rows = sel.index.to_numpy()
    sel = sel.reset_index(drop=True)
    print(f'  Selected voxels: {len(sel)} / {V}  '
          f'(top-{max_voxels} by r², no other filter)', flush=True)
    print(f'    sd: median {sel.sd.median():.3f}  q05/95 '
          f'{sel.sd.quantile(.05):.3f}/{sel.sd.quantile(.95):.3f}', flush=True)
    print(f'    r²: min {sel.r2.min():.3f}  median {sel.r2.median():.3f}  '
          f'max {sel.r2.max():.3f}', flush=True)
    if len(sel) < 20:
        raise RuntimeError(f'Too few voxels selected: {len(sel)}')

    pars_df = sel[pars_list].astype(np.float32)
    validate_prf_parameters(pars_df, sd_min=sd_min, model_label=f'm{model}',
                             source=str(npz_path))

    # BOLD + paradigm for this single run.
    bold_run = bold.reshape(n_runs, N_TR_PER_RUN, V)[row][:, npz_rows]
    paradigm = sub.get_bar_stimulus(
        session=session, run=run,
        resolution=resolution, grid_radius=grid_radius).astype(np.float32)
    T = paradigm.shape[0]
    print(f'  BOLD: {bold_run.shape}  paradigm: {paradigm.shape}', flush=True)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=session, run=run, grid_radius=grid_radius)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    paradigm_flat = paradigm.reshape(T, -1).astype(np.float32)
    bold_df = pd.DataFrame(bold_run.astype(np.float32),
                            index=pd.Index(np.arange(T), name='frame'))
    bold_df.columns.name = 'voxel'
    paradigm_df = pd.DataFrame(paradigm_flat,
                                index=pd.Index(np.arange(T), name='frame'))

    hrf = SPMHRFModel(tr=tr, delay=4.5, dispersion=0.75)

    def mk_model():
        return _make_model(model, grid_coordinates=grid,
                            paradigm=paradigm_flat, hrf_model=hrf,
                            data=bold_df, parameters=pars_df, sd_min=sd_min)

    print(f'Fitting residual covariance (noise_dist={noise_dist})...',
          flush=True)
    t0 = time.time()
    rf = ResidualFitter(model=mk_model(), data=bold_df, paradigm=paradigm_df,
                        parameters=pars_df)
    omega, dof = rf.fit(max_n_iterations=resid_max_iter, progressbar=True,
                         method=noise_dist)
    print(f'  omega: {omega.shape}  dof: {dof}  ({time.time() - t0:.0f}s)',
          flush=True)

    print('Decoding stimulus...', flush=True)
    t0 = time.time()
    sf = StimulusFitter(model=mk_model(), data=bold_df, omega=omega,
                        parameters=pars_df, dof=dof)
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
        paradigm=paradigm,
        grid=grid,
        voxel_idx=npz['voxel_idx'][npz_rows],
        session=np.int32(session),
        run=np.int32(run),
        l2_norm=np.float32(l2_norm), learning_rate=np.float32(learning_rate),
        resolution=np.int32(resolution),
        grid_radius=np.float32(grid_radius),
        tr=np.float32(tr),
        noise_dist=np.array(noise_dist),
        dof=np.float32(dof) if dof is not None else np.float32(np.nan),
    )
    print(f'Wrote: {out_path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--session', type=int, required=True)
    p.add_argument('--run', type=int, required=True)
    p.add_argument('--model', type=int, default=4, choices=sorted(PARS_BY_MODEL),
                   help='PRF model: 4 = DoG+HRF, 6 = Divisive Normalization+HRF.')
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--l2-norm', type=float, default=1.0,
                   help='L2 penalty on decoded stimulus (default 1.0; '
                        'matches the binary 0/1 paradigm scale via the '
                        'Gaussian-prior interpretation σ_prior = 1/√(2·L2) ≈ '
                        '0.71. Smaller L2 → noisy spikes; larger L2 → '
                        'crushed amplitude.).')
    p.add_argument('--learning-rate', type=float, default=0.5)
    p.add_argument('--max-n-iterations', type=int, default=1000)
    p.add_argument('--min-n-iterations', type=int, default=200)
    p.add_argument('--resid-max-iter', type=int, default=2000,
                   help='Max iters for ResidualFitter (default 2000; the '
                        'omega/dof landscape is shallow so converge with '
                        'plenty of headroom — early-stops on plateau).')
    p.add_argument('--max-voxels', type=int, default=200,
                   help='Number of top-r² voxels to include (default 200).')
    p.add_argument('--noise-dist', default='gauss', choices=['gauss', 't'],
                   help="Residual noise distribution (default 'gauss'). "
                        "'t' fits a Student-t and uses it in StimulusFitter "
                        'to down-weight heavy-tailed (small-PRF) voxels.')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()

    tag = f'ses-{args.session}_run-{args.run}_vox{args.max_voxels}'
    if args.model != 4:
        tag = f'm{args.model}_{tag}'
    if args.noise_dist == 't':
        tag = f'{tag}_t'
    out_path = args.out or (Path(__file__).resolve().parents[2]
                            / 'notes' / 'data' / 'v1_decode'
                            / f'sub-{args.subject:02d}'
                            / f'decoded_{tag}.npz')
    decode(subject=args.subject, session=args.session, run=args.run,
           model=args.model, bids_folder=args.bids_folder,
           l2_norm=args.l2_norm, learning_rate=args.learning_rate,
           max_n_iterations=args.max_n_iterations,
           min_n_iterations=args.min_n_iterations,
           resid_max_iter=args.resid_max_iter,
           max_voxels=args.max_voxels, noise_dist=args.noise_dist,
           out_path=out_path)


if __name__ == '__main__':
    main()
