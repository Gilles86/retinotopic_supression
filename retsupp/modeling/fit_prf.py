"""Whole-cortex PRF fit, FULL 8-item paradigm. One model per invocation.

Model 1 (Gaussian + fixed HRF) is the only "from scratch" fit (grid + GD).
Models 2-6 load model 1 (or model 3) parameters from disk as init, then GD only.
This lets each model be its own SLURM job — model 1 once, then 2/3/4/5/6
in any order or in parallel after model 1 lands.

Init dependency tree::

    1: grid + GD
    2: init from 1, +DoG params
    3: init from 1, +flex HRF
    4: init from 3, +DoG params         <- canonical
    5: init from 4, +DN params (fixed HRF)
    6: init from 4, +DN params (flex HRF)

Output: ``derivatives/prf/model{N}/sub-XX/sub-XX_desc-{par}.nii.gz``.

Voxel chunking: standard PRFs are massively univariate; chunk size sets
a VRAM-vs-throughput tradeoff. Default 10000 fits comfortably on L4.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, maskers
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import (
    GaussianPRF2DWithHRF,
    DifferenceOfGaussiansPRF2DWithHRF,
    DivisiveNormalizationGaussianPRF2DWithHRF,
)
from braincoder.optimize import ParameterFitter

from retsupp.utils.data import Subject


# Per-model config. ``extra_init`` values may be either constants or the
# name of an existing column to copy from (e.g. 'rf_amplitude' = 'amplitude').
MODEL_CFG: dict = {
    1: dict(cls=GaussianPRF2DWithHRF, flex_hrf=False, init_from=None,
            extra_init={}),
    2: dict(cls=DifferenceOfGaussiansPRF2DWithHRF, flex_hrf=False, init_from=1,
            extra_init={'srf_amplitude': 5e-2, 'srf_size': 2.0}),
    3: dict(cls=GaussianPRF2DWithHRF, flex_hrf=True, init_from=1,
            extra_init={'hrf_delay': 4.5, 'hrf_dispersion': 0.75}),
    4: dict(cls=DifferenceOfGaussiansPRF2DWithHRF, flex_hrf=True, init_from=3,
            extra_init={'srf_amplitude': 1e-3, 'srf_size': 2.0}),
    5: dict(cls=DivisiveNormalizationGaussianPRF2DWithHRF, flex_hrf=False,
            init_from=4,
            extra_init={'rf_amplitude': 'amplitude',
                        'srf_amplitude': 1e-2, 'srf_size': 2.0,
                        'neural_baseline': 1.0, 'surround_baseline': 1.0,
                        'bold_baseline': 0.0,
                        'hrf_delay': None, 'hrf_dispersion': None}),
    6: dict(cls=DivisiveNormalizationGaussianPRF2DWithHRF, flex_hrf=True,
            init_from=4,
            extra_init={'rf_amplitude': 'amplitude',
                        'srf_amplitude': 1e-2, 'srf_size': 2.0,
                        'neural_baseline': 1.0, 'surround_baseline': 1.0,
                        'bold_baseline': 0.0}),
}


def load_concatenated(sub: Subject, masker, resolution: int, kind: str):
    """Concatenate cleaned BOLD + per-run paradigm across all (ses, run).

    The HP-distractor location varies per run, so the search-array
    paradigm differs per run too. Averaging the BOLD across runs would
    average over different stimuli — instead we concatenate along time.
    Returns (bold (T_total, V), paradigm (T_total, G), grid_coords (G, 2)).
    """
    bold_chunks, par_chunks = [], []
    grid_coords = None
    for ses in (1, 2):
        for run in sub.get_runs(ses):
            bold_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                       / f'sub-{sub.subject_id:02d}' / f'ses-{ses}' / 'func'
                       / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                         f'desc-cleaned_run-{run}_bold.nii.gz')
            data = masker.transform(bold_fn).astype(np.float32)
            data = data[:258]  # cleaned BOLD is sometimes 259; crop.
            bold_chunks.append(data)

            if kind == 'full':
                par = sub.get_stimulus_with_distractors(
                    session=ses, run=run, resolution=resolution,
                    grid_radius=5.0, distractor_shape='rectangle',
                    distractor_long_side=1.5, distractor_short_side=0.375,
                ).astype(np.float32)
                gx, gy = sub.get_extended_grid_coordinates(
                    resolution=resolution, session=ses, run=run, grid_radius=5.0)
            elif kind == 'bar':
                par = sub.get_stimulus(session=ses, run=run,
                                        resolution=resolution).astype(np.float32)
                gx, gy = sub.get_grid_coordinates(session=ses, run=run,
                                                   resolution=resolution)
            else:
                raise ValueError(kind)
            par = par.reshape(par.shape[0], -1)
            par_chunks.append(par)
            if grid_coords is None:
                grid_coords = np.stack(
                    (gx.ravel(), gy.ravel()), axis=1).astype(np.float32)

    return (np.vstack(bold_chunks),
            np.vstack(par_chunks),
            grid_coords)


def chunked(work_fn, n_vox: int, chunk_size: int, label: str):
    """Apply ``work_fn(idx)`` per voxel-chunk, concat by index, sort.
    Outer tqdm across chunks shows total/elapsed/ETA."""
    chunks = np.array_split(np.arange(n_vox), max(1, n_vox // chunk_size))
    out = []
    bar = tqdm(chunks, total=len(chunks), desc=label,
               unit='chunk', mininterval=2.0)
    for idx in bar:
        df = work_fn(idx)
        df.index = idx
        out.append(df)
        bar.set_postfix(vox=len(idx))
    return pd.concat(out, axis=0).sort_index()


def grid_fit(model, data, paradigm, chunk_size, debug=False):
    """Gaussian (x, y, σ) grid + baseline/amplitude refine."""
    grid_x = np.linspace(-3, 3, 5 if debug else 12)
    grid_y = np.linspace(-3, 3, 5 if debug else 12)
    grid_sd = np.linspace(1.0, 4.0, 4 if debug else 8)

    def work(idx):
        f = ParameterFitter(model, data[:, idx], paradigm)
        pars = f.fit_grid(grid_x, grid_y, grid_sd, [0], [1],
                          use_correlation_cost=True)
        return f.refine_baseline_and_amplitude(pars, l2_alpha=0.001)

    return chunked(work, data.shape[1], chunk_size, 'grid')


def gd_fit(model_factory, data, paradigm, init_pars, chunk_size,
           max_iter, lr=0.005):
    """Per-chunk GD with a model factory rebuilt per chunk."""
    def work(idx):
        d = data[:, idx]
        m = model_factory(d, paradigm)
        f = ParameterFitter(m, d, paradigm)
        ip = init_pars.iloc[idx].reset_index(drop=True)
        pars = f.fit(init_pars=ip, max_n_iterations=max_iter, learning_rate=lr)
        pars['r2'] = f.get_rsq(pars).values
        return pars

    return chunked(work, data.shape[1], chunk_size, 'GD')


def load_prior_pars(subject: int, model_label: int, derivs: Path,
                    masker) -> pd.DataFrame:
    """Read NIfTI parameters of a previous model fit back into a DataFrame."""
    src = derivs / 'prf' / f'model{model_label}' / f'sub-{subject:02d}'
    if not src.exists():
        raise FileNotFoundError(
            f"Prior model {model_label} not found at {src}. "
            f"Fit it first.")
    pars = {}
    for nii in sorted(src.glob(f'sub-{subject:02d}_desc-*.nii.gz')):
        name = nii.name.split('desc-')[1].rsplit('.nii.gz', 1)[0]
        pars[name] = masker.transform(str(nii)).flatten()
    return pd.DataFrame(pars)


def save_pars(pars: pd.DataFrame, masker, target_dir: Path, subject: int):
    target_dir.mkdir(parents=True, exist_ok=True)
    p = pars.copy()
    if 'x' in p and 'y' in p:
        p['theta'] = np.arctan2(p['y'], p['x'])
        p['ecc'] = np.sqrt(p['x'] ** 2 + p['y'] ** 2)
    for col in p.columns:
        masker.inverse_transform(p[col].values).to_filename(
            target_dir / f'sub-{subject:02d}_desc-{col}.nii.gz')


def main(subject: int, model_label: int,
         bids_folder: str = '/data/ds-retsupp',
         resolution: int = 50, voxel_chunk_size: int = 10000,
         max_n_iterations: int = 2000, paradigm_kind: str = 'full',
         debug: bool = False,
         chunk_index: int | None = None,
         n_chunks: int | None = None):
    """Fit one PRF model. If chunk_index/n_chunks given, fit ONLY voxels
    in that chunk and save partial results to a chunks/ subdir; a
    separate merge step concatenates all chunks into final NIfTIs."""
    cfg = MODEL_CFG[model_label]
    bids = Path(bids_folder)
    sub = Subject(subject, bids)
    derivs = bids / 'derivatives'
    chunked_mode = chunk_index is not None
    if chunked_mode and n_chunks is None:
        raise ValueError("chunk_index requires n_chunks")

    # Some subjects (e.g. sub-01) lack a run-1 brain_mask; use whatever
    # the first available run is for that session.
    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    if debug:
        bold_mask = image.math_img(
            'np.where(m.astype(bool) & (np.random.rand(*m.shape) < 0.01), 1, 0)',
            m=bold_mask)
        max_n_iterations = 100
        resolution = 25

    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()  # Needed before .transform calls in load_concatenated.
    print("Loading + concatenating cleaned BOLD across all (ses, run)...")
    data, paradigm, grid_coords = load_concatenated(
        sub, masker, resolution, paradigm_kind)
    print(f"  BOLD: T={data.shape[0]} (= n_runs × 258), V={data.shape[1]}")
    print(f"  paradigm: {paradigm.shape}, grid: {grid_coords.shape}, "
          f"kind={paradigm_kind!r}")

    # If chunked mode: subset data to just THIS chunk's voxels.
    if chunked_mode:
        n_vox = data.shape[1]
        all_chunks = np.array_split(np.arange(n_vox), n_chunks)
        chunk_idx = all_chunks[chunk_index]
        print(f"  chunked mode: chunk {chunk_index}/{n_chunks}, "
              f"voxels {chunk_idx[0]}..{chunk_idx[-1]} ({len(chunk_idx)})")
    else:
        chunk_idx = np.arange(data.shape[1])

    hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)
    factory = lambda d, p: cfg['cls'](  # noqa: E731
        grid_coordinates=grid_coords, paradigm=p, hrf_model=hrf, data=d,
        flexible_hrf_parameters=cfg['flex_hrf'])

    if cfg['init_from'] is None:
        # Model 1: grid + GD.
        print("\n=== model 1: Gaussian + fixed HRF (grid + GD) ===")
        if chunked_mode:
            d_chunk = data[:, chunk_idx]
            init = grid_fit(factory(d_chunk, paradigm), d_chunk, paradigm,
                             voxel_chunk_size, debug=debug)
        else:
            init = grid_fit(factory(data, paradigm), data, paradigm,
                             voxel_chunk_size, debug=debug)
    else:
        # Models 2-6: load prior fit, add extras, GD only.
        print(f"\n=== model {model_label} (init from model {cfg['init_from']}) ===")
        init = load_prior_pars(subject, cfg['init_from'], derivs, masker)
        if chunked_mode:
            init = init.iloc[chunk_idx].reset_index(drop=True)
        # Drop r2/theta/ecc — recomputed on save.
        init = init.drop(columns=['r2', 'theta', 'ecc'], errors='ignore')
        for k, v in cfg['extra_init'].items():
            if v is None:
                init = init.drop(columns=[k], errors='ignore')
            elif isinstance(v, str) and v in init.columns:
                init[k] = init[v]
            else:
                init[k] = v

    if chunked_mode:
        d_chunk = data[:, chunk_idx]
        pars = gd_fit(factory, d_chunk, paradigm, init,
                      voxel_chunk_size, max_n_iterations)
    else:
        pars = gd_fit(factory, data, paradigm, init,
                      voxel_chunk_size, max_n_iterations)
    print(f"  median R²: {pars['r2'].median():.3f}")

    if chunked_mode:
        # Save partial chunk: an NPZ with the voxel indices and a column
        # array per parameter. Cheap; merger reassembles.
        chunks_dir = (derivs / ('prf.debug' if debug else 'prf')
                      / f'model{model_label}' / f'sub-{subject:02d}'
                      / 'chunks')
        chunks_dir.mkdir(parents=True, exist_ok=True)
        npz_path = chunks_dir / f'chunk-{chunk_index:04d}-of-{n_chunks:04d}.npz'
        cols = {f'col_{c}': pars[c].values for c in pars.columns}
        np.savez(npz_path, voxel_indices=chunk_idx, columns=list(pars.columns),
                 **cols)
        print(f"Saved chunk: {npz_path}")
        return

    target_dir = derivs / ('prf.debug' if debug else 'prf') \
        / f'model{model_label}' / f'sub-{subject:02d}'
    save_pars(pars, masker, target_dir, subject)
    print(f"Saved: {target_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--model', type=int, required=True, choices=list(MODEL_CFG))
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--resolution', type=int, default=50)
    p.add_argument('--voxel-chunk-size', type=int, default=10000)
    p.add_argument('--max-n-iterations', type=int, default=2000)
    p.add_argument('--paradigm-kind', choices=['full', 'bar'], default='full')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--chunk-index', type=int, default=None,
                   help='SLURM-array chunk index (0-based). When given, '
                        'fit only this chunk; --n-chunks must also be set. '
                        'Saves a partial NPZ instead of NIfTIs; run '
                        'merge_prf_chunks.py to assemble.')
    p.add_argument('--n-chunks', type=int, default=None,
                   help='Total number of voxel chunks across SLURM tasks.')
    a = p.parse_args()
    main(a.subject, a.model, bids_folder=a.bids_folder,
         resolution=a.resolution, voxel_chunk_size=a.voxel_chunk_size,
         max_n_iterations=a.max_n_iterations,
         paradigm_kind=a.paradigm_kind, debug=a.debug,
         chunk_index=a.chunk_index, n_chunks=a.n_chunks)
