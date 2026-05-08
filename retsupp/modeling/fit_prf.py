"""Whole-cortex PRF fits with the FULL 8-item paradigm, on GPU, chunked.

This is the canonical mean-PRF entry point. It replaces the earlier
bar-only single-model script: ALL six PRF model variants are fit in
one pass with a smart init pipeline so the Gaussian grid fit is done
exactly once, and DoG / DN / flexible-HRF variants reuse it as
warm-start parameters.

Pipeline
--------
1. **model 1 — Gaussian PRF, fixed HRF.** Grid search over
   (x, y, σ) on all cortical-ribbon voxels, then GD refine. Slow step.
2. **model 3 — Gaussian PRF, flexible HRF.** Init from model 1 +
   add (hrf_delay=4.5, hrf_dispersion=0.75). Per-voxel GD only.
3. **model 2 — DoG, fixed HRF.** Init from model 1 + add
   (srf_amplitude=5e-2, srf_size=2.). Per-voxel GD only.
4. **model 4 — DoG, flexible HRF.** Init from model 3 + add
   srf_*. Per-voxel GD only. THE canonical model.
5. **model 5 — Divisive Normalization, fixed HRF.** Init from model 4.
6. **model 6 — Divisive Normalization, flexible HRF.** Init from
   model 4 (or model 5).

Outputs go to ``derivatives/prf/model{N}/sub-XX/sub-XX_desc-{par}.nii.gz``,
one NIfTI per parameter. The pre-2026-05-08 single-model bar-only
fits are invalid (paradigm was missing 7 of 8 search-array items)
and have been moved to ``derivatives/.old_paradigm.bak/``.

Voxel chunking
--------------
The whole cortical ribbon is ~30-60k voxels. The GD fit's
activation memory dominates and scales linearly in voxel count. We
chunk voxels into batches of ``--voxel-chunk-size`` (default 10000)
and run grid + GD on each chunk independently — standard PRFs are
massively univariate (no shared params, no AF), so chunks are exactly
equivalent to the single-batch fit. Chunk size sets a
RAM-vs-throughput tradeoff: larger chunks use more VRAM but launch
fewer kernels.

Usage
-----
``python -m retsupp.modeling.fit_prf 3 --bids-folder /shares/zne.uzh/gdehol/ds-retsupp``
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, maskers

from braincoder.hrf import SPMHRFModel
from braincoder.models import (
    GaussianPRF2DWithHRF,
    DifferenceOfGaussiansPRF2DWithHRF,
    DivisiveNormalizationGaussianPRF2DWithHRF,
)
from braincoder.optimize import ParameterFitter

from retsupp.utils.data import Subject


# --- Paradigm builder -------------------------------------------------------

def build_paradigm(sub: Subject, resolution: int, paradigm_kind: str):
    """Return (paradigm (T, G), grid_coordinates (G, 2)) on the full grid.

    ``paradigm_kind='full'`` paints all 8 search-array rectangles per
    trial during the target window (rect 1.5° × 0.375°, ring r=4°).
    ``paradigm_kind='bar'`` keeps the legacy bar-only stimulus.
    """
    if paradigm_kind not in ('full', 'bar'):
        raise ValueError(f"paradigm_kind must be 'full' or 'bar', "
                         f"got {paradigm_kind!r}")

    if paradigm_kind == 'full':
        paradigm = sub.get_stimulus_with_distractors(
            session=1, run=1, resolution=resolution,
            grid_radius=5.0,
            distractor_shape='rectangle',
            distractor_long_side=1.5, distractor_short_side=0.375,
        ).astype(np.float32)
        gx, gy = sub.get_extended_grid_coordinates(
            resolution=resolution, session=1, run=1, grid_radius=5.0)
    else:
        paradigm = sub.get_stimulus(
            session=1, run=1, resolution=resolution).astype(np.float32)
        gx, gy = sub.get_grid_coordinates(
            session=1, run=1, resolution=resolution)

    paradigm = paradigm.reshape((paradigm.shape[0], -1))
    grid_coords = np.stack(
        (gx.ravel(), gy.ravel()), axis=1).astype(np.float32)
    return paradigm, grid_coords


# --- Chunked GD helper ------------------------------------------------------

def gd_fit_chunked(model_factory, data, paradigm, init_pars,
                   chunk_size, max_n_iterations, learning_rate=0.005):
    """Run GD per-voxel-chunk and concatenate the resulting parameter
    DataFrames. ``model_factory`` is a callable that builds a fresh model
    bound to the chunk's data (so we don't carry stale tensors).

    Returns the concatenated parameter DataFrame (V, n_pars).
    """
    n_vox = data.shape[1]
    chunks = np.array_split(np.arange(n_vox), max(1, n_vox // chunk_size))

    out = []
    for i, idx in enumerate(chunks):
        t0 = time.time()
        chunk_data = data[:, idx]
        chunk_init = init_pars.iloc[idx].reset_index(drop=True)
        chunk_model = model_factory(chunk_data, paradigm)
        fitter = ParameterFitter(chunk_model, chunk_data, paradigm)
        chunk_pars = fitter.fit(
            init_pars=chunk_init,
            max_n_iterations=max_n_iterations,
            learning_rate=learning_rate,
        )
        chunk_pars['r2'] = fitter.get_rsq(chunk_pars).values
        chunk_pars.index = idx
        out.append(chunk_pars)
        print(f"    chunk {i+1}/{len(chunks)}: {len(idx)} voxels, "
              f"{time.time() - t0:.1f}s, mean R²={chunk_pars['r2'].mean():.3f}")

    return pd.concat(out, axis=0).sort_index()


def grid_fit_chunked(model, data, paradigm, grid_x, grid_y, grid_sd,
                     chunk_size):
    """Grid fit per-voxel-chunk. Grid fits use correlation-based scoring
    so they're cheap; we still chunk to bound memory."""
    n_vox = data.shape[1]
    chunks = np.array_split(np.arange(n_vox), max(1, n_vox // chunk_size))

    grid_amp = [1]
    grid_baseline = [0]

    out = []
    for i, idx in enumerate(chunks):
        t0 = time.time()
        chunk_data = data[:, idx]
        fitter = ParameterFitter(model, chunk_data, paradigm)
        chunk_pars = fitter.fit_grid(
            grid_x, grid_y, grid_sd, grid_baseline, grid_amp,
            use_correlation_cost=True,
        )
        chunk_pars = fitter.refine_baseline_and_amplitude(
            chunk_pars, l2_alpha=0.001)
        chunk_pars.index = idx
        out.append(chunk_pars)
        print(f"    grid chunk {i+1}/{len(chunks)}: {len(idx)} vox, "
              f"{time.time() - t0:.1f}s")
    return pd.concat(out, axis=0).sort_index()


# --- Save helper ------------------------------------------------------------

def save_pars_as_niftis(pars: pd.DataFrame, masker, target_dir: Path,
                        subject: int):
    target_dir.mkdir(parents=True, exist_ok=True)
    pars = pars.copy()
    if 'x' in pars and 'y' in pars:
        pars['theta'] = np.arctan2(pars['y'], pars['x'])
        pars['ecc'] = np.sqrt(pars['x'] ** 2 + pars['y'] ** 2)
    for par in pars.columns:
        img = masker.inverse_transform(pars[par].values)
        img.to_filename(target_dir / f'sub-{subject:02d}_desc-{par}.nii.gz')


# --- Main pipeline ----------------------------------------------------------

def main(subject: int, bids_folder: str = '/data/ds-retsupp',
         resolution: int = 50, voxel_chunk_size: int = 10000,
         max_n_iterations: int = 2000,
         models: tuple[int, ...] = (1, 3, 2, 4, 5, 6),
         paradigm_kind: str = 'full',
         debug: bool = False):
    """Fit all requested PRF model variants for a subject.

    Models always run in the smart-init order: 1 → 3 → 2 → 4 → 5 → 6,
    so that DoG / DN / flexible-HRF variants warm-start from cheaper
    upstream fits.
    """
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)

    # Output: derivatives/prf/model{N}/sub-XX/...
    derivs = bids_folder / 'derivatives'

    # 1) Cortical-ribbon-ish mask. Subject.get_bold_mask() is the
    # canonical mask used by every other script in this project — keep it
    # for consistency. ~30-60k voxels per subject.
    bold_mask = sub.get_bold_mask()
    if debug:
        # Subsample mask to ~1% of voxels for quick smoke test.
        bold_mask = image.math_img(
            'np.where(m.astype(bool) & (np.random.rand(*m.shape) < 0.01), 1, 0)',
            m=bold_mask)
        max_n_iterations = 100
        resolution = 25

    masker = maskers.NiftiMasker(mask_img=bold_mask)
    mean_ts = (
        derivs / 'mean_signal' / f'sub-{subject:02d}'
        / f'sub-{subject:02d}_desc-mean_bold.nii.gz'
    )
    print(f"Loading mean BOLD: {mean_ts}")
    data = masker.fit_transform(mean_ts).astype(np.float32)
    n_vox = data.shape[1]
    n_T = data.shape[0]
    print(f"  shape: T={n_T}, V={n_vox}")

    # 2) Paradigm.
    paradigm, grid_coords = build_paradigm(sub, resolution, paradigm_kind)
    print(f"  paradigm shape: {paradigm.shape}, grid shape: {grid_coords.shape}, "
          f"kind={paradigm_kind!r}")

    hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)

    # ----- Step 1: model 1 — Gaussian PRF + fixed HRF (grid + GD). -----
    pars_per_model: dict[int, pd.DataFrame] = {}
    if 1 in models or any(m in models for m in (2, 3, 4, 5, 6)):
        # Even if only later models requested, we still need model-1 init.
        print("\n=== Model 1: Gaussian + fixed HRF ===")
        # Grid fit on the full set of voxels.
        gx_grid = np.linspace(-3, 3, 12)
        gy_grid = np.linspace(-3, 3, 12)
        sd_grid = np.linspace(1.0, 4.0, 8)
        if debug:
            gx_grid = np.linspace(-3, 3, 5)
            gy_grid = np.linspace(-3, 3, 5)
            sd_grid = np.linspace(1.0, 4.0, 4)

        gauss_model = GaussianPRF2DWithHRF(grid_coords, paradigm, hrf_model=hrf)
        print("  grid fit...")
        grid_pars = grid_fit_chunked(
            gauss_model, data, paradigm,
            gx_grid, gy_grid, sd_grid, chunk_size=voxel_chunk_size)
        print(f"  grid R² distribution: {grid_pars.shape[0]} voxels")

        print("  GD fit...")
        gauss_factory = lambda d, p: GaussianPRF2DWithHRF(  # noqa: E731
            grid_coords, p, hrf_model=hrf)
        pars_per_model[1] = gd_fit_chunked(
            gauss_factory, data, paradigm,
            init_pars=grid_pars,
            chunk_size=voxel_chunk_size,
            max_n_iterations=max_n_iterations)
        print(f"  median R²: {pars_per_model[1]['r2'].median():.3f}")

    # ----- Step 2: model 3 — Gaussian + flexible HRF (init from model 1). -----
    if 3 in models or any(m in models for m in (4, 5, 6)):
        print("\n=== Model 3: Gaussian + flexible HRF (init from model 1) ===")
        init = pars_per_model[1].copy()
        init['hrf_delay'] = 4.5
        init['hrf_dispersion'] = 0.75
        gauss_hrf_factory = lambda d, p: GaussianPRF2DWithHRF(  # noqa: E731
            grid_coords, p, hrf_model=hrf, flexible_hrf_parameters=True)
        pars_per_model[3] = gd_fit_chunked(
            gauss_hrf_factory, data, paradigm,
            init_pars=init.drop(columns=['r2'], errors='ignore'),
            chunk_size=voxel_chunk_size,
            max_n_iterations=max_n_iterations)
        print(f"  median R²: {pars_per_model[3]['r2'].median():.3f}")

    # ----- Step 3: model 2 — DoG + fixed HRF (init from model 1). -----
    if 2 in models:
        print("\n=== Model 2: DoG + fixed HRF (init from model 1) ===")
        init = pars_per_model[1].copy()
        init['srf_amplitude'] = 5e-2
        init['srf_size'] = 2.0
        dog_factory = lambda d, p: DifferenceOfGaussiansPRF2DWithHRF(  # noqa: E731
            grid_coordinates=grid_coords, paradigm=p, hrf_model=hrf,
            data=d, flexible_hrf_parameters=False)
        pars_per_model[2] = gd_fit_chunked(
            dog_factory, data, paradigm,
            init_pars=init.drop(columns=['r2'], errors='ignore'),
            chunk_size=voxel_chunk_size,
            max_n_iterations=max_n_iterations)
        print(f"  median R²: {pars_per_model[2]['r2'].median():.3f}")

    # ----- Step 4: model 4 — DoG + flexible HRF (init from model 3). -----
    if 4 in models or any(m in models for m in (5, 6)):
        print("\n=== Model 4: DoG + flexible HRF (init from model 3) ===")
        init = pars_per_model[3].copy()
        init['srf_amplitude'] = 1e-3
        init['srf_size'] = 2.0
        dog_hrf_factory = lambda d, p: DifferenceOfGaussiansPRF2DWithHRF(  # noqa: E731
            grid_coordinates=grid_coords, paradigm=p, hrf_model=hrf,
            data=d, flexible_hrf_parameters=True)
        pars_per_model[4] = gd_fit_chunked(
            dog_hrf_factory, data, paradigm,
            init_pars=init.drop(columns=['r2'], errors='ignore'),
            chunk_size=voxel_chunk_size,
            max_n_iterations=max_n_iterations)
        print(f"  median R²: {pars_per_model[4]['r2'].median():.3f}")

    # ----- Step 5: model 5 — DN + fixed HRF (init from model 4). -----
    if 5 in models:
        print("\n=== Model 5: DN + fixed HRF (init from model 4) ===")
        init = pars_per_model[4].copy()
        init['rf_amplitude'] = init['amplitude']
        init['srf_amplitude'] = init.get('srf_amplitude', 1e-2)
        init['srf_size'] = init.get('srf_size', 2.0)
        init['neural_baseline'] = 1.0
        init['surround_baseline'] = 1.0
        init['bold_baseline'] = 0.0
        dn_factory = lambda d, p: DivisiveNormalizationGaussianPRF2DWithHRF(  # noqa: E731
            grid_coordinates=grid_coords, paradigm=p, hrf_model=hrf,
            data=d, flexible_hrf_parameters=False)
        pars_per_model[5] = gd_fit_chunked(
            dn_factory, data, paradigm,
            init_pars=init.drop(columns=['r2'], errors='ignore'),
            chunk_size=voxel_chunk_size,
            max_n_iterations=max_n_iterations)
        print(f"  median R²: {pars_per_model[5]['r2'].median():.3f}")

    # ----- Step 6: model 6 — DN + flexible HRF (init from model 4). -----
    if 6 in models:
        print("\n=== Model 6: DN + flexible HRF (init from model 4) ===")
        init = pars_per_model[4].copy()
        init['rf_amplitude'] = init['amplitude']
        init['srf_amplitude'] = init.get('srf_amplitude', 1e-2)
        init['srf_size'] = init.get('srf_size', 2.0)
        init['neural_baseline'] = 1.0
        init['surround_baseline'] = 1.0
        init['bold_baseline'] = 0.0
        # hrf cols already in init from model 4.
        dn_hrf_factory = lambda d, p: DivisiveNormalizationGaussianPRF2DWithHRF(  # noqa: E731
            grid_coordinates=grid_coords, paradigm=p, hrf_model=hrf,
            data=d, flexible_hrf_parameters=True)
        pars_per_model[6] = gd_fit_chunked(
            dn_hrf_factory, data, paradigm,
            init_pars=init.drop(columns=['r2'], errors='ignore'),
            chunk_size=voxel_chunk_size,
            max_n_iterations=max_n_iterations)
        print(f"  median R²: {pars_per_model[6]['r2'].median():.3f}")

    # ----- Save outputs (only the requested models). -----
    print("\n=== Saving NIfTIs ===")
    suffix = '.debug' if debug else ''
    for label in models:
        if label not in pars_per_model:
            continue
        target_dir = derivs / f'prf{suffix}' / f'model{label}' / f'sub-{subject:02d}'
        save_pars_as_niftis(pars_per_model[label], masker, target_dir, subject)
        print(f"  model {label}: {target_dir}")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--resolution', type=int, default=50)
    p.add_argument('--voxel-chunk-size', type=int, default=10000)
    p.add_argument('--max-n-iterations', type=int, default=2000)
    p.add_argument('--models', type=int, nargs='+',
                   default=[1, 3, 2, 4, 5, 6],
                   help='Which model labels to fit (default: all 6).')
    p.add_argument('--paradigm-kind', choices=['full', 'bar'],
                   default='full',
                   help="'full' = 8-item paradigm (canonical, default); "
                        "'bar' = legacy bar-only.")
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()
    main(args.subject,
         bids_folder=args.bids_folder,
         resolution=args.resolution,
         voxel_chunk_size=args.voxel_chunk_size,
         max_n_iterations=args.max_n_iterations,
         models=tuple(args.models),
         paradigm_kind=args.paradigm_kind,
         debug=args.debug)
