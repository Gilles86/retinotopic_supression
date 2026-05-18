"""Cross-validated PRF fitting for model comparison (m1 .. m6).

Lives separately from the canonical ``retsupp/modeling/fit_prf.py``
pipeline — these CV fits are an extension used post-hoc to compare
models on a parameter-penalty-free basis, not part of the main
analysis chain.

Design:

- **3-fold CV** along the within-condition run order. Each subject has
  ~3 runs per HP-distractor condition (12 runs total ≈ 4 conditions ×
  3 runs). Fold ``K`` ∈ {0, 1, 2} holds out the ``K``-th run of each
  condition. Training set: the other ~8 runs.
- Fits the same ``MODEL_CFG`` schedules as ``fit_prf.py``, on the
  training runs only.
- After fitting, predicts the held-out runs' BOLD time-series from
  the trained model and writes per-voxel **test R²** to
  ``derivatives/prf_cv/model{N}/fold-{K}/sub-XX/sub-XX_fold-{K}_desc-r2_test.nii.gz``.
- Chunked-voxel mode (``--chunk-index`` / ``--n-chunks``) is supported
  for parallel GPU dispatch, matching ``fit_prf_l4_chunked.sh`` style.
  The chunks are merged by ``merge_prf_cv_chunks.py``.

Output is intentionally minimal: only the test-R² NIfTI per (subject,
model, fold). Parameter NIfTIs from the training fit are NOT saved —
the canonical mean fit already exists in ``derivatives/prf/`` for
inspection. The point here is the *generalisation score*, not the
parameters.

CLI shape mirrors ``fit_prf.py``:

    python -m retsupp.modeling.cv.fit_prf_cv 3 4 \\
        --bids-folder /data/ds-retsupp \\
        --fold 0 --n-folds 3 \\
        --chunk-index 0 --n-chunks 10
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image, maskers

# Reuse the canonical pipeline's schedules, helpers, and grid-fit /
# GD-fit functions — keeps the cv fits aligned with the main fits.
from retsupp.modeling.fit_prf import (
    MODEL_CFG,
    SPATIAL,
    HRF,
    build_model_factory,
    grid_fit,
    gd_fit_scheduled,
    load_prior_pars,
)
from retsupp.utils.data import Subject
from retsupp.utils.sentinels import assert_gpu_available_if_expected


def make_fold_assignments(sub: Subject, n_folds: int = 3):
    """Return ``folds[k] = list of (ses, run)`` held out in fold k.

    Group runs by HP-distractor condition (4 conditions, ~3 runs each).
    Within each condition, sort by (ses, run) and assign the ``i``-th
    run to fold ``i``. Subjects with fewer runs per condition (sub-20
    ses-1, sub-24 ses-2) just have fewer runs in the higher folds —
    the asymmetry is small and unbiased.
    """
    groups = sub.get_runs_by_hp()
    folds = [[] for _ in range(n_folds)]
    for label in sorted(groups.keys()):
        runs_sorted = sorted(groups[label])
        for i, run in enumerate(runs_sorted):
            folds[i % n_folds].append(run)
    return folds


def load_runs(sub: Subject, masker, resolution: int, kind: str,
              runs: list[tuple[int, int]]):
    """Concatenate cleaned BOLD + paradigm across a specified subset of runs.

    Adapted from ``fit_prf.load_concatenated`` (which always loads all
    runs); here we take an explicit list. No NPZ cache — the cv folds
    are bespoke; rebuilding cache files for every fold would just
    duplicate the BOLD on disk.
    """
    bold_chunks, par_chunks = [], []
    grid_coords = None
    for ses, run in runs:
        bold_fn = (sub.bids_folder / 'derivatives' / 'cleaned'
                   / f'sub-{sub.subject_id:02d}' / f'ses-{ses}' / 'func'
                   / f'sub-{sub.subject_id:02d}_ses-{ses}_task-search_'
                     f'desc-cleaned_run-{run}_bold.nii.gz')
        data = masker.transform(bold_fn).astype(np.float32)
        data = data[:258]
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


def per_voxel_r2(observed, predicted):
    """Per-voxel R² across timepoints.

    R² = 1 - SS_res / SS_tot, with SS_tot using each voxel's own mean
    over the test window. Returns a (V,) array clipped at -1 (anything
    below is "worse than the mean" — we cap to keep the NIfTI dtype
    sane and to make the R² band on summary plots usable).
    """
    obs_mean = observed.mean(axis=0, keepdims=True)
    ss_tot = np.sum((observed - obs_mean) ** 2, axis=0)
    ss_res = np.sum((observed - predicted) ** 2, axis=0)
    # Guard against zero-variance voxels (constant time series)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1.0 - ss_res / ss_tot
    r2[~np.isfinite(r2)] = 0.0
    return np.clip(r2, -1.0, 1.0).astype(np.float32)


def main(subject: int, model_label: int, fold: int,
         n_folds: int = 3,
         bids_folder: str = '/data/ds-retsupp',
         resolution: int = 50,
         voxel_chunk_size: int = 10000,
         paradigm_kind: str = 'full',
         chunk_index: int | None = None,
         n_chunks: int | None = None,
         output_suffix: str = '',
         sd_min: float = 0.3):
    assert_gpu_available_if_expected()
    cfg = MODEL_CFG[model_label]
    bids = Path(bids_folder)
    sub = Subject(subject, bids)
    derivs = bids / 'derivatives'
    chunked_mode = chunk_index is not None
    if chunked_mode and n_chunks is None:
        raise ValueError("chunk_index requires n_chunks")
    fit_t0 = time.time()

    # Fold split.
    folds = make_fold_assignments(sub, n_folds=n_folds)
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold must be in [0, {n_folds-1}], got {fold}")
    test_runs = sorted(folds[fold])
    train_runs = sorted(r for k, ff in enumerate(folds) if k != fold
                        for r in ff)
    print(f"CV fold {fold}/{n_folds-1}:")
    print(f"  train ({len(train_runs)} runs): {train_runs}")
    print(f"  test  ({len(test_runs)} runs): {test_runs}")

    # Mask + masker.
    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()

    # --- Load train + test. ---
    print("Loading train BOLD/paradigm...")
    train_bold, train_par, grid_coords = load_runs(
        sub, masker, resolution, paradigm_kind, train_runs)
    print(f"  train BOLD: T={train_bold.shape[0]}, V={train_bold.shape[1]}")

    print("Loading test BOLD/paradigm...")
    test_bold, test_par, _ = load_runs(
        sub, masker, resolution, paradigm_kind, test_runs)
    print(f"  test  BOLD: T={test_bold.shape[0]}")

    # Chunk subset.
    n_vox = train_bold.shape[1]
    if chunked_mode:
        all_chunks = np.array_split(np.arange(n_vox), n_chunks)
        vox_idx = all_chunks[chunk_index]
        print(f"  chunked: chunk {chunk_index}/{n_chunks}, "
              f"{len(vox_idx)} voxels")
    else:
        vox_idx = np.arange(n_vox)

    fit_train = train_bold[:, vox_idx]
    fit_test = test_bold[:, vox_idx]

    factory = build_model_factory(cfg, grid_coords, sd_min)
    print(f"  sd_min={sd_min}")

    # --- Fit on train. ---
    if cfg['init_from'] is None:
        print("  grid + GD (m1-style)")
        init = grid_fit(factory(fit_train, train_par), fit_train,
                        train_par, voxel_chunk_size, sd_min=sd_min)
    else:
        # Reuse the warm-start from the canonical *mean* fit (model
        # `init_from`); the CV training set is smaller but the warm-
        # start is still a reasonable seed and saves us re-running m1
        # within each cv fold. Same compromise the runwise /
        # conditionwise fits make.
        print(f"  warm-start from canonical m{cfg['init_from']} mean fit")
        init = load_prior_pars(subject, cfg['init_from'], derivs, masker,
                               sd_min=sd_min)
        init = init.iloc[vox_idx].reset_index(drop=True)
        init = init.drop(columns=['r2', 'theta', 'ecc'], errors='ignore')
        if cfg.get('adapt') is not None:
            init = cfg['adapt'](init)

    print("  scheduled GD refinement on training set")
    fitted = gd_fit_scheduled(factory, fit_train, train_par, init,
                              voxel_chunk_size, cfg['schedule'])

    # --- Predict test BOLD; compute per-voxel test R². ---
    print("  predicting held-out runs from training parameters")
    model = factory(fit_test, test_par)   # rebinds paradigm = test_par
    pred = model.predict(parameters=fitted).numpy().astype(np.float32)
    r2 = per_voxel_r2(fit_test, pred)
    print(f"  test R² stats: median={np.median(r2):.3f} "
          f"p90={np.quantile(r2, 0.9):.3f} "
          f"frac>0.1={np.mean(r2 > 0.1):.3f}")

    # --- Save. ---
    out_dir = (derivs / 'prf_cv' / f'model{model_label}'
               / f'fold-{fold}' / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)

    if chunked_mode:
        chunk_dir = out_dir / 'chunks'
        chunk_dir.mkdir(exist_ok=True)
        chunk_path = (chunk_dir
                      / f'chunk-{chunk_index:04d}-of-{n_chunks:04d}.npz')
        np.savez_compressed(chunk_path,
                            r2=r2, vox_idx=vox_idx,
                            n_total_vox=n_vox)
        print(f"wrote {chunk_path}")
    else:
        full_r2 = np.zeros(n_vox, dtype=np.float32)
        full_r2[vox_idx] = r2
        out_path = (out_dir
                    / f'sub-{subject:02d}_fold-{fold}_'
                      f'desc-r2_test{output_suffix}.nii.gz')
        img = masker.inverse_transform(full_r2)
        img.set_data_dtype(np.float32)
        img.header.set_slope_inter(slope=1, inter=0)
        img.to_filename(out_path)
        print(f"wrote {out_path}")

    print(f"total elapsed: {time.time() - fit_t0:.1f}s")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('subject', type=int)
    p.add_argument('model_label', type=int)
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--fold', type=int, required=True,
                   help='Which fold to HOLD OUT (0-indexed).')
    p.add_argument('--n-folds', type=int, default=3)
    p.add_argument('--resolution', type=int, default=50)
    p.add_argument('--voxel-chunk-size', type=int, default=10000)
    p.add_argument('--paradigm-kind', choices=['full', 'bar'],
                   default='full')
    p.add_argument('--chunk-index', type=int, default=None)
    p.add_argument('--n-chunks', type=int, default=None)
    p.add_argument('--output-suffix', default='')
    p.add_argument('--sd-min', type=float, default=0.3)
    args = p.parse_args()

    main(args.subject, args.model_label, args.fold,
         n_folds=args.n_folds, bids_folder=args.bids_folder,
         resolution=args.resolution,
         voxel_chunk_size=args.voxel_chunk_size,
         paradigm_kind=args.paradigm_kind,
         chunk_index=args.chunk_index, n_chunks=args.n_chunks,
         output_suffix=args.output_suffix, sd_min=args.sd_min)
