"""Whole-cortex PRF fit, FULL 8-item paradigm. One model per invocation.

Per-model recipe lives in ``MODEL_CFG``: each entry pairs a model class
with an ``init_from`` predecessor, an ``adapt`` callable (for the
parameter remapping between predecessor and target), and a ``schedule``
(a list of ``(fixed_pars, lr, max_iters, r2_atol)`` GD stages).

Init dependency chain (warm-start sandbox-validated)::

    1: grid + GD                         (single refine stage)
    2: init from 1, +DoG params          (spatial then refine)
    3: init from 1, +flex HRF            (spatial then HRF refine)
    4: init from 2, +flex HRF            (HRF, then spatial+surround refine)
    5: init from 2, +DN params           (spatial then refine)
    6: init from 5, +flex HRF            (HRF then spatial+DN refine)

m4/m5 chain off m2 (clean DoG surround), m6 off m5 (clean DN). HRF is
always introduced last on each branch so spatial parameters are
protected during the surround/DN warming stages.

The schedule's core invariant: spatial (x, y, sd) and HRF
(hrf_delay, hrf_dispersion) are NEVER unlocked in the same stage —
HRF gradients dominate spatial at lr=0.005 and corrupt retinotopy.
``_assert_no_joint_spatial_hrf`` catches violations at import time.

Output: ``derivatives/prf/model{N}/sub-XX/sub-XX_desc-{par}.nii.gz``.
Sentinel rows (zero-variance BOLD or non-finite predictions) are
marked post-hoc via ``mark_invalid_fits`` → NaN params + r²=0; the
output shape always matches the input brain mask.

Voxel chunking: standard PRFs are massively univariate; chunk size
sets a VRAM-vs-throughput tradeoff. Default 10000 fits on L4 / A100 /
H100.
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

from retsupp.utils.data import Subject, mark_invalid_fits, validate_prf_parameters  # noqa: F401


# Schedule primitives. A "schedule" is a list of stage tuples
#   (fixed_pars, learning_rate, max_iters, r2_atol)
# Each stage runs GD on the model with `fixed_pars` held constant.
# r2_atol is the plateau-stop threshold passed to ParameterFitter.fit;
# braincoder exits the stage early when median ΔR² over the last
# 100 iters falls below this. 1e-4 is the right scale for warm/coarse
# stages, 1e-5 for the final refine pass.
SPATIAL = ('x', 'y', 'sd')
HRF = ('hrf_delay', 'hrf_dispersion')
DN_PARAMS = ('rf_amplitude', 'srf_amplitude', 'srf_size',
             'neural_baseline', 'surround_baseline', 'bold_baseline')


# Adapters convert a prior-model parameter frame into the column set
# expected by the target model. Pair-specific because the
# parameterisations are not strictly nested.
def _adapt_m2_from_m1(init):
    """Seed DoG surround from a Gaussian m1 fit.

    ``srf_size = 4.0`` (surround σ = 4 × center σ) sits at the upper
    end of the V1 literature range (Zuiderbaan 2012, Aqil 2021) and
    matches empirical retsupp values, so GD almost always *shrinks*
    rather than grows the surround — safer descent path.
    """
    init = init.copy()
    init['srf_amplitude'] = 5e-2
    init['srf_size'] = 4.0
    return init


def _adapt_m3_from_m1(init):
    init = init.copy()
    init['hrf_delay'] = 4.5
    init['hrf_dispersion'] = 0.75
    return init


def _adapt_m4_from_m2(init):
    init = init.copy()
    init['hrf_delay'] = 4.5
    init['hrf_dispersion'] = 0.75
    return init


def _adapt_m5_from_m2(init):
    """m5 (DN, fixed HRF) reuses m2's clean DoG spatial+surround as
    a seed. Rename amplitude → rf_amplitude (signed; the DN model
    uses |rf_amplitude| in the denominator so negative PRFs are fine).
    Drop ``baseline``; seed the DN-specific scalars."""
    init = init.copy()
    if 'amplitude' in init.columns:
        init['rf_amplitude'] = init['amplitude']
    init = init.drop(columns=['amplitude', 'baseline'], errors='ignore')
    init['neural_baseline'] = 1.0
    init['surround_baseline'] = 1.0
    init['bold_baseline'] = 0.0
    return init


def _adapt_m6_from_m5(init):
    init = init.copy()
    init['hrf_delay'] = 4.5
    init['hrf_dispersion'] = 0.75
    return init


# Per-model config.
#   init_from: which model to load as warm-start seed (None = grid)
#   adapt:     callable (prior_pars_df) -> init_pars_df
#   schedule:  list of (fixed_pars, lr, max_iters, r2_atol) stages.
#              None means the legacy single-shot path is used
#              (currently only m1 grid + GD).
# CHAIN: m2/m3 from m1, m4/m5 from m2 (clean DoG seed),
#        m6 from m5 (clean DN seed). HRF is always introduced LAST in
#        each branch so spatial parameters are protected during the
#        surround/DN warming stages.
MODEL_CFG: dict = {
    1: dict(cls=GaussianPRF2DWithHRF, flex_hrf=False,
            init_from=None, adapt=None,
            schedule=[((), 0.005, 2000, 1e-5)]),
    2: dict(cls=DifferenceOfGaussiansPRF2DWithHRF, flex_hrf=False,
            init_from=1, adapt=_adapt_m2_from_m1,
            schedule=[(SPATIAL, 0.005, 3000, 1e-4),
                      ((),      0.001, 1500, 1e-5)]),
    3: dict(cls=GaussianPRF2DWithHRF, flex_hrf=True,
            init_from=1, adapt=_adapt_m3_from_m1,
            schedule=[(SPATIAL, 0.005, 3000, 1e-4),
                      (HRF,     0.001, 1500, 1e-5)]),
    4: dict(cls=DifferenceOfGaussiansPRF2DWithHRF, flex_hrf=True,
            init_from=2, adapt=_adapt_m4_from_m2,
            # Stage A: only HRF moves; spatial AND surround frozen.
            # Stage B: HRF frozen; spatial + surround refine together.
            schedule=[(SPATIAL + ('srf_amplitude', 'srf_size',
                                   'amplitude', 'baseline'),
                       0.005, 3000, 1e-4),
                      (HRF, 0.001, 1500, 1e-5)]),
    5: dict(cls=DivisiveNormalizationGaussianPRF2DWithHRF, flex_hrf=False,
            init_from=2, adapt=_adapt_m5_from_m2,
            schedule=[(SPATIAL, 0.005, 3000, 1e-4),
                      ((),      0.001, 1500, 1e-5)]),
    6: dict(cls=DivisiveNormalizationGaussianPRF2DWithHRF, flex_hrf=True,
            init_from=5, adapt=_adapt_m6_from_m5,
            schedule=[(SPATIAL + DN_PARAMS, 0.005, 3000, 1e-4),
                      (HRF, 0.001, 1500, 1e-5)]),
}


def _assert_no_joint_spatial_hrf():
    """Guard against schedules that free spatial AND HRF simultaneously.
    HRF gradients at lr=0.005 dominate the per-voxel spatial gradient
    and corrupt retinotopy when both are loose. Each stage of every
    flex-HRF schedule must therefore fix at least one of the two."""
    sp_set, hrf_set = set(SPATIAL), set(HRF)
    for m, cfg in MODEL_CFG.items():
        if not cfg.get('flex_hrf') or cfg.get('schedule') is None:
            continue
        for i, stage in enumerate(cfg['schedule'], start=1):
            fixed = set(stage[0])
            assert (sp_set & fixed) or (hrf_set & fixed), (
                f"Model {m} stage {i}: spatial AND HRF both free — "
                f"violates schedule core rule")


_assert_no_joint_spatial_hrf()


def _cache_path(sub: Subject, kind: str, resolution: int) -> Path:
    return (sub.bids_folder / 'derivatives' / 'cleaned_bold_cache'
            / f'sub-{sub.subject_id:02d}'
            / f'sub-{sub.subject_id:02d}_kind-{kind}_res-{resolution}.npz')


def load_concatenated(sub: Subject, masker, resolution: int, kind: str):
    """Concatenate cleaned BOLD + per-run paradigm across all (ses, run).

    Returns ``(bold (T_total, V), paradigm (T_total, G), grid_coords (G, 2))``.

    Uses ``derivatives/cleaned_bold_cache/sub-XX/...npz`` when present
    (built by ``build_cleaned_bold_cache.py``) — loads in ~5s instead
    of ~60-120s. Chunked fits get most of their wallclock back this way.
    The cache is invalidated by deleting the file; rebuild with the
    builder script's ``--force`` flag.
    """
    cache = _cache_path(sub, kind, resolution)
    if cache.exists():
        d = np.load(cache)
        print(f"  cache hit: {cache.name} "
              f"(bold {d['bold'].shape}, paradigm {d['paradigm'].shape})")
        return d['bold'], d['paradigm'], d['grid_coords']

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


def build_model_factory(cfg, grid_coords, sd_min):
    """Returns a closure ``(data, paradigm) -> model`` for an entry in
    ``MODEL_CFG``. Shared by fit_prf.main and the V1 sandbox so the
    model construction lives in exactly one place.
    """
    hrf = SPMHRFModel(tr=1.6, delay=4.5, dispersion=0.75)

    def factory(data, paradigm):
        return cfg['cls'](
            grid_coordinates=grid_coords, paradigm=paradigm,
            hrf_model=hrf, data=data,
            flexible_hrf_parameters=cfg['flex_hrf'],
            sd_min=sd_min)
    return factory


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


def grid_fit(model, data, paradigm, chunk_size, debug=False, sd_min=0.0):
    """Gaussian (x, y, σ) grid + baseline/amplitude refine.

    ``sd_min`` is the lower bound for σ enforced by the model's transform
    (see :func:`braincoder.models._sd_softplus_forward`). The grid's
    σ-axis lower bound is clamped to at least ``sd_min`` so the initial
    grid search never proposes a σ that would be rejected by the
    transform.
    """
    grid_x = np.linspace(-3, 3, 5 if debug else 12)
    grid_y = np.linspace(-3, 3, 5 if debug else 12)
    sd_low = float(max(1.0, sd_min))
    grid_sd = np.linspace(sd_low, 4.0, 4 if debug else 8)

    def work(idx):
        f = ParameterFitter(model, data[:, idx], paradigm)
        pars = f.fit_grid(grid_x, grid_y, grid_sd, [0], [1],
                          use_correlation_cost=True)
        return f.refine_baseline_and_amplitude(pars, l2_alpha=0.001)

    return chunked(work, data.shape[1], chunk_size, 'grid')


def gd_fit(model_factory, data, paradigm, init_pars, chunk_size,
           max_iter, lr=0.005):
    """Single-stage GD wrapper for callers (e.g. ``fit_condition``) that
    don't need the multi-stage schedule. Equivalent to
    ``gd_fit_scheduled`` with one fully-free stage.
    """
    schedule = [((), lr, max_iter, 1e-5)]
    return gd_fit_scheduled(model_factory, data, paradigm, init_pars,
                             chunk_size, schedule)


def gd_fit_scheduled(model_factory, data, paradigm, init_pars, chunk_size,
                     schedule):
    """Per-chunk staged GD according to ``schedule``.

    ``schedule`` is a list of ``(fixed_pars, lr, max_iter, r2_atol)``
    stages. Within each chunk, parameters are propagated stage-to-stage;
    only ``fixed_pars`` are held constant per stage. Derived columns
    (``r2``, ``theta``, ``ecc``) are stripped between stages so the
    optimizer never sees them.

    Spatial and HRF parameters are NEVER unlocked in the same stage —
    HRF gradients at lr=0.005 dominate the spatial-parameter gradient
    and corrupt retinotopy. The schedule design (see ``MODEL_CFG``) is
    responsible for enforcing this; ``_assert_no_joint_spatial_hrf``
    catches violations at import time.
    """
    def work(idx):
        d = data[:, idx]
        m = model_factory(d, paradigm)
        f = ParameterFitter(m, d, paradigm)
        pars = init_pars.iloc[idx].reset_index(drop=True)
        for stage_i, stage in enumerate(schedule, start=1):
            fixed, lr, n_iter, r2_atol = stage
            fixed_present = [p for p in fixed if p in pars.columns]
            kwargs = dict(init_pars=pars, max_n_iterations=n_iter,
                          learning_rate=lr,
                          fixed_pars=fixed_present if fixed_present else None,
                          r2_atol=r2_atol)
            pars = f.fit(**kwargs)
            # Drop derived columns so the next stage sees a clean frame.
            for derived in ('r2', 'theta', 'ecc'):
                if derived in pars.columns:
                    pars = pars.drop(columns=[derived])
        pars['r2'] = f.get_rsq(pars).values
        return pars

    return chunked(work, data.shape[1], chunk_size, 'GD-sched')


def load_prior_pars(subject: int, model_label: int, derivs: Path,
                    masker, sd_min: float | None = None) -> pd.DataFrame:
    """Read NIfTI parameters of a previous model fit into a DataFrame.

    If ``sd_min`` is given, clip σ-like columns to slightly above the
    floor so braincoder's `_sd_softplus_inverse` doesn't trip on
    numerical drift around the boundary. Sentinel zeros (invalid fits)
    are left alone — they get filtered out by the active-mask later.
    """
    src = derivs / 'prf' / f'model{model_label}' / f'sub-{subject:02d}'
    if not src.exists():
        raise FileNotFoundError(
            f"Prior model {model_label} not found at {src}. "
            f"Fit it first.")
    pars = {}
    for nii in sorted(src.glob(f'sub-{subject:02d}_desc-*.nii.gz')):
        name = nii.name.split('desc-')[1].rsplit('.nii.gz', 1)[0]
        pars[name] = masker.transform(str(nii)).flatten()
    df = pd.DataFrame(pars)
    if sd_min is not None:
        floor = sd_min * 1.01
        if 'sd' in df.columns:
            s = df['sd'].to_numpy()
            mask = (s > 0) & (s <= sd_min)
            if mask.any():
                df.loc[mask, 'sd'] = floor
        if 'srf_size' in df.columns:
            r = df['srf_size'].to_numpy()
            mask = (r > 0) & (r <= 1.0)
            if mask.any():
                df.loc[mask, 'srf_size'] = 1.01
        for pos_col in ('neural_baseline', 'surround_baseline',
                        'srf_amplitude'):
            if pos_col in df.columns:
                v = df[pos_col].to_numpy()
                mask = np.isfinite(v) & (v <= 0)
                if mask.any():
                    df.loc[mask, pos_col] = 1e-4
        validate_prf_parameters(df, sd_min=sd_min,
                                 model_label=model_label, source=str(src))
    return df


def save_pars(pars: pd.DataFrame, masker, target_dir: Path, subject: int):
    target_dir.mkdir(parents=True, exist_ok=True)
    p = pars.copy()
    if 'x' in p and 'y' in p:
        p['theta'] = np.arctan2(p['y'], p['x'])
        p['ecc'] = np.sqrt(p['x'] ** 2 + p['y'] ** 2)
    for col in p.columns:
        masker.inverse_transform(p[col].values).to_filename(
            target_dir / f'sub-{subject:02d}_desc-{col}.nii.gz')


def save_chunk(pars, chunk_idx, chunks_dir: Path, chunk_index, n_chunks,
               fit_meta):
    """Write per-chunk NPZ + JSON sidecar (merged later)."""
    import json
    chunks_dir.mkdir(parents=True, exist_ok=True)
    npz_path = chunks_dir / f'chunk-{chunk_index:04d}-of-{n_chunks:04d}.npz'
    cols = {f'col_{c}': pars[c].values for c in pars.columns}
    np.savez(npz_path, voxel_indices=chunk_idx,
             columns=list(pars.columns), **cols)
    with open(npz_path.with_suffix('.json'), 'w') as fh:
        json.dump(fit_meta, fh, indent=2)
    print(f"Saved chunk: {npz_path}")


def save_full(pars, masker, target_dir: Path, subject, fit_meta):
    """Write per-parameter NIfTIs + JSON sidecar (final, non-chunked)."""
    import json
    save_pars(pars, masker, target_dir, subject)
    meta_path = target_dir / f'sub-{subject:02d}_fit_metadata.json'
    with open(meta_path, 'w') as fh:
        json.dump(fit_meta, fh, indent=2)
    print(f"Saved: {target_dir}")
    print(f"Saved metadata: {meta_path}")


def build_fit_metadata(*, subject, model_label, paradigm_kind, resolution,
                       voxel_chunk_size, schedule, init_from, sd_min,
                       n_voxels, n_timepoints, chunked_mode, chunk_index,
                       n_chunks, elapsed_s):
    """Per-fit JSON sidecar payload."""
    import json, os, socket, platform  # noqa: F401
    return dict(
        model=int(model_label), subject=int(subject),
        paradigm_kind=paradigm_kind, resolution=int(resolution),
        voxel_chunk_size=int(voxel_chunk_size),
        schedule=[{'fixed': list(s[0]), 'lr': float(s[1]),
                   'max_iter': int(s[2]), 'r2_atol': float(s[3])}
                  for s in schedule],
        init_from=init_from, sd_min=float(sd_min),
        n_voxels=int(n_voxels), n_timepoints=int(n_timepoints),
        chunked_mode=bool(chunked_mode),
        chunk_index=(None if chunk_index is None else int(chunk_index)),
        n_chunks=(None if n_chunks is None else int(n_chunks)),
        elapsed_seconds=round(float(elapsed_s), 2),
        host=socket.gethostname(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
        python=platform.python_version(),
    )


def main(subject: int, model_label: int,
         bids_folder: str = '/data/ds-retsupp',
         resolution: int = 50, voxel_chunk_size: int = 10000,
         paradigm_kind: str = 'full',
         debug: bool = False,
         chunk_index: int | None = None,
         n_chunks: int | None = None,
         output_suffix: str = '',
         sd_min: float = 0.3):
    """Fit one PRF model with the per-model schedule defined in
    ``MODEL_CFG``. If ``chunk_index``/``n_chunks`` given, fit ONLY
    voxels in that chunk and save partial results to a ``chunks/``
    subdir; a separate merge step concatenates all chunks into final
    NIfTIs.

    Default ``sd_min=0.3`` enforces the σ floor that prevents the
    sigma-collapse pathology (NaN predictions / phantom R²=1) — the
    same value the V1 sandbox validated.
    """
    cfg = MODEL_CFG[model_label]
    bids = Path(bids_folder)
    sub = Subject(subject, bids)
    derivs = bids / 'derivatives'
    chunked_mode = chunk_index is not None
    if chunked_mode and n_chunks is None:
        raise ValueError("chunk_index requires n_chunks")
    fit_t0 = time.time()

    # Some subjects (e.g. sub-01) lack a run-1 brain_mask; use whatever
    # the first available run is for that session.
    first_run = sub.get_runs(1)[0]
    bold_mask = sub.get_bold_mask(session=1, run=first_run)
    if debug:
        bold_mask = image.math_img(
            'np.where(m.astype(bool) & (np.random.rand(*m.shape) < 0.01), 1, 0)',
            m=bold_mask)
        resolution = 25

    masker = maskers.NiftiMasker(mask_img=bold_mask)
    masker.fit()
    if paradigm_kind == 'bar':
        # Legacy comparison mode: mean BOLD across all runs + single
        # bar-only paradigm (T=258). Apples-to-apples with the original
        # PRF fits (which never modelled the search-array transients).
        mean_ts = (derivs / 'mean_signal' / f'sub-{subject:02d}'
                   / f'sub-{subject:02d}_desc-mean_bold.nii.gz')
        if not mean_ts.exists():
            raise FileNotFoundError(
                f"--paradigm-kind=bar needs precomputed mean BOLD at "
                f"{mean_ts}; run preprocess/mean_ts.py first.")
        print(f"Loading mean BOLD: {mean_ts}")
        data = masker.transform(mean_ts).astype(np.float32)
        # Single-run bar paradigm.
        par_one = sub.get_stimulus(session=1, run=1, resolution=resolution
                                    ).astype(np.float32)
        gx, gy = sub.get_grid_coordinates(session=1, run=1,
                                           resolution=resolution)
        paradigm = par_one.reshape(par_one.shape[0], -1)
        grid_coords = np.stack(
            (gx.ravel(), gy.ravel()), axis=1).astype(np.float32)
        print(f"  BOLD: T={data.shape[0]} (= mean across runs), V={data.shape[1]}")
    else:
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

    # Data view for whatever (chunked or not) the rest of the pipeline
    # operates on.
    fit_data = data[:, chunk_idx] if chunked_mode else data

    factory = build_model_factory(cfg, grid_coords, sd_min)
    print(f"  sd_min={sd_min} (σ lower bound for sd / srf_size / sigma_AF / ...)")

    if cfg['init_from'] is None:
        # Model 1: grid + GD. Grid does the heavy lift; the schedule
        # below is a single refine pass.
        print("\n=== model 1: Gaussian + fixed HRF (grid + schedule) ===")
        init = grid_fit(factory(fit_data, paradigm), fit_data, paradigm,
                         voxel_chunk_size, debug=debug, sd_min=sd_min)
    else:
        # Models 2-6: load prior fit, adapt for this model's column set,
        # then run the multi-stage schedule.
        print(f"\n=== model {model_label} (init from m{cfg['init_from']}) ===")
        # load_prior_pars validates σ > sd_min and raises on legacy
        # NIfTIs that predate the sd_min hook.
        init = load_prior_pars(subject, cfg['init_from'], derivs, masker,
                                sd_min=sd_min)
        if chunked_mode:
            init = init.iloc[chunk_idx].reset_index(drop=True)
        init = init.drop(columns=['r2', 'theta', 'ecc'], errors='ignore')
        if cfg.get('adapt') is not None:
            init = cfg['adapt'](init)
        print(f"  init cols: {list(init.columns)}")

    schedule = cfg['schedule']
    print(f"  schedule: {len(schedule)} stage(s)")
    for i, stage in enumerate(schedule, start=1):
        fixed, lr, n_iter, r2_atol = stage
        print(f"    stage {i}: fix={list(fixed)}  lr={lr}  "
              f"max_iter={n_iter}  r2_atol={r2_atol}")

    pars = gd_fit_scheduled(factory, fit_data, paradigm, init,
                             voxel_chunk_size, schedule)

    # Post-hoc: mark zero-variance / NaN-R² voxels with NaN params +
    # r²=0. Output shape stays aligned with the masker (NIfTI export
    # via masker.inverse_transform still works).
    mark_invalid_fits(pars, fit_data)
    print(f"  median R² (post-mark): {pars['r2'].median():.3f}")

    # Output goes to prf/ for full paradigm (canonical) or prf_bar/ for
    # legacy bar-only paradigm (kept separate so they coexist). The
    # output_suffix lets benchmark / experimental runs write to a
    # separate folder (e.g. prf.bench, prf_bar.bench) without
    # clobbering the canonical results.
    base_dir = 'prf_bar' if paradigm_kind == 'bar' else 'prf'
    if debug:
        base_dir += '.debug'
    if output_suffix:
        base_dir += f'.{output_suffix}'

    fit_meta = build_fit_metadata(
        subject=subject, model_label=model_label,
        paradigm_kind=paradigm_kind, resolution=resolution,
        voxel_chunk_size=voxel_chunk_size, schedule=schedule,
        init_from=cfg['init_from'], sd_min=sd_min,
        n_voxels=data.shape[1], n_timepoints=data.shape[0],
        chunked_mode=chunked_mode, chunk_index=chunk_index,
        n_chunks=n_chunks, elapsed_s=time.time() - fit_t0)

    model_dir = derivs / base_dir / f'model{model_label}' / f'sub-{subject:02d}'
    if chunked_mode:
        save_chunk(pars, chunk_idx, model_dir / 'chunks',
                   chunk_index, n_chunks, fit_meta)
    else:
        save_full(pars, masker, model_dir, subject, fit_meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subject', type=int)
    p.add_argument('--model', type=int, required=True, choices=list(MODEL_CFG))
    p.add_argument('--bids-folder', default='/data/ds-retsupp')
    p.add_argument('--resolution', type=int, default=50)
    p.add_argument('--voxel-chunk-size', type=int, default=10000)
    p.add_argument('--paradigm-kind', choices=['full', 'bar'], default='full')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--chunk-index', type=int, default=None,
                   help='SLURM-array chunk index (0-based). When given, '
                        'fit only this chunk; --n-chunks must also be set. '
                        'Saves a partial NPZ instead of NIfTIs; run '
                        'merge_prf_chunks.py to assemble.')
    p.add_argument('--n-chunks', type=int, default=None,
                   help='Total number of voxel chunks across SLURM tasks.')
    p.add_argument('--output-suffix', default='',
                   help='Append to the output base dir name (e.g. '
                        '--output-suffix=bench writes to prf.bench/ '
                        'instead of prf/). Use for benchmarks or '
                        'experimental runs that should not clobber '
                        'canonical results.')
    p.add_argument('--sd-min', type=float, default=0.2,
                   help='Strict lower bound for every σ-like parameter '
                        '(sd, srf_size, sigma_AF, sigma_dyn, ...) enforced '
                        'via shifted softplus σ = sd_min + softplus(raw). '
                        'Default 0.2° ≈ 1 grid-pixel at resolution=50, '
                        'which is the mathematical minimum (σ < grid '
                        'spacing is degenerate). Permits small V1-fovea '
                        'PRFs while still blocking the σ→0 collapse '
                        'pathology.')
    a = p.parse_args()
    main(a.subject, a.model, bids_folder=a.bids_folder,
         resolution=a.resolution, voxel_chunk_size=a.voxel_chunk_size,
         paradigm_kind=a.paradigm_kind, debug=a.debug,
         chunk_index=a.chunk_index, n_chunks=a.n_chunks,
         output_suffix=a.output_suffix,
         sd_min=a.sd_min)
