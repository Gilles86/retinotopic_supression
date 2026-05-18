"""Joint Dynamic Attention-Field + DoG-PRF fit at the BOLD signal level.

DoG-voxel-kernel counterpart to
:mod:`retsupp.modeling.fit_dynamic_af_braincoder`. The voxel kernel is
a Difference-of-Gaussians (centre + surround) — same shape as model 4
(``DoG + flexible HRF``) — and the AF modulation is one of the dynamic
variants (v2 or v3).

Two model versions are available, controlled by ``--model-version``.

v2 (default — shared σ, split dyn gain)
---------------------------------------
Uses :class:`braincoder.models.DoGDynamicAttentionFieldPRF2DWithHRF_v2`.

5 shared parameters: ``sigma_AF, g_HP, g_LP, g_HP_dyn, g_LP_dyn``.
The dynamic AF Gaussian uses the SAME ``sigma_AF`` as the sustained
term, and the per-TR phasic gain is split into HP-vs-LP.

v3 (separate σ_dyn AND split dyn gain)
--------------------------------------
Uses :class:`braincoder.models.DoGDynamicAttentionFieldPRF2DWithHRF_v3`.

6 shared parameters: ``sigma_AF, g_HP, g_LP, sigma_dyn, g_HP_dyn,
g_LP_dyn``. Most flexible variant.

Per-voxel free parameters: ``x, y, sd, baseline, amplitude,
srf_amplitude, srf_size`` (initialized from model 4, KEEPING
srf_size/srf_amplitude — mirrors :mod:`fit_dog_af_prf_braincoder`).

This always uses the FULL paradigm (bar + distractor disks at the 4
ring locations).

Output
------
A pickle and a TSV under
``derivatives/af_prf_joint_dynamic_v{2,3}_dog/sub-XX/sub-XX_roi-XX_*.{pkl,tsv}``.
The output filenames also include ``dog-dyn-v2`` / ``dog-dyn-v3`` so
they never collide with the Gaussian-AF dynamic outputs or the static
DoG-AF outputs.

Usage
-----
``python -m retsupp.modeling.fit_dog_dynamic_af_braincoder 2 --roi V3AB``
``python -m retsupp.modeling.fit_dog_dynamic_af_braincoder 2 --roi V3AB --model-version v3``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from nilearn import image, input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v2,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3,
)
from braincoder.optimize import ParameterFitter
from retsupp.modeling.local_models import (
    DivisiveNormalizationDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_oversampled,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_sharedDynGain,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma_sharedDynGain,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition_dynHP,
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_repeat,
)
from retsupp.utils.sentinels import assert_gpu_available_if_expected
from retsupp.utils.data import (
    Subject,
    distractor_locations,
)


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']


def get_ring_positions():
    """Return the (n_C, 2) array of ring positions matching CONDITIONS order."""
    keys = ['upper right', 'upper left', 'lower left', 'lower right']
    return np.array([list(distractor_locations[k]) for k in keys], dtype=np.float32)


def build_data_and_paradigm(sub: Subject, resolution: int = 50,
                            grid_radius: float = 5.0,
                            with_target: bool = False,
                            temporal_oversampling: int = 1,
                            with_run_position: bool = False,
                            with_repeat_split: bool = False,
                            distractor_shape: str = 'circle',
                            distractor_long_side: float = 1.5,
                            distractor_short_side: float = 0.5):
    """Load cleaned BOLD + paradigm + condition_indicator + dynamic_indicator.

    Always uses the FULL paradigm (bar + distractor disks). Identical to
    the Gaussian dynamic loader.

    Parameters
    ----------
    with_target : bool, default False
        If True, also assemble a per-TR target_indicator
        (n_T_total, 4) by concatenating
        :meth:`Subject.get_target_indicator` across runs (same channel
        order as ``dynamic_indicator``). Returned as the last element of
        the tuple; otherwise that slot is ``None``.
    temporal_oversampling : int, default 1
        Temporal oversampling factor ``N``. When ``N > 1``, the
        paradigm and condition_indicator are repeat-expanded along the
        time axis (each TR row repeated ``N`` times — paradigm and
        condition state are constant within a TR), and the
        dynamic_indicator / target_indicator are recomputed at fine
        timestep ``dt = TR / N``. The BOLD data remains at TR
        resolution. Returned shapes:

        - ``bold``                : ``(T,    V)``
        - ``paradigm_full``       : ``(T*N,  G)``
        - ``condition_indicator`` : ``(T*N,  4)``
        - ``dynamic_indicator``   : ``(T*N,  4)``
        - ``target_indicator``    : ``(T*N,  4)`` or ``None``
    """
    if int(temporal_oversampling) < 1:
        raise ValueError(
            f"temporal_oversampling must be >= 1, got {temporal_oversampling}")
    temporal_oversampling = int(temporal_oversampling)
    N = temporal_oversampling

    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius,
    )
    grid_coordinates = np.stack(
        (gx.ravel(), gy.ravel()), axis=1,
    ).astype(np.float32)

    hp_per_run = sub.get_hpd_locations()
    print('HP per run:', hp_per_run)

    # Cached cleaned BOLD: ~5s vs ~60-120s of per-run masker.transform.
    # The cache concatenates ALL (ses, run) at 258 TR/run regardless of
    # whether AF later skips a run for non-canonical HP, so we always
    # advance the cum_offset by CACHE_T_PER_RUN on every iteration of
    # the run loop. Distractor shape is NOT baked into BOLD — the cache
    # builder uses rectangle, but BOLD is the same regardless, so this
    # cache works for any distractor_shape argument.
    CACHE_T_PER_RUN = 258
    cache_path = (sub.bids_folder / 'derivatives' / 'cleaned_bold_cache'
                  / f'sub-{sub.subject_id:02d}'
                  / f'sub-{sub.subject_id:02d}_kind-full_res-{resolution}.npz')
    cached_bold = None
    if cache_path.exists():
        try:
            _c = np.load(cache_path)
            cached_bold = _c['bold']
            _c.close()
            print(f'  cached BOLD: {cache_path.name} ({cached_bold.shape})')
        except Exception as exc:
            print(f'  cache load failed ({exc}); falling back to NIfTIs')
            cached_bold = None

    bold_chunks = []
    paradigm_chunks = []
    cond_indicator_chunks = []
    dyn_indicator_chunks = []
    tgt_indicator_chunks = [] if with_target else None
    runpos_chunks = [] if with_run_position else None
    repeat_indicator_chunks = [] if with_repeat_split else None
    masker.fit(bold_mask)
    cum_offset = 0

    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    for session, run in tqdm(session_runs, desc='Loading runs'):
        hp = hp_per_run[(session, run)]
        if hp not in CONDITIONS:
            print(f'  ses-{session}_run-{run}: HP={hp!r}, skipping')
            # Advance the cache cursor even on skip — the cache concatenates
            # every (ses, run) pair regardless of HP filtering.
            if cached_bold is not None:
                cum_offset += CACHE_T_PER_RUN
            continue

        par_run = sub.get_stimulus_with_distractors(
            session=session, run=run, resolution=resolution,
            grid_radius=grid_radius,
            distractor_shape=distractor_shape,
            distractor_long_side=distractor_long_side,
            distractor_short_side=distractor_short_side,
        ).astype(np.float32)
        par_run_flat = par_run.reshape((par_run.shape[0], -1))
        n_T_run = par_run_flat.shape[0]

        if cached_bold is not None:
            data = cached_bold[cum_offset:cum_offset + CACHE_T_PER_RUN].astype(
                np.float32)
            cum_offset += CACHE_T_PER_RUN
        else:
            bold_fn = (
                sub.bids_folder / 'derivatives' / 'cleaned'
                / f'sub-{sub.subject_id:02d}' / f'ses-{session}' / 'func'
                / f'sub-{sub.subject_id:02d}_ses-{session}_task-search_'
                  f'desc-cleaned_run-{run}_bold.nii.gz'
            )
            data = masker.transform(bold_fn).astype(np.float32)
        if data.shape[0] < n_T_run:
            print(f'  ses-{session}_run-{run}: short by '
                  f'{n_T_run - data.shape[0]} TRs, padding with zeros')
            pad = np.zeros((n_T_run - data.shape[0], data.shape[1]),
                           dtype=np.float32)
            data = np.vstack([data, pad])
        elif data.shape[0] > n_T_run:
            data = data[:n_T_run]

        hp_idx = CONDITIONS.index(hp)
        cond_indicator = np.zeros((n_T_run, len(CONDITIONS)),
                                  dtype=np.float32)
        cond_indicator[:, hp_idx] = 1.0

        # Dynamic indicator at fine dt (or TR if N=1).
        dyn_indicator = sub.get_dynamic_indicator(
            session=session, run=run, oversampling=N,
        ).astype(np.float32)
        # Pad / crop to (n_T_run * N, 4).
        n_T_run_fine = n_T_run * N
        if dyn_indicator.shape[0] < n_T_run_fine:
            pad = np.zeros((n_T_run_fine - dyn_indicator.shape[0],
                            dyn_indicator.shape[1]), dtype=np.float32)
            dyn_indicator = np.vstack([dyn_indicator, pad])
        elif dyn_indicator.shape[0] > n_T_run_fine:
            dyn_indicator = dyn_indicator[:n_T_run_fine]

        if with_target:
            tgt_indicator = sub.get_target_indicator(
                session=session, run=run, oversampling=N,
            ).astype(np.float32)
            if tgt_indicator.shape[0] < n_T_run_fine:
                pad = np.zeros((n_T_run_fine - tgt_indicator.shape[0],
                                tgt_indicator.shape[1]), dtype=np.float32)
                tgt_indicator = np.vstack([tgt_indicator, pad])
            elif tgt_indicator.shape[0] > n_T_run_fine:
                tgt_indicator = tgt_indicator[:n_T_run_fine]
            tgt_indicator_chunks.append(tgt_indicator)

        if with_run_position:
            # Compute scalar run-position for this (session, run) and
            # broadcast to a (n_T_run, 3) one-hot per-TR indicator.
            pos_int = sub.get_run_position_per_tr(session, run, hp_per_run)
            runpos_run = np.zeros((n_T_run, 3), dtype=np.float32)
            runpos_run[:, pos_int] = 1.0
            print(f'  ses-{session}_run-{run}: HP={hp!r}, '
                  f'run-position={pos_int}')
            runpos_chunks.append(runpos_run)

        if with_repeat_split:
            rep_indicator = sub.get_repeat_distractor_indicator(
                session=session, run=run, oversampling=N,
            ).astype(np.float32)
            if rep_indicator.shape[0] < n_T_run_fine:
                pad = np.zeros((n_T_run_fine - rep_indicator.shape[0],
                                rep_indicator.shape[1]), dtype=np.float32)
                rep_indicator = np.vstack([rep_indicator, pad])
            elif rep_indicator.shape[0] > n_T_run_fine:
                rep_indicator = rep_indicator[:n_T_run_fine]
            repeat_indicator_chunks.append(rep_indicator)

        # Repeat-expand paradigm and condition_indicator along the time
        # axis when oversampling. Paradigm and HP-condition state are
        # constant within a TR. np.repeat with axis=0 turns row i of
        # length-T into rows i*N..(i+1)*N-1 of length-T*N.
        if N > 1:
            par_run_flat = np.repeat(par_run_flat, N, axis=0)
            cond_indicator = np.repeat(cond_indicator, N, axis=0)
            if with_run_position:
                # Run-position is also constant within a TR.
                runpos_chunks[-1] = np.repeat(runpos_chunks[-1], N, axis=0)

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)
        dyn_indicator_chunks.append(dyn_indicator)

    bold = np.vstack(bold_chunks)
    paradigm_full = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_indicator_chunks)
    dynamic_indicator = np.vstack(dyn_indicator_chunks)
    target_indicator = (np.vstack(tgt_indicator_chunks)
                        if with_target else None)
    run_position_indicator = (np.vstack(runpos_chunks)
                              if with_run_position else None)
    repeat_indicator = (np.vstack(repeat_indicator_chunks)
                        if with_repeat_split else None)

    print(f'Loaded BOLD: shape {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}, '
          f'dynamic_indicator {dynamic_indicator.shape} '
          f'(min={dynamic_indicator.min():.3f}, '
          f'max={dynamic_indicator.max():.3f}, '
          f'mean={dynamic_indicator.mean():.4f})')
    if with_target:
        print(f'target_indicator {target_indicator.shape} '
              f'(min={target_indicator.min():.3f}, '
              f'max={target_indicator.max():.3f}, '
              f'mean={target_indicator.mean():.4f})')
    if with_run_position:
        print(f'run_position_indicator {run_position_indicator.shape} '
              f'(per-position counts: '
              f'{run_position_indicator.sum(axis=0).astype(int).tolist()})')
    if with_repeat_split:
        rep_sum = repeat_indicator.sum()
        dyn_sum = dynamic_indicator.sum()
        rep_frac = (rep_sum / dyn_sum) if dyn_sum > 0 else 0.0
        print(f'repeat_indicator {repeat_indicator.shape} '
              f'(repeat fraction: {rep_frac:.3f}; '
              f'switch indicator implied)')
    if N > 1:
        print(f'temporal_oversampling: N={N} '
              f'(BOLD at TR={bold.shape[0]} rows; '
              f'paradigm/indicators at fine dt={paradigm_full.shape[0]} rows)')
    return (
        pd.DataFrame(bold),
        paradigm_full,
        condition_indicator,
        dynamic_indicator,
        grid_coordinates,
        masker,
        target_indicator,
        run_position_indicator,
        repeat_indicator,
    )


def select_roi_voxels(sub: Subject, roi: str, prf_pars: pd.DataFrame,
                       r2_thr: float = 0.05,
                       r2_max: float = 0.999,
                       sd_min: float = 0.05):
    """Boolean voxel mask: ROI ∧ (r2_thr < R² < r2_max) ∧ (sd > sd_min).

    Legacy R²-based selector. The ``r2_max`` upper bound drops phantom
    voxels — m4 (DoG flex-HRF) sometimes collapses σ → 0, producing NaN
    model predictions, which ``(resid**2).sum(skipna=True)`` reduces to
    0, giving R²=1.0. The ``sd_min`` filter also drops collapsed-σ
    voxels (some have R²<1 but σ ≈ 0).

    Use :func:`select_roi_voxels_psignal` when you want the posterior-
    based selector that consumes the GMM ``p_signal`` NIfTI directly.
    """
    roi_aliases = {
        'V3AB': ['V3A', 'V3B'],
        'LO': ['LO1', 'LO2'],
        'TO': ['TO1', 'TO2'],
        'VO': ['VO1', 'VO2'],
    }
    component_rois = roi_aliases.get(roi, [roi])

    masker_full = sub.get_bold_mask(return_masker=True)
    masker_full.fit()
    roi_arr = np.zeros(prf_pars.shape[0], dtype=bool)
    for r in component_rois:
        roi_img = sub.get_retinotopic_roi(roi=r, bold_space=True)
        roi_arr |= masker_full.transform(roi_img).astype(bool).flatten()
    r2 = prf_pars['r2'].values
    sd = prf_pars.get('sd', pd.Series(np.full(prf_pars.shape[0], np.inf))).values
    return (r2 > r2_thr) & (r2 < r2_max) & (sd > sd_min) & roi_arr


def select_roi_voxels_psignal(
        sub: Subject, roi: str, prf_pars: pd.DataFrame,
        *,
        model_label: int,
        p_signal_thr: float,
        aperture_mass_thr: float = 0.0,
        aperture_radius: float = 3.17):
    """Boolean voxel mask using the GMM per-voxel ``p_signal`` posterior.

    Drops all the legacy R²-band-aids (``r2_max``, ``sd_min``,
    fixed-α R²-FDR threshold) — the logit-Gaussian mixture's per-voxel
    posterior already handles phantom voxels (p_signal → 0 at R² → 1)
    and σ-collapsed voxels (low R² → low p_signal).

    Args:
        sub, roi, prf_pars: as :func:`select_roi_voxels`.
        model_label: PRF model whose ``desc-p_signal.nii.gz`` to read.
        p_signal_thr: keep voxels with P(signal | R²) > this. 0.95
            ≈ "individually ≥95% likely to be signal".
        aperture_mass_thr: optional aperture-mass filter. If > 0, also
            require ≥ this fraction of the PRF Gaussian's mass to fall
            inside the bar-paradigm aperture (Gaussian radial-CDF
            approximation; matches
            ``retsupp.utils.data.select_well_fit_voxels``). Set 0 to
            skip — useful for testing whether peripheral-RF voxels
            change the AF result.
        aperture_radius: aperture radius in deg (retsupp default 3.17).
    """
    from scipy.stats import norm

    roi_aliases = {
        'V3AB': ['V3A', 'V3B'],
        'LO': ['LO1', 'LO2'],
        'TO': ['TO1', 'TO2'],
        'VO': ['VO1', 'VO2'],
    }
    component_rois = roi_aliases.get(roi, [roi])

    masker_full = sub.get_bold_mask(return_masker=True)
    masker_full.fit()
    roi_arr = np.zeros(prf_pars.shape[0], dtype=bool)
    for r in component_rois:
        roi_img = sub.get_retinotopic_roi(roi=r, bold_space=True)
        roi_arr |= masker_full.transform(roi_img).astype(bool).flatten()

    # Per-voxel p_signal NIfTI written by run_r2_mixture_all (NaN for
    # voxels outside any per-ROI fit / below count threshold).
    p_sig_path = (sub.bids_folder / 'derivatives' / 'prf'
                   / f'model{model_label}' / f'sub-{sub.subject_id:02d}'
                   / f'sub-{sub.subject_id:02d}_desc-p_signal.nii.gz')
    if not p_sig_path.exists():
        raise FileNotFoundError(
            f'p_signal NIfTI missing for sub-{sub.subject_id:02d} '
            f'model {model_label}: {p_sig_path}. Run r2_mixture first.')
    p_sig = masker_full.transform(str(p_sig_path)).flatten()
    # NaN treated as False (not signal).
    sig_ok = np.isfinite(p_sig) & (p_sig > p_signal_thr)

    mask = roi_arr & sig_ok
    if aperture_mass_thr > 0:
        x = prf_pars['x'].to_numpy()
        y = prf_pars['y'].to_numpy()
        sd = prf_pars['sd'].to_numpy()
        sd_safe = np.where(np.isfinite(sd) & (sd > 0.05), sd, 0.05)
        eccen = np.sqrt(x ** 2 + y ** 2)
        mass_in = 1.0 - norm.cdf((eccen - aperture_radius) / sd_safe)
        mask &= np.isfinite(mass_in) & (mass_in >= aperture_mass_thr)
    return mask


def _write_dataset_description_if_missing(
        analysis_dir: Path, *,
        script_name: str, parameters: dict) -> None:
    """Write a BIDS-derivatives ``dataset_description.json`` at the
    top-level analysis dir, idempotently.

    Race-safe between sibling SLURM array tasks: open-with-'x' fails
    cleanly if another task got there first. Captures the current
    repo's git commit hash so anyone reading the dir later knows
    exactly which fitter version produced it.
    """
    import datetime
    import json
    import subprocess
    desc_path = analysis_dir / 'dataset_description.json'
    if desc_path.exists():
        return
    analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', '--short', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = 'unknown'
    desc = {
        'Name': analysis_dir.name,
        'BIDSVersion': '1.7.0',
        'DatasetType': 'derivative',
        'GeneratedBy': [{
            'Name': script_name,
            'Version': commit,
            'CodeURL': 'https://github.com/Gilles86/retinotopic_supression',
            'Parameters': parameters,
        }],
        'FirstWritten': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    try:
        with open(desc_path, 'x') as f:
            json.dump(desc, f, indent=2)
    except FileExistsError:
        pass  # another array task got there first; harmless.


def main(subject: int, bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V3AB',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         p_signal_thr: float = 0.5,
         aperture_mass_thr: float = 0.0,
         model_label: int = 4,
         max_voxels: int | None = 500,
         mode: str = 'signed',
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         output_subdir: str | None = None,
         model_version: str = 'v2',
         sigma_af_init: float = 2.0,
         sigma_dyn_init: float = 2.0,
         with_target: bool = False,
         g_t_dyn_init: float = 0.0,
         sigma_t_dyn_init: float = 2.0,
         temporal_oversampling: int | None = None,
         shared_target_sigma: bool = False,
         shared_dyn_gain: bool = False,
         all_shared_sigma: bool = False,
         per_run_position_gains: bool = False,
         per_run_position_dyn_hp: bool = False,
         with_repeat_split: bool = False,
         distractor_shape: str = 'circle',
         distractor_long_side: float = 1.5,
         distractor_short_side: float = 0.5):
    """Top-level fit driver.

    Parameters
    ----------
    temporal_oversampling : int or None, default None
        ``None`` means "do not run the oversampled code path" — the
        legacy path is used and no ``_tos{N}`` tag is appended to the
        output subdir, preserving exact behaviour for callers that
        don't pass ``--temporal-oversampling``. Any integer ``>= 1``
        routes through the temporally-oversampled subclass and tags the
        output subdir with ``_tos{N}`` (so even ``N=1`` lives in its
        own folder, segregated from the legacy fits). Only valid with
        ``model_version='v3'`` and ``with_target=True``.
    """
    assert_gpu_available_if_expected()   # fail fast on cuInit-race CPU fallback
    if model_version not in ('v2', 'v3'):
        raise ValueError(
            f"model_version must be 'v2' or 'v3', got {model_version!r}")
    if with_target and model_version != 'v3':
        raise ValueError(
            "--with-target is only supported with --model-version v3 "
            f"(got {model_version!r}).")
    if shared_target_sigma and not (with_target and model_version == 'v3'):
        raise ValueError(
            "--shared-target-sigma is only supported with "
            "--model-version v3 + --with-target.")
    if (shared_dyn_gain or all_shared_sigma) and not shared_target_sigma:
        raise ValueError(
            "--shared-dyn-gain / --all-shared-sigma require "
            "--shared-target-sigma (built on top of the sharedSigma family).")
    if per_run_position_gains and not (with_target and shared_target_sigma
                                       and model_version == 'v3'):
        raise ValueError(
            "--per-run-position-gains requires --model-version v3 + "
            "--with-target + --shared-target-sigma.")
    if per_run_position_dyn_hp and not per_run_position_gains:
        raise ValueError(
            "--per-run-position-dyn-hp requires --per-run-position-gains "
            "(it extends the runPosition model).")
    if with_repeat_split and not (with_target and shared_target_sigma
                                  and model_version == 'v3'):
        raise ValueError(
            "--with-repeat-split requires --model-version v3 + "
            "--with-target + --shared-target-sigma.")
    if with_repeat_split and per_run_position_gains:
        raise ValueError(
            "--with-repeat-split is not currently combined with "
            "--per-run-position-gains. Pick one extension.")
    use_oversampled_codepath = temporal_oversampling is not None
    if use_oversampled_codepath:
        if int(temporal_oversampling) < 1:
            raise ValueError(
                f"temporal_oversampling must be >= 1, "
                f"got {temporal_oversampling}")
        temporal_oversampling = int(temporal_oversampling)
        if not (with_target and model_version == 'v3'):
            raise ValueError(
                "--temporal-oversampling is only supported with "
                "--model-version v3 + --with-target (which routes "
                "through the oversampled local-models subclass).")
        if shared_target_sigma:
            raise ValueError(
                "--shared-target-sigma is not yet supported in the "
                "oversampled code path. Add a SharedSigmaOversampled "
                "subclass in local_models.py if you need both.")
        if per_run_position_gains:
            raise ValueError(
                "--per-run-position-gains is not yet supported in the "
                "oversampled code path.")
    else:
        # Legacy path -> always use N=1 internally for the few code
        # paths (build_data_and_paradigm) that take the kwarg.
        temporal_oversampling = 1
    if distractor_shape not in ('circle', 'rectangle'):
        raise ValueError(
            f"distractor_shape must be 'circle' or 'rectangle', "
            f"got {distractor_shape!r}")
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        if with_target:
            base = 'af_prf_joint_dynamic_v3_dog_with_target'
        else:
            base = {
                'v2': 'af_prf_joint_dynamic_v2_dog',
                'v3': 'af_prf_joint_dynamic_v3_dog',
            }[model_version]
        if shared_target_sigma:
            base = f'{base}_sharedSigma'
        if all_shared_sigma:
            base = f'{base}_allSharedSigma'
        if shared_dyn_gain:
            base = f'{base}_sharedDynGain'
        if per_run_position_gains:
            base = f'{base}_runPosition'
        if per_run_position_dyn_hp:
            base = f'{base}_dynHP'
        if with_repeat_split:
            base = f'{base}_repeat'
        if use_oversampled_codepath:
            base = f'{base}_tos{temporal_oversampling}'
        if distractor_shape == 'rectangle':
            # Keep rectangle fits in their own folder so they don't
            # overwrite the canonical circle baseline.
            base = f'{base}_rect'
        if p_signal_thr > 0:
            tag = f'pSig{p_signal_thr:g}'
            if aperture_mass_thr > 0:
                tag = f'{tag}_apt{aperture_mass_thr:g}'
            base = f'{base}_{tag}'
        if model_label != 4:
            base = f'{base}_base-m{model_label}'
        output_subdir = base
    analysis_dir = bids_folder / 'derivatives' / output_subdir
    out_dir = analysis_dir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_description_if_missing(
        analysis_dir,
        script_name='fit_dog_dynamic_af_braincoder.py',
        parameters=dict(
            model_version=model_version,
            spatial_model='dog',
            with_target=with_target,
            shared_target_sigma=shared_target_sigma,
            shared_dyn_gain=shared_dyn_gain,
            all_shared_sigma=all_shared_sigma,
            per_run_position_gains=per_run_position_gains,
            per_run_position_dyn_hp=per_run_position_dyn_hp,
            with_repeat_split=with_repeat_split,
            distractor_shape=distractor_shape,
            distractor_long_side=distractor_long_side,
            distractor_short_side=distractor_short_side,
            p_signal_thr=p_signal_thr,
            aperture_mass_thr=aperture_mass_thr,
            mode=mode,
            max_voxels=max_voxels,
            resolution=resolution,
            grid_radius=grid_radius,
            max_n_iterations=max_n_iterations,
            learning_rate=learning_rate,
            sigma_af_init=sigma_af_init,
            sigma_dyn_init=sigma_dyn_init,
            sigma_t_dyn_init=sigma_t_dyn_init,
            g_t_dyn_init=g_t_dyn_init,
            model_label=model_label,
            temporal_oversampling=temporal_oversampling,
        ),
    )

    tos_str = (f' | tos={temporal_oversampling}'
               if use_oversampled_codepath else '')
    shared_str = ' | sharedSigma' if shared_target_sigma else ''
    runpos_str = ' | runPosition' if per_run_position_gains else ''
    if per_run_position_dyn_hp:
        runpos_str = f'{runpos_str}+dynHP'
    repeat_str = ' | repeat-split' if with_repeat_split else ''
    shape_str = (f' | shape={distractor_shape}'
                 f'({distractor_long_side}x{distractor_short_side})'
                 if distractor_shape == 'rectangle'
                 else f' | shape={distractor_shape}')
    print(f'== sub-{subject:02d} | roi={roi} | mode={mode} | '
          f'model-version={model_version}'
          f'{" + target" if with_target else ""}'
          f'{shared_str}{runpos_str}{repeat_str}{tos_str}{shape_str} | '
          f'paradigm=full (DoG voxel kernel, dynamic AF) ==')

    # 1) Load BOLD + paradigm + condition_indicator + dynamic_indicator
    #    (+ target_indicator if requested).
    (bold_df, paradigm, condition_indicator, dynamic_indicator,
     grid_coords, masker, target_indicator,
     run_position_indicator, repeat_indicator) = build_data_and_paradigm(
        sub,
        resolution=resolution,
        grid_radius=grid_radius,
        with_target=with_target,
        temporal_oversampling=temporal_oversampling,
        with_run_position=per_run_position_gains,
        with_repeat_split=with_repeat_split,
        distractor_shape=distractor_shape,
        distractor_long_side=distractor_long_side,
        distractor_short_side=distractor_short_side,
    )

    # 2) Restrict to ROI voxels via the logit-GMM per-voxel p_signal
    #    posterior. No R²/σ band-aids — the mixture handles phantom and
    #    σ-collapsed voxels naturally (p_signal → 0 at R²→1 and at low R²).
    prf_pars = sub.get_prf_parameters_volume(model=model_label, return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)

    voxel_mask = select_roi_voxels_psignal(
        sub, roi, prf_pars,
        model_label=model_label,
        p_signal_thr=p_signal_thr,
        aperture_mass_thr=aperture_mass_thr)
    descr = (f'ROI {roi} | p_signal > {p_signal_thr}'
             + (f' AND mass_in ≥ {aperture_mass_thr}'
                if aperture_mass_thr > 0 else ''))
    print(f'{descr}: {voxel_mask.sum()} voxels')
    if voxel_mask.sum() == 0:
        raise RuntimeError(f'No voxels survive: {descr}.')

    if max_voxels is not None and voxel_mask.sum() > max_voxels:
        order = np.argsort(prf_pars['r2'].values)[::-1]
        keep = np.zeros_like(voxel_mask)
        ranked = [v for v in order if voxel_mask[v]][:max_voxels]
        keep[ranked] = True
        voxel_mask = keep
        print(f'  -> top {max_voxels} voxels by r²')

    bold_sub = bold_df.loc[:, voxel_mask].copy()

    # 3) Initialize from the base PRF model. The AF refit takes per-voxel
    # spatial estimates (x, y, sd, surround params) and the HRF estimate
    # from the base model; AF gains / σ_AF are added below with sane
    # defaults.
    #
    # We support two base-model families:
    # - DoG (m2, m4): rf is `Difference-of-Gaussians`. Uses the canonical
    #   DoG-AF model class. init_cols include baseline + amplitude.
    # - DN  (m5, m6): rf is `Divisive Normalization`. Uses the DN-AF
    #   model class; init_cols include rf_amplitude + neural_baseline +
    #   surround_baseline + bold_baseline.
    is_dn_base = model_label in (5, 6)
    if is_dn_base:
        init_cols = ['x', 'y', 'sd',
                     'rf_amplitude', 'srf_amplitude', 'srf_size',
                     'neural_baseline', 'surround_baseline', 'bold_baseline',
                     'hrf_delay', 'hrf_dispersion']
    else:
        init_cols = ['x', 'y', 'sd', 'baseline', 'amplitude',
                     'srf_amplitude', 'srf_size',
                     'hrf_delay', 'hrf_dispersion']
    missing = [c for c in init_cols if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model {model_label} is missing required params {missing}. '
            f'Available: {list(prf_pars.columns)}. '
            f'Only DoG (m2, m4) and DN+HRF (m5, m6) base models are '
            f'supported — Gaussian-only models lack srf_*.')
    init_pars = prf_pars.loc[voxel_mask, init_cols].copy()

    # AF inits.
    init_pars['sigma_AF'] = sigma_af_init
    init_pars['g_HP'] = 0.0 if mode == 'signed' else 0.30
    init_pars['g_LP'] = 0.0 if mode == 'signed' else 0.10

    if model_version == 'v2':
        # v2: shared sigma_AF + split HP/LP dynamic gains.
        init_pars['g_HP_dyn'] = 0.0 if mode == 'signed' else 0.10
        init_pars['g_LP_dyn'] = 0.0 if mode == 'signed' else 0.10
        shared_pars = ['sigma_AF', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn']
    else:
        # v3: separate sigma_dyn (smaller default — distractor ~0.4°) AND
        # split HP/LP dynamic gains. 6 shared parameters total.
        init_pars['sigma_dyn'] = sigma_dyn_init
        init_pars['g_HP_dyn'] = 0.0 if mode == 'signed' else 0.10
        init_pars['g_LP_dyn'] = 0.0 if mode == 'signed' else 0.10
        shared_pars = ['sigma_AF', 'g_HP', 'g_LP',
                       'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']
        if with_target:
            # v3 + target: 2 additional shared parameters.
            init_pars['g_T_dyn'] = g_t_dyn_init
            if shared_target_sigma:
                # Tie sigma_T_dyn to sigma_dyn at init so the (effectively
                # unused) σ_T_dyn raw variable starts in the right place.
                # The model's forward transform will overwrite slot 14
                # with slot 10 on every iteration anyway.
                init_pars['sigma_T_dyn'] = init_pars['sigma_dyn']
            else:
                init_pars['sigma_T_dyn'] = sigma_t_dyn_init
            shared_pars = shared_pars + ['g_T_dyn', 'sigma_T_dyn']
        if all_shared_sigma:
            # Tie sigma_AF to sigma_dyn at init. The model forward transform
            # overrides slot 7 with slot 10 on every iteration; this just
            # makes the (effectively unused) σ_AF raw variable start at
            # the right place.
            init_pars['sigma_AF'] = init_pars['sigma_dyn']
        if shared_dyn_gain:
            # Tie g_LP_dyn to g_HP_dyn at init. Model forward tying
            # handles the rest.
            init_pars['g_LP_dyn'] = init_pars['g_HP_dyn']

        if per_run_position_gains:
            # 6 new sustained gains, one pair per within-block run-position.
            # Originals (g_HP, g_LP) at slots 8/9 are kept (for slot
            # consistency with the parent) but unused in the forward.
            init_pars['g_HP'] = 0.0
            init_pars['g_LP'] = 0.0
            for r in (0, 1, 2):
                init_pars[f'g_HP_pos{r}'] = 0.0
                init_pars[f'g_LP_pos{r}'] = 0.0
            shared_pars = shared_pars + [
                'g_HP_pos0', 'g_HP_pos1', 'g_HP_pos2',
                'g_LP_pos0', 'g_LP_pos1', 'g_LP_pos2',
            ]

        if per_run_position_dyn_hp:
            # 3 new gains splitting g_HP_dyn by run-position.
            # The legacy g_HP_dyn slot is forced to 0 by the model.
            init_pars['g_HP_dyn'] = 0.0
            for r in (0, 1, 2):
                init_pars[f'g_HP_dyn_pos{r}'] = 0.0
            shared_pars = shared_pars + [
                'g_HP_dyn_pos0', 'g_HP_dyn_pos1', 'g_HP_dyn_pos2',
            ]

        if with_repeat_split:
            # 2 new gains splitting dyn HP/LP by repeat-vs-switch trial.
            # The existing g_HP_dyn / g_LP_dyn become the SWITCH gains.
            init_pars['g_HP_dyn_repeat'] = 0.0
            init_pars['g_LP_dyn_repeat'] = 0.0
            shared_pars = shared_pars + [
                'g_HP_dyn_repeat', 'g_LP_dyn_repeat',
            ]

    # 4) Build the dynamic DoG-AF + PRF model and the fitter.
    ring_positions = get_ring_positions()  # (4, 2)
    print('Ring positions:\n', ring_positions)

    # When oversampling, the HRF kernel must be sampled at the same
    # fine timestep as the paradigm and indicators.
    tr_orig = sub.get_tr(session=1, run=1)
    tr_for_hrf = tr_orig / temporal_oversampling
    hrf_model = SPMHRFModel(tr=tr_for_hrf,
                            delay=4.5, dispersion=0.75)

    if model_version == 'v2':
        ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v2
    elif with_target:
        if use_oversampled_codepath:
            ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_oversampled
        elif per_run_position_dyn_hp:
            ModelCls = (
                DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition_dynHP)
        elif per_run_position_gains:
            ModelCls = (
                DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition)
        elif with_repeat_split:
            ModelCls = (
                DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_repeat)
        elif shared_target_sigma:
            if all_shared_sigma and shared_dyn_gain:
                ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma_sharedDynGain
            elif all_shared_sigma:
                ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_allSharedSigma
            elif shared_dyn_gain:
                ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_sharedDynGain
            elif is_dn_base:
                # Divisive-Normalization counterpart of the canonical
                # DoG sharedSigma model. Used when the base PRF mean
                # fit is m5/m6 — same AF + dynamic + target + sharedSigma
                # modulation, but DN response math on the modulated drive.
                ModelCls = DivisiveNormalizationDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma
            else:
                ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma
        else:
            ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3_target
    else:
        ModelCls = DoGDynamicAttentionFieldPRF2DWithHRF_v3

    model_kwargs = dict(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        flexible_hrf_parameters=True,  # per-voxel HRF from m4; fixed during AF
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        ring_positions=ring_positions,
        mode=mode,
    )
    if with_target:
        model_kwargs['target_indicator'] = target_indicator
    if use_oversampled_codepath:
        model_kwargs['oversampling'] = temporal_oversampling
    if per_run_position_gains:
        model_kwargs['run_position_indicator'] = run_position_indicator
    if with_repeat_split:
        model_kwargs['repeat_indicator'] = repeat_indicator

    model = ModelCls(**model_kwargs)

    fitter = ParameterFitter(model, bold_sub, paradigm)

    # 5) Refine baseline/amplitude given current AF params.
    #
    # `refine_baseline_and_amplitude` is hardcoded to use the column
    # names 'baseline' and 'amplitude' (DoG convention). DN base
    # models have 'bold_baseline' and 'rf_amplitude' instead; the DN
    # response isn't linear in either of them so the closed-form least-
    # squares trick doesn't apply. Skip the pre-refine for DN base —
    # the joint GD fit handles initialisation on its own.
    if is_dn_base:
        refined_pars = init_pars
    else:
        refined_pars = fitter.refine_baseline_and_amplitude(init_pars, l2_alpha=1e-3)

    # 6) Joint fit. Shared pars depend on model version. HRF params
    #    are held fixed at m4's per-voxel estimates (we want AF to
    #    adapt only x/y/sd/baseline/amplitude + the surround on top
    #    of m4's HRF, not re-fit the HRF itself).
    fit_pars = fitter.fit(
        init_pars=refined_pars,
        max_n_iterations=max_n_iterations,
        shared_pars=shared_pars,
        fixed_pars=['hrf_delay', 'hrf_dispersion'],
        learning_rate=learning_rate,
    )

    r2 = fitter.get_rsq(fit_pars) if hasattr(fitter, 'get_rsq') else fitter.r2
    print(f'Mean R²: {np.nanmean(r2):.4f}')
    print('Shared parameters:')
    print(fit_pars[shared_pars].iloc[0])

    fit_pars['r2'] = r2.values if hasattr(r2, 'values') else r2

    # 7) Save outputs. Filename family flips from `dog-dyn-*` to
    # `dn-dyn-*` for DN-base fits so DoG and DN AF outputs don't share
    # filenames (paths already segregate by `_base-m{N}` subdir, but
    # the filename should be self-describing too).
    backbone = 'dn' if is_dn_base else 'dog'
    if with_target:
        dyn_tag = f'{backbone}-dyn-v3-target'
    else:
        dyn_tag = {'v2': f'{backbone}-dyn-v2',
                   'v3': f'{backbone}-dyn-v3'}[model_version]
    if shared_target_sigma:
        dyn_tag = f'{dyn_tag}-sharedSigma'
    if per_run_position_gains:
        dyn_tag = f'{dyn_tag}-runPosition'
    if per_run_position_dyn_hp:
        dyn_tag = f'{dyn_tag}-dynHP'
    if with_repeat_split:
        dyn_tag = f'{dyn_tag}-repeat'
    if use_oversampled_codepath:
        dyn_tag = f'{dyn_tag}-tos{temporal_oversampling}'
    if distractor_shape == 'rectangle':
        dyn_tag = f'{dyn_tag}-rect'
    out_tsv = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{dyn_tag}-af-prf-pars.tsv')
    fit_pars.to_csv(out_tsv, sep='\t')
    out_pkl = out_dir / (
        f'sub-{subject:02d}_roi-{roi}_mode-{mode}_{dyn_tag}-af-prf-fit.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'fit_pars': fit_pars,
            'r2': r2,
            'shared_pars': fit_pars[shared_pars].iloc[0].to_dict(),
            'shared_par_labels': shared_pars,
            'voxel_mask_indices': np.where(voxel_mask)[0],
            'mode': mode,
            'roi': roi,
            'resolution': resolution,
            'subject': subject,
            'paradigm_type': 'full',
            'grid_radius': grid_radius,
            'dynamic': True,
            'with_target': with_target,
            'voxel_kernel': 'DoG',
            'model_version': model_version,
            'model_label_init': model_label,
            'temporal_oversampling': (temporal_oversampling
                                      if use_oversampled_codepath else None),
            'shared_target_sigma': shared_target_sigma,
            'per_run_position_gains': per_run_position_gains,
            'per_run_position_dyn_hp': per_run_position_dyn_hp,
            'with_repeat_split': with_repeat_split,
            'distractor_shape': distractor_shape,
            'distractor_long_side': distractor_long_side,
            'distractor_short_side': distractor_short_side,
        }, f)
    print(f'Saved: {out_tsv}')
    print(f'Saved: {out_pkl}')

    return fit_pars, r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int,
                        help='Subject ID (zero-padded, e.g. 2 -> sub-02)')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V3AB',
                        help='Retinotopic ROI to fit (default V3AB).')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Stimulus grid resolution (default 50).')
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--p-signal-thr', type=float, default=0.5,
                        help='Posterior-based voxel selector. Load the '
                             'per-voxel GMM p_signal NIfTI and keep voxels '
                             'with P(signal | R²) > this. Default 0.5 '
                             '(matches FDR-α=0.05 pool size; the canonical '
                             'choice as of 2026-05-14).')
    parser.add_argument('--aperture-mass-thr', type=float, default=0.0,
                        help='Optional additional filter: keep voxels whose '
                             'PRF Gaussian has at least this fraction of mass '
                             'inside the bar aperture (3.17°, Gaussian '
                             'radial-CDF). 0 to skip (lets peripheral RFs '
                             'through — useful for comparing whether they '
                             'matter for the AF estimate).')
    parser.add_argument('--model-label', type=int, default=4,
                        help='Mean-model PRF used for DoG-init (x, y, sd, '
                             'amplitude, baseline, srf_amplitude, srf_size). '
                             'Model 4 (DoG + flexible HRF) is the canonical '
                             'choice.')
    parser.add_argument('--max-voxels', type=int, default=500,
                        help='Cap on voxels for the POC. Set 0 for no cap.')
    parser.add_argument('--mode',
                        choices=['suppression', 'attraction', 'signed'],
                        default='signed')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0,
                        help='Half-width of the extended grid in deg.')
    parser.add_argument('--output-subdir', default=None,
                        help='Output derivatives subdir. Default: '
                             "'af_prf_joint_dynamic_v2_dog' for v2, "
                             "'af_prf_joint_dynamic_v3_dog' for v3.")
    parser.add_argument('--sigma-af-init', type=float, default=2.0,
                        help='Initial value for sigma_AF (default 2.0).')
    parser.add_argument('--sigma-dyn-init', type=float, default=2.0,
                        help='Initial value for sigma_dyn (default 2.0; v3 only). '
                             'Both σ inits are now neutral/equal so the '
                             'optimizer must find σ_AF vs σ_dyn from data, '
                             'not the prior.')
    parser.add_argument('--model-version',
                        choices=['v2', 'v3'], default='v2',
                        help="v2 (default): shared sigma_AF + split HP/LP "
                             "dynamic gains (g_HP_dyn, g_LP_dyn). "
                             "v3: separate sigma_dyn AND split HP/LP "
                             "dynamic gains (6 shared params).")
    parser.add_argument('--with-target', action='store_true',
                        help='Add a phasic-target spatial-modulation term '
                             '(g_T_dyn, sigma_T_dyn). Only valid with '
                             '--model-version v3.')
    parser.add_argument('--g-t-dyn-init', type=float, default=0.0,
                        help='Initial value for g_T_dyn (default 0.0; signed '
                             'mode is neutral at 0).')
    parser.add_argument('--sigma-t-dyn-init', type=float, default=2.0,
                        help='Initial value for sigma_T_dyn (default 2.0; '
                             'matches the neutral sigma_AF / sigma_dyn '
                             'inits).')
    parser.add_argument('--all-shared-sigma', action='store_true',
                        help='Additionally tie sigma_AF := sigma_dyn so all '
                             'three AF Gaussians share a single width. '
                             'Requires --shared-target-sigma.')
    parser.add_argument('--shared-dyn-gain', action='store_true',
                        help='Additionally tie g_LP_dyn := g_HP_dyn so the '
                             'phasic-distractor transient has one gain '
                             'regardless of HP/LP location. Requires '
                             '--shared-target-sigma.')
    parser.add_argument('--shared-target-sigma', action='store_true',
                        help='Force sigma_T_dyn := sigma_dyn so the two '
                             'phasic Gaussians (distractor-onset and '
                             'target-onset) share a single spatial '
                             'extent. Only valid with --model-version v3 '
                             '+ --with-target. Routes through '
                             'DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma.')
    parser.add_argument('--per-run-position-gains', action='store_true',
                        help='Replace the single sustained (g_HP, g_LP) '
                             'pair with 6 sustained gains, one pair per '
                             'within-block run-position (0/1/2). Lets '
                             'us inspect arbitrary learning curves over '
                             'the 3-run HP blocks. Requires '
                             '--with-target --shared-target-sigma '
                             '--model-version v3. Routes through '
                             'DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition.')
    parser.add_argument('--per-run-position-dyn-hp', action='store_true',
                        help='ALSO split the dynamic-HP gain (g_HP_dyn) '
                             'into 3 per-run-position gains '
                             '(g_HP_dyn_pos0/1/2). Requires '
                             '--per-run-position-gains. Tests whether '
                             'phasic HP suppression deepens across the '
                             '3-run HP block. Routes through '
                             'DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_runPosition_dynHP.')
    parser.add_argument('--with-repeat-split', action='store_true',
                        help='Split the dynamic HP/LP gains by whether '
                             "this trial's distractor is at the SAME "
                             'ring location as the previous trial '
                             '(repeat) or not (switch). Adds 2 new '
                             'gains: g_HP_dyn_repeat, g_LP_dyn_repeat. '
                             'Requires --with-target --shared-target-sigma '
                             '--model-version v3. Cannot combine with '
                             '--per-run-position-gains.')
    parser.add_argument('--distractor-shape',
                        choices=['circle', 'rectangle'], default='circle',
                        help="Distractor footprint in the paradigm. "
                             "'circle' (default, backward-compatible) uses "
                             "a 0.4-deg disk. 'rectangle' uses an oriented "
                             "rectangle (long×short = "
                             "--distractor-long-side x --distractor-short-side) "
                             "rotated by the trial's distractor_orientation "
                             "(= 90 - target_orientation). Output subdir "
                             "is tagged with `_rect` so circle and "
                             "rectangle fits never collide.")
    parser.add_argument('--distractor-long-side', type=float, default=1.5,
                        help='Rectangle long-axis length, deg '
                             '(default 1.5). Only used with '
                             '--distractor-shape=rectangle.')
    parser.add_argument('--distractor-short-side', type=float, default=0.5,
                        help='Rectangle short-axis length, deg '
                             '(default 0.5). Only used with '
                             '--distractor-shape=rectangle.')
    parser.add_argument('--temporal-oversampling', type=int, default=None,
                        help='Temporal oversampling factor N for the HRF '
                             'convolution (only valid with v3 + '
                             '--with-target). When omitted, the legacy '
                             'TR-resolution code path is used. When '
                             'given (e.g. 1, 4, 8), the paradigm and '
                             'indicators are built at fine dt = TR/N, '
                             'the HRF is constructed at fine dt, and '
                             'the model output is subsampled by N '
                             'before computing the BOLD-space loss. '
                             'Output subdir is tagged with `_tos{N}` '
                             'so even N=1 lives in its own folder.')
    args = parser.parse_args()
    main(
        args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        p_signal_thr=args.p_signal_thr,
        aperture_mass_thr=args.aperture_mass_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        mode=args.mode,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
        model_version=args.model_version,
        sigma_af_init=args.sigma_af_init,
        sigma_dyn_init=args.sigma_dyn_init,
        with_target=args.with_target,
        g_t_dyn_init=args.g_t_dyn_init,
        sigma_t_dyn_init=args.sigma_t_dyn_init,
        temporal_oversampling=args.temporal_oversampling,
        shared_target_sigma=args.shared_target_sigma,
        shared_dyn_gain=args.shared_dyn_gain,
        all_shared_sigma=args.all_shared_sigma,
        per_run_position_gains=args.per_run_position_gains,
        per_run_position_dyn_hp=args.per_run_position_dyn_hp,
        with_repeat_split=args.with_repeat_split,
        distractor_shape=args.distractor_shape,
        distractor_long_side=args.distractor_long_side,
        distractor_short_side=args.distractor_short_side,
    )
