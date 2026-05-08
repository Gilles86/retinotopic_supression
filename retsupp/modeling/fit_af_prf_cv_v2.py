"""Cross-validated DoG-dyn-v3 + target sharedSigma fits — 18-class factorial.

CV-v2 is the successor to ``fit_af_prf_cv.py`` (CV-v1).  Three big
changes versus CV-v1:

1. **Target-onset term included optionally.**  The fifth gain
   ``g_T_dyn`` is either signed-unconstrained (``--target-gain free``)
   or fixed = 0 (``--target-gain zero``).  This crosses on top of the
   distractor-class factorial and lets us test "does AF help?" and
   "does target help?" simultaneously.
2. **σ_T_dyn := σ_dyn (shared).**  Hard-tied at every forward pass via
   :class:`DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial`
   in :mod:`retsupp.modeling.local_models`.  This avoids the σ_T_dyn
   drifting to ~9° identifiability artifact we saw in CV-v1.
3. **σ initializers all = 5.0** (σ_AF, σ_dyn).  The earlier robustness
   check showed σ_dyn is poorly identifiable with small inits; large
   inits give the optimizer enough gradient signal to shrink as the
   data wants.
4. **9 distractor-classes only**: cross of
   ``sus ∈ {(zero,zero), (minus,zero), (minus,minus)} ×
    dyn ∈ {(zero,zero), (minus,zero), (minus,minus)}``.
   ``(plus, plus)`` and ``signed-control`` are dropped — both failed
   badly in CV-v1.

Sign-constraint mechanism
-------------------------
For each of the five gains ``g_HP, g_LP, g_HP_dyn, g_LP_dyn, g_T_dyn``
we pick one of four constraints, encoded as words to avoid shell
escaping::

    plus    g = +softplus(raw)   — TF Variable, optimised, sign ≥ 0
    minus   g = -softplus(raw)   — TF Variable, optimised, sign ≤ 0
    zero    g = 0.0              — Python constant, NOT optimised
    free    g = raw              — TF Variable, optimised, no sign

The submitter ``submit_all_cv_v2.sh`` only sweeps a 3-pattern subset on
the sustained and dynamic pairs --- ``(zero,zero)``, ``(minus,zero)``,
``(minus,minus)`` --- and a 2-option choice on the target gain ---
``free``, ``zero`` --- but the script accepts the full 4-choice space
on every gain so we can probe other patterns ad hoc.

CV scheme: leave-one-CONDITION-out, 4 folds.  Each invocation of this
script processes ONE (subject, ROI, sign-class) and loops over all 4
folds internally to amortise the per-ROI BOLD load.

Output
------
``derivatives/af_prf_cv_v2/<class_dirname>/sub-XX/
    sub-XX_roi-{ROI}_cv-fits.pkl``

The class_dirname concatenates the distractor-pair labels with the
target-gain tag, e.g.::

    sus-minus-zero_dyn-minus-minus_tgt-free
    sus-zero-zero_dyn-zero-zero_tgt-zero

The pickle contains, per fold:
    shared_pars_per_fold : list[dict]   # 4 dicts of shared-par means
    cv_r2_per_fold       : list[ndarray]  # per-voxel CV-R², per fold
    train_r2_per_fold    : list[ndarray]
    fit_pars_per_fold    : list[DataFrame]

Usage
-----
``python -m retsupp.modeling.fit_af_prf_cv_v2 2 --roi V3AB \\
    --sus-hp-sign minus --sus-lp-sign zero \\
    --dyn-hp-sign minus --dyn-lp-sign minus \\
    --target-gain free``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter
from retsupp.modeling.cv_helpers import (
    CONDITIONS,
    RunMeta,
    held_out_condition,
    per_voxel_r2,
    summarize_split,
)
from retsupp.modeling.local_models import (
    DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial,
)
from retsupp.utils.data import Subject, distractor_locations


SIGN_CHOICES = ('plus', 'zero', 'minus', 'free')
TARGET_GAIN_CHOICES = ('free', 'zero')


# All 5 gain names used by the factorial sharedSigma model.
GAIN_NAMES = ('g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn')

# All shared parameters that the fitter optimises jointly across voxels.
# σ_AF and σ_dyn are always free; the five gains may be free, fixed-zero,
# or sign-constrained per the factorial.  We do NOT include sigma_T_dyn
# because it is tied to sigma_dyn at every forward pass — its raw
# variable still exists in the parameter tensor but receives no gradient.
ALL_SHARED_PARS = ['sigma_AF', 'g_HP', 'g_LP',
                   'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
                   'g_T_dyn', 'sigma_T_dyn']


# ---------------------------------------------------------------------------
# Sign-class encoding & directory naming.
# ---------------------------------------------------------------------------

def class_dirname(sus_hp: str, sus_lp: str,
                  dyn_hp: str, dyn_lp: str,
                  target_gain: str) -> str:
    """Convert the 5 sign args into the output directory name.

    Format: ``sus-{sus_hp}-{sus_lp}_dyn-{dyn_hp}-{dyn_lp}_tgt-{target}``.
    """
    return (f'sus-{sus_hp}-{sus_lp}_'
            f'dyn-{dyn_hp}-{dyn_lp}_'
            f'tgt-{target_gain}')


def get_ring_positions():
    keys = ['upper right', 'upper left', 'lower left', 'lower right']
    return np.array(
        [list(distractor_locations[k]) for k in keys], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading.  Same as CV-v1 except we ALSO load target_indicator.
# ---------------------------------------------------------------------------

def build_data_and_paradigm(
    sub: Subject,
    resolution: int = 50, grid_radius: float = 5.0,
):
    """Load BOLD + paradigm + condition / dynamic / target indicators.

    Returns
    -------
    bold : (T_total, V) ndarray
    paradigm_full : (T_total, G) ndarray
    condition_indicator : (T_total, n_C) ndarray
    dynamic_indicator : (T_total, n_C) ndarray
    target_indicator : (T_total, n_C) ndarray
    grid_coordinates : (G, 2) ndarray
    masker : NiftiMasker
    run_meta : list[RunMeta]
    """
    bold_mask = sub.get_bold_mask()
    masker = input_data.NiftiMasker(mask_img=bold_mask)
    masker.fit(bold_mask)

    gx, gy = sub.get_extended_grid_coordinates(
        resolution=resolution, session=1, run=1, grid_radius=grid_radius,
    )
    grid_coordinates = np.stack(
        (gx.ravel(), gy.ravel()), axis=1,
    ).astype(np.float32)

    hp_per_run = sub.get_hpd_locations()
    print('HP per run:', hp_per_run)

    bold_chunks = []
    paradigm_chunks = []
    cond_indicator_chunks = []
    dyn_indicator_chunks = []
    tgt_indicator_chunks = []
    run_meta: list[RunMeta] = []
    cursor = 0

    session_runs = [(s, r) for s in [1, 2] for r in sub.get_runs(s)]
    for session, run in tqdm(session_runs, desc='Loading runs'):
        hp = hp_per_run[(session, run)]
        if hp not in CONDITIONS:
            print(f'  ses-{session}_run-{run}: HP={hp!r}, skipping')
            continue

        par_run = sub.get_stimulus_with_distractors(
            session=session, run=run, resolution=resolution,
            grid_radius=grid_radius,
        ).astype(np.float32)
        par_run_flat = par_run.reshape((par_run.shape[0], -1))
        n_T_run = par_run_flat.shape[0]

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

        dyn = sub.get_dynamic_indicator(
            session=session, run=run,
        ).astype(np.float32)
        dyn = _pad_or_crop(dyn, n_T_run)

        tgt = sub.get_target_indicator(
            session=session, run=run,
        ).astype(np.float32)
        tgt = _pad_or_crop(tgt, n_T_run)

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)
        dyn_indicator_chunks.append(dyn)
        tgt_indicator_chunks.append(tgt)
        run_meta.append(RunMeta(
            session=session, run=run, hp=hp,
            start=cursor, n_T=n_T_run,
        ))
        cursor += n_T_run

    bold = np.vstack(bold_chunks)
    paradigm_full = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_indicator_chunks)
    dynamic_indicator = np.vstack(dyn_indicator_chunks)
    target_indicator = np.vstack(tgt_indicator_chunks)

    print(f'Loaded BOLD: shape {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}, '
          f'dynamic_indicator {dynamic_indicator.shape}, '
          f'target_indicator {target_indicator.shape} '
          f'(target mean={target_indicator.mean():.4f})')
    return (
        bold, paradigm_full, condition_indicator,
        dynamic_indicator, target_indicator,
        grid_coordinates, masker, run_meta,
    )


def _pad_or_crop(arr: np.ndarray, n_T: int) -> np.ndarray:
    if arr.shape[0] < n_T:
        pad = np.zeros((n_T - arr.shape[0], arr.shape[1]),
                       dtype=np.float32)
        return np.vstack([arr, pad])
    if arr.shape[0] > n_T:
        return arr[:n_T]
    return arr


def select_roi_voxels(sub: Subject, roi: str, prf_pars: pd.DataFrame,
                      r2_thr: float = 0.05):
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
    return (prf_pars['r2'].values > r2_thr) & roi_arr


# ---------------------------------------------------------------------------
# Init / split helpers.
# ---------------------------------------------------------------------------

def make_init_pars(prf_pars: pd.DataFrame,
                   voxel_mask: np.ndarray,
                   sign_pattern: dict[str, str],
                   sigma_af_init: float,
                   sigma_dyn_init: float) -> pd.DataFrame:
    """Initialise per-voxel + shared parameters for the sharedSigma factorial.

    σ_AF and σ_dyn are initialised at the values given (defaults 5.0).
    σ_T_dyn is set equal to σ_dyn (the model ties them at every forward
    pass anyway).  Sign-constrained gain inits are picked to match each
    gain's sign with a small magnitude (so the optimiser starts near
    zero but on the correct side).
    """
    init_cols = ['x', 'y', 'sd', 'baseline', 'amplitude',
                 'srf_amplitude', 'srf_size']
    missing = [c for c in init_cols if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model is missing DoG params {missing}. '
            'Use model 4 for DoG init.')
    init_pars = prf_pars.loc[voxel_mask, init_cols].copy()
    init_pars = init_pars.reset_index(drop=True)

    init_pars['sigma_AF'] = sigma_af_init
    init_pars['sigma_dyn'] = sigma_dyn_init
    # σ_T_dyn raw variable starts at the same place as σ_dyn — the
    # model overwrites it on every forward pass anyway, but keeping the
    # raw inits aligned avoids any first-iteration weirdness.
    init_pars['sigma_T_dyn'] = sigma_dyn_init

    def _init_usable(sign: str) -> float:
        if sign == 'plus':
            return 0.10
        if sign == 'minus':
            return -0.10
        # 'zero' and 'free' both start at 0.
        return 0.0

    for gain_name in GAIN_NAMES:
        init_pars[gain_name] = _init_usable(sign_pattern[gain_name])

    return init_pars


def split_by_condition_with_target(
    bold: np.ndarray, paradigm: np.ndarray,
    condition_indicator: np.ndarray,
    dynamic_indicator: np.ndarray,
    target_indicator: np.ndarray,
    run_meta: list[RunMeta], held_out_cond: str,
):
    """Local re-implementation of cv_helpers.split_by_condition that
    also splits ``target_indicator`` along the time axis.

    We deliberately avoid modifying ``cv_helpers.py`` so CV-v1 stays
    pinned to its current semantics.
    """
    if held_out_cond not in CONDITIONS:
        raise ValueError(
            f'held_out_cond must be in {CONDITIONS!r} (got {held_out_cond!r}).')

    expected_T = sum(rm.n_T for rm in run_meta)
    if bold.shape[0] != expected_T:
        raise ValueError(
            f'run_meta sums to {expected_T} rows but bold has '
            f'{bold.shape[0]} rows.')

    train_rows = []
    held_rows = []
    train_run_meta: list[RunMeta] = []
    held_run_meta: list[RunMeta] = []
    train_offset = 0
    held_offset = 0
    for rm in run_meta:
        rows = np.arange(rm.start, rm.stop)
        if rm.hp == held_out_cond:
            held_rows.append(rows)
            held_run_meta.append(RunMeta(
                session=rm.session, run=rm.run, hp=rm.hp,
                start=held_offset, n_T=rm.n_T,
            ))
            held_offset += rm.n_T
        else:
            train_rows.append(rows)
            train_run_meta.append(RunMeta(
                session=rm.session, run=rm.run, hp=rm.hp,
                start=train_offset, n_T=rm.n_T,
            ))
            train_offset += rm.n_T

    if not train_rows:
        raise RuntimeError(
            f'No training runs left after holding out {held_out_cond!r}.')
    if not held_rows:
        raise RuntimeError(
            f'No held-out runs found for condition {held_out_cond!r}. '
            f'(Subject may not have any runs with this HP.)')

    train_idx = np.concatenate(train_rows)
    held_idx = np.concatenate(held_rows)

    train = dict(
        bold=bold[train_idx],
        paradigm=paradigm[train_idx],
        condition_indicator=condition_indicator[train_idx],
        dynamic_indicator=dynamic_indicator[train_idx],
        target_indicator=target_indicator[train_idx],
        run_meta=train_run_meta,
    )
    held = dict(
        bold=bold[held_idx],
        paradigm=paradigm[held_idx],
        condition_indicator=condition_indicator[held_idx],
        dynamic_indicator=dynamic_indicator[held_idx],
        target_indicator=target_indicator[held_idx],
        run_meta=held_run_meta,
    )
    return {'train': train, 'held': held}


def build_model(*, grid_coords, paradigm,
                condition_indicator, dynamic_indicator,
                target_indicator,
                ring_positions, hrf_model, sign_pattern):
    """Construct the sharedSigma + factorial model for one CV fold."""
    return DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        target_indicator=target_indicator,
        ring_positions=ring_positions,
        sign_pattern=sign_pattern,
    )


def fixed_pars_for(sign_pattern: dict[str, str]) -> list[str] | None:
    """Names of gains whose sign is 'zero' (passed to ParameterFitter).

    The model already clamps zero-signed gains to 0 in the forward
    transform; we ALSO mark them fixed so the optimiser doesn't waste
    effort updating their (irrelevant) raw variables.

    Note: we do NOT add ``sigma_T_dyn`` to ``fixed_pars`` even though it
    is effectively dead (tied to σ_dyn).  Adding it would fight with
    the shared_pars list.  Its raw variable simply receives zero
    gradient through the model.
    """
    fixed = [name for name in GAIN_NAMES if sign_pattern[name] == 'zero']
    return fixed if fixed else None


def shared_pars_for(sign_pattern: dict[str, str]) -> list[str]:
    """Optimiser-shared parameters that are NOT zero-fixed.

    σ_AF, σ_dyn, σ_T_dyn are always shared across voxels (σ_T_dyn even
    though dead — keeping it shared keeps the shape of the fit_pars
    DataFrame consistent across cells).  Each gain is shared unless
    its sign is 'zero' (which would conflict with ``fixed_pars``).
    """
    keep = ['sigma_AF', 'sigma_dyn', 'sigma_T_dyn']
    for gain_name in GAIN_NAMES:
        if sign_pattern[gain_name] != 'zero':
            keep.append(gain_name)
    return keep


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(subject: int,
         bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V3AB',
         sus_hp_sign: str = 'minus',
         sus_lp_sign: str = 'zero',
         dyn_hp_sign: str = 'minus',
         dyn_lp_sign: str = 'zero',
         target_gain: str = 'free',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         model_label: int = 4,
         max_voxels: int | None = None,
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         sigma_af_init: float = 5.0,
         sigma_dyn_init: float = 5.0,
         output_subdir: str | None = None):

    for s in (sus_hp_sign, sus_lp_sign, dyn_hp_sign, dyn_lp_sign):
        if s not in SIGN_CHOICES:
            raise ValueError(
                f'sign args must be one of {SIGN_CHOICES} (got {s!r}).')
    if target_gain not in TARGET_GAIN_CHOICES:
        raise ValueError(
            f'--target-gain must be one of {TARGET_GAIN_CHOICES} '
            f'(got {target_gain!r}).')

    sign_pattern = {
        'g_HP': sus_hp_sign,
        'g_LP': sus_lp_sign,
        'g_HP_dyn': dyn_hp_sign,
        'g_LP_dyn': dyn_lp_sign,
        'g_T_dyn': target_gain,
    }
    cls_dir = class_dirname(sus_hp_sign, sus_lp_sign,
                            dyn_hp_sign, dyn_lp_sign,
                            target_gain)

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        output_subdir = f'af_prf_cv_v2/{cls_dir}'
    out_dir = (bids_folder / 'derivatives' / output_subdir
               / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | class={cls_dir} ==')
    print(f'   sign pattern: {sign_pattern}')
    print(f'   sigma inits:  σ_AF={sigma_af_init}, σ_dyn={sigma_dyn_init} '
          f'(σ_T_dyn := σ_dyn at every forward pass)')

    # 1) Load BOLD + paradigm + indicators (incl. target_indicator).
    (bold, paradigm, condition_indicator,
     dynamic_indicator, target_indicator,
     grid_coords, _masker, run_meta) = build_data_and_paradigm(
        sub, resolution=resolution, grid_radius=grid_radius,
    )

    # 2) Restrict to ROI voxels with decent mean-model PRF R².
    prf_pars = sub.get_prf_parameters_volume(model=model_label,
                                             return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)
    voxel_mask = select_roi_voxels(sub, roi, prf_pars, r2_thr=r2_thr)
    print(f'ROI {roi} | r2>{r2_thr}: {voxel_mask.sum()} voxels')
    if voxel_mask.sum() == 0:
        raise RuntimeError(f'No voxels survive: ROI={roi}, r2>{r2_thr}.')

    if max_voxels is not None and voxel_mask.sum() > max_voxels:
        order = np.argsort(prf_pars['r2'].values)[::-1]
        keep = np.zeros_like(voxel_mask)
        ranked = [v for v in order if voxel_mask[v]][:max_voxels]
        keep[ranked] = True
        voxel_mask = keep
        print(f'  -> top {max_voxels} voxels by r²')

    init_pars_template = make_init_pars(
        prf_pars, voxel_mask, sign_pattern,
        sigma_af_init=sigma_af_init,
        sigma_dyn_init=sigma_dyn_init,
    )
    ring_positions = get_ring_positions()
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)

    fixed_pars = fixed_pars_for(sign_pattern)
    shared_pars = shared_pars_for(sign_pattern)
    print(f'   fixed_pars : {fixed_pars}')
    print(f'   shared_pars: {shared_pars}')

    # 3) Loop over CV folds (4 held-out HP conditions).
    shared_pars_per_fold: list[dict] = []
    cv_r2_per_fold: list[np.ndarray] = []
    train_r2_per_fold: list[np.ndarray] = []
    fit_pars_per_fold: list[pd.DataFrame] = []
    train_run_meta_per_fold: list[list[tuple[int, int, str]]] = []
    held_run_meta_per_fold: list[list[tuple[int, int, str]]] = []

    n_voxels_fit = int(voxel_mask.sum())

    for cv_fold in range(4):
        held_out = held_out_condition(cv_fold)
        print()
        print(f'=== CV fold {cv_fold} : held-out = {held_out} ===')

        try:
            split = split_by_condition_with_target(
                bold=bold, paradigm=paradigm,
                condition_indicator=condition_indicator,
                dynamic_indicator=dynamic_indicator,
                target_indicator=target_indicator,
                run_meta=run_meta, held_out_cond=held_out,
            )
        except RuntimeError as e:
            print(f'  -> SKIPPING fold {cv_fold}: {e}')
            nan_arr = np.full(n_voxels_fit, np.nan, dtype=np.float32)
            shared_pars_per_fold.append(None)
            cv_r2_per_fold.append(nan_arr)
            train_r2_per_fold.append(nan_arr.copy())
            fit_pars_per_fold.append(None)
            train_run_meta_per_fold.append([])
            held_run_meta_per_fold.append([])
            continue

        train, held = split['train'], split['held']
        print(summarize_split(run_meta, held_out))

        bold_train_sub = pd.DataFrame(train['bold'][:, voxel_mask])
        bold_held_sub = pd.DataFrame(held['bold'][:, voxel_mask])

        model_train = build_model(
            grid_coords=grid_coords,
            paradigm=train['paradigm'],
            condition_indicator=train['condition_indicator'],
            dynamic_indicator=train['dynamic_indicator'],
            target_indicator=train['target_indicator'],
            ring_positions=ring_positions,
            hrf_model=hrf_model,
            sign_pattern=sign_pattern,
        )

        fitter = ParameterFitter(model_train, bold_train_sub,
                                 train['paradigm'])
        refined_pars = fitter.refine_baseline_and_amplitude(
            init_pars_template.copy(), l2_alpha=1e-3)

        fit_pars = fitter.fit(
            init_pars=refined_pars,
            max_n_iterations=max_n_iterations,
            shared_pars=shared_pars,
            fixed_pars=fixed_pars,
            learning_rate=learning_rate,
        )
        train_r2 = fitter.get_rsq(fit_pars)
        train_r2 = (train_r2.values if hasattr(train_r2, 'values')
                    else np.asarray(train_r2))
        print(f'Train R² mean: {np.nanmean(train_r2):.4f}')

        # Held-out prediction.
        model_held = build_model(
            grid_coords=grid_coords,
            paradigm=held['paradigm'],
            condition_indicator=held['condition_indicator'],
            dynamic_indicator=held['dynamic_indicator'],
            target_indicator=held['target_indicator'],
            ring_positions=ring_positions,
            hrf_model=hrf_model,
            sign_pattern=sign_pattern,
        )
        pred_held = model_held.predict(paradigm=held['paradigm'],
                                       parameters=fit_pars)
        pred_held_arr = (pred_held.values if hasattr(pred_held, 'values')
                         else np.asarray(pred_held))
        cv_r2 = per_voxel_r2(bold_held_sub.values, pred_held_arr)
        print(f'Held-out CV-R² (n={cv_r2.size}) | '
              f'mean={np.nanmean(cv_r2):.4f} | '
              f'median={np.nanmedian(cv_r2):.4f}')

        # All shared-par columns reported (always include all 8
        # ALL_SHARED_PARS; zero-fixed gains will be 0 because the
        # forward transform clamps them; σ_T_dyn will equal σ_dyn).
        shared_dict = fit_pars[ALL_SHARED_PARS].iloc[0].to_dict()

        shared_pars_per_fold.append(shared_dict)
        cv_r2_per_fold.append(cv_r2)
        train_r2_per_fold.append(train_r2)
        fit_pars_per_fold.append(fit_pars)
        train_run_meta_per_fold.append(
            [(rm.session, rm.run, rm.hp) for rm in train['run_meta']])
        held_run_meta_per_fold.append(
            [(rm.session, rm.run, rm.hp) for rm in held['run_meta']])

    # 4) Save one pickle per (subject, ROI, class).
    out_pkl = out_dir / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
    payload = {
        'shared_pars_per_fold': shared_pars_per_fold,
        'cv_r2_per_fold': cv_r2_per_fold,
        'train_r2_per_fold': train_r2_per_fold,
        'fit_pars_per_fold': fit_pars_per_fold,
        'class_label': (sus_hp_sign, sus_lp_sign,
                        dyn_hp_sign, dyn_lp_sign, target_gain),
        'class_dirname': cls_dir,
        'sign_pattern': sign_pattern,
        'shared_par_labels': ALL_SHARED_PARS,
        'fitted_shared_pars': shared_pars,
        'fixed_pars': fixed_pars or [],
        'voxel_mask_indices': np.where(voxel_mask)[0],
        'roi': roi,
        'resolution': resolution,
        'subject': subject,
        'paradigm_type': 'full',
        'grid_radius': grid_radius,
        'model_name': 'dog-dyn-v3-target-sharedSigma-factorial',
        'cv_folds': list(range(4)),
        'cv_held_out_conditions': list(CONDITIONS),
        'train_run_meta_per_fold': train_run_meta_per_fold,
        'held_run_meta_per_fold': held_run_meta_per_fold,
        'sigma_af_init': sigma_af_init,
        'sigma_dyn_init': sigma_dyn_init,
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Saved: {out_pkl}')

    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V3AB')
    parser.add_argument('--sus-hp-sign', choices=list(SIGN_CHOICES),
                        required=True,
                        help='Sign constraint on g_HP (sustained).')
    parser.add_argument('--sus-lp-sign', choices=list(SIGN_CHOICES),
                        required=True,
                        help='Sign constraint on g_LP (sustained).')
    parser.add_argument('--dyn-hp-sign', choices=list(SIGN_CHOICES),
                        required=True,
                        help='Sign constraint on g_HP_dyn (dynamic).')
    parser.add_argument('--dyn-lp-sign', choices=list(SIGN_CHOICES),
                        required=True,
                        help='Sign constraint on g_LP_dyn (dynamic).')
    parser.add_argument('--target-gain', choices=list(TARGET_GAIN_CHOICES),
                        required=True,
                        help='Sign constraint on g_T_dyn (target onset). '
                             "'free' = signed-unconstrained, 'zero' = no "
                             'target term.')
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--model-label', type=int, default=4)
    parser.add_argument('--max-voxels', type=int, default=0,
                        help='Cap on voxels (0 = no cap, default).')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0)
    parser.add_argument('--sigma-af-init', type=float, default=5.0,
                        help='Initial σ_AF (default 5.0; CV-v2 default).')
    parser.add_argument('--sigma-dyn-init', type=float, default=5.0,
                        help='Initial σ_dyn (default 5.0; CV-v2 default).')
    parser.add_argument('--output-subdir', default=None)
    args = parser.parse_args()
    main(
        subject=args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        sus_hp_sign=args.sus_hp_sign,
        sus_lp_sign=args.sus_lp_sign,
        dyn_hp_sign=args.dyn_hp_sign,
        dyn_lp_sign=args.dyn_lp_sign,
        target_gain=args.target_gain,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        sigma_af_init=args.sigma_af_init,
        sigma_dyn_init=args.sigma_dyn_init,
        output_subdir=args.output_subdir,
    )
