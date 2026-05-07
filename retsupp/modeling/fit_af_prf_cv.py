"""Cross-validated DoG-dyn-v3 AF + PRF fits — 17-class factorial design.

Hardcoded model: ``DoGDynamicAttentionFieldPRF2DWithHRF_v3``
(:mod:`braincoder.models`). Per-voxel free parameters: ``x, y, sd,
baseline, amplitude, srf_amplitude, srf_size``. Shared parameters:
``sigma_AF, g_HP, g_LP, sigma_dyn, g_HP_dyn, g_LP_dyn`` (σ_AF and σ_dyn
are always free).

Sign-constraint factorial
-------------------------
For each of the four gains ``g_HP, g_LP, g_HP_dyn, g_LP_dyn`` we pick
one of four constraints, encoded as words to avoid shell escaping::

    plus    g = +softplus(raw)   — TF Variable, optimised, sign ≥ 0
    minus   g = -softplus(raw)   — TF Variable, optimised, sign ≤ 0
    zero    g = 0.0              — Python constant, NOT optimised
    free    g = raw              — TF Variable, optimised, no sign

The factorial restricts the SUSTAINED pair (g_HP, g_LP) and the DYNAMIC
pair (g_HP_dyn, g_LP_dyn) to one of four sign-patterns each:

    (zero, zero), (minus, zero), (minus, minus), (plus, plus)

Crossing 4 sustained × 4 dynamic = 16 cells. Plus one ``free, free,
free, free`` "signed-control" cell where all four gains are signed
unconstrained. **17 classes total.** The (zero, zero, zero, zero) cell
is the no-AF null model.

CV scheme: leave-one-CONDITION-out, 4 folds. Each invocation of this
script processes ONE (subject, ROI, sign-class) and loops over all 4
folds internally to amortise the per-ROI BOLD load.

Per fold: fit on the 3 training-condition runs, build a fresh model
with the held-out paradigm / condition_indicator / dynamic_indicator,
predict the held-out BOLD, compute per-voxel CV-R².

Output
------
``derivatives/af_prf_cv_factorial/<class_label>/sub-XX/
    sub-XX_roi-{ROI}_cv-fits.pkl``

The pickle contains:
    shared_pars_per_fold : list[dict]   # 4 dicts of shared-par means
    cv_r2_per_fold       : list[ndarray]  # per-voxel CV-R², per fold
    train_r2_per_fold    : list[ndarray]
    fit_pars_per_fold    : list[DataFrame]
    class_label          : tuple[str, str, str, str]
    class_dirname        : str

The ``<class_label>`` directory name is built from the four sign args
as ``sus-{sus_hp}-{sus_lp}_dyn-{dyn_hp}-{dyn_lp}``, e.g.
``sus-minus-zero_dyn-minus-minus``. The signed-unconstrained class is
named ``signed-control``.

Usage
-----
``python -m retsupp.modeling.fit_af_prf_cv 2 --roi V3AB \\
    --sus-hp-sign minus --sus-lp-sign zero \\
    --dyn-hp-sign minus --dyn-lp-sign minus``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from nilearn import input_data
from tqdm import tqdm

from braincoder.hrf import SPMHRFModel
from braincoder.models import DoGDynamicAttentionFieldPRF2DWithHRF_v3
from braincoder.optimize import ParameterFitter
from retsupp.modeling.cv_helpers import (
    CONDITIONS,
    RunMeta,
    held_out_condition,
    per_voxel_r2,
    restrict_to_train_condition,
    summarize_split,
)
from retsupp.utils.data import Subject, distractor_locations


SIGN_CHOICES = ('plus', 'zero', 'minus', 'free')

# All 13 parameters of DoGDynamicAttentionFieldPRF2DWithHRF_v3, in order:
#   0  x
#   1  y
#   2  sd
#   3  baseline
#   4  amplitude
#   5  srf_amplitude
#   6  srf_size
#   7  sigma_AF
#   8  g_HP
#   9  g_LP
#  10  sigma_dyn
#  11  g_HP_dyn
#  12  g_LP_dyn
GAIN_INDEX = {
    'g_HP': 8,
    'g_LP': 9,
    'g_HP_dyn': 11,
    'g_LP_dyn': 12,
}

# All shared parameters that the fitter optimises jointly across voxels.
# σ_AF and σ_dyn are always free; the four gains may be free, fixed-zero,
# or sign-constrained per the factorial.
ALL_SHARED_PARS = ['sigma_AF', 'g_HP', 'g_LP',
                   'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']


# ---------------------------------------------------------------------------
# Sign-constrained model wrapper.
# ---------------------------------------------------------------------------

def _gain_forward(raw, sign):
    """Apply per-gain sign constraint in the forward (raw -> usable) direction.

    Parameters
    ----------
    raw : (B, 1) tensor — the unconstrained variable.
    sign : str — one of {'plus', 'minus', 'zero', 'free'}.
    """
    if sign == 'plus':
        return tf.math.softplus(raw)
    if sign == 'minus':
        return -tf.math.softplus(raw)
    if sign == 'zero':
        # Force to exact 0 regardless of the underlying variable. We still
        # keep the variable in the parameter tensor (the fitter will mark
        # it fixed via fixed_pars) but the forward transform clamps it.
        return tf.zeros_like(raw)
    if sign == 'free':
        return raw
    raise ValueError(f'Unknown sign {sign!r}')


def _gain_backward(usable, sign):
    """Inverse transform: usable value -> raw variable initialisation.

    For ``zero`` we just return zero (the variable is fixed and the
    forward transform clamps to 0 anyway).
    For ``free`` we return the value as-is.
    For ``plus``/``minus`` we softplus_inverse the magnitude (which
    requires |usable| > 0 — the caller passes a small positive seed).
    """
    if sign == 'plus':
        # softplus_inverse(usable) — usable expected ≥ 0.
        return tfp.math.softplus_inverse(tf.maximum(usable, 1e-4))
    if sign == 'minus':
        # We want forward(raw) = -softplus(raw) = usable (≤ 0), so
        # raw = softplus_inverse(-usable). Caller passes usable ≤ 0.
        return tfp.math.softplus_inverse(tf.maximum(-usable, 1e-4))
    if sign == 'zero':
        return tf.zeros_like(usable)
    if sign == 'free':
        return usable
    raise ValueError(f'Unknown sign {sign!r}')


class FactorialDoGDynV3(DoGDynamicAttentionFieldPRF2DWithHRF_v3):
    """``DoGDynamicAttentionFieldPRF2DWithHRF_v3`` with per-gain sign constraints.

    Each of the four gains (``g_HP``, ``g_LP``, ``g_HP_dyn``, ``g_LP_dyn``)
    is mapped through one of four constraint transforms (see module
    docstring). σ_AF and σ_dyn keep their canonical softplus transform.
    All non-gain parameters fall through to the parent.

    The class is constructed with ``mode='signed'`` regardless — the
    ``_signed_gains`` flag in the parent disables the parent's softplus
    on the gains so our overridden transforms take effect cleanly.
    """

    def __init__(self, *, sign_pattern, **kwargs):
        kwargs['mode'] = 'signed'
        super().__init__(**kwargs)
        self._sign_pattern = sign_pattern  # dict: 'g_HP'/'g_LP'/'g_HP_dyn'/'g_LP_dyn' -> str

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Mirror the parent forward transform but apply per-gain sign
        # constraints to slots 8, 9, 11, 12. Slot 10 (sigma_dyn) and the
        # first 8 slots are unchanged from parent behaviour.
        # First 7 columns: DoG basis transforms; col 7 = softplus(sigma_AF).
        x = parameters[:, 0][:, tf.newaxis]
        y = parameters[:, 1][:, tf.newaxis]
        sd = tf.math.softplus(parameters[:, 2][:, tf.newaxis])
        baseline = parameters[:, 3][:, tf.newaxis]
        amplitude = parameters[:, 4][:, tf.newaxis]
        srf_amp = tf.math.softplus(parameters[:, 5][:, tf.newaxis])
        srf_size = tf.math.softplus(parameters[:, 6][:, tf.newaxis])
        sigma_AF = tf.math.softplus(parameters[:, 7][:, tf.newaxis])

        g_hp = _gain_forward(parameters[:, 8][:, tf.newaxis],
                             self._sign_pattern['g_HP'])
        g_lp = _gain_forward(parameters[:, 9][:, tf.newaxis],
                             self._sign_pattern['g_LP'])
        sigma_dyn = tf.math.softplus(parameters[:, 10][:, tf.newaxis])
        g_hp_dyn = _gain_forward(parameters[:, 11][:, tf.newaxis],
                                 self._sign_pattern['g_HP_dyn'])
        g_lp_dyn = _gain_forward(parameters[:, 12][:, tf.newaxis],
                                 self._sign_pattern['g_LP_dyn'])

        encoding = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_hp, g_lp, sigma_dyn, g_hp_dyn, g_lp_dyn,
        ], axis=1)

        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf:])
            return tf.concat([encoding, hrf_pars], axis=1)
        return encoding

    @tf.function
    def _transform_parameters_backward(self, parameters):
        x = parameters[:, 0][:, tf.newaxis]
        y = parameters[:, 1][:, tf.newaxis]
        sd = tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis])
        baseline = parameters[:, 3][:, tf.newaxis]
        amplitude = parameters[:, 4][:, tf.newaxis]
        srf_amp = tfp.math.softplus_inverse(
            parameters[:, 5][:, tf.newaxis])
        srf_size = tfp.math.softplus_inverse(
            parameters[:, 6][:, tf.newaxis])
        sigma_AF = tfp.math.softplus_inverse(
            parameters[:, 7][:, tf.newaxis])

        g_hp = _gain_backward(parameters[:, 8][:, tf.newaxis],
                              self._sign_pattern['g_HP'])
        g_lp = _gain_backward(parameters[:, 9][:, tf.newaxis],
                              self._sign_pattern['g_LP'])
        sigma_dyn = tfp.math.softplus_inverse(
            parameters[:, 10][:, tf.newaxis])
        g_hp_dyn = _gain_backward(parameters[:, 11][:, tf.newaxis],
                                  self._sign_pattern['g_HP_dyn'])
        g_lp_dyn = _gain_backward(parameters[:, 12][:, tf.newaxis],
                                  self._sign_pattern['g_LP_dyn'])

        encoding = tf.concat([
            x, y, sd, baseline, amplitude, srf_amp, srf_size,
            sigma_AF, g_hp, g_lp, sigma_dyn, g_hp_dyn, g_lp_dyn,
        ], axis=1)

        if self.flexible_hrf_parameters:
            n_hrf = len(self.hrf_model.parameter_labels)
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf:])
            return tf.concat([encoding, hrf_pars], axis=1)
        return encoding


# ---------------------------------------------------------------------------
# Sign-class encoding & directory naming.
# ---------------------------------------------------------------------------

def class_dirname(sus_hp: str, sus_lp: str,
                  dyn_hp: str, dyn_lp: str) -> str:
    """Convert the 4-tuple of sign args into the output directory name."""
    if (sus_hp, sus_lp, dyn_hp, dyn_lp) == ('free', 'free', 'free', 'free'):
        return 'signed-control'
    return f'sus-{sus_hp}-{sus_lp}_dyn-{dyn_hp}-{dyn_lp}'


def get_ring_positions():
    keys = ['upper right', 'upper left', 'lower left', 'lower right']
    return np.array(
        [list(distractor_locations[k]) for k in keys], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def build_data_and_paradigm(
    sub: Subject,
    resolution: int = 50, grid_radius: float = 5.0,
):
    """Load BOLD + FULL paradigm + condition_indicator + dynamic_indicator.

    Always loads dynamic_indicator (DoG-dyn-v3 needs it). Returns the
    arrays plus a list of :class:`cv_helpers.RunMeta` describing the
    run-chunk boundaries.
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
        if dyn.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - dyn.shape[0], dyn.shape[1]),
                           dtype=np.float32)
            dyn = np.vstack([dyn, pad])
        elif dyn.shape[0] > n_T_run:
            dyn = dyn[:n_T_run]
        dyn_indicator_chunks.append(dyn)

        bold_chunks.append(data)
        paradigm_chunks.append(par_run_flat)
        cond_indicator_chunks.append(cond_indicator)
        run_meta.append(RunMeta(
            session=session, run=run, hp=hp,
            start=cursor, n_T=n_T_run,
        ))
        cursor += n_T_run

    bold = np.vstack(bold_chunks)
    paradigm_full = np.vstack(paradigm_chunks)
    condition_indicator = np.vstack(cond_indicator_chunks)
    dynamic_indicator = np.vstack(dyn_indicator_chunks)

    print(f'Loaded BOLD: shape {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}, '
          f'dynamic_indicator {dynamic_indicator.shape}')
    return (
        bold, paradigm_full, condition_indicator, dynamic_indicator,
        grid_coordinates, masker, run_meta,
    )


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


def make_init_pars(prf_pars: pd.DataFrame,
                   voxel_mask: np.ndarray,
                   sign_pattern: dict[str, str]) -> pd.DataFrame:
    """Initialise per-voxel + shared parameters for DoG-dyn-v3.

    The init values for sign-constrained gains are picked so that the
    backward transform produces a small valid raw value (e.g. ≈0.05 in
    usable space). For 'zero' gains we set 0 (the model also clamps).
    For 'free' we use 0 as init.
    """
    init_cols = ['x', 'y', 'sd', 'baseline', 'amplitude',
                 'srf_amplitude', 'srf_size']
    missing = [c for c in init_cols if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model is missing DoG params {missing}. '
            'Use model 4 for DoG init.')
    init_pars = prf_pars.loc[voxel_mask, init_cols].copy()

    init_pars['sigma_AF'] = 2.0
    init_pars['sigma_dyn'] = 0.5

    # Initial usable-space values per gain — picked to match each gain's
    # sign constraint with a small magnitude so the optimiser starts
    # near zero but on the correct side.
    def _init_usable(sign: str) -> float:
        if sign == 'plus':
            return 0.10
        if sign == 'minus':
            return -0.10
        # 'zero' and 'free' both start at 0.
        return 0.0

    init_pars['g_HP'] = _init_usable(sign_pattern['g_HP'])
    init_pars['g_LP'] = _init_usable(sign_pattern['g_LP'])
    init_pars['g_HP_dyn'] = _init_usable(sign_pattern['g_HP_dyn'])
    init_pars['g_LP_dyn'] = _init_usable(sign_pattern['g_LP_dyn'])

    return init_pars


def build_model(*, grid_coords, paradigm,
                condition_indicator, dynamic_indicator,
                ring_positions, hrf_model, sign_pattern):
    """Build a FactorialDoGDynV3 (signed mode) with the given sign pattern."""
    return FactorialDoGDynV3(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        ring_positions=ring_positions,
        sign_pattern=sign_pattern,
    )


def fixed_pars_for(sign_pattern: dict[str, str]) -> list[str] | None:
    """Return the list of gain-names whose sign is 'zero' (so we fix them).

    The model already clamps zero-signed gains to 0 in the forward
    transform, but we also pass them through ``fixed_pars`` so the
    optimiser doesn't waste effort updating their (irrelevant) raw
    variables.
    """
    fixed = [name for name, sign in sign_pattern.items() if sign == 'zero']
    return fixed if fixed else None


def shared_pars_for(sign_pattern: dict[str, str]) -> list[str]:
    """The optimiser-shared parameters that are NOT zero-fixed.

    σ_AF and σ_dyn are always shared. Each gain is shared unless it's
    'zero' (in which case it's also in fixed_pars and the fitter would
    raise on the overlap).
    """
    keep = ['sigma_AF', 'sigma_dyn']
    for gain_name in ('g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn'):
        if sign_pattern[gain_name] != 'zero':
            keep.append(gain_name)
    return keep


# ---------------------------------------------------------------------------
# Main: 1 (subject, ROI, sign-class) job, 4 folds inside.
# ---------------------------------------------------------------------------

def main(subject: int,
         bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V3AB',
         sus_hp_sign: str = 'minus',
         sus_lp_sign: str = 'zero',
         dyn_hp_sign: str = 'minus',
         dyn_lp_sign: str = 'zero',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         model_label: int = 4,
         max_voxels: int | None = None,
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         output_subdir: str | None = None):

    for s in (sus_hp_sign, sus_lp_sign, dyn_hp_sign, dyn_lp_sign):
        if s not in SIGN_CHOICES:
            raise ValueError(
                f'sign args must be one of {SIGN_CHOICES} (got {s!r}).')

    sign_pattern = {
        'g_HP': sus_hp_sign,
        'g_LP': sus_lp_sign,
        'g_HP_dyn': dyn_hp_sign,
        'g_LP_dyn': dyn_lp_sign,
    }
    cls_dir = class_dirname(sus_hp_sign, sus_lp_sign,
                            dyn_hp_sign, dyn_lp_sign)

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        output_subdir = f'af_prf_cv_factorial/{cls_dir}'
    out_dir = (bids_folder / 'derivatives' / output_subdir
               / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | class={cls_dir} ==')
    print(f'   sign pattern: {sign_pattern}')

    # 1) Load BOLD + paradigm + condition_indicator + dynamic_indicator.
    (bold, paradigm, condition_indicator, dynamic_indicator,
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

    init_pars_template = make_init_pars(prf_pars, voxel_mask, sign_pattern)
    ring_positions = get_ring_positions()
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)

    fixed_pars = fixed_pars_for(sign_pattern)
    shared_pars = shared_pars_for(sign_pattern)

    # 3) Loop over CV folds (4 held-out HP conditions).
    shared_pars_per_fold: list[dict] = []
    cv_r2_per_fold: list[np.ndarray] = []
    train_r2_per_fold: list[np.ndarray] = []
    fit_pars_per_fold: list[pd.DataFrame] = []
    train_run_meta_per_fold: list[list[tuple[int, int, str]]] = []
    held_run_meta_per_fold: list[list[tuple[int, int, str]]] = []

    for cv_fold in range(4):
        held_out = held_out_condition(cv_fold)
        print()
        print(f'=== CV fold {cv_fold} : held-out = {held_out} ===')

        split = restrict_to_train_condition(
            bold=bold, paradigm=paradigm,
            condition_indicator=condition_indicator,
            run_meta=run_meta, held_out_cond_idx=cv_fold,
            dynamic_indicator=dynamic_indicator,
        )
        train, held = split['train'], split['held']
        print(summarize_split(run_meta, held_out))

        bold_train_sub = pd.DataFrame(train['bold'][:, voxel_mask])
        bold_held_sub = pd.DataFrame(held['bold'][:, voxel_mask])

        model_train = build_model(
            grid_coords=grid_coords,
            paradigm=train['paradigm'],
            condition_indicator=train['condition_indicator'],
            dynamic_indicator=train['dynamic_indicator'],
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

        # All shared-par columns we report (always include all 6 ALL_SHARED_PARS;
        # zero-fixed gains will be 0 because the forward transform clamps them).
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
                        dyn_hp_sign, dyn_lp_sign),
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
        'model_name': 'dog-dyn-v3-factorial',
        'cv_folds': list(range(4)),
        'cv_held_out_conditions': list(CONDITIONS),
        'train_run_meta_per_fold': train_run_meta_per_fold,
        'held_run_meta_per_fold': held_run_meta_per_fold,
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
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--model-label', type=int, default=4)
    parser.add_argument('--max-voxels', type=int, default=0,
                        help='Cap on voxels (0 = no cap, default).')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0)
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
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
    )
