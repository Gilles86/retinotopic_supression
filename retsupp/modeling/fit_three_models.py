"""Fit one of three Gaussian-PRF AF formulations on retsupp BOLD data.

This script is the comparison-sweep driver for three formulations of
attention-field-modulated PRFs on a *Gaussian* (not DoG) backbone:

- ``drive``      → :class:`local_models.GaussianAFDriveModelWithHRF`
- ``analytical`` → :class:`local_models.GaussianAFAnalyticalShiftModelWithHRF`
- ``numerical``  → :class:`local_models.GaussianAFNumericalShiftModelWithHRF`

All three share the same shared-parameter set (8 parameters)::

    sigma_AF, g_HP, g_LP, sigma_dyn, g_HP_dyn, g_LP_dyn,
    g_T_dyn, sigma_T_dyn

with ``sigma_T_dyn := sigma_dyn`` (sharedSigma constraint).

Voxel parameters are initialised from the model-1 (Gaussian PRF) mean
fit (``x``, ``y``, ``sd``, ``baseline``, ``amplitude``).

The full paradigm (bar + distractor disks at the 4 ring locations) is
used. Output goes to::

    derivatives/af_three_models/{drive,analytical,numerical}/sub-XX/
        sub-XX_roi-{ROI}_fit.pkl
        sub-XX_roi-{ROI}_pars.tsv

Usage
-----
``python -m retsupp.modeling.fit_three_models 28 --roi V3AB --model-version drive``
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
from braincoder.optimize import ParameterFitter
from retsupp.modeling.local_models import (
    GaussianAFAnalyticalShiftModelWithHRF,
    GaussianAFDriveModelWithHRF,
    GaussianAFNumericalShiftModelWithHRF,
)
from retsupp.utils.data import (
    Subject,
    distractor_locations,
)


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']

MODEL_CLASSES = {
    'drive': GaussianAFDriveModelWithHRF,
    'analytical': GaussianAFAnalyticalShiftModelWithHRF,
    'numerical': GaussianAFNumericalShiftModelWithHRF,
}


def get_ring_positions():
    """Return the (n_C, 2) array of ring positions matching CONDITIONS order."""
    keys = ['upper right', 'upper left', 'lower left', 'lower right']
    return np.array(
        [list(distractor_locations[k]) for k in keys], dtype=np.float32)


def build_data_and_paradigm(sub: Subject, resolution: int = 50,
                            grid_radius: float = 5.0,
                            with_target: bool = True):
    """Load cleaned BOLD + paradigm + indicators (full paradigm).

    Returns
    -------
    bold_df : pd.DataFrame, shape (T, V)
    paradigm_full : np.ndarray, shape (T, G)
    condition_indicator : np.ndarray, shape (T, 4)  one-hot HP per TR
    dynamic_indicator : np.ndarray, shape (T, 4)    per-ring distractor on-fraction
    grid_coordinates : np.ndarray, shape (G, 2)
    masker : NiftiMasker (fit to bold mask)
    target_indicator : np.ndarray, shape (T, 4) or None
    """
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

    bold_chunks = []
    paradigm_chunks = []
    cond_indicator_chunks = []
    dyn_indicator_chunks = []
    tgt_indicator_chunks = [] if with_target else None
    masker.fit(bold_mask)

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

        dyn_indicator = sub.get_dynamic_indicator(
            session=session, run=run, oversampling=1,
        ).astype(np.float32)
        if dyn_indicator.shape[0] < n_T_run:
            pad = np.zeros((n_T_run - dyn_indicator.shape[0],
                            dyn_indicator.shape[1]), dtype=np.float32)
            dyn_indicator = np.vstack([dyn_indicator, pad])
        elif dyn_indicator.shape[0] > n_T_run:
            dyn_indicator = dyn_indicator[:n_T_run]

        if with_target:
            tgt_indicator = sub.get_target_indicator(
                session=session, run=run, oversampling=1,
            ).astype(np.float32)
            if tgt_indicator.shape[0] < n_T_run:
                pad = np.zeros((n_T_run - tgt_indicator.shape[0],
                                tgt_indicator.shape[1]), dtype=np.float32)
                tgt_indicator = np.vstack([tgt_indicator, pad])
            elif tgt_indicator.shape[0] > n_T_run:
                tgt_indicator = tgt_indicator[:n_T_run]
            tgt_indicator_chunks.append(tgt_indicator)

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

    print(f'Loaded BOLD: {bold.shape}, paradigm {paradigm_full.shape}, '
          f'condition_indicator {condition_indicator.shape}, '
          f'dynamic_indicator {dynamic_indicator.shape}')
    if with_target:
        print(f'target_indicator {target_indicator.shape}')

    return (
        pd.DataFrame(bold),
        paradigm_full,
        condition_indicator,
        dynamic_indicator,
        grid_coordinates,
        masker,
        target_indicator,
    )


def select_roi_voxels(sub: Subject, roi: str, prf_pars: pd.DataFrame,
                       r2_thr: float = 0.05):
    """Boolean voxel mask: ROI ∧ (mean-model R² > thr)."""
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
    r2_mask = (prf_pars['r2'].values > r2_thr) & roi_arr
    return r2_mask


def main(subject: int, bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V3AB',
         model_version: str = 'drive',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         max_voxels: int | None = 500,
         mode: str = 'signed',
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         output_subdir: str | None = None,
         sigma_af_init: float = 2.0,
         sigma_dyn_init: float = 2.0,
         with_target: bool = True,
         g_t_dyn_init: float = 0.0):
    """Top-level fit driver."""
    if model_version not in MODEL_CLASSES:
        raise ValueError(
            f"model_version must be one of {list(MODEL_CLASSES)}, "
            f"got {model_version!r}")

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)

    if output_subdir is None:
        output_subdir = f'af_three_models/{model_version}'
    out_dir = bids_folder / 'derivatives' / output_subdir / f'sub-{subject:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | mode={mode} | '
          f'model-version={model_version} | with_target={with_target} '
          f'(Gaussian PRF, three-models AF sweep) ==')

    # 1) Load BOLD + paradigm + indicators.
    (bold_df, paradigm, condition_indicator, dynamic_indicator,
     grid_coords, masker, target_indicator) = build_data_and_paradigm(
        sub, resolution=resolution, grid_radius=grid_radius,
        with_target=with_target,
    )

    # 2) Restrict to ROI voxels with decent mean-model PRF R².
    # Use model 1 (Gaussian PRF) as the canonical Gaussian baseline.
    prf_pars = sub.get_prf_parameters_volume(model=1, return_images=False)
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

    bold_sub = bold_df.loc[:, voxel_mask].copy()

    # 3) Initialise from model 1 (Gaussian PRF). 5 voxel params.
    init_cols = ['x', 'y', 'sd', 'baseline', 'amplitude']
    missing = [c for c in init_cols if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model 1 is missing Gaussian params {missing}.')
    init_pars = prf_pars.loc[voxel_mask, init_cols].copy()

    # AF inits.
    init_pars['sigma_AF'] = sigma_af_init
    init_pars['g_HP'] = 0.0 if mode == 'signed' else 0.30
    init_pars['g_LP'] = 0.0 if mode == 'signed' else 0.10
    init_pars['sigma_dyn'] = sigma_dyn_init
    init_pars['g_HP_dyn'] = 0.0 if mode == 'signed' else 0.10
    init_pars['g_LP_dyn'] = 0.0 if mode == 'signed' else 0.10
    init_pars['g_T_dyn'] = g_t_dyn_init
    # sigma_T_dyn is tied to sigma_dyn at every forward pass; but we
    # still init it for the optimiser's raw variable.
    init_pars['sigma_T_dyn'] = init_pars['sigma_dyn']

    shared_pars = ['sigma_AF', 'g_HP', 'g_LP',
                   'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn',
                   'g_T_dyn', 'sigma_T_dyn']

    # 4) Build the model.
    ring_positions = get_ring_positions()
    print('Ring positions:\n', ring_positions)

    tr_orig = sub.get_tr(session=1, run=1)
    hrf_model = SPMHRFModel(tr=tr_orig, delay=4.5, dispersion=0.75)

    ModelCls = MODEL_CLASSES[model_version]
    print(f'Using model class: {ModelCls.__name__}')

    model = ModelCls(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        target_indicator=target_indicator,
        ring_positions=ring_positions,
        mode=mode,
    )

    fitter = ParameterFitter(model, bold_sub, paradigm)

    # 5) Refine baseline/amplitude given current AF params.
    refined_pars = fitter.refine_baseline_and_amplitude(
        init_pars, l2_alpha=1e-3)

    # 6) Joint fit.
    fit_pars = fitter.fit(
        init_pars=refined_pars,
        max_n_iterations=max_n_iterations,
        shared_pars=shared_pars,
        learning_rate=learning_rate,
    )

    r2 = fitter.get_rsq(fit_pars) if hasattr(fitter, 'get_rsq') else fitter.r2
    print(f'Mean R²: {np.nanmean(r2):.4f}')
    print('Shared parameters:')
    print(fit_pars[shared_pars].iloc[0])

    fit_pars['r2'] = r2.values if hasattr(r2, 'values') else r2

    # 7) Save outputs.
    out_tsv = out_dir / f'sub-{subject:02d}_roi-{roi}_pars.tsv'
    fit_pars.to_csv(out_tsv, sep='\t')
    out_pkl = out_dir / f'sub-{subject:02d}_roi-{roi}_fit.pkl'
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
            'voxel_kernel': 'Gaussian',
            'model_version': model_version,
            'with_target': with_target,
        }, f)
    print(f'Saved: {out_tsv}')
    print(f'Saved: {out_pkl}')

    return fit_pars, r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int,
                        help='Subject ID (e.g. 28 -> sub-28).')
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V3AB',
                        help='Retinotopic ROI to fit (default V3AB).')
    parser.add_argument('--model-version',
                        choices=list(MODEL_CLASSES), default='drive',
                        help="Which AF formulation: drive (Model A, "
                             "amplitude AF+), analytical (Model B, "
                             "precision-weighted-mean shift), or "
                             "numerical (Model C, AF+ then refit "
                             "Gaussian to COM).")
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--max-voxels', type=int, default=500,
                        help='Cap on voxels. 0 = no cap.')
    parser.add_argument('--mode',
                        choices=['suppression', 'attraction', 'signed'],
                        default='signed')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0)
    parser.add_argument('--output-subdir', default=None)
    parser.add_argument('--sigma-af-init', type=float, default=2.0)
    parser.add_argument('--sigma-dyn-init', type=float, default=2.0)
    parser.add_argument('--with-target', action='store_true',
                        default=True,
                        help='Include the phasic-target term '
                             '(g_T_dyn). Default ON.')
    parser.add_argument('--no-target', dest='with_target',
                        action='store_false',
                        help='Disable the phasic-target term.')
    parser.add_argument('--g-t-dyn-init', type=float, default=0.0)
    args = parser.parse_args()
    main(
        args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        model_version=args.model_version,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        mode=args.mode,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        output_subdir=args.output_subdir,
        sigma_af_init=args.sigma_af_init,
        sigma_dyn_init=args.sigma_dyn_init,
        with_target=args.with_target,
        g_t_dyn_init=args.g_t_dyn_init,
    )
