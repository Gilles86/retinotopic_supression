"""Cross-validated DoG-Klein-shift (6-σ) fits — SHIFT arm of shift-vs-gain.

Twin of :mod:`retsupp.modeling.fit_af_prf_cv_v2` (the GAIN arm).  The two
scripts share EVERYTHING that is model-agnostic — data loading, voxel
selection, fold geometry, scoring — by importing those pieces directly
from CV-v2 and :mod:`retsupp.modeling.cv_helpers`.  The ONLY difference is
the forward model:

* GAIN arm  : ``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target_sharedSigma_factorial``
              with all five gains FREE (multiplicative AF modulation).
* SHIFT arm : ``DoGKleinShift_v3_target_6sigma`` — NO gains; per-ring
              "strength" is encoded purely by the 6 shared σ's
              (sigma_{HP,LP}_{sus,dyn,T}).  Klein's Gaussian-product
              precision-weighted mean shifts each voxel's DoG center +
              surround toward the attended ring.

Fairness constraints (must match the GAIN arm exactly)
------------------------------------------------------
1. **Identical voxel selection.**  We import and call CV-v2's
   :func:`choose_voxel_mask`, which defaults to the GMM ``p_signal``
   selector (``--p-signal-thr 0.5``, matching the non-CV klein best-fit
   voxels) plus the same ``--max-voxels 500`` top-by-r² ranking, so both
   arms fit the IDENTICAL voxel set per (subject, ROI).
2. **Identical (fixed canonical) HRF.**  Both arms build a single fixed
   ``SPMHRFModel(tr=..., delay=4.5, dispersion=0.75)`` and do NOT fit the
   HRF.  The klein class is constructed with
   ``flexible_hrf_parameters=False`` (its default) so it carries no HRF
   parameters in the fit — exactly mirroring CV-v2's ``build_model``,
   which also leaves ``flexible_hrf_parameters`` at its False default.
   Neither arm passes any HRF parameter to ``fixed_pars``.
3. **Identical fold geometry / scoring / iterations.**  Same
   :func:`split_by_condition_with_target`, :func:`per_voxel_r2`,
   :func:`held_out_condition`, and ``max_n_iterations`` across arms.

CV scheme: leave-one-CONDITION-out, 4 folds (one per HP-distractor ring).
Each invocation processes ONE (subject, ROI) and loops over all 4 folds.

Output
------
``derivatives/af_prf_cv_shiftvsgain/shift/sub-XX/
    sub-XX_roi-{ROI}_cv-fits.pkl``

Same payload schema as CV-v2 (so ``aggregate_shiftvsgain_cv.py`` reads
both trees with one loader), minus the gain/sign bookkeeping.  Canonical
output is the TSV/JSON triple written by ``cv_helpers.write_cv_tsvs``
(``sub-XX_roi-{ROI}_cv-r2.tsv`` / ``_cv-params.tsv`` / ``_meta.json``);
the pickle is kept as a convenience side-product.

Usage
-----
``python -m retsupp.modeling.fit_klein_shift_cv 3 --roi V3AB \\
    --bids-folder /shares/zne.uzh/gdehol/ds-retsupp \\
    --max-voxels 500 --max-n-iterations 1500``
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter

# Reusable, model-agnostic pieces imported straight from the GAIN arm so
# the two scripts cannot drift apart on data loading / voxel selection /
# fold geometry.
from retsupp.modeling.fit_af_prf_cv_v2 import (
    build_data_and_paradigm,
    choose_voxel_mask,
    get_ring_positions,
    split_by_condition_with_target,
)
from retsupp.modeling.cv_helpers import (
    CONDITIONS,
    held_out_condition,
    per_voxel_r2,
    summarize_split,
    write_cv_tsvs,
)
from retsupp.modeling.local_models import DoGKleinShift_v3_target_6sigma
from retsupp.utils.data import Subject


# The 6 shared σ's — ALL shared across voxels (no gains in this model).
SHARED_PARS = ['sigma_HP_sus', 'sigma_LP_sus',
               'sigma_HP_dyn', 'sigma_LP_dyn',
               'sigma_HP_T',   'sigma_LP_T']

# Per-voxel init columns (DoG layout from the m4 mean model).
INIT_COLS = ['x', 'y', 'sd', 'baseline', 'amplitude',
             'srf_amplitude', 'srf_size']


def make_init_pars(prf_pars: pd.DataFrame,
                   voxel_mask: np.ndarray,
                   sigma_init: float) -> pd.DataFrame:
    """Per-voxel DoG init + the 6 shared σ's, all initialised at ``sigma_init``.

    No gains anywhere — the klein model has none.
    """
    missing = [c for c in INIT_COLS if c not in prf_pars.columns]
    if missing:
        raise RuntimeError(
            f'Mean model is missing DoG params {missing}. '
            'Use model 4 for DoG init.')
    init_pars = prf_pars.loc[voxel_mask, INIT_COLS].copy()
    init_pars = init_pars.reset_index(drop=True)
    for s in SHARED_PARS:
        init_pars[s] = sigma_init
    return init_pars


def build_model(*, grid_coords, paradigm,
                condition_indicator, dynamic_indicator,
                target_indicator,
                ring_positions, hrf_model):
    """Construct the Klein-shift 6-σ model for one CV fold.

    NOTE: ``flexible_hrf_parameters`` is left at its False default — the
    HRF is FIXED canonical, matching the GAIN arm.  No ``sign_pattern``,
    no gains.
    """
    return DoGKleinShift_v3_target_6sigma(
        grid_coordinates=grid_coords,
        paradigm=paradigm,
        hrf_model=hrf_model,
        condition_indicator=condition_indicator,
        dynamic_indicator=dynamic_indicator,
        target_indicator=target_indicator,
        ring_positions=ring_positions,
    )


def main(subject: int,
         bids_folder: str = '/data/ds-retsupp',
         roi: str = 'V3AB',
         resolution: int = 50,
         max_n_iterations: int = 1500,
         r2_thr: float = 0.05,
         model_label: int = 4,
         max_voxels: int | None = None,
         learning_rate: float = 0.01,
         grid_radius: float = 5.0,
         sigma_init: float = 2.0,
         use_psignal: bool = True,
         p_signal_thr: float = 0.5,
         output_subdir: str | None = None):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder)
    if output_subdir is None:
        output_subdir = 'af_prf_cv_shiftvsgain/shift'
    out_dir = (bids_folder / 'derivatives' / output_subdir
               / f'sub-{subject:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'== sub-{subject:02d} | roi={roi} | SHIFT (klein 6sigma) ==')
    print(f'   sigma init: {sigma_init} (all 6 sigmas)')

    # 1) Load BOLD + paradigm + indicators (incl. target_indicator).
    #    Reused verbatim from the GAIN arm.
    (bold, paradigm, condition_indicator,
     dynamic_indicator, target_indicator,
     grid_coords, _masker, run_meta) = build_data_and_paradigm(
        sub, resolution=resolution, grid_radius=grid_radius,
    )

    # 2) Restrict to ROI voxels — IDENTICAL selector + ranking as GAIN arm
    #    (choose_voxel_mask is shared, so both arms select the same voxels).
    prf_pars = sub.get_prf_parameters_volume(model=model_label,
                                             return_images=False)
    if not isinstance(prf_pars, pd.DataFrame):
        prf_pars = pd.DataFrame(prf_pars)
    voxel_mask, descr = choose_voxel_mask(
        sub, roi, prf_pars,
        use_psignal=use_psignal, p_signal_thr=p_signal_thr,
        r2_thr=r2_thr, model_label=model_label, max_voxels=max_voxels)
    print(f'{descr}: {voxel_mask.sum()} voxels')

    init_pars_template = make_init_pars(prf_pars, voxel_mask,
                                        sigma_init=sigma_init)
    ring_positions = get_ring_positions()
    # FIXED canonical HRF — identical to the GAIN arm.
    hrf_model = SPMHRFModel(tr=sub.get_tr(session=1, run=1),
                            delay=4.5, dispersion=0.75)

    print(f'   shared_pars: {SHARED_PARS}')

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
        )

        fitter = ParameterFitter(model_train, bold_train_sub,
                                 train['paradigm'])
        refined_pars = fitter.refine_baseline_and_amplitude(
            init_pars_template.copy(), l2_alpha=1e-3)

        fit_pars = fitter.fit(
            init_pars=refined_pars,
            max_n_iterations=max_n_iterations,
            shared_pars=SHARED_PARS,
            learning_rate=learning_rate,
        )
        train_r2 = fitter.get_rsq(fit_pars)
        train_r2 = (train_r2.values if hasattr(train_r2, 'values')
                    else np.asarray(train_r2))
        print(f'Train R2 mean: {np.nanmean(train_r2):.4f}')

        # Held-out prediction.
        model_held = build_model(
            grid_coords=grid_coords,
            paradigm=held['paradigm'],
            condition_indicator=held['condition_indicator'],
            dynamic_indicator=held['dynamic_indicator'],
            target_indicator=held['target_indicator'],
            ring_positions=ring_positions,
            hrf_model=hrf_model,
        )
        pred_held = model_held.predict(paradigm=held['paradigm'],
                                       parameters=fit_pars)
        pred_held_arr = (pred_held.values if hasattr(pred_held, 'values')
                         else np.asarray(pred_held))
        cv_r2 = per_voxel_r2(bold_held_sub.values, pred_held_arr)
        print(f'Held-out CV-R2 (n={cv_r2.size}) | '
              f'mean={np.nanmean(cv_r2):.4f} | '
              f'median={np.nanmedian(cv_r2):.4f}')

        shared_dict = fit_pars[SHARED_PARS].iloc[0].to_dict()

        shared_pars_per_fold.append(shared_dict)
        cv_r2_per_fold.append(cv_r2)
        train_r2_per_fold.append(train_r2)
        fit_pars_per_fold.append(fit_pars)
        train_run_meta_per_fold.append(
            [(rm.session, rm.run, rm.hp) for rm in train['run_meta']])
        held_run_meta_per_fold.append(
            [(rm.session, rm.run, rm.hp) for rm in held['run_meta']])

    # 4) Save one pickle per (subject, ROI).  Schema mirrors CV-v2.
    out_pkl = out_dir / f'sub-{subject:02d}_roi-{roi}_cv-fits.pkl'
    payload = {
        'shared_pars_per_fold': shared_pars_per_fold,
        'cv_r2_per_fold': cv_r2_per_fold,
        'train_r2_per_fold': train_r2_per_fold,
        'fit_pars_per_fold': fit_pars_per_fold,
        'shared_par_labels': SHARED_PARS,
        'fitted_shared_pars': SHARED_PARS,
        'fixed_pars': [],
        'voxel_mask_indices': np.where(voxel_mask)[0],
        'roi': roi,
        'resolution': resolution,
        'subject': subject,
        'paradigm_type': 'full',
        'grid_radius': grid_radius,
        'model_name': 'dog-klein-shift-6sigma',
        'cv_folds': list(range(4)),
        'cv_held_out_conditions': list(CONDITIONS),
        'train_run_meta_per_fold': train_run_meta_per_fold,
        'held_run_meta_per_fold': held_run_meta_per_fold,
        'sigma_init': sigma_init,
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Saved: {out_pkl}')

    # 5) Canonical TSV output (read by aggregate_shiftvsgain_cv).
    voxel_ids = np.where(voxel_mask)[0]
    per_voxel_par_cols = INIT_COLS  # x, y, sd, baseline, amplitude, srf_*
    tsv_paths = write_cv_tsvs(
        out_dir,
        subject=subject, roi=roi, model='shift',
        voxel_ids=voxel_ids,
        cv_r2_per_fold=cv_r2_per_fold,
        train_r2_per_fold=train_r2_per_fold,
        fit_pars_per_fold=fit_pars_per_fold,
        shared_pars_per_fold=shared_pars_per_fold,
        shared_par_labels=SHARED_PARS,
        per_voxel_par_cols=per_voxel_par_cols,
        train_run_meta_per_fold=train_run_meta_per_fold,
        held_run_meta_per_fold=held_run_meta_per_fold,
        selector='psignal' if use_psignal else 'r2',
        p_signal_thr=p_signal_thr if use_psignal else -1.0,
        extra_meta=dict(
            model_name='dog-klein-shift-6sigma',
            sigma_init=sigma_init,
            max_voxels=max_voxels,
        ),
    )
    print(f'Saved TSVs: {tsv_paths["r2"]}, {tsv_paths["params"]}, '
          f'{tsv_paths["meta"]}')

    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-folder', default='/data/ds-retsupp')
    parser.add_argument('--roi', default='V3AB')
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--max-n-iterations', type=int, default=1500)
    parser.add_argument('--r2-thr', type=float, default=0.05)
    parser.add_argument('--model-label', type=int, default=4)
    parser.add_argument('--max-voxels', type=int, default=0,
                        help='Cap on voxels (0 = no cap, default).')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--grid-radius', type=float, default=5.0)
    parser.add_argument('--sigma-init', type=float, default=2.0,
                        help='Initial value for all 6 sigmas (default 2.0).')
    parser.add_argument('--use-psignal', dest='use_psignal',
                        action='store_true', default=True,
                        help='Use the GMM p_signal voxel selector '
                             '(default ON; matches the non-CV klein fit).')
    parser.add_argument('--no-psignal', dest='use_psignal',
                        action='store_false',
                        help='Fall back to the legacy r²-threshold selector.')
    parser.add_argument('--p-signal-thr', type=float, default=0.5,
                        help='p_signal posterior threshold (default 0.5).')
    parser.add_argument('--output-subdir', default=None)
    args = parser.parse_args()
    main(
        subject=args.subject,
        bids_folder=args.bids_folder,
        roi=args.roi,
        resolution=args.resolution,
        max_n_iterations=args.max_n_iterations,
        r2_thr=args.r2_thr,
        model_label=args.model_label,
        max_voxels=None if args.max_voxels == 0 else args.max_voxels,
        learning_rate=args.learning_rate,
        grid_radius=args.grid_radius,
        sigma_init=args.sigma_init,
        use_psignal=args.use_psignal,
        p_signal_thr=args.p_signal_thr,
        output_subdir=args.output_subdir,
    )
